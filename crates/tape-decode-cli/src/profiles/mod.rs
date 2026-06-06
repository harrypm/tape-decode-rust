//! Decode profile loading: parsing the embedded profile tables and expanding
//! each profile's `format` and `sys_params` `base` inheritance chains into a
//! flat [`DecodeProfile`].

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context as _, Result};
use serde_json::{Map, Value};

use tape_decode::DecodeProfile;

/// Clone an entry's key/value pairs, dropping the `base` inheritance marker.
fn without_base(entry: &Map<String, Value>) -> Map<String, Value> {
    entry
        .iter()
        .filter(|(key, _)| key.as_str() != "base")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Flatten a named `sys_params` definition by walking its `base` inheritance
/// chain, with later (more specific) entries overriding earlier ones. The
/// `base` bookkeeping key is dropped. Results are memoized in `cache`.
fn resolve_sys_params(
    name: &str,
    defs: &HashMap<String, Map<String, Value>>,
    cache: &mut HashMap<String, Map<String, Value>>,
) -> Result<Map<String, Value>> {
    if let Some(cached) = cache.get(name) {
        return Ok(cached.clone());
    }
    let entry = defs
        .get(name)
        .with_context(|| format!("unknown sys_params base: {name}"))?;
    let mut resolved = match entry.get("base").and_then(Value::as_str) {
        Some(base) => resolve_sys_params(base, defs, cache)?,
        None => Map::new(),
    };
    resolved.extend(without_base(entry));
    cache.insert(name.to_string(), resolved.clone());
    Ok(resolved)
}

/// Recursively merge `over` into `base`: object-valued keys are merged
/// key-by-key, while scalars, arrays, and nulls replace wholesale. Used to
/// overlay the nested `decoder_params` overrides of a `format` definition (and
/// of an individual profile) on top of its inherited base.
fn deep_merge(base: &mut Map<String, Value>, over: &Map<String, Value>) {
    for (key, value) in over {
        match (base.get_mut(key), value) {
            (Some(Value::Object(base_obj)), Value::Object(over_obj)) => {
                deep_merge(base_obj, over_obj);
            }
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Flatten a named `format` definition (its `decoder_params`)
/// by walking its `base` inheritance chain, deep-merging each more specific
/// entry over the inherited result. The `base` bookkeeping key is dropped.
/// Results are memoized in `cache`.
fn resolve_format(
    name: &str,
    defs: &HashMap<String, Map<String, Value>>,
    cache: &mut HashMap<String, Map<String, Value>>,
) -> Result<Map<String, Value>> {
    if let Some(cached) = cache.get(name) {
        return Ok(cached.clone());
    }
    let entry = defs
        .get(name)
        .with_context(|| format!("unknown format base: {name}"))?;
    let mut resolved = match entry.get("base").and_then(Value::as_str) {
        Some(base) => resolve_format(base, defs, cache)?,
        None => Map::new(),
    };
    deep_merge(&mut resolved, &without_base(entry));
    cache.insert(name.to_string(), resolved.clone());
    Ok(resolved)
}

/// Parse a named-definitions file (`sys_params.json` / `format_params.json`)
/// into a lookup table keyed by definition name.
fn load_defs(json: &str, kind: &str) -> Result<HashMap<String, Map<String, Value>>> {
    let parsed: Map<String, Value> = serde_json::from_str(json)?;
    let mut defs = HashMap::new();
    for (name, entry) in parsed {
        let Value::Object(obj) = entry else {
            bail!("{kind} entry {name} must be an object");
        };
        defs.insert(name, obj);
    }
    Ok(defs)
}

/// Expand a profile's optional `format` and `sys_params` `base` references
/// against the embedded definition tables. Profiles loaded from a file may also
/// be fully resolved already; in that case their top-level `decoder_params`
/// and `sys_params` fields are left intact.
fn expand_profile(
    key: &str,
    profile: &mut Value,
    sys_param_defs: &HashMap<String, Map<String, Value>>,
    format_defs: &HashMap<String, Map<String, Value>>,
    sys_param_cache: &mut HashMap<String, Map<String, Value>>,
    format_cache: &mut HashMap<String, Map<String, Value>>,
) -> Result<()> {
    let profile_obj = profile
        .as_object_mut()
        .with_context(|| format!("profile {key} must be an object"))?;

    if let Some(format_value) = profile_obj.get("format") {
        let format = format_value
            .as_object()
            .cloned()
            .with_context(|| format!("profile {key} `format` must be an object"))?;
        let format_base = format
            .get("base")
            .and_then(Value::as_str)
            .with_context(|| format!("profile {key} format missing `base`"))?;
        let mut resolved_format = resolve_format(format_base, format_defs, format_cache)?;
        deep_merge(&mut resolved_format, &without_base(&format));
        let decoder_params = resolved_format
            .remove("decoder_params")
            .with_context(|| format!("profile {key} format missing `decoder_params`"))?;
        profile_obj.insert("decoder_params".to_string(), decoder_params);
        let mut decode_options = match resolved_format.remove("decode_options") {
            Some(Value::Object(options)) => options,
            Some(_) => bail!("profile {key} format `decode_options` must be an object"),
            None => Map::new(),
        };
        let profile_decode_options = profile_obj
            .get("decode_options")
            .map(|value| {
                value
                    .as_object()
                    .cloned()
                    .with_context(|| format!("profile {key} `decode_options` must be an object"))
            })
            .transpose()?;
        if let Some(profile_decode_options) = &profile_decode_options {
            deep_merge(&mut decode_options, profile_decode_options);
        }
        profile_obj.insert("decode_options".to_string(), Value::Object(decode_options));
    }

    let sys_params = profile_obj
        .get("sys_params")
        .and_then(Value::as_object)
        .cloned()
        .with_context(|| format!("profile {key} missing sys_params object"))?;
    if let Some(base_value) = sys_params.get("base") {
        let base = base_value
            .as_str()
            .with_context(|| format!("profile {key} sys_params `base` must be a string"))?;
        let mut resolved = resolve_sys_params(base, sys_param_defs, sys_param_cache)?;
        resolved.extend(without_base(&sys_params));
        profile_obj.insert("sys_params".to_string(), Value::Object(resolved));
    }

    Ok(())
}

/// Load every decode profile, expanding each profile's `format` and `sys_params`
/// `base` references against the named definitions in `format_params.json` and
/// `sys_params.json`.
fn load_profiles() -> Result<Map<String, Value>> {
    let sys_param_defs = load_defs(include_str!("sys_params.json"), "sys_params")?;
    let format_defs = load_defs(include_str!("format_params.json"), "format")?;

    let mut profiles: Map<String, Value> = serde_json::from_str(include_str!("profiles.json"))?;
    let mut sys_param_cache: HashMap<String, Map<String, Value>> = HashMap::new();
    let mut format_cache: HashMap<String, Map<String, Value>> = HashMap::new();
    for (key, profile) in &mut profiles {
        expand_profile(
            key,
            profile,
            &sys_param_defs,
            &format_defs,
            &mut sys_param_cache,
            &mut format_cache,
        )?;
    }

    Ok(profiles)
}

fn deserialize_profile(profile: Value, source: &str) -> Result<DecodeProfile> {
    serde_json::from_value(profile).with_context(|| format!("failed to load profile {source}"))
}

/// Look up a profile by name and deserialize it into a [`DecodeProfile`].
pub fn load_profile(name: &str) -> Result<DecodeProfile> {
    deserialize_profile(flatten_profile(Some(name), None)?, name)
}

/// Load one profile directly from a JSON file.
pub fn load_profile_file(path: &Path) -> Result<DecodeProfile> {
    let source = format!("file {}", path.display());
    deserialize_profile(flatten_profile(None, Some(path))?, &source)
}

/// Resolve a profile to its flattened JSON object: `base` inheritance chains in
/// `format` and `sys_params` are expanded in place, leaving top-level
/// `decoder_params`, `decode_options`, and `sys_params` with no `base` markers.
pub fn flatten_profile(profile: Option<&str>, profile_file: Option<&Path>) -> Result<Value> {
    match (profile, profile_file) {
        (Some(name), None) => {
            let mut profiles = load_profiles()?;
            profiles
                .remove(name)
                .with_context(|| format!("unknown profile: {name}"))
        }
        (None, Some(path)) => {
            let data = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read profile file {}", path.display()))?;
            let mut profile: Value = serde_json::from_str(&data)
                .with_context(|| format!("failed to parse profile file {}", path.display()))?;
            let sys_param_defs = load_defs(include_str!("sys_params.json"), "sys_params")?;
            let format_defs = load_defs(include_str!("format_params.json"), "format")?;
            let mut sys_param_cache: HashMap<String, Map<String, Value>> = HashMap::new();
            let mut format_cache: HashMap<String, Map<String, Value>> = HashMap::new();
            expand_profile(
                &format!("file {}", path.display()),
                &mut profile,
                &sys_param_defs,
                &format_defs,
                &mut sys_param_cache,
                &mut format_cache,
            )?;
            Ok(profile)
        }
        _ => bail!("exactly one of --profile or --profile-file is required"),
    }
}
