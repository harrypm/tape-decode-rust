//! Command-line front end: argument parsing, profile lookup, and wiring the
//! parsed options into a `DecoderSpec` before running the decode.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context as _, Result};
use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};

use crate::decode::{decode_all, decode_all_mt, MtParams};
use crate::fields_match::{f32_msre, wrapped_u16_msre};
use crate::metadata::{PcmAudioParameters, TbcMetadataFull, VideoParameters};
use crate::os;
use crate::profiles::{flatten_profile, load_profile, load_profile_file};
use crate::reader::{open_source, DecodeReader, SampleFormat};
use crate::writer::DecodeWriter;
use tape_decode::{
    DecodeRequest, DecoderSpec, DropOuts, FieldInfoEntry, FieldOrderAction, NotchFilter,
    WowInterpolation,
};

const DEFAULT_THRESHOLD_P_DDD: f32 = 0.18;
const DEFAULT_HYSTERESIS: f32 = 1.25;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliFieldOrderAction {
    Detect,
    Duplicate,
    Drop,
    None,
}

impl From<CliFieldOrderAction> for FieldOrderAction {
    fn from(value: CliFieldOrderAction) -> Self {
        match value {
            CliFieldOrderAction::Detect => FieldOrderAction::Detect,
            CliFieldOrderAction::Duplicate => FieldOrderAction::Duplicate,
            CliFieldOrderAction::Drop => FieldOrderAction::Drop,
            CliFieldOrderAction::None => FieldOrderAction::None,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliWowInterpolation {
    Linear,
    Quadratic,
    Cubic,
}

impl From<CliWowInterpolation> for WowInterpolation {
    fn from(value: CliWowInterpolation) -> Self {
        match value {
            CliWowInterpolation::Linear => WowInterpolation::Linear,
            CliWowInterpolation::Quadratic => WowInterpolation::Quadratic,
            CliWowInterpolation::Cubic => WowInterpolation::Cubic,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliSampleFormat {
    /// Unsigned 8-bit samples.
    U8,
    /// Signed 8-bit samples.
    S8,
    /// Signed little-endian 16-bit samples.
    S16LE,
    /// Unsigned little-endian 16-bit samples.
    U16LE,
    /// Little-endian 32-bit float samples.
    F32LE,
    /// Mono FLAC stream (decoded to its native bit depth).
    Flac,
}

impl From<CliSampleFormat> for SampleFormat {
    fn from(value: CliSampleFormat) -> Self {
        match value {
            CliSampleFormat::U8 => SampleFormat::U8,
            CliSampleFormat::S8 => SampleFormat::S8,
            CliSampleFormat::S16LE => SampleFormat::S16LE,
            CliSampleFormat::U16LE => SampleFormat::U16LE,
            CliSampleFormat::F32LE => SampleFormat::F32LE,
            CliSampleFormat::Flac => SampleFormat::Flac,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "tape-decode")]
#[command(
    about = "Extracts video from RAW RF captures of colour-under & composite modulated tapes"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Decode an RF capture.
    Decode(DecodeArgs),
    /// Flatten a named profile and write it to a JSON file.
    WriteProfile(WriteProfileArgs),
    /// Compare two decode outputs.
    Compare(CompareArgs),
}

#[derive(Args, Debug)]
#[command(group(
    ArgGroup::new("profile_source")
        .required(true)
        .args(["profile", "profile_file"]),
))]
#[command(group(
    ArgGroup::new("raw_or_chroma")
        .args(["export_raw_tbc"])
        .conflicts_with("chroma_out"),
))]
struct DecodeArgs {
    /// Profile name.
    #[arg(long)]
    profile: Option<String>,
    /// Path to a profile to load instead of an embedded profile.
    #[arg(long, alias = "profile_file")]
    profile_file: Option<PathBuf>,
    /// Input RF sample rate in MHz; Hz, kHz, MHz, M, and k suffixes are accepted.
    #[arg(long, value_parser = parse_frequency)]
    frequency: Option<f64>,
    /// Input sample offset to seek before decoding.
    #[arg(long)]
    offset: Option<u64>,
    /// Input format.
    #[arg(long, value_enum, ignore_case = true, default_value = "u8")]
    input_format: CliSampleFormat,
    /// Allow overwriting outputs.
    #[arg(long)]
    overwrite: bool,
    /// Enable debug-level logging unless RUST_LOG supplies an explicit filter.
    #[arg(long)]
    debug: bool,

    /// Export raw f32 TBC luma.
    #[arg(long)]
    export_raw_tbc: bool,

    /// Luma output path, or `-` to write to standard output.
    #[arg(long)]
    luma_out: PathBuf,
    /// Chroma output path, or `-` to write to standard output.
    #[arg(long)]
    chroma_out: Option<PathBuf>,
    /// Metadata output path
    #[arg(long)]
    metadata_out: Option<PathBuf>,

    /// Apply a chroma trap to the luma path.
    #[arg(long)]
    chroma_trap: bool,
    /// Video EQ sharpening amount; 0 disables sharpening.
    #[arg(long, default_value_t = 0)]
    sharpness: i64,
    /// Apply an RF notch filter at the given frequency in MHz.
    #[arg(long, value_parser = parse_frequency)]
    notch: Option<f64>,
    /// Q factor for the RF notch filter.
    #[arg(long, default_value_t = 10.0)]
    notch_q: f64,
    /// Treat NTSC black level as 0 IRE instead of 7.5 IRE.
    #[arg(long)]
    ntscj: bool,
    /// MAD multiplier used when suppressing wow level-adjustment outliers.
    #[arg(long, default_value_t = 0.1)]
    level_adjust: f32,
    /// Adjust RF IRE0 from measured picture content.
    #[arg(long)]
    ire0_adjust: bool,
    /// Override the profile's RF high-boost multiplier.
    #[arg(long)]
    high_boost: Option<f64>,
    /// Disable differential video demodulation.
    #[arg(long)]
    disable_diff_demod: bool,
    /// Enable FM audio notch filters; omitting the value uses Q=10.0.
    #[arg(long, num_args = 0..=1, default_missing_value = "10.0")]
    fm_audio_notch: Option<f64>,
    /// Apply detected RF DC-offset correction to the decoded video.
    #[arg(long)]
    enable_dc_offset: bool,
    /// Enable nonlinear deemphasis.
    #[arg(long)]
    nldeemp: bool,
    /// Enable sub-deemphasis.
    #[arg(long)]
    subdeemp: bool,
    /// Enable luma Y comb filtering with the given IRE clamp; omitting the value uses 1.5.
    #[arg(long, num_args = 0..=1, default_missing_value = "1.5", default_value = "0.0")]
    y_comb: f32,
    /// Override chroma automatic frequency control; omitting the value enables it.
    #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    cafc: Option<bool>,
    /// Force a chroma track phase instead of using the detected/default phase.
    #[arg(long)]
    track_phase: Option<i64>,
    /// Detect chroma track phase from the decoded signal.
    #[arg(long)]
    detect_chroma_track_phase: bool,
    /// Disable chroma phase correction.
    #[arg(long)]
    disable_phase_correction: bool,
    /// Disable burst-based hsync correction.
    #[arg(long)]
    disable_burst_hsync: bool,
    /// Disable chroma comb filtering.
    #[arg(long)]
    no_comb: bool,
    /// Disable the right-edge hsync zero-crossing refinement.
    #[arg(long)]
    disable_right_hsync: bool,
    /// Divisor for the level-detection pass sample rate.
    #[arg(long, default_value_t = 3)]
    level_detect_divisor: i64,
    /// Override fallback vsync recovery; omitting the value enables it.
    #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    fallback_vsync: Option<bool>,
    /// Relax line-0 recovery checks when sync is difficult to lock.
    #[arg(long)]
    relaxed_line0: bool,
    /// Required confidence percentage for accepting detected field order.
    #[arg(long, default_value_t = 100)]
    field_order_confidence: i64,
    /// Action to take when consecutive fields have the same detected order.
    #[arg(long, value_enum, ignore_case = true)]
    field_order_action: Option<CliFieldOrderAction>,
    /// Reuse previously detected levels until sync issues require recalculation.
    #[arg(long)]
    use_saved_levels: bool,
    /// Skip hsync location refinement after initial pulse detection.
    #[arg(long)]
    skip_hsync_refine: bool,
    /// Disable RF dropout detection.
    #[arg(long)]
    no_dod: bool,
    /// RF dropout threshold as a fraction of the field average envelope.
    #[arg(long, default_value_t = DEFAULT_THRESHOLD_P_DDD)]
    dod_threshold_p: f32,
    /// Absolute RF dropout threshold, overriding the percentage threshold.
    #[arg(long)]
    dod_threshold_a: Option<f32>,
    /// Hysteresis ratio used by RF dropout detection.
    #[arg(long, default_value_t = DEFAULT_HYSTERESIS)]
    dod_hysteresis: f32,
    /// Smoothing window, in lines, for wow level adjustment.
    #[arg(long)]
    wow_level_adjust_smoothing: Option<f32>,
    /// Interpolation method for wow correction.
    #[arg(long, value_enum, ignore_case = true, default_value = "linear")]
    wow_interpolation_method: CliWowInterpolation,

    /// Number of decoding threads; 0 decodes serially on a single thread.
    #[arg(long, default_value_t = 0)]
    mt_threads: usize,
    /// Fields of distance between each thread's start, and the overlap width searched for a stitch.
    #[arg(long, default_value_t = 20)]
    mt_distance_size: u64,
    /// Consecutive matching fields required to stitch one thread onto the next.
    #[arg(long, default_value_t = 2)]
    mt_overlap_count: usize,
    /// Per-field MSRE threshold below which overlapping fields are treated as matching.
    #[arg(long, default_value_t = 64.0)]
    mt_threshold: f64,
    /// Fraction of the largest per-sample deviations discarded before averaging when matching fields.
    #[arg(long, default_value_t = 0.10)]
    mt_trim_fraction: f64,

    /// Input RF capture file, or `-` to read from standard input.
    infile: PathBuf,
}

#[derive(Args, Debug)]
#[command(group(
    ArgGroup::new("write_profile_source")
        .required(true)
        .args(["profile", "profile_file"]),
))]
struct WriteProfileArgs {
    /// Profile key from the embedded profile table.
    #[arg(long)]
    profile: Option<String>,
    /// Path to a JSON profile object to load instead of an embedded profile key.
    #[arg(long, alias = "profile_file")]
    profile_file: Option<PathBuf>,
    /// Allow replacing an existing output file.
    #[arg(long)]
    overwrite: bool,
    /// Destination path for the flattened profile JSON.
    out: PathBuf,
}

#[derive(Args, Debug)]
struct CompareArgs {
    /// Reference and candidate metadata sidecars (`.tbc.json`).
    #[arg(long, num_args = 2, value_names = ["REFERENCE", "CANDIDATE"])]
    metadata: Vec<PathBuf>,
    /// Reference and candidate luma `.tbc` files.
    #[arg(long, num_args = 2, value_names = ["REFERENCE", "CANDIDATE"])]
    luma: Vec<PathBuf>,
    /// Reference and candidate chroma `_chroma.tbc` files.
    #[arg(long, num_args = 2, value_names = ["REFERENCE", "CANDIDATE"])]
    chroma: Option<Vec<PathBuf>>,
    /// Per-field MSRE threshold below which TBC fields are considered matching.
    #[arg(long, default_value_t = 64.0)]
    threshold: f64,
    /// Fraction of the largest per-sample squared deviations discarded before averaging.
    #[arg(long, default_value_t = 0.10)]
    trim_fraction: f64,
    /// Absolute tolerance for comparing JSON float values.
    #[arg(long, default_value_t = 0.11)]
    float_abs_tol: f64,
    /// Relative tolerance for comparing JSON float values.
    #[arg(long, default_value_t = 1.0e-9)]
    float_rel_tol: f64,
}

fn parse_frequency(value: &str) -> Result<f64> {
    let value = value.trim();
    let suffix_start = value
        .find(|ch: char| !matches!(ch, '0'..='9' | '.' | '-' | '+' | 'e' | 'E'))
        .unwrap_or(value.len());
    let (number, suffix) = value.split_at(suffix_start);
    let base = number.parse::<f64>().context("invalid frequency value")?;
    let multiplier = match suffix.to_ascii_lowercase().as_str() {
        "" | "m" | "mhz" => 1.0,
        "k" | "khz" => 1.0e-3,
        "hz" => 1.0e-6,
        _ => bail!("unknown frequency suffix: {suffix}"),
    };
    Ok(base * multiplier)
}


pub fn run_cli() -> Result<()> {
    match Cli::parse().command {
        Command::Decode(args) => run_decode(args),
        Command::WriteProfile(args) => run_write_profile(args),
        Command::Compare(args) => run_compare(args),
    }
}

fn run_decode(cli: DecodeArgs) -> Result<()> {
    let filter = if cli.debug { "debug" } else { "info" };
    let _ = tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .try_init();

    let profile = match (cli.profile.as_deref(), cli.profile_file.as_deref()) {
        (Some(name), None) => load_profile(name)?,
        (None, Some(path)) => load_profile_file(path)?,
        _ => bail!("exactly one of --profile or --profile-file is required"),
    };
    let request = DecodeRequest {
        inputfreq: cli.frequency.unwrap_or(40.0),

        chroma_trap: cli.chroma_trap,
        sharpness: cli.sharpness,
        notch: cli.notch.map(|freq| NotchFilter {
            freq,
            q: cli.notch_q,
        }),
        dod_threshold_p: cli.dod_threshold_p,
        dod_threshold_a: cli.dod_threshold_a,
        dod_hysteresis: cli.dod_hysteresis,
        track_phase: cli.track_phase,
        high_boost: cli.high_boost,
        video_disable_diff_demod: cli.disable_diff_demod,
        fm_audio_notch: cli
            .fm_audio_notch
            .unwrap_or(profile.decode_options.fm_audio_notch),
        rf_disable_dc_offset: !cli.enable_dc_offset,
        disable_comb: cli.no_comb,
        skip_chroma: cli.chroma_out.is_none(),
        video_nldeemp_enabled: cli.nldeemp,
        subdeemp: cli.subdeemp,
        y_comb: cli.y_comb,
        cafc: cli.cafc.unwrap_or(profile.decode_options.cafc),
        rf_disable_right_hsync: cli.disable_right_hsync,
        level_detect_divisor: cli.level_detect_divisor,
        fallback_vsync: cli
            .fallback_vsync
            .unwrap_or(profile.decode_options.fallback_vsync),
        rf_relaxed_line0: cli.relaxed_line0,
        rf_field_order_confidence: cli.field_order_confidence.clamp(0, 100),
        rf_saved_levels: cli.use_saved_levels,
        rf_skip_hsync_refine: cli.skip_hsync_refine,
        rf_export_raw_tbc: cli.export_raw_tbc,
        rf_ire0_adjust: cli.ire0_adjust,
        rf_detect_chroma_track_phase: cli.detect_chroma_track_phase,
        rf_disable_burst_hsync: cli.disable_burst_hsync,
        rf_disable_phase_correction: cli.disable_phase_correction,

        wow_level_adjust_smoothing: cli.wow_level_adjust_smoothing,
        wow_interpolation_method: cli.wow_interpolation_method.into(),

        field_order_action: cli
            .field_order_action
            .map(FieldOrderAction::from)
            .unwrap_or(profile.decode_options.field_order_action),
        level_adjust: cli.level_adjust,
        do_dod: !cli.no_dod,
        decode_profile: profile,
        ntscj: cli.ntscj,
    };

    if cli.mt_threads != 0 {
        if cli.mt_distance_size == 0 {
            bail!("--mt-distance-size must be at least 1");
        }
        if cli.mt_overlap_count == 0 {
            bail!("--mt-overlap-count must be at least 1");
        }
        if !(0.0..1.0).contains(&cli.mt_trim_fraction) {
            bail!("--mt-trim-fraction must be in [0.0, 1.0)");
        }
    }

    let input_file = if cli.infile.as_os_str() == "-" {
        os::stdin_file()?
    } else {
        OpenOptions::new()
            .read(true)
            .open(&cli.infile)
            .with_context(|| format!("failed to open input {}", cli.infile.display()))?
    };

    let mut open_options = OpenOptions::new();
    if cli.overwrite {
        open_options.write(true).create(true).truncate(true)
    } else {
        open_options.write(true).create_new(true)
    };

    let luma_out = if cli.luma_out.as_os_str() == "-" {
        os::stdout_file().with_context(|| "failed to open standard output for luma output")?
    } else {
        open_options.clone().open(&cli.luma_out).with_context(|| {
            format!("failed to open luma output file {}", cli.luma_out.display())
        })?
    };

    let chroma_out = match cli.chroma_out {
        Some(path) if path.as_os_str() != "-" => Some(
            open_options
                .clone()
                .open(&path)
                .with_context(|| format!("failed to open chroma output file {}", path.display()))?,
        ),
        Some(_) => Some(
            os::stdout_file()
                .with_context(|| "failed to open standard output for chroma output")?,
        ),
        _ => None,
    };

    let metadata_out = match cli.metadata_out {
        Some(path) if path.as_os_str() != "-" => {
            Some(open_options.clone().open(&path).with_context(|| {
                format!("failed to open metadata output file {}", path.display())
            })?)
        }
        Some(_) => bail!("metadata output cannot be standard output"),
        _ => None,
    };

    let spec = Arc::new(DecoderSpec::new(&request)?);
    let mut reader = DecodeReader::new(open_source(input_file, cli.input_format.into())?);
    let mut writer = DecodeWriter::new(luma_out, chroma_out, metadata_out)?;
    let start_offset = cli.offset.unwrap_or(0);
    // Both paths stream the input once from the start (so they work on non-seekable
    // inputs) and take `start_offset` directly.
    if cli.mt_threads == 0 {
        decode_all(&mut reader, &mut writer, spec, start_offset)?;
    } else {
        let mt = MtParams {
            threads: cli.mt_threads,
            distance_size: cli.mt_distance_size,
            overlap_count: cli.mt_overlap_count,
            threshold: cli.mt_threshold,
            trim_fraction: cli.mt_trim_fraction,
        };
        decode_all_mt(reader, &mut writer, spec, mt, start_offset)?;
    }
    Ok(())
}


fn run_write_profile(args: WriteProfileArgs) -> Result<()> {
    let profile = flatten_profile(args.profile.as_deref(), args.profile_file.as_deref())?;
    let mut open_options = OpenOptions::new();
    if args.overwrite {
        open_options.write(true).create(true).truncate(true);
    } else {
        open_options.write(true).create_new(true);
    }
    let file = open_options
        .open(&args.out)
        .with_context(|| format!("failed to open profile output file {}", args.out.display()))?;
    serde_json::to_writer(file, &profile)
        .with_context(|| format!("failed to write profile to {}", args.out.display()))?;
    Ok(())
}

// --- `compare` subcommand ----------------------------------------------------

/// Boundaries of a matched dropout interval may each shift by this many samples.
const DOD_SHIFT_TOLERANCE: i64 = 10;
/// Cap on the number of individual difference messages collected for reporting.
const MAX_REPORTED_ERRORS: usize = 20;

/// Field geometry.
struct Geometry {
    field_width: usize,
    field_samples: usize,
    sequential_fields: usize,
}

fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compare one pair of `.tbc` field-data files, appending a message per failing
/// field (and any structural mismatch) to `errors`. The reference file's size
/// picks the sample format: u16 fields (wrapped MSRE) or f32 fields (raw MSRE).
fn compare_tbc(
    reference: &Path,
    candidate: &Path,
    geometry: &Geometry,
    threshold: f64,
    trim_fraction: f64,
    errors: &mut Vec<String>,
) -> Result<()> {
    let reference_len = std::fs::metadata(reference)
        .with_context(|| format!("failed to stat {}", reference.display()))?
        .len();
    let candidate_len = std::fs::metadata(candidate)
        .with_context(|| format!("failed to stat {}", candidate.display()))?
        .len();

    let u16_field_bytes = (geometry.field_samples * 2) as u64;
    let f32_field_bytes = (geometry.field_samples * 4) as u64;
    let expected_fields = geometry.sequential_fields as u64;

    let (field_bytes, is_f32) = if reference_len == expected_fields * u16_field_bytes {
        (u16_field_bytes, false)
    } else if reference_len == expected_fields * f32_field_bytes {
        (f32_field_bytes, true)
    } else {
        errors.push(format!(
            "{}: size {} does not match metadata as u16 ({} bytes) or f32 ({} bytes) field data",
            reference.display(),
            reference_len,
            expected_fields * u16_field_bytes,
            expected_fields * f32_field_bytes,
        ));
        return Ok(());
    };

    if field_bytes == 0 || candidate_len % field_bytes != 0 {
        errors.push(format!(
            "{}: size {} is not a multiple of field size {field_bytes}",
            candidate.display(),
            candidate_len,
        ));
        return Ok(());
    }

    let candidate_fields = candidate_len / field_bytes;
    if candidate_fields != expected_fields {
        errors.push(format!(
            "{}: contains {candidate_fields} fields, reference metadata says {expected_fields}",
            candidate.display(),
        ));
        return Ok(());
    }

    let mut reference_reader = BufReader::new(
        File::open(reference).with_context(|| format!("failed to open {}", reference.display()))?,
    );
    let mut candidate_reader = BufReader::new(
        File::open(candidate).with_context(|| format!("failed to open {}", candidate.display()))?,
    );

    let field_bytes = field_bytes as usize;
    let mut reference_field = vec![0u8; field_bytes];
    let mut candidate_field = vec![0u8; field_bytes];
    for index in 0..geometry.sequential_fields {
        reference_reader
            .read_exact(&mut reference_field)
            .with_context(|| {
                format!(
                    "failed to read field {} from {}",
                    index + 1,
                    reference.display()
                )
            })?;
        candidate_reader
            .read_exact(&mut candidate_field)
            .with_context(|| {
                format!(
                    "failed to read field {} from {}",
                    index + 1,
                    candidate.display()
                )
            })?;
        let msre = if is_f32 {
            f32_msre(
                &bytes_to_f32(&reference_field),
                &bytes_to_f32(&candidate_field),
                trim_fraction,
            )
        } else {
            wrapped_u16_msre(
                &bytes_to_u16(&reference_field),
                &bytes_to_u16(&candidate_field),
                trim_fraction,
            )
        };
        if msre >= threshold {
            errors.push(format!(
                "field {}: MSRE {msre:.9} >= {threshold:.9}",
                index + 1
            ));
        }
    }
    Ok(())
}


fn read_tbc_json(path: &Path) -> Result<TbcMetadataFull> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("failed to parse {}", path.display()))
}

fn is_close(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
    (a - b).abs() <= f64::max(rel_tol * f64::max(a.abs(), b.abs()), abs_tol)
}

/// Per-field metadata difference accumulator
struct JsonDiff {
    errors: Vec<String>,
    total: usize,
    abs_tol: f64,
    rel_tol: f64,
}

impl JsonDiff {
    fn new(abs_tol: f64, rel_tol: f64) -> Self {
        Self {
            errors: Vec::new(),
            total: 0,
            abs_tol,
            rel_tol,
        }
    }

    fn record(&mut self, message: String) {
        if self.errors.len() < MAX_REPORTED_ERRORS {
            self.errors.push(message);
        }
        self.total += 1;
    }

    fn int<T: PartialEq + std::fmt::Display>(&mut self, path: &str, expected: T, actual: T) {
        if expected != actual {
            self.record(format!("{path}: expected {expected}, got {actual}"));
        }
    }

    fn bool(&mut self, path: &str, expected: bool, actual: bool) {
        if expected != actual {
            self.record(format!("{path}: expected {expected}, got {actual}"));
        }
    }

    fn float(&mut self, path: &str, expected: f64, actual: f64) {
        let close = expected.is_finite()
            && actual.is_finite()
            && is_close(expected, actual, self.abs_tol, self.rel_tol);
        if !close {
            self.record(format!("{path}: expected {expected}, got {actual}"));
        }
    }

    fn opt_float(&mut self, path: &str, expected: Option<f64>, actual: Option<f64>) {
        match (expected, actual) {
            (Some(e), Some(a)) => self.float(path, e, a),
            (None, None) => {}
            (Some(_), None) => self.record(format!("{path}: missing key")),
            (None, Some(_)) => self.record(format!("{path}: unexpected key")),
        }
    }

    fn opt_int(&mut self, path: &str, expected: Option<i64>, actual: Option<i64>) {
        match (expected, actual) {
            (Some(e), Some(a)) => self.int(path, e, a),
            (None, None) => {}
            (Some(_), None) => self.record(format!("{path}: missing key")),
            (None, Some(_)) => self.record(format!("{path}: unexpected key")),
        }
    }
}

/// Flatten per-line dropout segments into merged, contiguous sample intervals in
/// whole-field coordinates (`fieldLine * stride + x`)
fn dropout_intervals(dropouts: &DropOuts, stride: i64) -> Result<Vec<(i64, i64)>> {
    if dropouts.field_line.len() != dropouts.startx.len()
        || dropouts.field_line.len() != dropouts.endx.len()
    {
        bail!("dropOuts arrays have mismatched lengths");
    }
    let mut spans: Vec<(i64, i64)> = dropouts
        .field_line
        .iter()
        .zip(&dropouts.startx)
        .zip(&dropouts.endx)
        .map(|((&line, &sx), &ex)| {
            let line = line as i64;
            (line * stride + sx as i64, line * stride + ex as i64)
        })
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 => {
                if end > last.1 {
                    last.1 = end;
                }
            }
            _ => merged.push((start, end)),
        }
    }
    Ok(merged)
}

/// Compare two `dropOuts` objects by their per-sample dropout *state*: at every
/// sample position the "are we inside a dropout" flags of the two sides must
/// converge within [`DOD_SHIFT_TOLERANCE`] samples. Concretely, the regions
/// where exactly one side is in a dropout (boundary shifts, or a short interval
/// present on only one side) form contiguous runs of disagreement; any run
/// longer than the tolerance is a real difference.
fn compare_dropouts(
    expected: &DropOuts,
    actual: &DropOuts,
    path: &str,
    stride: i64,
    diff: &mut JsonDiff,
) {
    let (exp_intervals, act_intervals) = match (
        dropout_intervals(expected, stride),
        dropout_intervals(actual, stride),
    ) {
        (Ok(exp), Ok(act)) => (exp, act),
        (Err(err), _) | (_, Err(err)) => {
            diff.record(format!("{path}: malformed dropOuts ({err})"));
            return;
        }
    };

    // Sweep the merged boundary timeline of both sides, tracking whether each
    // side is currently inside a dropout. `+1`/`-1` deltas open and close an
    // interval; the merged intervals are disjoint, so each depth stays in 0/1.
    let mut events: Vec<(i64, i32, i32)> =
        Vec::with_capacity((exp_intervals.len() + act_intervals.len()) * 2);
    for &(start, end) in &exp_intervals {
        events.push((start, 1, 0));
        events.push((end, -1, 0));
    }
    for &(start, end) in &act_intervals {
        events.push((start, 0, 1));
        events.push((end, 0, -1));
    }
    events.sort_unstable_by_key(|event| event.0);

    let mut exp_depth = 0;
    let mut act_depth = 0;
    // Start of the current contiguous run where exactly one side is in dropout.
    let mut run_start: Option<i64> = None;
    let mut idx = 0;
    while idx < events.len() {
        let coord = events[idx].0;
        while idx < events.len() && events[idx].0 == coord {
            exp_depth += events[idx].1;
            act_depth += events[idx].2;
            idx += 1;
        }
        // State for the segment starting at `coord` (until the next event).
        let disagree = (exp_depth > 0) != (act_depth > 0);
        match (run_start, disagree) {
            (None, true) => run_start = Some(coord),
            (Some(start), false) => {
                let length = coord - start;
                if length > DOD_SHIFT_TOLERANCE {
                    diff.record(format!(
                        "{path}: dropout state diverges over [{start}, {coord}) ({length} samples), exceeding DOD_SHIFT_TOLERANCE ({DOD_SHIFT_TOLERANCE} samples)"
                    ));
                }
                run_start = None;
            }
            _ => {}
        }
    }
}

/// Compare the two deserialized sidecars field by field.
fn compare_metadata(
    expected: &TbcMetadataFull,
    actual: &TbcMetadataFull,
    stride: i64,
    diff: &mut JsonDiff,
) {
    compare_pcm_parameters(
        &expected.pcm_audio_parameters,
        &actual.pcm_audio_parameters,
        diff,
    );
    compare_video_parameters(&expected.video_parameters, &actual.video_parameters, diff);
    let e_fields = &expected.fields;
    let a_fields = &actual.fields;
    if e_fields.len() != a_fields.len() {
        diff.record(format!(
            "$.fields: expected length {}, got {}",
            e_fields.len(),
            a_fields.len()
        ));
    }
    for (index, (expected_field, actual_field)) in e_fields.iter().zip(a_fields).enumerate() {
        compare_field(expected_field, actual_field, index, stride, diff);
    }
}

#[rustfmt::skip]
fn compare_pcm_parameters(expected: &PcmAudioParameters, actual: &PcmAudioParameters, diff: &mut JsonDiff) {
    let p = "$.pcmAudioParameters";
    diff.int(&format!("{p}.bits"), expected.bits, actual.bits);
    diff.bool(&format!("{p}.isLittleEndian"), expected.is_little_endian, actual.is_little_endian);
    diff.bool(&format!("{p}.isSigned"), expected.is_signed, actual.is_signed);
    diff.int(&format!("{p}.sampleRate"), expected.sample_rate, actual.sample_rate);
}

#[rustfmt::skip]
fn compare_video_parameters(expected: &VideoParameters, actual: &VideoParameters, diff: &mut JsonDiff) {
    let p = "$.videoParameters";
    diff.int(&format!("{p}.numberOfSequentialFields"), expected.number_of_sequential_fields, actual.number_of_sequential_fields);
    // not checked: osInfo
    // not checked: gitBranch
    // not checked: gitCommit
    // not checked: system
    diff.int(&format!("{p}.fieldWidth"), expected.field_width, actual.field_width);
    diff.float(&format!("{p}.sampleRate"), expected.sample_rate, actual.sample_rate);
    diff.float(&format!("{p}.black16bIRE"), expected.black_16b_ire, actual.black_16b_ire);
    diff.float(&format!("{p}.white16bIRE"), expected.white_16b_ire, actual.white_16b_ire);
    diff.int(&format!("{p}.fieldHeight"), expected.field_height, actual.field_height);
    diff.int(&format!("{p}.colourBurstStart"), expected.colour_burst_start, actual.colour_burst_start);
    diff.int(&format!("{p}.colourBurstEnd"), expected.colour_burst_end, actual.colour_burst_end);
    diff.int(&format!("{p}.activeVideoStart"), expected.active_video_start, actual.active_video_start);
    diff.int(&format!("{p}.activeVideoEnd"), expected.active_video_end, actual.active_video_end);
    // not checked: tapeFormat
}

#[rustfmt::skip]
fn compare_field(expected: &FieldInfoEntry, actual: &FieldInfoEntry, index: usize, stride: i64, diff: &mut JsonDiff) {
    let p = format!("$.fields[{index}]");
    diff.bool(&format!("{p}.isFirstField"), expected.is_first_field, actual.is_first_field);
    diff.bool(&format!("{p}.detectedFirstField"), expected.detected_first_field, actual.detected_first_field);
    diff.bool(&format!("{p}.isDuplicateField"), expected.is_duplicate_field, actual.is_duplicate_field);
    diff.int(&format!("{p}.syncConf"), expected.sync_conf, actual.sync_conf);
    diff.int(&format!("{p}.seqNo"), expected.seq_no, actual.seq_no);
    diff.float(&format!("{p}.diskLoc"), expected.disk_loc, actual.disk_loc);
    diff.int(&format!("{p}.fileLoc"), expected.file_loc, actual.file_loc);
    diff.int(&format!("{p}.fieldPhaseID"), expected.field_phase_id, actual.field_phase_id);
    diff.opt_float(&format!("{p}.vitsMetrics.wSNR"), expected.vits_metrics.w_snr, actual.vits_metrics.w_snr);
    diff.opt_float(&format!("{p}.vitsMetrics.bPSNR"), expected.vits_metrics.b_psnr, actual.vits_metrics.b_psnr);
    diff.opt_int(&format!("{p}.decodeFaults"), expected.decode_faults, actual.decode_faults);
    // Dropouts are not written if there isn't any, but an empty one may still match within
    // tolerance to a non-empty one, so we treat missing `dropOuts` as empty.
    let empty_dropouts = DropOuts {
        field_line: Vec::new(),
        startx: Vec::new(),
        endx: Vec::new(),
    };
    let expected_dropouts = expected.drop_outs.as_ref().unwrap_or(&empty_dropouts);
    let actual_dropouts = actual.drop_outs.as_ref().unwrap_or(&empty_dropouts);
    compare_dropouts(expected_dropouts, actual_dropouts, &format!("{p}.dropOuts"), stride, diff)
}

fn run_compare(args: CompareArgs) -> Result<()> {
    if !(0.0..1.0).contains(&args.trim_fraction) {
        bail!("--trim-fraction must be in [0.0, 1.0)");
    }

    let [reference_meta_path, candidate_meta_path] = <[PathBuf; 2]>::try_from(args.metadata)
        .map_err(|_| {
            anyhow::anyhow!("--metadata requires exactly REFERENCE and CANDIDATE paths")
        })?;
    let [reference_luma, candidate_luma] = <[PathBuf; 2]>::try_from(args.luma)
        .map_err(|_| anyhow::anyhow!("--luma requires exactly REFERENCE and CANDIDATE paths"))?;

    let reference_meta = read_tbc_json(&reference_meta_path)?;
    let candidate_meta = read_tbc_json(&candidate_meta_path)?;
    let video = &reference_meta.video_parameters;
    let width = video.field_width;
    let height = video.field_height;
    if width == 0 || height == 0 {
        bail!("invalid reference field geometry width={width} height={height}");
    }
    let geometry = Geometry {
        field_width: width,
        field_samples: width * height,
        sequential_fields: video.number_of_sequential_fields,
    };

    let mut failed = false;

    let mut luma_errors = Vec::new();
    compare_tbc(
        &reference_luma,
        &candidate_luma,
        &geometry,
        args.threshold,
        args.trim_fraction,
        &mut luma_errors,
    )?;
    report_tbc("luma", &candidate_luma, &luma_errors, &mut failed);

    if let Some(chroma) = args.chroma {
        let [reference_chroma, candidate_chroma] =
            <[PathBuf; 2]>::try_from(chroma).map_err(|_| {
                anyhow::anyhow!("--chroma requires exactly REFERENCE and CANDIDATE paths")
            })?;
        let mut chroma_errors = Vec::new();
        compare_tbc(
            &reference_chroma,
            &candidate_chroma,
            &geometry,
            args.threshold,
            args.trim_fraction,
            &mut chroma_errors,
        )?;
        report_tbc("chroma", &candidate_chroma, &chroma_errors, &mut failed);
    }

    let stride = i64::try_from(geometry.field_width).unwrap_or(0);
    let mut diff = JsonDiff::new(args.float_abs_tol, args.float_rel_tol);
    compare_metadata(&reference_meta, &candidate_meta, stride, &mut diff);
    if diff.total > diff.errors.len() {
        diff.errors.push(format!(
            "... {} more JSON difference(s) not shown",
            diff.total - diff.errors.len()
        ));
    }
    report_json(&candidate_meta_path, diff.total, &diff.errors, &mut failed);

    if failed {
        bail!("comparison failed");
    }
    Ok(())
}

fn report_tbc(kind: &str, candidate: &Path, errors: &[String], failed: &mut bool) {
    if errors.is_empty() {
        println!("{} ({}): OK", kind, candidate.display());
    } else {
        println!("{} ({}): FAILED", kind, candidate.display());
        for error in errors {
            eprintln!("  {error}");
        }
        *failed = true;
    }
}

fn report_json(candidate: &Path, total_errors: usize, errors: &[String], failed: &mut bool) {
    if total_errors == 0 {
        println!("metadata ({}): OK", candidate.display());
    } else {
        println!("metadata ({}): FAILED", candidate.display());
        for error in errors {
            eprintln!("  {error}");
        }
        *failed = true;
    }
}
