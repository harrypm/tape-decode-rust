use super::*;

fn lineslice_tbc_bounds(spec: &DecoderSpec, loc: &[usize; 3], keepphase: bool) -> (usize, usize) {
    // return a slice corresponding with pre-TBC line
    let outlinelen = spec.sys_outlinelen as f64;
    let mut begin_px = outlinelen * (loc[0] as f64 - 1.0);
    let mut begin_offset = spec.sys_outfreq * loc[1] as f64;
    if keepphase {
        begin_offset = (begin_offset / 4.0).floor() * 4.0;
    }
    begin_px += begin_offset;
    let length_px = spec.sys_outfreq * loc[2] as f64;
    (
        begin_px.round_ties_even().max(0.0) as usize,
        (begin_px + length_px).round_ties_even().max(0.0) as usize,
    )
}

fn encoded_output_to_ire(field: &DecodedField, spec: &DecoderSpec, output: &[u16]) -> Vec<f64> {
    let output_zero = spec.sys_output_zero as u16;
    let out_scale = field.out_scale.unwrap();
    let vsync_ire = f64::from(spec.sys_vsync_ire);
    output
        .iter()
        .map(|&value| (value.wrapping_sub(output_zero) as f64 / out_scale) + vsync_ire)
        .collect()
}

fn raw_output_to_ire<T>(field: &DecodedField, spec: &DecoderSpec, output: &[T]) -> Vec<f64>
where
    T: Float,
{
    let output_zero = spec.sys_output_zero as f64;
    let out_scale = field.out_scale.unwrap();
    let vsync_ire = f64::from(spec.sys_vsync_ire);
    output
        .iter()
        .map(|&value| ((value.to_f64().unwrap() - output_zero) / out_scale) + vsync_ire)
        .collect()
}

fn clamped_slice<T>(values: &[T], start: usize, end: usize) -> &[T] {
    &values[start.min(values.len())..end.min(values.len())]
}

fn luma_slice_to_ire(
    field: &DecodedField,
    spec: &DecoderSpec,
    luma: &LumaOutput,
    start: usize,
    end: usize,
) -> Vec<f64> {
    match luma {
        LumaOutput::Encoded(values) => {
            encoded_output_to_ire(field, spec, clamped_slice(values, start, end))
        }
        LumaOutput::Raw(values) => {
            raw_output_to_ire(field, spec, clamped_slice(values, start, end))
        }
    }
}

fn luma_slice_samples(luma: &LumaOutput, start: usize, end: usize) -> Vec<f64> {
    fn to_f64<T: Into<f64> + Copy>(values: &[T], start: usize, end: usize) -> Vec<f64> {
        clamped_slice(values, start, end)
            .iter()
            .map(|&v| v.into())
            .collect()
    }
    match luma {
        LumaOutput::Encoded(values) => to_f64(values, start, end),
        LumaOutput::Raw(values) => to_f64(values, start, end),
    }
}

// SNR (dB) of a luma slice relative to a 100 IRE reference, from the standard
// deviation of its IRE-converted samples. Returns 0 for a noiseless slice.
fn snr_at(
    field: &DecodedField,
    spec: &DecoderSpec,
    luma: &LumaOutput,
    start: usize,
    end: usize,
) -> f64 {
    let data = raw_output_to_ire(field, spec, &luma_slice_samples(luma, start, end));
    let noise = rms(&data);
    if noise == 0.0 {
        0.0
    } else {
        20.0 * (100.0 / noise).log10()
    }
}

// VITS-derived white SNR / black PSNR for a decoded field's luma picture.
pub(crate) fn compute_vits_metrics(
    spec: &DecoderSpec,
    metrics_field: &DecodedField,
    metrics_luma: &LumaOutput,
) -> Result<VitsMetrics> {
    let mut wsnr_raw: Option<f64> = None;
    for wl in &spec.sys_ld_vits_whitelocs {
        let (start, end) = lineslice_tbc_bounds(spec, wl, false);
        let ire_values = luma_slice_to_ire(metrics_field, spec, metrics_luma, start, end);
        if inrange(mean(&ire_values), 90.0, 110.0) {
            wsnr_raw = Some(snr_at(metrics_field, spec, metrics_luma, start, end));
            break;
        }
    }
    let (black_start, black_end) = lineslice_tbc_bounds(spec, &spec.sys_blacksnr_slice, false);
    let bpsnr_raw = snr_at(metrics_field, spec, metrics_luma, black_start, black_end);
    let round_finite = |value: f64| {
        let rounded = roundfloat(value, 1);
        rounded.is_finite().then_some(rounded)
    };
    Ok(VitsMetrics {
        w_snr: wsnr_raw.and_then(round_finite),
        b_psnr: round_finite(bpsnr_raw),
    })
}
