use std::sync::Arc;

use crate::optimized::{
    exp_fast, narrow_sos, powf_fast_nonneg, scale_field, sosfilt_f32, sosfiltfilt_f32,
    sum_algebraic, unwrap_angles, ScaleFieldParams,
};
use crate::request::{ColorSystem, FieldOrderAction, LineSystem, WowInterpolation};
use crate::spec::DecoderSpec;
use crate::DeterministicHashMap;
use anyhow::{bail, Context as _, Result};
use num_traits::Float;
use realfft::{ComplexToReal, RealToComplex};
use rustfft::num_complex::Complex32;
use rustfft::Fft;
use sci_rs::signal::filter::design::{FilterBandType, Sos};
use serde::{Deserialize, Serialize};

// Submodules split out of the original monolithic decode.rs.
mod chroma;
mod demodblock;
mod dropouts;
mod field;
mod sync;
mod vits;

use chroma::decode_chroma;
use demodblock::decode_video_block;
use dropouts::detect_dropouts_rf;
use field::predecode_field_from_rawdecode;
use sync::ResyncState;
use vits::compute_vits_metrics;

pub(crate) fn iretohz(ire0: f64, hz_ire: f64, ire: f64) -> f64 {
    ire0 + (hz_ire * ire)
}

fn hztoire(ire0: f64, hz_ire: f64, hz: f64) -> f64 {
    (hz - ire0) / hz_ire
}

fn inrange(a: f64, mi: f64, ma: f64) -> bool {
    a >= mi && a <= ma
}

fn hz_to_output_array(spec: &DecoderSpec, input: &[f32], ire0: f64, out_scale: f64) -> Vec<u16> {
    let out_scale = out_scale as f32;
    let scale = out_scale / spec.sys_hz_ire;
    let offset =
        spec.sys_output_zero as f32 - spec.sys_vsync_ire * out_scale - (ire0 as f32) * scale;
    input
        .iter()
        .map(|&sample| {
            let value = sample * scale + offset;
            value.clamp(0.0, 65535.0) as u16
        })
        .collect()
}

fn y_comb(input: &[f32], line_len: usize, limit: f32) -> Vec<f32> {
    let len = input.len();
    if len == 0 {
        return Vec::new();
    }

    // The two delayed taps wrap at i = len - shift and i = shift respectively,
    // so cutting the walk at both points leaves segments where each tap is one
    // contiguous slice: no per-sample modulo, and the loops vectorize.
    let shift = line_len % len;
    let mut output = Vec::with_capacity(len);
    let mut emit = |range: std::ops::Range<usize>| {
        let back_start = (range.start + shift) % len;
        let fwd_start = (range.start + len - shift) % len;
        let current = &input[range.clone()];
        let back = &input[back_start..back_start + current.len()];
        let fwd = &input[fwd_start..fwd_start + current.len()];
        output.extend(current.iter().zip(back).zip(fwd).map(
            |((&current, &back), &fwd)| {
                let diffb = current - back;
                let difff = current - fwd;
                let clipped = (diffb + difff).clamp(-limit, limit);
                current - clipped / 2.0
            },
        ));
    };
    let cut0 = shift.min(len - shift);
    let cut1 = shift.max(len - shift);
    emit(0..cut0);
    emit(cut0..cut1);
    emit(cut1..len);
    output
}

fn roundfloat(fl: f64, places: i64) -> f64 {
    let scale = 10.0f64.powi(places as i32);
    (fl * scale).round_ties_even() / scale
}

fn resolve_slice_bound(len: usize, index: isize) -> usize {
    if index >= 0 {
        usize::try_from(index).unwrap_or(usize::MAX).min(len)
    } else {
        len.saturating_sub(index.unsigned_abs())
    }
}

fn mean_slice<T>(values: &[T]) -> f64
where
    T: Float,
{
    if values.is_empty() {
        return 0.0;
    }
    // Lane accumulators so the conversion and sum vectorize instead of running
    // as one serial carried add over field-sized inputs.
    const LANES: usize = 8;
    let mut acc = [0.0f64; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, &value) in acc.iter_mut().zip(chunk) {
            *lane += value.to_f64().unwrap();
        }
    }
    let tail: f64 = chunks
        .remainder()
        .iter()
        .map(|&value| value.to_f64().unwrap())
        .sum();
    (acc.iter().sum::<f64>() + tail) / values.len() as f64
}

fn median_from_values<T: Float>(values: &mut [T]) -> T {
    if values.is_empty() {
        return T::nan();
    }
    // Quickselect (O(n) avg) instead of a full sort: we only need the middle
    // order statistic(s), and selecting the k-th element yields the same value
    // a full sort would place at index k for the same comparator.
    let cmp = |a: &T, b: &T| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        let (left, &mut hi, _) = values.select_nth_unstable_by(mid, cmp);
        // Lower middle element is the maximum of the partition below `mid`,
        // ranked by the same comparator used for selection.
        let lo = *left.iter().max_by(|a, b| cmp(a, b)).unwrap();
        (lo + hi) / (T::one() + T::one())
    } else {
        let (_, &mut median, _) = values.select_nth_unstable_by(mid, cmp);
        median
    }
}

fn fft_in_place_f32(buffer: &mut [Complex32], fft: &dyn Fft<f32>, inverse: bool) {
    assert_eq!(buffer.len(), fft.len());
    if buffer.is_empty() {
        return;
    }

    fft.process(buffer);

    if inverse {
        let inv_scale = 1.0 / buffer.len() as f32;
        for sample in buffer {
            *sample *= inv_scale;
        }
    }
}

fn fft_real_f32(input: &[f32], forward_fft: &dyn Fft<f32>) -> Vec<Complex32> {
    let mut output = input
        .iter()
        .map(|&sample| Complex32::new(sample, 0.0))
        .collect::<Vec<_>>();
    fft_in_place_f32(&mut output, forward_fft, false);
    output
}

fn ifft_complex_owned_f32(
    mut buffer: Vec<Complex32>,
    inverse_fft: &dyn Fft<f32>,
) -> Vec<Complex32> {
    fft_in_place_f32(&mut buffer, inverse_fft, true);
    buffer
}

fn hilbert_f32(
    input: &[f32],
    forward_fft: &dyn Fft<f32>,
    inverse_fft: &dyn Fft<f32>,
) -> Vec<Complex32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut spectrum = fft_real_f32(input, forward_fft);
    if n.is_multiple_of(2) {
        for sample in &mut spectrum[1..(n / 2)] {
            *sample *= 2.0;
        }
        for sample in &mut spectrum[(n / 2 + 1)..] {
            *sample = Complex32::new(0.0, 0.0);
        }
    } else if n > 1 {
        for sample in &mut spectrum[1..=((n - 1) / 2)] {
            *sample *= 2.0;
        }
        for sample in &mut spectrum[n.div_ceil(2)..] {
            *sample = Complex32::new(0.0, 0.0);
        }
    }

    ifft_complex_owned_f32(spectrum, inverse_fft)
}

fn cafc_fft_center_freq(spec: &DecoderSpec, data: &[f32]) -> Result<(f64, f64)> {
    if data.len() < 3 {
        bail!("cafc_fft_center_freq requires at least three samples");
    }

    let sig_fft = fft_real_f32(data, spec.fft_field_forward_f32.as_ref());
    // The squared-magnitude spectrum is a field-sized buffer; store it as f32
    // (each magnitude is still computed in f64) — it only feeds the local-peak
    // search for the cafc carrier bin, which compares well-separated spectral
    // lines.
    let power: Vec<f32> = sig_fft
        .iter()
        .map(|sample| sample.re.mul_add(sample.re, sample.im * sample.im))
        .collect();
    let max_power = power.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let clip_min = (f64::from(max_power) * DecoderSpec::CHROMA_AFC_POWER_THRESHOLD) as f32;

    let time_step = 1.0 / (spec.sys_fsc_mhz * 4.0 * 1e6);
    let scale = 1.0 / (data.len() as f64 * time_step);
    let positive_end = data.len().div_ceil(2);
    if positive_end <= 2 {
        bail!("cafc_fft_center_freq has no interior positive-frequency bins");
    }

    let clipped_power: Vec<f32> = (1..positive_end)
        .map(|index| power[index].clamp(clip_min, max_power))
        .collect();

    let mut peak_index = None;
    let mut peak_delta = f64::INFINITY;
    for local_index in 1..clipped_power.len().saturating_sub(1) {
        if clipped_power[local_index] > clipped_power[local_index - 1]
            && clipped_power[local_index] > clipped_power[local_index + 1]
        {
            let fft_index = local_index + 1;
            let frequency = fft_index as f64 * scale;
            let delta = (frequency - spec.decoder_color_under_carrier).abs();
            if delta < peak_delta {
                peak_delta = delta;
                peak_index = Some(fft_index);
            }
        }
    }

    let peak_index = peak_index.context("cafc_fft_center_freq found no local peak")?;
    let peak_freq = peak_index as f64 * scale;

    let fine_tune_threshold = spec.chroma_afc_fh() * spec.chroma_afc_fine_tune_fh_ratio;

    let carrier_freq = fine_tune_frequency(
        peak_freq,
        spec.decoder_color_under_carrier,
        fine_tune_threshold,
    );

    let mut cc_phase = 0.0;
    for (index, sample) in sig_fft.iter().enumerate() {
        let sample_freq = if index < positive_end {
            index as f64 * scale
        } else {
            (index as isize - data.len() as isize) as f64 * scale
        };
        if sample_freq == carrier_freq {
            cc_phase = f64::from(sample.im).atan2(f64::from(sample.re));
            break;
        }
    }

    Ok((carrier_freq, cc_phase))
}
fn fine_tune_frequency(freq: f64, color_under: f64, max_step: f64) -> f64 {
    let specs_distance = |frequency: f64| (frequency - color_under).abs();
    let mut tune_freq = freq;
    while specs_distance(tune_freq) >= max_step {
        tune_freq -= if tune_freq > color_under {
            max_step
        } else {
            -max_step
        };
    }

    let one_step_more = tune_freq + max_step;
    let one_step_less = tune_freq - max_step;

    if specs_distance(tune_freq) < specs_distance(one_step_less)
        && specs_distance(tune_freq) < specs_distance(one_step_more)
    {
        tune_freq
    } else if specs_distance(one_step_more) < specs_distance(one_step_less) {
        one_step_more
    } else {
        one_step_less
    }
}

pub(crate) fn gen_chroma_heterodyne(
    het_wave_scale: f64,
    phase_drift: f64,
    field_len: usize,
) -> Vec<Vec<f32>> {
    use std::f64::consts::TAU;
    let angle_step = TAU * het_wave_scale;

    let mut phase0 = Vec::with_capacity(field_len);
    let mut phase1 = Vec::with_capacity(field_len);
    let mut phase2 = Vec::with_capacity(field_len);
    let mut phase3 = Vec::with_capacity(field_len);

    for i in 0..field_len {
        // Reduce the (large, accumulating) carrier phase modulo a turn before
        // narrowing, so the carrier itself is evaluated over a small argument.
        let reduced = (angle_step * i as f64 + phase_drift).rem_euclid(TAU) as f32;
        let (sin, cos) = reduced.sin_cos();
        phase0.push(-cos);
        phase1.push(sin);
        phase2.push(cos);
        phase3.push(-sin);
    }

    vec![phase0, phase1, phase2, phase3]
}

pub(crate) fn butter_sos(
    order: usize,
    wn: &[f64],
    band_type: FilterBandType,
) -> Result<Vec<Sos<f64>>> {
    use sci_rs::signal::filter::design::{butter_dyn, DigitalFilter, FilterOutputType};

    let filter = butter_dyn::<f64>(
        order,
        wn.to_vec(),
        Some(band_type),
        Some(false),
        Some(FilterOutputType::Sos),
        None,
    );

    match filter {
        DigitalFilter::Sos(sos) => Ok(sos.sos),
        _ => bail!("sci-rs returned an unexpected Butterworth SOS representation"),
    }
}

fn rms<T>(samples: &[T]) -> f64
where
    T: Float,
{
    let len = samples.len() as f64;
    let mean = samples
        .iter()
        .map(|&sample| sample.to_f64().unwrap())
        .sum::<f64>()
        / len;
    let square_mean = samples
        .iter()
        .map(|&sample| {
            let centered = sample.to_f64().unwrap() - mean;
            centered * centered
        })
        .sum::<f64>()
        / len;
    square_mean.sqrt()
}

fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

fn shift_chroma_and_remove_dc(mut output: Vec<f32>, move_by: isize) -> Vec<f32> {
    if output.is_empty() {
        return output;
    }

    roll(&mut output, move_by);

    let sum: f32 = output.iter().copied().sum();
    let mean = sum / output.len() as f32;
    for sample in output.iter_mut() {
        *sample -= mean;
    }

    output
}

fn get_linefreq(
    linelen: f64,
    samplesperline: f64,
    linecount: Option<usize>,
    lineoffset: usize,
    line: Option<usize>,
    linelocs: Option<&[f32]>,
) -> f64 {
    let mut length =
        if let (Some(line), Some(linecount), Some(linelocs)) = (line, linecount, linelocs) {
            if line >= linecount + lineoffset {
                f64::from(linelocs[line]) - f64::from(linelocs[line - 1])
            } else if line > 0 {
                (f64::from(linelocs[line + 1]) - f64::from(linelocs[line - 1])) / 2.0
            } else {
                f64::from(linelocs[1]) - f64::from(linelocs[0])
            }
        } else {
            linelen
        };

    if length <= 0.0 {
        length = linelen;
    }

    samplesperline * length
}

fn usectoinpx(
    linelen: f64,
    samplesperline: f64,
    linecount: Option<usize>,
    lineoffset: usize,
    x: f64,
    line: Option<usize>,
    linelocs: Option<&[f32]>,
) -> f64 {
    x * get_linefreq(
        linelen,
        samplesperline,
        linecount,
        lineoffset,
        line,
        linelocs,
    )
}

fn hz_to_output_scalar(spec: &DecoderSpec, input: f64, out_scale: f64) -> f64 {
    if spec.rf_export_raw_tbc {
        return input;
    }

    let mut reduced = (input - f64::from(spec.sys_ire0)) / f64::from(spec.sys_hz_ire);
    reduced -= f64::from(spec.sys_vsync_ire);
    (((reduced * out_scale) + spec.sys_output_zero as f64).clamp(0.0, 65535.0) + 0.5) as u16 as f64
}

fn sync_confidence_from_linelocs(field: &DecodedField) -> Result<i64> {
    let linelocs = field.linelocs.as_ref().context("missing linelocs")?;
    let linecount = field.linecount.unwrap_or(0);
    let end = (field.lineoffset + linecount).min(linelocs.len());
    if end.saturating_sub(field.lineoffset) < 3 {
        return Ok(field.sync_confidence);
    }

    let mut lld2max = f32::NEG_INFINITY;
    for index in field.lineoffset..end - 2 {
        let lld2 = linelocs[index + 2] - (2.0 * linelocs[index + 1]) + linelocs[index];
        lld2max = lld2max.max(lld2);
    }

    let newconf = if lld2max > 4.0 { 45 } else { 100 };
    Ok(field.sync_confidence.min(newconf))
}

fn ire0_adjust_from_picture(picture_luma: &[f32], field: &DecodedField) -> f64 {
    let ire0_adjust_padding = 4usize;
    let hsync_start = field.ire0_backporch.0 + ire0_adjust_padding;
    let hsync_end = field.ire0_backporch.1 - ire0_adjust_padding;
    if field.outlinecount == 0 || field.outlinelen == 0 || hsync_start >= hsync_end {
        return f64::NAN;
    }

    let mut blank_levels = Vec::with_capacity(field.outlinecount);
    for line in 0..field.outlinecount {
        let line_start = line * field.outlinelen;
        let start = line_start + hsync_start;
        let end = line_start + hsync_end;
        if end > picture_luma.len() || start >= end {
            blank_levels.push(f64::NAN);
            continue;
        }

        let mut values = picture_luma[start..end]
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        blank_levels.push(median_from_values(&mut values));
    }

    blank_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let start = field.outlinecount / 3;
    let end = (field.outlinecount * 2) / 3;
    mean_slice(&blank_levels[start..end])
}

fn demod_slice_bounds(len: usize, start: i64, end: i64) -> Option<(usize, usize)> {
    let len = len as i64;
    let start = start.clamp(0, len) as usize;
    let end = end.clamp(0, len) as usize;
    (start < end).then_some((start, end))
}

fn demod_mean(data: &[f32], start: i64, end: i64) -> f64 {
    let Some((start, end)) = demod_slice_bounds(data.len(), start, end) else {
        return f64::NAN;
    };
    let slice = &data[start..end];
    f64::from(slice.iter().sum::<f32>() / slice.len() as f32)
}

type PhaseSequenceEntry = (usize, usize, f64, f64, f64, f64);

// Whether any sample in the chunk sits on the given side of the threshold,
// as a branch-free OR-reduction the compiler vectorizes.
#[inline]
fn chunk_crosses(chunk: &[f32], high: f32, above: bool) -> bool {
    if above {
        chunk.iter().fold(false, |acc, &value| acc | (value > high))
    } else {
        chunk
            .iter()
            .fold(false, |acc, &value| acc | (value <= high))
    }
}

fn findpulses_raw(
    sync_ref: &[f32],
    high: f32,
    min_synclen: f64,
    max_synclen: f64,
) -> (Vec<i64>, Vec<i64>) {
    let mut in_pulse = sync_ref[0] <= high;
    let mut starts = Vec::new();
    let mut lengths = Vec::new();
    let mut cur_start = 0usize;

    // The signal crosses the threshold only a few hundred times per field, so
    // first test whole chunks for a crossing out of the current state and run
    // the per-sample edge logic only on the chunks that contain one.
    const CHUNK: usize = 64;
    let mut pos = 0usize;
    let n = sync_ref.len();
    while pos < n {
        let end = (pos + CHUNK).min(n);
        let chunk = &sync_ref[pos..end];
        if !chunk_crosses(chunk, high, in_pulse) {
            pos = end;
            continue;
        }
        for (offset, &value) in chunk.iter().enumerate() {
            if in_pulse {
                if value > high {
                    let length = pos + offset - cur_start;
                    if (length as f64) >= min_synclen
                        && (length as f64) <= max_synclen
                        && cur_start != 0
                    {
                        starts.push(cur_start as i64);
                        lengths.push(length as i64);
                    }
                    in_pulse = false;
                }
            } else if value <= high {
                cur_start = pos + offset;
                in_pulse = true;
            }
        }
        pos = end;
    }

    (starts, lengths)
}

pub const BLOCKSIZE: usize = 32 * 1024;
pub(crate) const BLOCKCUT: usize = 1024;
pub(crate) const BLOCKCUT_END: usize = 1024;
const DOD_MERGE_THRESHOLD: isize = 30;
const DOD_MIN_LENGTH: isize = 10;
const BADJ: f64 = -1.4;

#[derive(Clone)]
pub enum LumaOutput {
    Encoded(Vec<u16>),
    Raw(Vec<f32>),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct VitsMetrics {
    #[serde(rename = "wSNR", skip_serializing_if = "Option::is_none")]
    pub w_snr: Option<f64>,
    #[serde(rename = "bPSNR", skip_serializing_if = "Option::is_none")]
    pub b_psnr: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DropOuts {
    pub field_line: Vec<usize>,
    pub startx: Vec<usize>,
    pub endx: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldInfoEntry {
    pub is_first_field: bool,
    pub detected_first_field: bool,
    pub is_duplicate_field: bool,
    pub sync_conf: i64,
    pub seq_no: usize,
    pub disk_loc: f32,
    pub file_loc: u64,
    #[serde(rename = "fieldPhaseID")]
    pub field_phase_id: i64,
    pub vits_metrics: VitsMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop_outs: Option<DropOuts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_faults: Option<i64>,
}

#[derive(Clone)]
pub struct DecoderMetadata {
    pub system: &'static str,
    pub field_width: usize,
    pub sample_rate: f64,
    pub black_16b_ire: f64,
    pub white_16b_ire: f64,
    pub field_height: usize,
    pub colour_burst_start: i64,
    pub colour_burst_end: i64,
    pub active_video_start: i64,
    pub active_video_end: i64,
}

#[derive(Clone)]
struct StackableMa {
    window_average: usize,
    min_watermark: usize,
    stack: Vec<f64>,
}

impl StackableMa {
    fn new(min_watermark: usize, window_average: usize) -> Self {
        Self {
            window_average,
            min_watermark,
            stack: Vec::new(),
        }
    }

    fn push(&mut self, value: f64) {
        self.stack.push(value);
    }

    fn pull(&mut self) -> Option<f64> {
        let keep_from = self.stack.len().saturating_sub(self.window_average);
        self.stack.drain(..keep_from);
        (!self.stack.is_empty()).then(|| mean(&self.stack))
    }

    fn has_values(&self) -> bool {
        self.stack.len() > self.min_watermark
    }

    fn size(&self) -> usize {
        self.stack.len()
    }
}

// Mutable ChromaAFC carrier-tracking state; immutable filters and constants live
// in DecoderSpec.

#[derive(Clone)]
struct ChromaAfcState {
    cc_phase: f64,
    cc_freq_mhz: f64,
    chroma_heterodyne: Vec<Vec<f32>>,
    // CAFC-only drift stacks (None when CAFC is disabled).
    meas_stack: Option<StackableMa>,
    chroma_log_drift: Option<StackableMa>,
}

fn chroma_afc_chainfiltfilt(config: &DecoderSpec, data: &[f32]) -> Vec<f32> {
    // The narrowband SOS chain keeps its field-sized output in f32. This only
    // feeds the cafc center-frequency measurement.
    let mut iter = config.chroma_afc_narrowband.iter();
    let Some(sos0) = iter.next() else {
        return data.to_vec();
    };
    let mut filtered = sosfiltfilt_f32(sos0, data);
    for sos in iter {
        filtered = sosfiltfilt_f32(sos, &filtered);
    }
    filtered
}

impl ChromaAfcState {
    fn new(config: &DecoderSpec) -> Self {
        let mut state = ChromaAfcState {
            cc_phase: 0.0,
            cc_freq_mhz: 0.0,
            chroma_heterodyne: Vec::new(),
            meas_stack: None,
            chroma_log_drift: None,
        };
        if config.chroma_afc_enabled() {
            state.meas_stack = Some(StackableMa::new(0, 8192));
            state.chroma_log_drift = Some(StackableMa::new(0, 8192));
        }
        state.set_cc(config, config.decoder_color_under_carrier);
        state
    }

    // fcc in Hz (the current dwc subcarrier freq)
    fn set_cc(&mut self, config: &DecoderSpec, fcc_hz: f64) {
        self.cc_freq_mhz = fcc_hz / 1e6;
        self.gen_het_c(config);
    }

    // Generates the heterodyning carrier. The resulting signal is a
    // -cosine of (fcc + fsc) frequency with cc_phase phase.
    fn gen_het_c(&mut self, config: &DecoderSpec) {
        let het_freq = config.sys_fsc_mhz + self.cc_freq_mhz;
        let het_wave_scale = het_freq / (config.sys_fsc_mhz * 4.0);
        let field_lines = config.sys_field_lines[0].max(config.sys_field_lines[1]);
        self.chroma_heterodyne = gen_chroma_heterodyne(
            het_wave_scale,
            // This is the last cc phase measured as it comes from the tape
            self.cc_phase,
            config.sys_outlinelen * field_lines as usize,
        );
    }

    fn measure_center_freq(&mut self, config: &DecoderSpec, data: &[f32]) -> Result<f64> {
        let filtered = chroma_afc_chainfiltfilt(config, data);
        let (carrier_freq, cc_phase) = cafc_fft_center_freq(config, &filtered)?;
        self.cc_phase = cc_phase;
        Ok(carrier_freq)
    }

    // returns the downconverted chroma carrier offset
    fn freq_offset(&mut self, config: &DecoderSpec, chroma: &[f32], adjustf: bool) -> Result<()> {
        let (min_f, max_f) = config.chroma_afc_band_tolerance();
        let measured = self.measure_center_freq(config, chroma)?;
        let freq_cc_x = measured.clamp(
            config.decoder_color_under_carrier * min_f,
            config.decoder_color_under_carrier * max_f,
        );

        if measured != freq_cc_x {
            tracing::warn!(clipped = freq_cc_x, measured, "Chroma PLL range clipped");
        }
        let freq_cc = if adjustf {
            self.meas_stack.as_mut().unwrap().push(freq_cc_x);
            freq_cc_x
        } else {
            self.cc_freq_mhz * 1e6
        };
        self.set_cc(config, freq_cc);
        let drift_stack = self.chroma_log_drift.as_mut().unwrap();
        drift_stack.push(freq_cc - config.decoder_color_under_carrier);
        // Advance the drift window for its trimming side effect; value unused.
        drift_stack.pull();
        Ok(())
    }

    // Filter to pick out color-under chroma component (about twice the carrier).
    // Note: the effective order doubles since it is applied forward and backward.
    fn get_chroma_bandpass(&self, config: &DecoderSpec) -> Result<Vec<Sos<f32>>> {
        let freq_hz_half = config.freq_hz() / 2.0;
        let chroma_bpf_under_ratio =
            config.decoder_chroma_bpf_upper / config.decoder_color_under_carrier;
        let sos = butter_sos(
            config.decoder_chroma_bpf_order,
            &[
                config.decoder_chroma_bpf_lower / freq_hz_half,
                self.cc_freq_mhz * 1e6 * chroma_bpf_under_ratio / freq_hz_half,
            ],
            FilterBandType::Bandpass,
        )?;
        Ok(narrow_sos(&sos))
    }
}

type ChromaArray = Vec<f32>;

fn roll<T>(values: &mut [T], shift: isize) {
    if values.is_empty() {
        return;
    }
    let len = values.len() as isize;
    let shift = shift.rem_euclid(len) as usize;
    values.rotate_right(shift);
}

fn active_chroma_heterodyne<'a>(
    spec: &'a DecoderSpec,
    chroma_afc_state: &'a ChromaAfcState,
) -> &'a [Vec<f32>] {
    if spec.chroma_afc_enabled() {
        &chroma_afc_state.chroma_heterodyne
    } else {
        &spec.rf_chroma_heterodyne
    }
}

#[derive(Clone)]
struct VideoChannels {
    demod: Vec<f32>,
    demod_05: Vec<f32>,
    demod_burst: Vec<f32>,
    envelope: Vec<f32>,
}

#[derive(Clone)]
struct FieldData {
    video: VideoChannels,
    startloc: u64,
    input_len: usize,
}

#[derive(Clone)]
struct PrevFieldState {
    readloc: u64,
    field_number: i64,
    phase_adjust_median: f64,
}

#[derive(Clone)]
struct InterFieldState {
    prev_first_hsync_readloc: i64,
    prev_first_hsync_loc: f32,
    prev_first_hsync_diff: f32,
    prev_first_field: i64,
    track_phase: Option<i64>,
    compute_linelocs_issues: bool,
}

impl InterFieldState {
    fn new(track_phase: Option<i64>) -> Self {
        Self {
            prev_first_hsync_readloc: -1,
            prev_first_hsync_loc: -1.0,
            prev_first_hsync_diff: -1.0,
            prev_first_field: -1,
            track_phase,
            compute_linelocs_issues: false,
        }
    }
}

#[derive(Clone)]
struct DecodedField {
    data: FieldData,
    prevfield: Option<PrevFieldState>,
    readloc: u64,
    inlinelen: f64,
    outlinelen: usize,
    outlinecount: usize,
    ire0_backporch: (usize, usize),
    wow_level_adjust_smoothing: f32,
    wow_interpolation_method: WowInterpolation,
    validpulses: Vec<i64>,
    is_first_field: Option<bool>,
    linebad: Option<Vec<u8>>,
    nextfieldoffset: Option<f64>,
    vblank_next: Option<f64>,
    lt_vsync: Option<(f64, f64)>,
    is_progressive_field: Option<bool>,
    field_number: i64,
    linelocs: Option<Vec<f32>>,
    lineoffset: usize,
    linecount: Option<usize>,
    out_scale: Option<f64>,
    field_phase_id: Option<i64>,
    phase_adjust_median: f64,
    valid: bool,
    sync_confidence: i64,
    phase_sequence: Option<Vec<PhaseSequenceEntry>>,
    burst_phase_avg: Option<f64>,
    /// Cached `(median, mad)` of the field's wow-factor distribution. These
    /// depend only on the line-location spline, so they are identical for every
    /// channel; `downscale_raw_vec` fills this on the first call and reuses it.
    wow_analysis: Option<(f64, f64)>,
}

struct DecodeFieldResult {
    field: DecodedField,
    offset: f64,
}
#[derive(Clone)]
pub struct WriteableField {
    pub info: FieldInfoEntry,
    picture: Arc<WriteablePicture>,
    /// The signal-derived field phase ID, or `None` when the phase ID in `info`
    /// was instead derived from the running sequence number. Multithreaded
    /// stitching needs this to recompute `info.field_phase_id` against a global
    /// sequence number; it is not serialized and does not affect serial output.
    pub field_phase_id_raw: Option<i64>,
}

struct WriteablePicture {
    luma: LumaOutput,
    chroma: Option<Vec<u16>>,
}

impl WriteableField {
    fn new(
        info: FieldInfoEntry,
        luma: LumaOutput,
        chroma: Option<Vec<u16>>,
        field_phase_id_raw: Option<i64>,
    ) -> Self {
        Self {
            info,
            picture: Arc::new(WriteablePicture { luma, chroma }),
            field_phase_id_raw,
        }
    }

    #[inline]
    pub fn luma(&self) -> &LumaOutput {
        &self.picture.luma
    }

    #[inline]
    pub fn chroma(&self) -> Option<&[u16]> {
        self.picture.chroma.as_deref()
    }
}

#[derive(Clone, Copy)]
struct MetadataFieldState {
    out_scale: f64,
    outlinecount: usize,
}

fn demod_chroma_filt_array(
    data: &[f32],
    spec: &DecoderSpec,
    filter: &[Sos<f32>],
    blocklen: usize,
    move_by: Option<isize>,
) -> ChromaArray {
    let end = data.len().min(blocklen);
    // The chroma is f32 throughout this block-sized buffer; feed the input slice
    // straight to the SOS filters and keep the output f32 for the downstream
    // chroma pipeline.
    let mut out_chroma = sosfiltfilt_f32(filter, &data[..end]);
    if let Some(sos) = spec.chroma_filter_audio_notch.as_ref() {
        out_chroma = sosfiltfilt_f32(sos, &out_chroma);
    }
    // f_video_notch is populated exactly when the user passed --notch.
    if let Some(sos) = spec.chroma_filter_video_notch.as_ref() {
        out_chroma = sosfiltfilt_f32(sos, &out_chroma);
    }
    shift_chroma_and_remove_dc(out_chroma, move_by.unwrap_or_else(|| spec.chroma_offset()))
}

/// One phase of the rational cubic resampler: the integer tap base for output
/// `i` and the four Catmull-Rom weights, exactly as `cubic_resample` derived
/// them (the fractional offset narrows to f32 before the weight polynomial).
fn resample_phase_taps(input_rate: usize, output_rate: usize, i: usize) -> (isize, [f32; 4]) {
    let scale = input_rate as f64 / output_rate as f64;
    let pos = (i % output_rate) as f64 * scale;
    let idx_off = pos.floor();
    let f = (pos - idx_off) as f32;
    let f2 = f * f;
    let f3 = f2 * f;
    let weights = [
        -0.5 * f3 + f2 - 0.5 * f,
        1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
        0.5 * f3 - 0.5 * f2,
    ];
    let base = (i / output_rate * input_rate) as isize + idx_off as isize;
    (base, weights)
}

pub(crate) struct ChromaSepClass {
    ratio_den: usize,
    /// Offset of the first fused tap relative to the output index.
    win_lo: isize,
    /// Fused tap weights, one tap-offset row of `ratio_den` phase entries each.
    fused_taps: Vec<Vec<f32>>,
}

impl ChromaSepClass {
    pub(crate) fn new(fs: f64, fsc: f64) -> Self {
        let multiplier = 8usize;
        let delay = multiplier / 2;
        let fscx = (fsc * multiplier as f64 * 1e6) as usize;
        let (ratio_num, ratio_den) = limit_denominator(fscx as f64 / fs, 1000);

        // The chain this implements — cubic upsample to multiplier*fsc, the
        // half-cycle comb average, cubic downsample back — is linear and
        // periodic with the output phase `i % ratio_den`, so it composes into
        // one short FIR per phase over a contiguous input window. Probe each
        // phase far enough into an imagined infinite stream that no composed
        // index wraps, and accumulate the products of the stage weights per
        // input tap; the offsets relative to the output index are
        // phase-periodic by construction.
        let probe_cycle = delay + 2;
        let mut per_phase: Vec<std::collections::BTreeMap<isize, f64>> =
            Vec::with_capacity(ratio_den);
        for p in 0..ratio_den {
            let i = p + probe_cycle * ratio_den;
            let mut taps = std::collections::BTreeMap::new();
            let (down_base, down_w) = resample_phase_taps(ratio_num, ratio_den, i);
            for (k, &dw) in down_w.iter().enumerate() {
                let j = down_base - 1 + k as isize;
                for branch in [0, delay as isize] {
                    let (up_base, up_w) =
                        resample_phase_taps(ratio_den, ratio_num, (j - branch) as usize);
                    for (m, &uw) in up_w.iter().enumerate() {
                        let xi = up_base - 1 + m as isize - i as isize;
                        *taps.entry(xi).or_insert(0.0) += f64::from(dw) * 0.5 * f64::from(uw);
                    }
                }
            }
            per_phase.push(taps);
        }
        let win_lo = per_phase
            .iter()
            .filter_map(|taps| taps.keys().next().copied())
            .min()
            .unwrap_or(0);
        let win_hi = per_phase
            .iter()
            .filter_map(|taps| taps.keys().next_back().copied())
            .max()
            .unwrap_or(0);
        let width = (win_hi - win_lo + 1) as usize;
        let mut fused_taps = vec![vec![0.0f32; ratio_den]; width];
        for (p, taps) in per_phase.iter().enumerate() {
            for (&offset, &coeff) in taps {
                fused_taps[(offset - win_lo) as usize][p] = coeff as f32;
            }
        }
        Self {
            ratio_den,
            win_lo,
            fused_taps,
        }
    }

    // It resamples the luminance data to self.multiplier * fsc, applies the
    // comb filter, then resamples it back — evaluated as the precomposed
    // per-phase FIR. The few outputs whose window leaves the buffer (where
    // the old chain clamped or wrapped) use clamped taps; they sit well
    // inside the discarded block-cut margins.
    fn work(&self, luminance: &[f32]) -> Vec<f32> {
        let len = luminance.len();
        let den = self.ratio_den;
        let width = self.fused_taps.len();
        let win_lo = self.win_lo;

        let edge_at = |i: usize| -> f32 {
            let p = i % den;
            let mut acc = 0.0f32;
            for (u, taps) in self.fused_taps.iter().enumerate() {
                acc += taps[p] * sample_clamped(luminance, i as isize + win_lo + u as isize);
            }
            acc
        };

        let head_end = ((-win_lo).max(0) as usize).min(len);
        let tail_start = (len as isize - win_lo - width as isize + 1)
            .clamp(head_end as isize, len as isize) as usize;

        let mut out = Vec::with_capacity(len);
        out.extend((0..head_end).map(edge_at));
        // The interior walks whole phase cycles; every tap row pairs a
        // contiguous weight run with a contiguous input window, so the
        // accumulation passes vectorize with no per-sample indexing.
        let mut i = head_end;
        while i < tail_start {
            let p = i % den;
            let run = (tail_start - i).min(den - p);
            let x0 = (i as isize + win_lo) as usize;
            let start = out.len();
            out.extend(
                self.fused_taps[0][p..p + run]
                    .iter()
                    .zip(&luminance[x0..x0 + run])
                    .map(|(&w, &x)| w * x),
            );
            let dst = &mut out[start..start + run];
            for (u, taps) in self.fused_taps.iter().enumerate().skip(1) {
                let xs = &luminance[x0 + u..x0 + u + run];
                for ((acc, &w), &x) in dst.iter_mut().zip(&taps[p..p + run]).zip(xs) {
                    *acc += w * x;
                }
            }
            i += run;
        }
        out.extend((tail_start..len).map(edge_at));
        out
    }
}

fn limit_denominator(x: f64, max_den: usize) -> (usize, usize) {
    let mut best_num = 0usize;
    let mut best_den = 1usize;
    let mut best_err = f64::INFINITY;
    for den in 1..=max_den {
        let num = (x * den as f64).round() as usize;
        let err = (x - num as f64 / den as f64).abs();
        if err < best_err {
            best_err = err;
            best_num = num;
            best_den = den;
        }
    }
    (best_num, best_den)
}

fn sample_clamped(data: &[f32], idx: isize) -> f32 {
    data[idx.clamp(0, data.len() as isize - 1) as usize]
}

fn downscale_raw_vec(
    field: &mut DecodedField,
    lineinfo: Option<&[f32]>,
    linesout: Option<usize>,
    outwidth: Option<usize>,
    use_burst_channel: bool,
) -> Result<Vec<f32>> {
    let actual_linelocs = lineinfo
        .or(field.linelocs.as_deref())
        .context("missing linelocs")?;
    let outwidth = outwidth.unwrap_or(field.outlinelen);
    let linesout = linesout.unwrap_or(field.outlinecount);
    let k = field.wow_interpolation_method.spline_degree();
    let out_len = linesout * outwidth;
    // The line-location spline is solved in f64; widen the f32 sync output here.
    let actual_linelocs = actual_linelocs
        .iter()
        .map(|&v| f64::from(v))
        .collect::<Vec<f64>>();
    let actual_linelocs = actual_linelocs.as_slice();
    let expected_linelocs = (0..actual_linelocs.len())
        .map(|i| i as f64 * field.inlinelen)
        .collect::<Vec<_>>();
    let outscale = field.inlinelen / field.outlinelen as f64;
    let outsamples = field.outlinecount * field.outlinelen;
    let outline_offset = (field.lineoffset + 1) * field.outlinelen;
    let eval_count = outsamples + outline_offset;
    let (dsout, median, mad) = {
        let channel = if use_burst_channel {
            field.data.video.demod_burst.as_slice()
        } else {
            field.data.video.demod.as_slice()
        };
        scale_field(
            channel,
            out_len,
            &expected_linelocs,
            actual_linelocs,
            k,
            ScaleFieldParams {
                eval_scale: outscale,
                eval_count,
                lineoffset: field.lineoffset,
                outwidth,
                wow_level_adjust_smoothing: field.wow_level_adjust_smoothing,
                level_adjust_threshold: 15.0,
                cached_median_mad: field.wow_analysis,
            },
        )?
    };
    field.wow_analysis = Some((median, mad));
    Ok(dsout)
}

/// Map a (first-field, second-phase) pair to the 1..=4 fieldPhaseID metadata value.
fn field_phase_id(first_field: bool, second_phase: bool) -> i64 {
    match (first_field, second_phase) {
        (true, true) => 1,
        (false, false) => 2,
        (true, false) => 3,
        (false, true) => 4,
    }
}

// Color burst window with the small padding used by both the phase-rotation
// lock and the chroma upconvert.
fn padded_burst_area(spec: &DecoderSpec) -> (isize, isize) {
    (
        (spec.sys_color_burst_us[0] * spec.sys_outfreq).floor() as isize - 5,
        (spec.sys_color_burst_us[1] * spec.sys_outfreq).ceil() as isize + 10,
    )
}

// Minimal state needed to progress a decode: the immutable spec, the running
// input offset, and the speculative-predecode/field-ordering memory.
pub struct Decoder {
    spec: Arc<DecoderSpec>,
    fdoffset: u64,
    fdoffset_frac: f64,
    inter_field_state: InterFieldState,
    resync_state: ResyncState,
    chroma_afc_state: ChromaAfcState,
    fields: Vec<FieldInfoEntry>,
    seen_first_field: bool,
    metadata_field: Option<MetadataFieldState>,
    lastvalidfield: Vec<Option<WriteableField>>,
    pending_result: Option<DecodeFieldResult>,
    has_pending: bool,
    duplicate_prev_field: bool,
}

impl Decoder {
    pub fn new(spec: Arc<DecoderSpec>, fdoffset: u64) -> Self {
        let inter_field_state = InterFieldState::new(spec.track_phase);
        let resync_state = ResyncState::new(&spec);
        let chroma_afc_state = ChromaAfcState::new(&spec);
        Self {
            spec,
            fdoffset,
            fdoffset_frac: 0.0,
            inter_field_state,
            resync_state,
            chroma_afc_state,
            fields: Vec::new(),
            seen_first_field: false,
            metadata_field: None,
            lastvalidfield: vec![None, None],
            pending_result: None,
            has_pending: false,
            duplicate_prev_field: true,
        }
    }

    // Decode as many fields as the window `data` allows. `data` is a window of the
    // input samples starting at absolute sample offset `data_start`;
    // `final_chunk` is set only when it reaches the true end of input. Returns the
    // absolute offset before which input is no longer needed (so the caller can drop
    // or skip past it) together with the fields produced. When more input is needed
    // mid-field, decoding pauses with state intact so a later call with an extended
    // window resumes it bit-identically.
    pub fn decode(
        &mut self,
        data: &[f32],
        data_start: u64,
        final_chunk: bool,
    ) -> Result<(u64, Vec<WriteableField>)> {
        let data_start = data_start as usize;
        let readlen = self.spec.readlen();
        let blocksize = BLOCKSIZE;
        let usable_blocksize = self.spec.usable_blocksize();
        let output_lines = self.spec.output_lines();
        let bytes_per_field = self.spec.bytes_per_field();

        let mut output: Vec<WriteableField> = Vec::new();

        let mut done = false;
        while !done {
            let mut field_done = false;
            let mut picture_luma: Option<LumaOutput> = None;
            let mut picture_chroma: Option<Vec<u16>> = None;
            let mut field: Option<DecodedField> = None;

            while !field_done {
                let (decoded_field, decoded_offset) = if !self.has_pending {
                    (None, Some(0.0))
                } else if let Some(pending) = self.pending_result.take() {
                    (Some(pending.field), Some(pending.offset))
                } else {
                    (None, None)
                };

                let scheduled_prevfield = decoded_field
                    .as_ref()
                    .filter(|field_obj| field_obj.valid)
                    .map(|field_obj| PrevFieldState {
                        readloc: field_obj.readloc,
                        field_number: field_obj.field_number,
                        phase_adjust_median: field_obj.phase_adjust_median,
                    });
                let pending_frac = self.fdoffset_frac + decoded_offset.unwrap_or(0.0);
                let scheduled_readloc_value = (self.fdoffset as i64 - BLOCKCUT as i64
                    + pending_frac.floor() as i64)
                    .max(0) as u64;
                let readloc_block = scheduled_readloc_value as usize / blocksize;
                let numblocks = (readlen / blocksize) + 2;
                let block_begin = readloc_block * blocksize;
                let block_end = block_begin + (numblocks * blocksize);
                let requested_begin = block_begin / usable_blocksize;
                let requested_end = (block_end / usable_blocksize) + 1;

                // The field needs blocks [requested_begin, requested_end), each a
                // BLOCKSIZE window at absolute offset `b * usable_blocksize`. If the
                // current window does not cover all of them and more input may still
                // arrive, pause here: hand the pending field back untouched and
                // report `requested_begin`'s block as the earliest offset still
                // needed. `decode_video_block` advances the video-EQ state, so we
                // must not run it on a partial block set that a resumed call reruns.
                let needed_end = (requested_end - 1) * usable_blocksize + BLOCKSIZE;
                if needed_end > data_start + data.len() && !final_chunk {
                    if let Some(field) = decoded_field {
                        self.pending_result = Some(DecodeFieldResult {
                            field,
                            offset: decoded_offset.unwrap_or(0.0),
                        });
                    }
                    return Ok(((requested_begin * usable_blocksize) as u64, output));
                }

                // Every block contributes exactly `usable_blocksize` samples per
                // channel, so size the field buffers up front and let each block
                // append straight into them. This drops the per-block channel
                // Vecs and the field-wide concatenation copy that followed.
                let field_capacity = (requested_end - requested_begin) * usable_blocksize;
                let mut video = VideoChannels {
                    demod: Vec::with_capacity(field_capacity),
                    demod_05: Vec::with_capacity(field_capacity),
                    demod_burst: Vec::with_capacity(field_capacity),
                    envelope: Vec::with_capacity(field_capacity),
                };
                let mut completed_blocks = true;
                for b in requested_begin..requested_end {
                    // Only decode a full BLOCKSIZE window; a short tail at the true
                    // end of input leaves the sequence incomplete and ends decoding.
                    let start = b * usable_blocksize - data_start;
                    let Some(rawdata) = data.get(start..start + BLOCKSIZE) else {
                        completed_blocks = false;
                        break;
                    };
                    decode_video_block(rawdata, &self.spec, &mut video)?;
                }
                self.pending_result = if completed_blocks {
                    let rawdecode = FieldData {
                        startloc: ((block_begin / usable_blocksize) * usable_blocksize) as u64,
                        input_len: video.demod.len(),
                        video,
                    };
                    Some(predecode_field_from_rawdecode(
                        rawdecode,
                        &self.spec,
                        scheduled_prevfield,
                        &mut self.inter_field_state,
                        scheduled_readloc_value,
                        &mut self.resync_state,
                        &self.chroma_afc_state,
                    )?)
                } else {
                    None
                };
                self.has_pending = true;

                if decoded_field.is_some() {
                    let advanced = self.fdoffset_frac + decoded_offset.unwrap_or(0.0);
                    let whole = advanced.floor();
                    self.fdoffset = self.fdoffset.saturating_add_signed(whole as i64);
                    self.fdoffset_frac = advanced - whole;
                }

                let decoded_was_none = decoded_field.is_none();
                field = decoded_field;

                if let Some(mut field_obj) = field.take() {
                    if field_obj.valid {
                        // Predecode populated the wow-analysis cache before
                        // refining linelocs/lineoffset, so it is stale now.
                        // Drop it; the luma pass below recomputes with the final
                        // geometry and the chroma pass then reuses that.
                        field_obj.wow_analysis = None;
                        let mut luma = downscale_raw_vec(
                            &mut field_obj,
                            None,
                            Some(output_lines),
                            None,
                            false,
                        )?;
                        if self.spec.rf_y_comb != 0.0 {
                            luma = y_comb(&luma, field_obj.outlinelen, self.spec.rf_y_comb);
                        }

                        if self.spec.rf_export_raw_tbc {
                            picture_luma = Some(LumaOutput::Raw(luma));
                        } else {
                            let mut ire0 = f64::from(self.spec.sys_ire0);
                            if self.spec.rf_ire0_adjust
                                && luma.len() == field_obj.outlinecount * field_obj.outlinelen
                            {
                                ire0 = ire0_adjust_from_picture(&luma, &field_obj);
                                tracing::debug!(ire0, "calculated ire0");
                            }
                            if let Some(track_phase) = self.inter_field_state.track_phase {
                                let idx = (track_phase ^ (field_obj.field_number % 2)) as usize;
                                ire0 += self.spec.sys_track_ire0_offset[idx];
                            }
                            picture_luma = Some(LumaOutput::Encoded(hz_to_output_array(
                                &self.spec,
                                &luma,
                                ire0,
                                field_obj.out_scale.unwrap(),
                            )));
                        }
                        self.metadata_field = Some(MetadataFieldState {
                            out_scale: field_obj.out_scale.unwrap(),
                            outlinecount: field_obj.outlinecount,
                        });

                        picture_chroma =
                            decode_chroma(&mut field_obj, &self.spec, &mut self.chroma_afc_state)?;

                        field_obj.prevfield = None;
                        field_done = true;
                    }
                    field = Some(field_obj);
                }
                if decoded_was_none && decoded_offset.is_none() {
                    field = None;
                    break;
                }
            }

            let Some(field_obj) = field else {
                done = true;
                continue;
            };
            if !field_obj.valid {
                done = true;
                continue;
            }

            if !self.fields.is_empty() || field_obj.is_first_field.unwrap_or(false) {
                let prevfi_1 = self.fields.last().cloned();
                let prevfi_2 = self.fields.iter().rev().nth(1).cloned();

                // --- Seed values (may be mutated below) ---
                let mut is_first_field = field_obj.is_first_field.unwrap_or(false);
                let detected_first_field = is_first_field;
                let mut is_duplicate_field = false;
                let mut sync_conf = sync_confidence_from_linelocs(&field_obj)?;
                let seq_no = self.fields.len() + 1;
                let disk_loc = roundfloat(field_obj.readloc as f64 / bytes_per_field as f64, 1);
                let file_loc = field_obj.readloc;

                // Field phase ID.
                let field_phase_id = field_obj.field_phase_id.unwrap_or_else(|| {
                    field_phase_id(is_first_field, (seq_no / 2).is_multiple_of(2))
                });
                let mut write_field = true;

                // Dropout detection.
                let mut drop_outs: Option<DropOuts> = None;
                if self.spec.do_dod {
                    let (_field_average, dropout_lines, dropout_starts, dropout_ends) =
                        detect_dropouts_rf(
                            &self.spec,
                            &field_obj,
                            DOD_MERGE_THRESHOLD,
                            DOD_MIN_LENGTH,
                        )?;
                    if !dropout_lines.is_empty() {
                        drop_outs = Some(DropOuts {
                            field_line: dropout_lines,
                            startx: dropout_starts,
                            endx: dropout_ends,
                        });
                    }
                }

                // Decode-fault bitmap.
                let mut decode_faults = 0i64;
                let vits_metrics = {
                    let metrics_luma = picture_luma
                        .as_ref()
                        .context("valid field missing luma picture")?;
                    compute_vits_metrics(&self.spec, &field_obj, metrics_luma)?
                };

                // Interlaced video requires alternating fields. Handle cases where
                // fields repeat (recording breaks, progressive content, etc.).
                if let Some(prevfi) = prevfi_1
                    .as_ref()
                    .filter(|prevfi| prevfi.is_first_field == is_first_field)
                {
                    // Measure the field spacing against the previous field's exact
                    // sample position rather than its rounded reported location, so
                    // the progressive-content heuristic does not hinge on the
                    // reported value's precision.
                    let distance_from_previous_field =
                        disk_loc - roundfloat(prevfi.file_loc as f64 / bytes_per_field as f64, 1);
                    if prevfi.detected_first_field == detected_first_field
                        && prevfi_2
                            .as_ref()
                            .is_some_and(|prev| prev.detected_first_field)
                            == prevfi.detected_first_field
                        && inrange(distance_from_previous_field, 0.9, 1.1)
                    {
                        // treat this as progressive, and manually flip the field order.
                        tracing::warn!("Detected progressive video content..., manually flipping the field order to compensate");
                        decode_faults |= 1;
                        sync_conf = 10;
                        is_first_field = !prevfi.is_first_field;
                    } else {
                        match self.spec.field_order_action {
                            FieldOrderAction::Duplicate => self.duplicate_prev_field = true,
                            FieldOrderAction::Drop => self.duplicate_prev_field = false,
                            FieldOrderAction::Detect => {
                                if distance_from_previous_field > 1.1 {
                                    self.duplicate_prev_field = true;
                                } else if distance_from_previous_field < 0.9 {
                                    self.duplicate_prev_field = false;
                                } else {
                                    self.duplicate_prev_field = !self.duplicate_prev_field;
                                }
                            }
                            FieldOrderAction::None => {}
                        }
                        // Every same-order outcome marks a skipped-field fault and
                        // zeroes confidence; the branches differ only in the remedy.
                        decode_faults |= 4;
                        sync_conf = 0;
                        if self.spec.field_order_action == FieldOrderAction::None {
                            tracing::warn!("Possibly skipped field (Two fields with same isFirstField in a row), manually flipping the field order to compensate");
                            is_first_field = !prevfi.is_first_field;
                        } else if self.duplicate_prev_field {
                            tracing::warn!("Possibly skipped field (Two fields with same isFirstField in a row), duplicating the last field to compensate...");
                            is_duplicate_field = true;
                        } else {
                            tracing::warn!("Possibly skipped field (Two fields with same isFirstField in a row), dropping the last field to compensate...");
                            write_field = false;
                        }
                    }
                } else if field_obj.is_first_field.unwrap_or(false) {
                    self.seen_first_field = true;
                }

                let info = FieldInfoEntry {
                    is_first_field,
                    detected_first_field,
                    is_duplicate_field,
                    sync_conf,
                    seq_no,
                    disk_loc: disk_loc as f32,
                    file_loc,
                    field_phase_id,
                    vits_metrics,
                    drop_outs,
                    decode_faults: (decode_faults != 0).then_some(decode_faults),
                };

                // Slot fields by their detected order so the duplicate path can pair
                // the current field with the most recent opposite-order field.
                let idx = usize::from(field_obj.is_first_field.unwrap_or(false));
                if write_field {
                    let dataset = WriteableField::new(
                        info,
                        picture_luma
                            .take()
                            .context("valid field missing luma picture")?,
                        picture_chroma.take(),
                        field_obj.field_phase_id,
                    );
                    self.lastvalidfield[idx] = Some(dataset.clone());
                    if is_duplicate_field {
                        if let Some(other) = self.lastvalidfield[1 - idx].clone() {
                            self.fields.push(other.info.clone());
                            output.push(other);
                            self.fields.push(dataset.info.clone());
                            output.push(dataset);
                        }
                    } else {
                        self.fields.push(dataset.info.clone());
                        output.push(dataset);
                    }
                }
            }
        }

        Ok((self.fdoffset, output))
    }

    pub fn metadata(&self) -> Option<DecoderMetadata> {
        let spec: &DecoderSpec = &self.spec;
        let metadata_field = self.metadata_field?;
        let ire_to_output = |ire: f64| {
            hz_to_output_scalar(
                spec,
                iretohz(f64::from(spec.sys_ire0), f64::from(spec.sys_hz_ire), ire),
                metadata_field.out_scale,
            )
        };
        let black = ire_to_output(spec.black_ire());
        let white = ire_to_output(100.0);
        let to_sample = |us: f64| (us * spec.sys_outfreq + BADJ).round_ties_even() as i64;
        let system = match (spec.sys_frame_lines, spec.color_system) {
            (LineSystem::Line525, ColorSystem::Pal) => "PAL-M",
            (LineSystem::Line525, _) => "NTSC",
            _ => "PAL",
        };
        Some(DecoderMetadata {
            system,
            field_width: spec.sys_outlinelen,
            sample_rate: spec.sys_outfreq * 1_000_000.0,
            black_16b_ire: black * (1.0 - spec.level_adjust as f64),
            white_16b_ire: white * (1.0 + spec.level_adjust as f64),
            field_height: metadata_field.outlinecount,
            colour_burst_start: to_sample(spec.sys_color_burst_us[0]),
            colour_burst_end: to_sample(spec.sys_color_burst_us[1]),
            active_video_start: to_sample(spec.sys_active_video_us[0]),
            active_video_end: to_sample(spec.sys_active_video_us[1]),
        })
    }
}
