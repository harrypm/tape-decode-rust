use super::*;

fn unwrap_hilbert(input: &[Complex32], freq: f64, offset: f32) -> Vec<f32> {
    // `unwrap_angles` computes each phase derivative in f32, so the result is an
    // f32-precision signal; store it as f32 instead of widening to f64. The
    // downstream `rfft_f32` narrows to f32 regardless, so this is exact while
    // halving the block-sized demod buffer's write and read traffic.
    //
    // `offset` recenters the demod (the 0-IRE carrier frequency `sys_ire0`) so
    // the signal sits near zero rather than on a multi-MHz pedestal. The first
    // sample is the only slot `unwrap_angles` leaves untouched, so seed it with
    // the same `-offset` the rest receive (its baseline value was 0) to keep the
    // whole buffer consistently centered.
    let mut output = vec![-offset; input.len()];
    unwrap_angles(input, &mut output, freq as f32, offset);
    output
}

fn slice_max(values: &[f32]) -> f32 {
    let mut max = values[0];
    for &value in &values[1..] {
        if value > max {
            max = value;
        }
    }
    max
}

fn replace_spikes(
    demod: &mut [f32],
    demod_diffed: &[f32],
    max_value: f32,
    replace_start: usize,
    replace_end: usize,
) {
    let to_fix: Vec<usize> = demod
        .iter()
        .enumerate()
        .filter_map(|(i, &sample)| (sample > max_value).then_some(i))
        .collect();

    for i in to_fix {
        let start = i.saturating_sub(replace_start);
        let end = (i + replace_end).min(demod_diffed.len() - 1);
        if slice_max(&demod_diffed[start..end]) < slice_max(&demod[start..end]) {
            demod[start..end].copy_from_slice(&demod_diffed[start..end]);
        }
    }
}

/// Forward FFT of a real signal to its `n/2 + 1` unique spectrum bins.
/// The r2c transform consumes its input as scratch, so the buffer is taken by
/// value; callers whose signal must survive pass a copy via `rfft_f32`.
fn rfft_owned_f32(mut buffer: Vec<f32>, r2c: &dyn RealToComplex<f32>) -> Vec<Complex32> {
    assert_eq!(buffer.len(), r2c.len());
    let mut output = r2c.make_output_vec();
    r2c.process(&mut buffer, &mut output)
        .expect("r2c forward FFT failed");
    output
}

fn rfft_f32(input: &[f32], r2c: &dyn RealToComplex<f32>) -> Vec<Complex32> {
    rfft_owned_f32(input.to_vec(), r2c)
}

/// See `rfft_owned_f32` for the ownership contract.
fn irfft_owned_f32(
    mut spectrum: Vec<Complex32>,
    n: Option<usize>,
    c2r: &dyn ComplexToReal<f32>,
) -> Vec<f32> {
    if spectrum.is_empty() {
        return Vec::new();
    }

    let n = n.unwrap_or_else(|| 2 * (spectrum.len() - 1));
    assert_eq!(n, c2r.len());
    assert_eq!(spectrum.len(), (n / 2) + 1);

    // The c2r transform rebuilds the signal from the unique bins directly
    // (half-length inner FFT) instead of mirroring out the full Hermitian
    // spectrum and running a full-length complex inverse. It rejects residual
    // imaginary parts on the DC/Nyquist bins, which the filters can leave
    // behind; the full-spectrum path simply dropped them (only the real part
    // was kept), so zero them to match.
    spectrum[0].im = 0.0;
    spectrum[n / 2].im = 0.0;
    let mut output = c2r.make_output_vec();
    c2r.process(&mut spectrum, &mut output)
        .expect("c2r inverse FFT failed");
    let inv_scale = 1.0 / n as f32;
    for sample in &mut output {
        *sample *= inv_scale;
    }
    output
}

fn irfft_f32(input: &[Complex32], n: Option<usize>, c2r: &dyn ComplexToReal<f32>) -> Vec<f32> {
    irfft_owned_f32(input.to_vec(), n, c2r)
}

/// Analytic-signal spectrum from the `n/2 + 1` unique bins: keep DC and
/// Nyquist, double the interior bins, zero the negative frequencies.
fn analytic_spectrum(half: &[Complex32], n: usize) -> Vec<Complex32> {
    assert_eq!(half.len(), (n / 2) + 1);
    let mut spectrum = Vec::with_capacity(n);
    spectrum.push(half[0]);
    spectrum.extend(half[1..half.len() - 1].iter().map(|&bin| bin * 2.0));
    spectrum.push(half[half.len() - 1]);
    spectrum.resize(n, Complex32::new(0.0, 0.0));
    spectrum
}

fn sub_deemphasis(
    spec: &DecoderSpec,
    out_video: &[f32],
    out_video_fft: &[Complex32],
) -> Result<Vec<f32>> {
    let nl_high_pass_f = spec
        .video_nl_high_pass_f
        .as_ref()
        .context("missing nonlinear high-pass filter")?;
    if out_video_fft.len() != nl_high_pass_f.len() {
        bail!("sub_deemphasis FFT inputs must have equal length");
    }

    let mut deviation = spec.video_sub_deemphasis_deviation();

    let high_pass_fft = spectrum_times_filter(out_video_fft, nl_high_pass_f);

    deviation /= 2.0;
    if deviation == 0.0 {
        bail!("sub_deemphasis deviation must be non-zero");
    }

    let static_gain = spec
        .decoder_nonlinear_static_factor
        .filter(|&value| value != 0.0);

    // Get the instantaneous amplitude of the signal using the hilbert transform
    // and divide by the formats specified deviation so we get a amplitude compared to the specifications references.
    // The forward transform of `hf_part` is `high_pass_fft` itself (the c2r
    // inverse below is its exact counterpart), so build the analytic spectrum
    // straight from it instead of running a redundant r2c. The round trip
    // would drop the DC and Nyquist imaginary parts; match that here.
    let analytic = {
        let n = out_video.len();
        let mut spectrum = analytic_spectrum(&high_pass_fft, n);
        spectrum[0].im = 0.0;
        spectrum[n / 2].im = 0.0;
        ifft_complex_owned_f32(spectrum, spec.fft_block_inverse_f32.as_ref())
    };
    let hf_part = irfft_owned_f32(
        high_pass_fft,
        Some(out_video.len()),
        spec.fft_block_c2r_f32.as_ref(),
    );
    // `Complex::norm` calls libm `hypot`, whose overflow-safe scaling is wasted
    // on these bounded analytic samples. Computing the magnitude directly is far
    // cheaper (and matches the `re*re + im*im` idiom used elsewhere).
    let inv_deviation = 1.0 / deviation;
    // The amplitude buffer is block-sized; the analytic samples are already f32,
    // so compute each magnitude directly in f32 and store it as f32.
    let mut amplitude: Vec<f32> = analytic
        .iter()
        .map(|sample| sample.re.mul_add(sample.re, sample.im * sample.im).sqrt() * inv_deviation)
        .collect();

    // Clip the value after filtering to make sure we don't go negative.
    amplitude = sosfiltfilt_f32(&spec.video_nl_amplitude_lpf, &amplitude);
    // Hoist the per-spec tuning out of the loop so the body is loop-invariant
    // arithmetic the compiler can unswitch and vectorize; a multiply by 1.0 is
    // an exact identity, so the optional scales fold into unconditional muls.
    let scaling_1 = spec.decoder_nonlinear_scaling_1.unwrap_or(1.0);
    let exp_scaling = spec.decoder_nonlinear_exp_scaling;
    let scaling_2 = spec.decoder_nonlinear_scaling_2.unwrap_or(1.0);
    let logistic = spec
        .decoder_nonlinear_logistic
        .filter(|&(_, rate)| rate > 0.0);
    for sample in &mut amplitude {
        // The nonlinear chain runs in f32; its inputs (the analytic amplitude
        // and the amplitude-LPF output) are already f32-precision.
        let mut value = *sample;
        if value < 0.0 {
            value = 0.0;
        }
        value *= scaling_1;
        // Scale the amplitude by a exponential factore (typically less than 1 so it ends up being a root function of sorts)
        value = powf_fast_nonneg(value, exp_scaling);
        value *= scaling_2;
        if let Some((mid, rate)) = logistic {
            value *= 1.0 / (1.0 + exp_fast(-rate * (value - mid)));
        }
        *sample = value;
    }

    // Scale the band-pass filtered signal by one minus the resulting referenc
    // e.g this means it get scaled more at lower amplitudes.
    // Folding the optional static gain into an unconditional multiply (0.0 when
    // absent) and collecting through the exact-size zip keeps the loop free of
    // per-sample branches and capacity checks, so it vectorizes.
    let static_gain = static_gain.unwrap_or(0.0);
    let output = out_video
        .iter()
        .zip(&hf_part)
        .zip(&amplitude)
        .map(|((&video, &hf), &amp)| {
            let scaled_hf = hf * (1.0 - amp);
            // And subtract it from the output signal.
            video - scaled_hf - hf * static_gain
        })
        .collect();

    Ok(output)
}

// A single blanket impl over `ComplexFloat` covers both the real (`f32`/`f64`)
// and complex (`Complex32`/`Complex64`) filter coefficients; `re()`/`im()` make
// the real case a complex value with zero imaginary part. `SpectrumValue`
// spectrum bins reuse the same accessors.
trait SpectrumFilterSample: Copy {
    fn re_f32(self) -> f32;
    fn im_f32(self) -> f32;
}

impl<T> SpectrumFilterSample for T
where
    T: ComplexFloat,
    T::Real: Float,
{
    #[inline(always)]
    fn re_f32(self) -> f32 {
        self.re().to_f32().unwrap()
    }

    #[inline(always)]
    fn im_f32(self) -> f32 {
        self.im().to_f32().unwrap()
    }
}

#[inline(always)]
fn multiply_spectrum_sample<T>(sample: Complex32, filter: T) -> Complex32
where
    T: SpectrumFilterSample,
{
    let sample_re = sample.re;
    let sample_im = sample.im;
    let filter_re = filter.re_f32();
    let filter_im = filter.im_f32();
    Complex32::new(
        sample_re.mul_add(filter_re, -(sample_im * filter_im)),
        sample_re.mul_add(filter_im, sample_im * filter_re),
    )
}

fn multiply_spectrum_real<T>(spectrum: &mut [Complex32], filter: &[T])
where
    T: Float,
{
    assert_eq!(filter.len(), spectrum.len(), "length mismatch");
    for (sample, &gain) in spectrum.iter_mut().zip(filter) {
        let gain = gain.to_f32().unwrap();
        *sample = Complex32::new(sample.re * gain, sample.im * gain);
    }
}

fn spectrum_times_filter<T>(spectrum: &[Complex32], filter: &[T]) -> Vec<Complex32>
where
    T: SpectrumFilterSample,
{
    assert_eq!(filter.len(), spectrum.len(), "length mismatch");
    spectrum
        .iter()
        .zip(filter)
        .map(|(&sample, &gain)| multiply_spectrum_sample(sample, gain))
        .collect()
}

fn slice_vec<'a, T>(
    values: &'a [T],
    blockcut: usize,
    blockcut_end: usize,
    name: &'static str,
) -> &'a [T] {
    let len = values.len();
    assert!(
        blockcut + blockcut_end <= len,
        "{name} length {len} is shorter than block cuts {blockcut}+{blockcut_end}"
    );
    &values[blockcut..len - blockcut_end]
}

fn max_excluding_edges(values: &[f32], edge: usize) -> f32 {
    if values.len() <= edge * 2 {
        return f32::NEG_INFINITY;
    }
    values[edge..values.len() - edge]
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &value| acc.max(value))
}

fn ediff1d_complex_to_begin_zero(values: &[Complex32]) -> Vec<Complex32> {
    let mut out = Vec::with_capacity(values.len());
    out.push(Complex32::new(0.0, 0.0));
    for pair in values.windows(2) {
        out.push(pair[1] - pair[0]);
    }
    out
}

pub(crate) fn decode_video_block(
    rawdata: &[f32],
    spec: &DecoderSpec,
    out: &mut VideoChannels,
) -> Result<()> {
    if rawdata.len() < BLOCKSIZE {
        bail!(
            "rawdata length {} is shorter than blocklen {}",
            rawdata.len(),
            BLOCKSIZE
        );
    }
    // The whole RF conditioning chain works on the unique half spectrum; the
    // full (Hermitian) spectrum is only spelled out where the analytic signal
    // needs its one-sided complex inverse.
    let mut indata_fft = rfft_f32(&rawdata[..BLOCKSIZE], spec.fft_block_r2c_f32.as_ref());
    let half_bins = indata_fft.len();

    if let Some(video_notch_f) = &spec.video_notch_filter {
        multiply_spectrum_real(&mut indata_fft, &video_notch_f[..half_bins]);
    }
    multiply_spectrum_real(&mut indata_fft, &spec.video_rf_filter[..half_bins]);

    // Analytic signal (zeroed negative frequencies).
    let hilbert_spectrum = analytic_spectrum(&indata_fft, BLOCKSIZE);
    let mut hilbert = ifft_complex_owned_f32(hilbert_spectrum, spec.fft_block_inverse_f32.as_ref());

    // The rectified envelope is |Re(analytic signal)|; derive it from `hilbert`
    // rather than repeating the hilbert-filter multiply and inverse FFT. `c.re`
    // is f32 and abs() only clears the sign bit, so keeping `raw_env` in f32 is
    // exact and halves this block-sized buffer.
    let mut raw_env: Vec<f32> = hilbert.iter().map(|c| c.re.abs()).collect();
    roll(&mut raw_env, 4);

    // `env` is a block-sized buffer that feeds the envelope output channel, the
    // mean below, and the high-boost gain.
    let env = sosfiltfilt_f32(&spec.video_env_post_filter, &raw_env);
    let env_mean = sum_algebraic(&env) / env.len() as f32;
    let env_all_nonzero = env.iter().all(|&value| value != 0.0);

    if env_all_nonzero {
        if let Some(high_boost_value) = spec.video_high_boost_value {
            let rf_top = spec
                .video_rf_top_fft_gain
                .as_ref()
                .context("high_boost_value requires rf_top")?;
            // The zero-phase band extraction is a real spectrum gain, applied
            // to the RF spectrum already in hand; only the per-sample envelope
            // gain below needs the time domain.
            let mut high_spectrum = indata_fft.clone();
            multiply_spectrum_real(&mut high_spectrum, rf_top);
            let mut high_part =
                irfft_owned_f32(high_spectrum, None, spec.fft_block_c2r_f32.as_ref());
            assert_eq!(env.len(), high_part.len(), "env length mismatch");
            // The per-sample gain numerator is constant across the block, so
            // fold it into a single scale and only divide by the envelope level
            // per sample.
            let gain_numerator = env_mean * 0.9 * high_boost_value;
            for (sample, &level) in high_part.iter_mut().zip(&env) {
                *sample *= gain_numerator / level;
            }
            let high_part_fft = rfft_f32(&high_part, spec.fft_block_r2c_f32.as_ref());
            assert_eq!(
                high_part_fft.len(),
                indata_fft.len(),
                "high_part_fft length mismatch"
            );
            for (sample, boost) in indata_fft.iter_mut().zip(high_part_fft) {
                *sample += boost;
            }
            // The high boost rewrote indata_fft; recompute the analytic signal.
            let hilbert_spectrum = analytic_spectrum(&indata_fft, BLOCKSIZE);
            hilbert = ifft_complex_owned_f32(hilbert_spectrum, spec.fft_block_inverse_f32.as_ref());
        }
    } else {
        tracing::warn!("RF signal is weak. Is your deck tracking properly?");
    }

    // The demod is recentered by `sys_ire0` (the 0-IRE carrier frequency) so the
    // whole block-level luma chain runs near zero instead of on a ~4 MHz DC
    // pedestal; the offset is added back to the stored luma channels at the end.
    // This keeps the IIR/FFT filtering well-conditioned (the pedestal would
    // otherwise dominate the small high-frequency detail the filters extract).
    let ire0 = f64::from(spec.sys_ire0);
    let mut demod = unwrap_hilbert(&hilbert, spec.freq_hz(), spec.sys_ire0);

    // The diff-demod spike check compares against an absolute-Hz threshold;
    // shift it into the recentered domain by the same `ire0`.
    let diff_demod_check_value = iretohz(ire0, f64::from(spec.sys_hz_ire), 100.0) * 2.0 - ire0;
    if !spec.video_disable_diff_demod
        && f64::from(max_excluding_edges(&demod, 20)) > diff_demod_check_value
    {
        let hilbert_diffed = ediff1d_complex_to_begin_zero(&hilbert);
        let demod_b = unwrap_hilbert(&hilbert_diffed, spec.freq_hz(), spec.sys_ire0);
        replace_spikes(&mut demod, &demod_b, diff_demod_check_value as f32, 8, 30);
    }

    // The video-EQ and chroma-trap stages are only active for a few formats.
    // The EQ (sharpness) adds back a zero-phase highpass of the demod, which is
    // a real |H|^2 spectrum gain folded into one r2c/c2r round trip; the edge
    // ringing this trades against the time-domain cascade lands in the
    // BLOCKCUT margins, which are discarded.
    if let Some(fft_gain) = spec.video_eq_fft_gain.as_ref() {
        let mut spectrum =
            rfft_owned_f32(std::mem::take(&mut demod), spec.fft_block_r2c_f32.as_ref());
        multiply_spectrum_real(&mut spectrum, fft_gain);
        demod = irfft_owned_f32(spectrum, None, spec.fft_block_c2r_f32.as_ref());
    }

    if let Some(chroma_trap) = &spec.video_chroma_trap {
        demod = chroma_trap.work(&demod);
    }

    let demod_fft = rfft_f32(&demod, spec.fft_block_r2c_f32.as_ref());
    let out_video_fft = spectrum_times_filter(&demod_fft, &spec.video_filter);
    let mut out_video = irfft_f32(&out_video_fft, None, spec.fft_block_c2r_f32.as_ref());

    if spec.video_nldeemp_enabled {
        let highpass = spec
            .video_nl_high_pass_f
            .as_ref()
            .context("missing nonlinear high-pass filter")?;
        let hf_spectrum = spectrum_times_filter(&out_video_fft, highpass);
        let hf_part = irfft_owned_f32(hf_spectrum, None, spec.fft_block_c2r_f32.as_ref());
        // Clamp and subtract in a single pass so `hf_part` is read once instead
        // of being rewritten by a separate clamp pass and then read again.
        for (sample, hf) in out_video.iter_mut().zip(hf_part) {
            *sample -= hf.clamp(
                spec.decoder_nonlinear_highpass_limit_l,
                spec.decoder_nonlinear_highpass_limit_h,
            );
        }
    }

    if spec.video_subdeemp_enabled {
        out_video = sub_deemphasis(spec, &out_video, &out_video_fft)?;
    }

    if let Some(sos) = spec.video_fsc_notch.as_ref() {
        out_video = sosfiltfilt_f32(sos, &out_video);
    }

    let out_video05_fft = spectrum_times_filter(&demod_fft, &spec.video05_filter);
    let mut out_video05 = irfft_owned_f32(out_video05_fft, None, spec.fft_block_c2r_f32.as_ref());
    roll(
        &mut out_video05,
        -(DecoderSpec::VIDEO05_FILTER_OFFSET as isize),
    );

    // Append this block's burst channel straight onto the field buffer.
    if spec.chroma_afc_enabled() {
        out.demod_burst
            .extend_from_slice(slice_vec(rawdata, BLOCKCUT, BLOCKCUT_END, "chroma"));
    } else {
        // The burst chain (bandpass plus optional notches) is zero-phase, so it
        // runs as the precomputed |H|^2 spectrum gain in one r2c/c2r round trip
        // instead of cascaded time-domain forward/backward filters.
        let source = if spec.color_system != ColorSystem::Monochrome {
            &rawdata[..BLOCKSIZE]
        } else {
            out_video.as_slice()
        };
        let mut spectrum = rfft_f32(source, spec.fft_block_r2c_f32.as_ref());
        multiply_spectrum_real(&mut spectrum, &spec.chroma_burst_block_fft_gain);
        let filtered = irfft_owned_f32(spectrum, None, spec.fft_block_c2r_f32.as_ref());
        let out_chroma = shift_chroma_and_remove_dc(filtered, spec.chroma_offset());
        out.demod_burst
            .extend_from_slice(slice_vec(&out_chroma, BLOCKCUT, BLOCKCUT_END, "chroma"));
    }

    // For the raw-tbc export path the demodulated signal is emitted directly;
    // otherwise the filtered video is used. Both buffers are already f32, so the
    // raw path moves `demod` straight into the shared output buffer instead of
    // copying it element-by-element.
    let output_video: Vec<f32> = if spec.rf_export_raw_tbc {
        demod
    } else {
        out_video
    };
    // The three output channels share the identical `[BLOCKCUT .. len-BLOCKCUT_END]`
    // index range. Append each channel with a single bulk extend: the
    // exact-length iterators reserve once and the loops vectorize, unlike a
    // fused loop of per-sample pushes with their capacity checks.
    let demod_slice = slice_vec(&output_video, BLOCKCUT, BLOCKCUT_END, "demod");
    let demod_05_slice = slice_vec(&out_video05, BLOCKCUT, BLOCKCUT_END, "demod_05");
    let envelope_slice = slice_vec(&env, BLOCKCUT, BLOCKCUT_END, "envelope");
    // The luma channels carry the recentered demod; restore the absolute-Hz
    // pedestal here, as they leave the block, so the rest of the pipeline
    // (levels, scaling, output) is unchanged. The envelope is amplitude data and
    // was never recentered.
    let ire0_f32 = spec.sys_ire0;
    out.demod.extend(demod_slice.iter().map(|&v| v + ire0_f32));
    out.demod_05
        .extend(demod_05_slice.iter().map(|&v| v + ire0_f32));
    out.envelope.extend_from_slice(envelope_slice);

    Ok(())
}
