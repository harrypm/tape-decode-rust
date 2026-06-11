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
    max_value: f64,
    replace_start: usize,
    replace_end: usize,
) {
    let to_fix: Vec<usize> = demod
        .iter()
        .enumerate()
        .filter_map(|(i, &sample)| (f64::from(sample) > max_value).then_some(i))
        .collect();

    for i in to_fix {
        let start = i.saturating_sub(replace_start);
        let end = (i + replace_end).min(demod_diffed.len() - 1);
        if slice_max(&demod_diffed[start..end]) < slice_max(&demod[start..end]) {
            demod[start..end].copy_from_slice(&demod_diffed[start..end]);
        }
    }
}

fn ifft_complex_real_owned_f32(mut buffer: Vec<Complex32>, inverse_fft: &dyn Fft<f32>) -> Vec<f32> {
    if buffer.is_empty() {
        return Vec::new();
    }
    // Run the inverse transform without the trailing in-place 1/N scaling pass
    // and instead fold that scale into the real-part extraction. This collapses
    // two full passes over the block-sized complex buffer (scale, then map .re)
    // into a single pass, which matters on the bandwidth-bound MT path where
    // irfft runs several times per block.
    let inv_scale = 1.0 / buffer.len() as f32;
    inverse_fft.process(&mut buffer);
    convert_vec_in_place(buffer, |sample| sample.re * inv_scale)
}

fn rfft_f32(input: &[f32], forward_fft: &dyn Fft<f32>) -> Vec<Complex32> {
    let mut output = fft_real_f32(input, forward_fft);
    output.truncate((input.len() / 2) + 1);
    output
}

fn irfft_f32(input: &[Complex32], n: Option<usize>, inverse_fft: &dyn Fft<f32>) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let n = n.unwrap_or_else(|| 2 * (input.len() - 1));
    let expected_len = if n.is_multiple_of(2) {
        (n / 2) + 1
    } else {
        n.div_ceil(2)
    };
    assert_ne!(n, 0);
    assert_eq!(input.len(), expected_len);

    let mirror_end = if n.is_multiple_of(2) {
        input.len() - 1
    } else {
        input.len()
    };
    // Build the Hermitian-symmetric spectrum sequentially instead of
    // zero-filling `n` complex slots and overwriting them with a forward copy
    // plus a backward-scattered mirror. The mirror tail is exactly
    // input[1..mirror_end] reversed and conjugated, so two sequential extends
    // write every slot once (total = input.len() + mirror_end - 1 == n) and
    // skip the wasted memset that showed up as from_elem on the MT path.
    let mut spectrum = Vec::with_capacity(n);
    spectrum.extend_from_slice(input);
    spectrum.extend(
        input[1..mirror_end]
            .iter()
            .rev()
            .map(|sample| sample.conj()),
    );
    debug_assert_eq!(spectrum.len(), n);

    ifft_complex_real_owned_f32(spectrum, inverse_fft)
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
    let hf_part = irfft_f32(
        &high_pass_fft,
        Some(out_video.len()),
        spec.fft_block_inverse_f32.as_ref(),
    );

    deviation /= 2.0;
    if deviation == 0.0 {
        bail!("sub_deemphasis deviation must be non-zero");
    }

    let static_gain = spec
        .decoder_nonlinear_static_factor
        .filter(|&value| value != 0.0);

    // Get the instantaneous amplitude of the signal using the hilbert transform
    // and divide by the formats specified deviation so we get a amplitude compared to the specifications references.
    let analytic = hilbert_f32(
        &hf_part,
        spec.fft_block_forward_f32.as_ref(),
        spec.fft_block_inverse_f32.as_ref(),
    );
    // `Complex::norm` calls libm `hypot`, whose overflow-safe scaling is wasted
    // on these bounded analytic samples. Computing the magnitude directly is far
    // cheaper (and matches the `re*re + im*im` idiom used elsewhere).
    let inv_deviation = 1.0 / f64::from(deviation);
    // The amplitude buffer is block-sized; store it as f32 (the sosfiltfilt
    // recurrence still runs in f64 internally) to halve it.
    let mut amplitude: Vec<f32> = analytic
        .iter()
        .map(|sample| {
            let re = f64::from(sample.re);
            let im = f64::from(sample.im);
            (re.mul_add(re, im * im).sqrt() * inv_deviation) as f32
        })
        .collect();

    // Clip the value after filtering to make sure we don't go negative.
    amplitude = sosfiltfilt_f32(&spec.video_nl_amplitude_lpf, &amplitude);
    for sample in &mut amplitude {
        // The nonlinear chain runs in f32; its inputs (the analytic amplitude
        // and the amplitude-LPF output) are already f32-precision.
        let mut value = *sample;
        if value < 0.0 {
            value = 0.0;
        }
        if let Some(scale) = spec.decoder_nonlinear_scaling_1 {
            value *= scale;
        }
        // Scale the amplitude by a exponential factore (typically less than 1 so it ends up being a root function of sorts)
        value = value.powf(spec.decoder_nonlinear_exp_scaling);
        if let Some(scale) = spec.decoder_nonlinear_scaling_2 {
            value *= scale;
        }
        if let Some((mid, rate)) = spec.decoder_nonlinear_logistic {
            if rate > 0.0 {
                value *= 1.0 / (1.0 + (-rate * (value - mid)).exp());
            }
        }
        *sample = value;
    }

    // Scale the band-pass filtered signal by one minus the resulting referenc
    // e.g this means it get scaled more at lower amplitudes.
    let mut output = Vec::with_capacity(out_video.len());
    for ((&video, &hf), &amp) in out_video.iter().zip(hf_part.iter()).zip(amplitude.iter()) {
        let static_part = static_gain.map_or(0.0, |gain| hf * gain);
        let scaled_hf = hf * (1.0 - amp);
        // And subtract it from the output signal.
        output.push(video - scaled_hf - static_part);
    }

    Ok(output)
}

// A single blanket impl over `ComplexFloat` covers both the real (`f32`/`f64`)
// and complex (`Complex32`/`Complex64`) filter coefficients; `re()`/`im()` make
// the real case a complex value with zero imaginary part. `SpectrumValue`
// spectrum bins reuse the same accessors.
trait SpectrumFilterSample: Copy {
    fn re_f64(self) -> f64;
    fn im_f64(self) -> f64;
}

impl<T> SpectrumFilterSample for T
where
    T: ComplexFloat,
    T::Real: Float,
{
    #[inline(always)]
    fn re_f64(self) -> f64 {
        self.re().to_f64().unwrap()
    }

    #[inline(always)]
    fn im_f64(self) -> f64 {
        self.im().to_f64().unwrap()
    }
}

#[inline(always)]
fn multiply_spectrum_sample<T>(sample: Complex32, filter: T) -> Complex32
where
    T: SpectrumFilterSample,
{
    let sample_re = sample.re_f64();
    let sample_im = sample.im_f64();
    let filter_re = filter.re_f64();
    let filter_im = filter.im_f64();
    Complex32::new(
        sample_re.mul_add(filter_re, -(sample_im * filter_im)) as f32,
        sample_re.mul_add(filter_im, sample_im * filter_re) as f32,
    )
}

fn multiply_spectrum_real<T>(spectrum: &mut [Complex32], filter: &[T])
where
    T: Float,
{
    assert_eq!(filter.len(), spectrum.len(), "length mismatch");
    for (sample, &gain) in spectrum.iter_mut().zip(filter) {
        let gain = gain.to_f64().unwrap();
        *sample = Complex32::new(
            (sample.re_f64() * gain) as f32,
            (sample.im_f64() * gain) as f32,
        );
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

fn max_excluding_edges(values: &[f32], edge: usize) -> f64 {
    if values.len() <= edge * 2 {
        return f64::NEG_INFINITY;
    }
    values[edge..values.len() - edge]
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &value| acc.max(f64::from(value)))
}

fn ediff1d_complex_to_begin_zero(values: &[Complex32]) -> Vec<Complex32> {
    let mut out = Vec::with_capacity(values.len());
    out.push(Complex32::new(0.0, 0.0));
    for pair in values.windows(2) {
        out.push(Complex32::new(
            (pair[1].re_f64() - pair[0].re_f64()) as f32,
            (pair[1].im_f64() - pair[0].im_f64()) as f32,
        ));
    }
    out
}

pub(crate) fn decode_video_block(
    rawdata: &[f32],
    spec: &DecoderSpec,
    video_eq_state: Option<&mut VideoEqState>,
    out: &mut VideoChannels,
) -> Result<()> {
    if rawdata.len() < BLOCKSIZE {
        bail!(
            "rawdata length {} is shorter than blocklen {}",
            rawdata.len(),
            BLOCKSIZE
        );
    }
    let mut indata_fft = fft_real_f32(&rawdata[..BLOCKSIZE], spec.fft_block_forward_f32.as_ref());

    if let Some(video_notch_f) = &spec.video_notch_filter {
        multiply_spectrum_real(&mut indata_fft, video_notch_f);
    }
    multiply_spectrum_real(&mut indata_fft, &spec.video_rf_filter);

    // Analytic signal (the hilbert filter zeroes the negative frequencies).
    let hilbert_spectrum = spectrum_times_filter(&indata_fft, &spec.video_hilbert_filter);
    let mut hilbert = ifft_complex_owned_f32(hilbert_spectrum, spec.fft_block_inverse_f32.as_ref());

    // The rectified envelope is |Re(analytic signal)|; derive it from `hilbert`
    // rather than repeating the hilbert-filter multiply and inverse FFT. `c.re`
    // is f32 and abs() only clears the sign bit, so keeping `raw_env` in f32
    // (instead of widening to f64 here) is exact — sosfiltfilt widens each
    // sample back to f64 internally — and halves this block-sized buffer.
    let mut raw_env: Vec<f32> = hilbert.iter().map(|c| c.re.abs()).collect();
    roll(&mut raw_env, 4);

    // `env` is a block-sized buffer that is written to the f32 envelope output
    // channel and otherwise only feeds the f64 mean below and the high-boost
    // gain; store it as f32 (the sosfiltfilt recurrence still runs in f64) to
    // halve this per-block buffer. The mean stays an f64 accumulation.
    let env = sosfiltfilt_f32(&spec.video_env_post_filter, &raw_env);
    // Sum and the all-nonzero test in one scan over `env` instead of a `mean`
    // pass followed by a separate `.all()` pass.
    let mut env_sum = 0.0;
    let mut env_all_nonzero = true;
    for &value in &env {
        env_sum += f64::from(value);
        env_all_nonzero &= value != 0.0;
    }
    let env_mean = env_sum / env.len() as f64;

    if env_all_nonzero {
        if let Some(high_boost_value) = spec.video_high_boost_value {
            let rf_top = spec
                .video_rf_top_filter
                .as_ref()
                .context("high_boost_value requires rf_top")?;
            let data_filtered = ifft_complex_real_owned_f32(
                indata_fft.to_vec(),
                spec.fft_block_inverse_f32.as_ref(),
            );
            // `high_part` is re-narrowed to f32 by `fft_real_f32` below, so store
            // it as f32 (the sosfiltfilt recurrence still runs in f64) to halve
            // this block-sized buffer.
            let mut high_part = sosfiltfilt_f32(rf_top, &data_filtered);
            assert_eq!(env.len(), high_part.len(), "env length mismatch");
            for (sample, &level) in high_part.iter_mut().zip(&env) {
                let gain = (env_mean * 0.9) / f64::from(level);
                *sample *= (gain * high_boost_value) as f32;
            }
            let high_part_fft = fft_real_f32(&high_part, spec.fft_block_forward_f32.as_ref());
            assert_eq!(
                high_part_fft.len(),
                indata_fft.len(),
                "high_part_fft length mismatch"
            );
            for (sample, boost) in indata_fft.iter_mut().zip(high_part_fft) {
                *sample = Complex32::new(
                    (f64::from(sample.re) + f64::from(boost.re)) as f32,
                    (f64::from(sample.im) + f64::from(boost.im)) as f32,
                );
            }
            // The high boost rewrote indata_fft; recompute the analytic signal.
            let hilbert_spectrum = spectrum_times_filter(&indata_fft, &spec.video_hilbert_filter);
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
    if !spec.video_disable_diff_demod && max_excluding_edges(&demod, 20) > diff_demod_check_value {
        let hilbert_diffed = ediff1d_complex_to_begin_zero(&hilbert);
        let demod_b = unwrap_hilbert(&hilbert_diffed, spec.freq_hz(), spec.sys_ire0);
        replace_spikes(&mut demod, &demod_b, diff_demod_check_value, 8, 30);
    }

    // The video-EQ and chroma-trap stages are only active for a few formats.
    // Both consume and produce the f32 demod directly (their filter recurrences
    // and cubic resampling still evaluate in f64 per sample), so no widening copy
    // of the block-sized demod is needed around them.
    if let (Some(config), Some(state)) = (spec.video_eq_config.as_ref(), video_eq_state) {
        demod = state.filter_video(config, &demod);
    }

    if let Some(chroma_trap) = &spec.video_chroma_trap {
        demod = chroma_trap.work(&demod);
    }

    let demod_fft = rfft_f32(&demod, spec.fft_block_forward_f32.as_ref());
    let out_video_fft = spectrum_times_filter(&demod_fft, &spec.video_filter);
    let mut out_video = irfft_f32(&out_video_fft, None, spec.fft_block_inverse_f32.as_ref());

    if spec.video_nldeemp_enabled {
        let highpass = spec
            .video_nl_high_pass_f
            .as_ref()
            .context("missing nonlinear high-pass filter")?;
        let hf_spectrum = spectrum_times_filter(&out_video_fft, highpass);
        let hf_part = irfft_f32(&hf_spectrum, None, spec.fft_block_inverse_f32.as_ref());
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

    if let Some((b, a)) = spec.video_fsc_notch.as_ref() {
        out_video = filtfilt_f32(b, a, &out_video);
    }

    let out_video05_fft = spectrum_times_filter(&demod_fft, &spec.video05_filter);
    let mut out_video05 = irfft_f32(&out_video05_fft, None, spec.fft_block_inverse_f32.as_ref());
    roll(
        &mut out_video05,
        -(DecoderSpec::VIDEO05_FILTER_OFFSET as isize),
    );

    // Append this block's burst channel straight onto the field buffer.
    if spec.chroma_afc_enabled() {
        out.demod_burst
            .extend_from_slice(slice_vec(rawdata, BLOCKCUT, BLOCKCUT_END, "chroma"));
    } else {
        let out_chroma = if spec.color_system != ColorSystem::Monochrome {
            demod_chroma_filt_array(
                &rawdata[..BLOCKSIZE],
                spec,
                &spec.chroma_filter_video_burst,
                BLOCKSIZE,
                None,
            )
        } else {
            demod_chroma_filt_array(
                &out_video,
                spec,
                &spec.chroma_filter_video_burst,
                BLOCKSIZE,
                None,
            )
        };
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
    // index range. Append all three onto the shared field buffers in one fused
    // loop so the reads and writes stay interleaved and each sample lands in its
    // final place without a separate per-block buffer and concatenation copy.
    let demod_slice = slice_vec(&output_video, BLOCKCUT, BLOCKCUT_END, "demod");
    let demod_05_slice = slice_vec(&out_video05, BLOCKCUT, BLOCKCUT_END, "demod_05");
    let envelope_slice = slice_vec(&env, BLOCKCUT, BLOCKCUT_END, "envelope");
    // The luma channels carry the recentered demod; restore the absolute-Hz
    // pedestal here, as they leave the block, so the rest of the pipeline
    // (levels, scaling, output) is unchanged. The envelope is amplitude data and
    // was never recentered.
    let ire0_f32 = spec.sys_ire0;
    for ((&v, &v05), &env_sample) in demod_slice
        .iter()
        .zip(demod_05_slice.iter())
        .zip(envelope_slice.iter())
    {
        out.demod.push(v + ire0_f32);
        out.demod_05.push(v05 + ire0_f32);
        out.envelope.push(env_sample);
    }

    Ok(())
}
