use anyhow::{bail, Context as _, Result};
use rustfft::num_complex::{Complex32, Complex64};
use rustfft::{Fft, FftPlanner};
use sci_rs::signal::filter::design::{FilterBandType, Sos};
use std::sync::Arc;

use crate::decode::{
    butter_sos, gen_chroma_heterodyne, lfilter_zi, ChromaSepClass, BLOCKCUT, BLOCKCUT_END,
    BLOCKSIZE,
};
use crate::request::{
    BoostRampFilter, ColorSystem, DecodeRequest, DeemphasisParams, FieldOrderAction, LineSystem,
    NonlinearParams, ShelfKind, VideoLumaFilter, WowInterpolation,
};
use crate::vec_utils::convert_vec_in_place;

const CHROMA_AUDIO_NOTCH_Q: f64 = 10.0;

fn store_real_filter(values: Vec<f64>) -> Vec<f32> {
    convert_vec_in_place(values, |value| value as f32)
}

fn store_complex_filter(values: Vec<Complex64>) -> Vec<Complex32> {
    convert_vec_in_place(values, |value| {
        Complex32::new(value.re as f32, value.im as f32)
    })
}

// Sharpness control based on format parameters and sharpness setting.
//
// The immutable filter coefficients/gains live in VideoEqConfig; the mutable
// IIR delay state that carries across blocks/fields lives in VideoEqState.

pub(crate) struct VideoEqConfig {
    pub(crate) b: Vec<f64>,
    pub(crate) a: Vec<f64>,
    pub(crate) gain: f64,
    pub(crate) sharpness_level: f64,
    pub(crate) initial_z: Vec<f64>,
    _private: (),
}

pub struct DecoderSpec {
    pub(crate) freq: f64,

    pub(crate) color_system: ColorSystem,

    pub(crate) sys_fsc_mhz: f64,
    pub(crate) sys_frame_lines: LineSystem,
    pub(crate) sys_field_lines: [i64; 2],
    pub(crate) sys_line_period: f64,
    pub(crate) sys_active_video_us: [f64; 2],
    pub(crate) sys_fps: f64,
    pub(crate) sys_ire0: f32,
    pub(crate) sys_hz_ire: f32,
    pub(crate) sys_vsync_ire: f32,
    pub(crate) sys_color_burst_us: [f64; 2],
    pub(crate) sys_blacksnr_slice: [usize; 3],
    pub(crate) sys_num_pulses: usize,
    pub(crate) sys_hsync_pulse_us: f64,
    pub(crate) sys_eq_pulse_us: f64,
    pub(crate) sys_vsync_pulse_us: f64,
    pub(crate) sys_output_zero: i64,
    pub(crate) sys_outlinelen: usize,
    pub(crate) sys_outfreq: f64,
    pub(crate) sys_ld_vits_whitelocs: Vec<[usize; 3]>,
    pub(crate) sys_burst_abs_ref: Option<f32>,
    pub(crate) sys_track_ire0_offset: [f64; 2],
    pub(crate) sys_nonlinear_deviation: Option<f32>,

    pub(crate) decoder_color_under_carrier: f64,
    pub(crate) decoder_chroma_bpf_upper: f64,
    pub(crate) decoder_chroma_bpf_order: usize,
    pub(crate) decoder_chroma_bpf_lower: f64,
    pub(crate) decoder_chroma_rotation: Option<[i64; 2]>,
    pub(crate) decoder_chroma_offset: f64,
    pub(crate) decoder_nonlinear_highpass_limit_l: f32,
    pub(crate) decoder_nonlinear_highpass_limit_h: f32,
    pub(crate) decoder_nonlinear_exp_scaling: f32,
    pub(crate) decoder_nonlinear_scaling_1: Option<f32>,
    pub(crate) decoder_nonlinear_scaling_2: Option<f32>,
    pub(crate) decoder_nonlinear_logistic: Option<(f32, f32)>,
    pub(crate) decoder_nonlinear_static_factor: Option<f32>,

    pub(crate) field_order_action: FieldOrderAction,
    pub(crate) video_eq_config: Option<VideoEqConfig>,

    pub(crate) chroma_afc_narrowband: Vec<(Vec<f64>, Vec<f64>)>,
    pub(crate) chroma_afc_fine_tune_fh_ratio: f64,

    pub(crate) dod_threshold_p: f32,
    pub(crate) dod_threshold_a: Option<f32>,
    pub(crate) dod_hysteresis: f32,

    pub(crate) rf_chroma_heterodyne: Vec<Vec<f32>>,
    pub(crate) rf_fsc_wave: Vec<(f32, f32)>,

    pub(crate) rf_disable_comb: bool,
    pub(crate) rf_disable_right_hsync: bool,
    pub(crate) rf_disable_dc_offset: bool,
    pub(crate) rf_fallback_vsync: bool,
    pub(crate) rf_field_order_confidence: i64,
    pub(crate) rf_saved_levels: bool,
    pub(crate) rf_y_comb: f32,
    pub(crate) rf_write_chroma: bool,
    pub(crate) rf_skip_hsync_refine: bool,
    pub(crate) rf_export_raw_tbc: bool,
    pub(crate) rf_ire0_adjust: bool,
    pub(crate) rf_relaxed_line0: bool,
    pub(crate) rf_detect_chroma_track_phase: bool,
    pub(crate) rf_disable_burst_hsync: bool,
    pub(crate) rf_disable_phase_correction: bool,

    pub(crate) chroma_filter_video_burst: Vec<Sos<f64>>,
    pub(crate) chroma_filter_video_notch: Option<(Vec<f64>, Vec<f64>)>,
    pub(crate) chroma_filter_deemphasis: Option<(Vec<f64>, Vec<f64>)>,
    pub(crate) chroma_filter_audio_notch: Option<(Vec<f64>, Vec<f64>)>,
    pub(crate) chroma_filter_final: Vec<Sos<f64>>,

    pub(crate) video_rf_filter: Vec<f32>,
    pub(crate) video_notch_filter: Option<Vec<f32>>,
    pub(crate) video_hilbert_filter: Vec<f32>,
    pub(crate) video_env_post_filter: Vec<Sos<f64>>,
    pub(crate) video_rf_top_filter: Option<Vec<Sos<f64>>>,
    pub(crate) video_high_boost_value: Option<f64>,
    pub(crate) video_disable_diff_demod: bool,
    pub(crate) video_chroma_trap: Option<ChromaSepClass>,
    pub(crate) video_filter: Vec<Complex32>,
    pub(crate) video_nl_amplitude_lpf: Vec<Sos<f64>>,
    pub(crate) video_nl_high_pass_f: Option<Vec<Complex32>>,
    pub(crate) video_nldeemp_enabled: bool,
    pub(crate) video_subdeemp_enabled: bool,
    pub(crate) video_fsc_notch: Option<(Vec<f64>, Vec<f64>)>,
    pub(crate) video05_filter: Vec<Complex32>,

    pub(crate) fft_block_forward_f32: Arc<dyn Fft<f32>>,
    pub(crate) fft_block_inverse_f32: Arc<dyn Fft<f32>>,
    pub(crate) fft_field_forward_f32: Arc<dyn Fft<f32>>,
    pub(crate) fft_field_inverse_f32: Arc<dyn Fft<f32>>,

    pub(crate) resync_divisor: usize,
    pub(crate) resync_vsync_env_filter: (Vec<f64>, Vec<f64>),
    pub(crate) resync_serration_filter_base: [(Vec<f64>, Vec<f64>); 2],
    pub(crate) resync_serration_filter_envelope: (Vec<f64>, Vec<f64>),

    pub(crate) ntscj: bool,
    pub(crate) track_phase: Option<i64>,
    pub(crate) wow_level_adjust_smoothing: f32,
    pub(crate) wow_interpolation_method: WowInterpolation,
    pub(crate) do_dod: bool,
    pub(crate) level_adjust: f32,
    _private: (),
}

impl DecoderSpec {
    pub fn new(request: &DecodeRequest) -> Result<Self> {
        let sys_params = &request.decode_profile.sys_params;
        let decoder_params = &request.decode_profile.decoder_params;
        let decode_options = &request.decode_profile.decode_options;
        let rf_freq = request.inputfreq;
        let freq_half = rf_freq / 2.0;
        let freq_hz = rf_freq * 1_000_000.0;
        let freq_hz_half = freq_hz / 2.0;
        let rf_color_system = if decoder_params.is_composite_color {
            ColorSystem::Monochrome
        } else {
            sys_params.color_system
        };
        let do_cafc = request.cafc;
        // Validated and clamped once here; ResyncCore only reads the result.
        let level_detect_divisor = {
            let requested = request.level_detect_divisor;
            if !(1..=10).contains(&requested) {
                bail!(
                    "requested level detection divisor {} is not supported",
                    requested
                );
            } else if request.inputfreq / (requested as f64) < 4.0 {
                bail!(
                    "requested level detection divisor {} too high for frequency {} MHz",
                    requested,
                    rf_freq
                );
            } else {
                requested
            }
        };
        // Controls the sharpness EQ gain.
        let sharpness_level = request.sharpness as f64 / 100.0;
        let video_eq_config = if sharpness_level > f64::EPSILON {
            let loband = &decoder_params
                .video_eq
                .as_ref()
                .context("sharpness requires video_eq params")?
                .loband;
            let (b, a) = iir_highpass(
                freq_hz,
                loband.corner,
                loband.transition,
                loband.order_limit,
            )?;
            let initial_z = lfilter_zi(&b, &a);
            Some(VideoEqConfig {
                b,
                a,
                gain: loband.order_limit as f64,
                sharpness_level,
                initial_z,
                _private: (),
            })
        } else {
            None
        };
        let fh = sys_params.fps * sys_params.frame_lines.line_count() as f64;
        let color_under = decoder_params.color_under_carrier;
        let out_sample_rate_mhz = sys_params.fsc_mhz * 4.0;
        let fieldlen =
            sys_params.outlinelen * *sys_params.field_lines.iter().max().unwrap() as usize;

        // Standard frequency color carrier wave.
        let fsc_wave = gen_wave_at_frequency(sys_params.fsc_mhz, out_sample_rate_mhz, fieldlen);

        let chroma_afc_bpf_under_ratio = decoder_params.chroma_bpf_upper / color_under;
        let chroma_afc_out_frequency_half = out_sample_rate_mhz / 2.0;
        let chroma_afc_narrowband = if do_cafc {
            const TRANSITION_EXPAND: f64 = 12.0;
            // Carrier-frequency tolerance (percent), expanded into a transition
            // width. The high and low transitions are symmetric, so one value.
            let percent = 200.0 * fh / color_under;
            let trans = color_under * TRANSITION_EXPAND * percent / 100.0;
            let samp_rate = out_sample_rate_mhz * 1e6;
            vec![
                iir_highpass(samp_rate, color_under, trans, 200)?,
                iir_lowpass(samp_rate, color_under, trans, 200)?,
            ]
        } else {
            Vec::new()
        };

        let blocklen = BLOCKSIZE;
        let mut fft_planner_f32 = FftPlanner::<f32>::new();
        let fft_block_forward_f32 = fft_planner_f32.plan_fft_forward(blocklen);
        let fft_block_inverse_f32 = fft_planner_f32.plan_fft_inverse(blocklen);
        let fft_field_forward_f32 = fft_planner_f32.plan_fft_forward(fieldlen);
        let fft_field_inverse_f32 = fft_planner_f32.plan_fft_inverse(fieldlen);

        let is_color_under = rf_color_system != ColorSystem::Monochrome;

        let video_subdeemp_enabled =
            decoder_params.nonlinear.use_sub_deemphasis || request.subdeemp;
        let chroma_deemphasis_enabled = decoder_params.chroma_deemphasis_enabled;
        let fm_audio_notch = request.fm_audio_notch;
        let rf_write_chroma = is_color_under && !request.rf_export_raw_tbc && !request.skip_chroma;

        let rf_disable_comb = request.disable_comb || rf_color_system == ColorSystem::Secam;
        let rf_fallback_vsync = request.fallback_vsync;
        let rf_y_comb = request.y_comb * sys_params.hz_ire;

        let video_hilbert_filter = build_hilbert(blocklen);

        // Filter for rf before demodulating.
        // Only use bpf if video_bpf is defined - otherwise skip.
        let y_fm = if let Some(dp_video_bpf) = &decoder_params.video_bpf {
            let (b, a) = butter_ba(
                dp_video_bpf.order,
                &[
                    dp_video_bpf.low / freq_hz_half,
                    dp_video_bpf.high / freq_hz_half,
                ],
                FilterBandType::Bandpass,
                None,
            )?;
            Some(filtfft(&b, &a, blocklen, true))
        } else {
            None
        };
        let y_fm_lowpass = sosfiltfft(
            &butter_sos(
                decoder_params.video_lpf_extra_order,
                &[decoder_params.video_lpf_extra / freq_hz_half],
                FilterBandType::Lowpass,
            )?,
            blocklen,
        );
        let y_fm_highpass = sosfiltfft(
            &butter_sos(
                decoder_params.video_hpf_extra_order,
                &[decoder_params.video_hpf_extra / freq_hz_half],
                FilterBandType::Highpass,
            )?,
            blocklen,
        );
        let y_fm_lowpass_abs = abs_complex_owned(y_fm_lowpass);
        let y_fm_highpass_abs = abs_complex_owned(y_fm_highpass);
        let mut video_rf_filter = if let Some(y_fm) = y_fm {
            let y_fm_abs = abs_complex_owned(y_fm);
            multiply(&multiply(&y_fm_abs, &y_fm_lowpass_abs), &y_fm_highpass_abs)
        } else {
            multiply(&y_fm_lowpass_abs, &y_fm_highpass_abs)
        };

        // Add optional rf peaking filter.
        if let Some(dp_video_rf_peak) = &decoder_params.video_rf_peak {
            let peaking_filter = peaking(
                dp_video_rf_peak.freq / freq_hz_half,
                dp_video_rf_peak.gain,
                None,
                Some(dp_video_rf_peak.bandwidth / freq_hz_half),
            )?;
            let peaking_fft = filtfft(&peaking_filter.0, &peaking_filter.1, blocklen, true);
            video_rf_filter = multiply(&video_rf_filter, &abs_complex_owned(peaking_fft));
        }

        // Make sure this is an int in case it could be passed in as a string via the gui.
        if fm_audio_notch > f64::EPSILON {
            if let Some(fm_audio_channels) = &decoder_params.fm_audio_channels {
                let notch0 = gen_fft_notch(
                    fm_audio_channels.channel_0_freq,
                    fm_audio_notch,
                    freq_hz_half,
                    blocklen,
                )?;
                let notch1 = gen_fft_notch(
                    fm_audio_channels.channel_1_freq,
                    fm_audio_notch,
                    freq_hz_half,
                    blocklen,
                )?;
                let audio_fm_notch_filter = multiply(&notch0, &notch1);
                video_rf_filter =
                    multiply(&video_rf_filter, &abs_complex_owned(audio_fm_notch_filter));
            } else {
                bail!("Requested audio notch even though the format does not have audio channels specified!");
            }
        }

        if let Some(boost_ramp) = &decoder_params.boost_ramp {
            let ramp = gen_ramp_filter(boost_ramp, freq_hz_half, blocklen);
            video_rf_filter = multiply(&video_rf_filter, &ramp);
        }

        let video_rf_top_filter = if let Some(boost_bpf) = &decoder_params.boost_bpf {
            Some(butter_sos(
                1,
                &[boost_bpf.low / freq_hz_half, boost_bpf.high / freq_hz_half],
                FilterBandType::Bandpass,
            )?)
        } else {
            None
        };

        let filter_deemp = if let Some(deemph) = &decoder_params.deemph {
            gen_video_main_deemp_fft(deemph, freq_hz, blocklen)?
        } else {
            vec![Complex64::new(1.0, 0.0); blocklen / 2 + 1]
        };

        let filter_video_lpf_real = if decoder_params.video_lpf_supergauss {
            gen_video_lpf_supergauss(
                decoder_params.video_lpf_freq,
                decoder_params.video_lpf_order,
                freq_hz_half,
                blocklen,
            )
        } else {
            gen_video_lpf(
                decoder_params.video_lpf_freq,
                decoder_params.video_lpf_order,
                freq_hz_half,
                blocklen,
            )?
        };

        let custom_video =
            if let Some(custom_luma_filters) = &decoder_params.video_custom_luma_filters {
                gen_custom_video_filters(custom_luma_filters, freq_hz, blocklen)?
            } else {
                vec![Complex64::new(1.0, 0.0); blocklen / 2 + 1]
            };

        // Sync detection uses a fixed 0.5 MHz low-pass FIR so the delay and
        // coefficient response stay stable.
        let f0_5 = firwin_lowpass(65, 0.5 / freq_half);
        let filter_05 = filtfft(&f0_5, &[1.0], blocklen, false);

        let deemp_lpf = multiply(&filter_deemp, &filter_video_lpf_real);
        let video_filter = multiply(&deemp_lpf, &custom_video);
        let video05_filter = multiply(&deemp_lpf, &filter_05);

        // The post-envelope smoother is intentionally first-order; higher
        // orders alter the detector response too much.
        let video_env_post_filter =
            butter_sos(1, &[700000.0 / freq_hz_half], FilterBandType::Lowpass)?;

        let video_nl_amplitude_lpf = butter_sos(
            1,
            &[decoder_params.nonlinear.amp_lpf_freq / freq_hz_half],
            FilterBandType::Lowpass,
        )?;
        let video_fsc_notch = if decode_options.use_fsc_notch_filter {
            Some(ba_to_vec(iirnotch(sys_params.fsc_mhz / freq_half, 2.0)?))
        } else {
            None
        };
        let video_nl_high_pass_f = if request.video_nldeemp_enabled || video_subdeemp_enabled {
            Some(gen_nonlinear_bandpass(
                &decoder_params.nonlinear,
                freq_hz_half,
                blocklen,
            )?)
        } else {
            None
        };
        let video_high_boost_value = decoder_params
            .boost_bpf
            .as_ref()
            .map(|boost_bpf| request.high_boost.unwrap_or(boost_bpf.mult));

        let chroma_bandpass_final = |color_under_format: bool| -> Result<Vec<Sos<f64>>> {
            let (lower, upper) = if color_under_format {
                ((color_under / 1e6) * 0.9, (color_under / 1e6) * 0.75)
            } else {
                // Using a narrow filter atm as this is just used for picking out
                // burst signal in this case.
                (0.1, 0.1)
            };
            butter_sos(
                4,
                &[
                    (sys_params.fsc_mhz - lower) / chroma_afc_out_frequency_half,
                    (sys_params.fsc_mhz + upper) / chroma_afc_out_frequency_half,
                ],
                FilterBandType::Bandpass,
            )
        };

        // --- Build luma notch filter for the RF/video path ---
        let (video_notch_filter, chroma_filter_video_notch) = if let Some(notch) = &request.notch {
            let video_notch_ba = ba_to_vec(iirnotch(notch.freq / freq_half, notch.q)?);
            let video_notch_filter = abs_complex_owned(filtfft(
                &video_notch_ba.0,
                &video_notch_ba.1,
                blocklen,
                true,
            ));
            let chroma_filter_video_notch = if do_cafc {
                ba_to_vec(iirnotch(
                    notch.freq / chroma_afc_out_frequency_half,
                    notch.q,
                )?)
            } else {
                video_notch_ba
            };
            (Some(video_notch_filter), Some(chroma_filter_video_notch))
        } else {
            (None, None)
        };

        // --- Build chroma filters ---
        let chroma_filter_video_burst = if is_color_under {
            butter_sos(
                decoder_params.chroma_bpf_order,
                &[
                    decoder_params.chroma_bpf_lower / freq_hz_half,
                    (color_under / 1e6) * 1e6 * chroma_afc_bpf_under_ratio / freq_hz_half,
                ],
                FilterBandType::Bandpass,
            )?
        } else {
            chroma_bandpass_final(false)?
        };
        let chroma_filter_deemphasis = if chroma_deemphasis_enabled {
            Some(peaking(
                sys_params.fsc_mhz / chroma_afc_out_frequency_half,
                3.4,
                None,
                Some(0.5 / chroma_afc_out_frequency_half),
            )?)
        } else {
            None
        };

        let chroma_filter_audio_notch = if decoder_params.chroma_audio_notch_freq > f64::EPSILON {
            let nyquist = if do_cafc {
                chroma_afc_out_frequency_half * 1e6
            } else {
                freq_hz_half
            };

            Some(ba_to_vec(iirnotch(
                decoder_params.chroma_audio_notch_freq / nyquist,
                CHROMA_AUDIO_NOTCH_Q,
            )?))
        } else {
            None
        };
        // Post-TBC chroma filter at output sample rate (4fsc).
        let chroma_filter_final = chroma_bandpass_final(is_color_under)?;
        let (rf_chroma_heterodyne, rf_fsc_wave) = if is_color_under {
            let cc_freq_mhz = color_under / 1e6;
            let het_freq = sys_params.fsc_mhz + cc_freq_mhz;
            let het_wave_scale = het_freq / out_sample_rate_mhz;
            (
                gen_chroma_heterodyne(het_wave_scale, 0.0, fieldlen),
                fsc_wave,
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // Increase the cutoff at the end of blocks to avoid edge distortion from filters making it through.

        let video_chroma_trap = if request.chroma_trap {
            Some(ChromaSepClass::new(freq_hz, sys_params.fsc_mhz))
        } else {
            None
        };

        let resync_divisor = level_detect_divisor as usize;
        let samp_rate = freq_hz / resync_divisor as f64;
        let fv = sys_params.fps * 2.0;
        let fh = sys_params.fps * sys_params.frame_lines.line_count() as f64;
        let venv_limit = 5.0;
        let serration_limit = 3.0;
        let resync_vsync_env_filter = iir_lowpass(samp_rate, fv * venv_limit, 1e3, 20)?;
        let resync_serration_filter_base = [
            iir_highpass(samp_rate, fh, fh, 20)?,
            iir_lowpass(samp_rate, fh, fh, 20)?,
        ];
        let resync_serration_filter_envelope =
            iir_lowpass(samp_rate, fh / serration_limit, fh / 2.0, 20)?;

        let track_phase = request.track_phase;
        if let Some(track_phase) = track_phase {
            if rf_color_system == ColorSystem::Secam {
                bail!("Track phase is not supported for SECAM");
            }
            if track_phase != 0 && track_phase != 1 {
                bail!("Track phase can only be 0, 1 or None");
            }
        }

        let wow_level_adjust_smoothing = request
            .wow_level_adjust_smoothing
            .unwrap_or(sys_params.frame_lines.line_count() as f32 / 2.0);

        Ok(Self {
            field_order_action: request.field_order_action,
            video_eq_config,

            chroma_afc_narrowband,
            chroma_afc_fine_tune_fh_ratio: decode_options.chroma_afc_fine_tune_fh_ratio,

            freq: rf_freq,
            color_system: rf_color_system,
            sys_fsc_mhz: sys_params.fsc_mhz,
            sys_frame_lines: sys_params.frame_lines,
            sys_field_lines: sys_params.field_lines,
            sys_line_period: sys_params.line_period,
            sys_active_video_us: sys_params.active_video_us,
            sys_fps: sys_params.fps,
            sys_ire0: sys_params.ire0,
            sys_hz_ire: sys_params.hz_ire,
            sys_vsync_ire: sys_params.vsync_ire,
            sys_color_burst_us: sys_params.color_burst_us,
            sys_blacksnr_slice: sys_params.blacksnr_slice,
            sys_num_pulses: sys_params.num_pulses,
            sys_hsync_pulse_us: sys_params.hsync_pulse_us,
            sys_eq_pulse_us: sys_params.eq_pulse_us,
            sys_vsync_pulse_us: sys_params.vsync_pulse_us,
            sys_output_zero: sys_params.output_zero,
            sys_outlinelen: sys_params.outlinelen,
            sys_outfreq: sys_params.outfreq,
            sys_ld_vits_whitelocs: sys_params.ld_vits_whitelocs.clone(),
            sys_burst_abs_ref: sys_params.burst_abs_ref,
            sys_track_ire0_offset: sys_params.track_ire0_offset,
            sys_nonlinear_deviation: sys_params.nonlinear_deviation,

            decoder_color_under_carrier: decoder_params.color_under_carrier,
            decoder_chroma_bpf_upper: decoder_params.chroma_bpf_upper,
            decoder_chroma_bpf_order: decoder_params.chroma_bpf_order,
            decoder_chroma_bpf_lower: decoder_params.chroma_bpf_lower,
            decoder_chroma_rotation: decoder_params.chroma_rotation,
            decoder_chroma_offset: decoder_params.chroma_offset,
            decoder_nonlinear_highpass_limit_l: decoder_params.nonlinear.highpass_limit_l,
            decoder_nonlinear_highpass_limit_h: decoder_params.nonlinear.highpass_limit_h,
            decoder_nonlinear_exp_scaling: decoder_params.nonlinear.exp_scaling,
            decoder_nonlinear_scaling_1: decoder_params.nonlinear.scaling_1,
            decoder_nonlinear_scaling_2: decoder_params.nonlinear.scaling_2,
            decoder_nonlinear_logistic: decoder_params
                .nonlinear
                .logistic
                .as_ref()
                .map(|value| (value.mid, value.rate)),
            decoder_nonlinear_static_factor: decoder_params.nonlinear.static_factor,
            rf_chroma_heterodyne,
            rf_fsc_wave,

            dod_threshold_p: request.dod_threshold_p,
            dod_threshold_a: request.dod_threshold_a,
            dod_hysteresis: request.dod_hysteresis,

            rf_disable_comb,
            rf_disable_right_hsync: request.rf_disable_right_hsync,
            rf_disable_dc_offset: request.rf_disable_dc_offset,
            rf_fallback_vsync,
            rf_field_order_confidence: request.rf_field_order_confidence,
            rf_saved_levels: request.rf_saved_levels,
            rf_y_comb,
            rf_write_chroma,
            rf_skip_hsync_refine: request.rf_skip_hsync_refine,
            rf_export_raw_tbc: request.rf_export_raw_tbc,
            rf_ire0_adjust: request.rf_ire0_adjust,
            rf_relaxed_line0: request.rf_relaxed_line0,
            rf_detect_chroma_track_phase: request.rf_detect_chroma_track_phase,
            rf_disable_burst_hsync: request.rf_disable_burst_hsync,
            rf_disable_phase_correction: request.rf_disable_phase_correction,

            chroma_filter_video_burst,
            chroma_filter_video_notch,
            chroma_filter_deemphasis,
            chroma_filter_audio_notch,
            chroma_filter_final,

            video_rf_filter: store_real_filter(video_rf_filter),
            video_notch_filter: video_notch_filter.map(store_real_filter),
            video_hilbert_filter,
            video_env_post_filter,
            video_rf_top_filter,
            video_high_boost_value,
            video_disable_diff_demod: request.video_disable_diff_demod,
            video_chroma_trap,
            video_filter: store_complex_filter(video_filter),
            video_nl_amplitude_lpf,
            video_nl_high_pass_f: video_nl_high_pass_f.map(store_complex_filter),
            video_nldeemp_enabled: request.video_nldeemp_enabled,
            video_subdeemp_enabled,
            video_fsc_notch,
            video05_filter: store_complex_filter(video05_filter),

            fft_block_forward_f32,
            fft_block_inverse_f32,
            fft_field_forward_f32,
            fft_field_inverse_f32,

            resync_divisor,
            resync_vsync_env_filter,
            resync_serration_filter_base,
            resync_serration_filter_envelope,

            ntscj: request.ntscj,
            track_phase,
            wow_level_adjust_smoothing,
            wow_interpolation_method: request.wow_interpolation_method,
            do_dod: request.do_dod,
            level_adjust: request.level_adjust,
            _private: (),
        })
    }

    pub(crate) const CHROMA_AFC_POWER_THRESHOLD: f64 = 1.0 / 3.0;
    // 65-tap FIR low-pass introduces a 32-sample group delay.
    pub(crate) const VIDEO05_FILTER_OFFSET: usize = 32;

    #[inline]
    pub(crate) fn freq_hz(&self) -> f64 {
        self.freq * 1_000_000.0
    }

    #[inline]
    pub(crate) fn linelen(&self) -> usize {
        (self.freq_hz() / (1_000_000.0 / self.sys_line_period)).round_ties_even() as usize
    }

    #[inline]
    pub(crate) fn samplesperline(&self) -> f64 {
        self.freq / self.linelen() as f64
    }

    #[inline]
    pub(crate) fn black_ire(&self) -> f64 {
        // NTSC uses a 7.5 IRE black pedestal; every other standard (including
        // 525-line PAL-M variants) sets up black at 0 IRE.
        if self.color_system == ColorSystem::Ntsc && !self.ntscj {
            7.5
        } else {
            0.0
        }
    }

    #[inline]
    pub(crate) fn video_sub_deemphasis_deviation(&self) -> f32 {
        self.sys_nonlinear_deviation
            .unwrap_or(self.sys_hz_ire * (100.0 - self.sys_vsync_ire))
    }

    #[inline]
    pub(crate) fn chroma_offset(&self) -> isize {
        (self.decoder_chroma_offset * (self.freq / 40.0)) as isize
    }

    #[inline]
    pub(crate) fn chroma_afc_fh(&self) -> f64 {
        self.sys_fps * self.sys_frame_lines.line_count() as f64
    }

    #[inline]
    pub(crate) fn chroma_afc_band_tolerance(&self) -> (f64, f64) {
        let color_under = self.decoder_color_under_carrier;
        let percent = 200.0 * self.chroma_afc_fh() / color_under;
        ((100.0 - percent) / 100.0, (100.0 + percent) / 100.0)
    }

    #[inline]
    pub(crate) fn chroma_afc_enabled(&self) -> bool {
        !self.chroma_afc_narrowband.is_empty()
    }

    #[inline]
    pub(crate) fn resync_divisor(&self) -> i64 {
        self.resync_divisor as i64
    }

    #[inline]
    fn resync_samp_rate(&self) -> f64 {
        self.freq_hz() / self.resync_divisor as f64
    }

    #[inline]
    fn resync_fv(&self) -> f64 {
        self.sys_fps * 2.0
    }

    #[inline]
    fn resync_fh(&self) -> f64 {
        self.sys_fps * self.sys_frame_lines.line_count() as f64
    }

    #[inline]
    pub(crate) fn resync_eq_pulselen_downsampled(&self) -> usize {
        t_to_samples(self.resync_samp_rate(), self.sys_eq_pulse_us * 1e-6).round_ties_even()
            as usize
    }

    #[inline]
    pub(crate) fn resync_vsynclen_downsampled(&self) -> usize {
        (self.resync_samp_rate() / self.resync_fv()).round_ties_even() as usize
    }

    #[inline]
    pub(crate) fn resync_linelen_downsampled(&self) -> usize {
        (self.resync_samp_rate() / self.resync_fh()).round_ties_even() as usize
    }

    #[inline]
    pub(crate) fn resync_vbi_time_range(&self) -> (f64, f64) {
        let line_time = 1.0 / self.resync_fh();
        let vbi_time = 6.5 * line_time;
        (
            t_to_samples(self.resync_samp_rate(), vbi_time * 3.0 / 4.0),
            t_to_samples(self.resync_samp_rate(), vbi_time * 5.0 / 4.0),
        )
    }

    #[inline]
    pub(crate) fn resync_field_ma_depth(&self) -> usize {
        let fv = self.resync_fv();
        if fv < 60.0 {
            (fv / 5.0).round_ties_even() as usize
        } else {
            (fv / 6.0).round_ties_even() as usize
        }
    }

    #[inline]
    pub(crate) fn resync_eq_pulselen(&self) -> usize {
        self.resync_eq_pulselen_downsampled() * self.resync_divisor
    }

    #[inline]
    fn resync_linelen(&self) -> usize {
        self.resync_linelen_downsampled() * self.resync_divisor
    }

    #[inline]
    pub(crate) fn resync_long_pulse_max(&self) -> f64 {
        self.resync_linelen() as f64 * 5.0
    }

    #[inline]
    pub fn readlen(&self) -> usize {
        let linelen = self.linelen();
        match self.sys_frame_lines {
            LineSystem::Line819 => linelen * 500,
            LineSystem::Line525 => ((linelen * 350) / 16384) * 16384,
            _ => linelen * 400,
        }
    }

    #[inline]
    pub(crate) fn usable_blocksize(&self) -> usize {
        BLOCKSIZE - (BLOCKCUT + BLOCKCUT_END)
    }

    #[inline]
    pub(crate) fn output_lines(&self) -> usize {
        (self.sys_frame_lines.line_count() / 2) + 1
    }

    #[inline]
    pub(crate) fn bytes_per_field(&self) -> i64 {
        (self.freq_hz() / (self.sys_fps * 2.0)) as i64 + 1
    }

    /// Input samples spanned by one field, used to translate a field distance
    /// into a seek offset for multithreaded decoding.
    #[inline]
    pub fn samples_per_field(&self) -> u64 {
        self.bytes_per_field().max(1) as u64
    }

    #[inline]
    pub fn write_chroma(&self) -> bool {
        self.rf_write_chroma
    }
}

// Filter-design helpers, moved here from decode.rs because spec.rs
// (DecoderSpec construction) is their sole consumer.

fn t_to_samples(samp_rate: f64, time: f64) -> f64 {
    samp_rate / (1.0 / time)
}

fn build_hilbert(fft_size: usize) -> Vec<f32> {
    assert_ne!(fft_size, 0);
    assert!(fft_size.is_multiple_of(2));

    let mut output = vec![0.0f32; fft_size];
    let half_size = fft_size / 2;
    output[0] = 1.0;
    output[half_size] = 1.0;
    for sample in &mut output[1..half_size] {
        *sample = 2.0;
    }
    output
}

fn gen_wave_at_frequency(
    frequency: f64,
    sample_frequency: f64,
    num_samples: usize,
) -> Vec<(f32, f32)> {
    use std::f64::consts::TAU;
    let angle_step = TAU * frequency / sample_frequency;
    (0..num_samples)
        .map(|i| {
            let (sin, cos) = (angle_step * i as f64).sin_cos();
            (sin as f32, cos as f32)
        })
        .collect()
}

fn iirnotch(w0: f64, q: f64) -> Result<([f64; 3], [f64; 3])> {
    if !(0.0..1.0).contains(&w0) {
        bail!("notch frequency must be greater than 0 and less than Nyquist");
    }
    if q <= 0.0 {
        bail!("notch Q must be positive");
    }

    let w0_radians = std::f64::consts::PI * w0;
    let bandwidth = std::f64::consts::PI * (w0 / q);
    let beta = (bandwidth / 2.0).tan();
    let gain = 1.0 / (1.0 + beta);
    let cos_w0 = w0_radians.cos();

    Ok((
        [gain, -2.0 * gain * cos_w0, gain],
        [1.0, -2.0 * gain * cos_w0, (2.0 * gain) - 1.0],
    ))
}

/// Multiply two polynomials given in ascending-power coefficient order.
fn poly_mul(p: &[f64], q: &[f64]) -> Vec<f64> {
    let mut r = vec![0.0; p.len() + q.len() - 1];
    for (i, &pi) in p.iter().enumerate() {
        for (j, &qj) in q.iter().enumerate() {
            r[i + j] += pi * qj;
        }
    }
    r
}

/// Raise a polynomial (ascending-power coefficients) to an integer power.
fn poly_pow(p: &[f64], n: usize) -> Vec<f64> {
    let mut r = vec![1.0];
    for _ in 0..n {
        r = poly_mul(&r, p);
    }
    r
}

/// Accumulate `scale * p` into `acc`, both in ascending-power coefficient order.
fn poly_add_scaled(acc: &mut Vec<f64>, p: &[f64], scale: f64) {
    if p.len() > acc.len() {
        acc.resize(p.len(), 0.0);
    }
    for (i, &pi) in p.iter().enumerate() {
        acc[i] += scale * pi;
    }
}

/// Normalize numerator/denominator coefficients of a transfer function.
fn normalize(b: &[f64], a: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Trim leading zeros in denominator, leave at least one.
    let den_start = a
        .iter()
        .position(|&v| v != 0.0)
        .unwrap_or(a.len().saturating_sub(1));
    let den = &a[den_start..];

    // Normalize transfer function.
    let a0 = den[0];
    let num: Vec<f64> = b.iter().map(|&v| v / a0).collect();
    let den: Vec<f64> = den.iter().map(|&v| v / a0).collect();

    // Trim leading near-zero numerator coefficients, leaving at least one.
    let num_start = num
        .iter()
        .position(|&v| v.abs() > 1e-14)
        .unwrap_or(num.len().saturating_sub(1));
    let num = num[num_start..].to_vec();

    (num, den)
}

/// Transform a lowpass analog prototype to a different cutoff frequency (BA form).
fn lp2lp(b: &[f64], a: &[f64], wo: f64) -> (Vec<f64>, Vec<f64>) {
    let d = a.len();
    let n = b.len();
    let m = d.max(n);
    // pwo = wo ** arange(M - 1, -1, -1)
    let pwo: Vec<f64> = (0..m).map(|i| wo.powi((m - 1 - i) as i32)).collect();
    let start1 = n.saturating_sub(d);
    let start2 = d.saturating_sub(n);
    let b_new: Vec<f64> = b
        .iter()
        .enumerate()
        .map(|(i, &bv)| bv * pwo[start1] / pwo[start2 + i])
        .collect();
    let a_new: Vec<f64> = a
        .iter()
        .enumerate()
        .map(|(i, &av)| av * pwo[start1] / pwo[start1 + i])
        .collect();
    normalize(&b_new, &a_new)
}

/// Calculate a digital IIR filter from an analog transfer function (BA form)
/// using the bilinear transform.
fn bilinear(b: &[f64], a: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    // remove leading zeros
    let b: Vec<f64> = {
        let start = b.iter().position(|&v| v != 0.0).unwrap_or(b.len());
        b[start..].to_vec()
    };
    let a: Vec<f64> = {
        let start = a.iter().position(|&v| v != 0.0).unwrap_or(a.len());
        a[start..].to_vec()
    };

    // Splitting the factor fs*2 between numerator and denominator reduces the chance of
    // numeric overflow for large fs and large N.
    let fac = (fs * 2.0).sqrt();
    // zp1 = (z + 1) / fac, zm1 = (z - 1) * fac (ascending-power coefficients)
    let zp1 = [1.0 / fac, 1.0 / fac];
    let zm1 = [-fac, fac];

    let n = a.len().max(b.len()) - 1;
    let mut numerator: Vec<f64> = vec![0.0];
    for (q, &b_) in b.iter().rev().enumerate() {
        let term = poly_mul(&poly_pow(&zp1, n - q), &poly_pow(&zm1, q));
        poly_add_scaled(&mut numerator, &term, b_);
    }
    let mut denominator: Vec<f64> = vec![0.0];
    for (p, &a_) in a.iter().rev().enumerate() {
        let term = poly_mul(&poly_pow(&zp1, n - p), &poly_pow(&zm1, p));
        poly_add_scaled(&mut denominator, &term, a_);
    }

    // Polynomial helpers use ascending powers; BA transfer functions use descending powers.
    numerator.reverse();
    denominator.reverse();
    normalize(&numerator, &denominator)
}

/// Convert an analog prototype filter to a digital BA transfer function.
fn biquad_transform(b: &[f64], a: &[f64], wn: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !(0.0..=1.0).contains(&wn) {
        bail!("Digital filter critical frequencies must be 0 <= Wn <= 1");
    }
    let fs = 2.0;
    let warped = 2.0 * fs * (std::f64::consts::PI * wn / fs).tan();

    // Shift frequency
    let (b, a) = lp2lp(b, a, warped);
    // Find discrete equivalent
    let (b, a) = bilinear(&b, &a, fs);
    Ok((b, a))
}

/// Design a digital biquad peaking filter with variable Q (BA output).
fn peaking(wn: f64, db_gain: f64, q: Option<f64>, bw: Option<f64>) -> Result<(Vec<f64>, Vec<f64>)> {
    let bw = if q.is_none() && bw.is_none() {
        Some(1.0) // octave
    } else {
        bw
    };

    let q = match q {
        Some(q) => q,
        None => {
            // analog filter prototype
            1.0 / (2.0 * (2.0_f64.ln() / 2.0 * bw.unwrap()).sinh())
        }
    };

    let (az, ap) = {
        let a = 10.0_f64.powf(db_gain / 20.0);
        if db_gain > 0.0 {
            (a, 1.0) // boost
        } else {
            (1.0, a) // cut
        }
    };

    // H(s) = (s**2 + s*(Az/Q) + 1) / (s**2 + s/(Ap*Q) + 1)
    let b = [1.0, az / q, 1.0];
    let a = [1.0, 1.0 / (ap * q), 1.0];

    biquad_transform(&b, &a, wn)
}

fn gen_shelf(
    f0: f64,
    dbgain: f64,
    shelf_type: ShelfKind,
    fs: f64,
    qfactor: Option<f64>,
    bandwidth: Option<f64>,
    slope: Option<f64>,
) -> Result<([f64; 3], [f64; 3])> {
    let a = 10.0_f64.powf(dbgain / 40.0);
    let w0 = std::f64::consts::TAU * (f0 / fs);
    let sinw0 = w0.sin();

    let alpha = if let Some(qfactor) = qfactor {
        sinw0 / (2.0 * qfactor)
    } else if let Some(bandwidth) = bandwidth {
        sinw0 * ((std::f64::consts::LN_2 / 2.0) * bandwidth * (w0 / sinw0)).sinh()
    } else if let Some(slope) = slope {
        (w0 / 2.0).sin() * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt()
    } else {
        bail!("Must specify one value for either qfactor, bandwidth, or slope");
    };

    let cosw0 = w0.cos();
    let asquared = a.sqrt();

    match shelf_type {
        ShelfKind::Low => Ok((
            [
                a * ((a + 1.0) - (a - 1.0) * cosw0 + 2.0 * asquared * alpha),
                2.0 * a * ((a - 1.0) - (a + 1.0) * cosw0),
                a * ((a + 1.0) - (a - 1.0) * cosw0 - 2.0 * asquared * alpha),
            ],
            [
                (a + 1.0) + (a - 1.0) * cosw0 + 2.0 * asquared * alpha,
                -2.0 * ((a - 1.0) + (a + 1.0) * cosw0),
                (a + 1.0) + (a - 1.0) * cosw0 - 2.0 * asquared * alpha,
            ],
        )),
        ShelfKind::High => Ok((
            [
                a * ((a + 1.0) + (a - 1.0) * cosw0 + 2.0 * asquared * alpha),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cosw0),
                a * ((a + 1.0) + (a - 1.0) * cosw0 - 2.0 * asquared * alpha),
            ],
            [
                (a + 1.0) - (a - 1.0) * cosw0 + 2.0 * asquared * alpha,
                2.0 * ((a - 1.0) - (a + 1.0) * cosw0),
                (a + 1.0) - (a - 1.0) * cosw0 - 2.0 * asquared * alpha,
            ],
        )),
    }
}

fn butter_ba(
    order: usize,
    wn: &[f64],
    band_type: FilterBandType,
    fs: Option<f64>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    use sci_rs::signal::filter::design::{butter_dyn, DigitalFilter, FilterOutputType};

    let filter = butter_dyn::<f64>(
        order,
        wn.to_vec(),
        Some(band_type),
        Some(false),
        Some(FilterOutputType::Ba),
        fs,
    );

    match filter {
        DigitalFilter::Ba(mut ba) => {
            while ba.b.len() > 1
                && ba.a.len() > 1
                && ba.b.last().is_some_and(|value| value.abs() <= f64::EPSILON)
                && ba.a.last().is_some_and(|value| value.abs() <= f64::EPSILON)
            {
                ba.b.pop();
                ba.a.pop();
            }
            Ok((ba.b, ba.a))
        }
        _ => bail!("sci-rs returned an unexpected Butterworth BA representation"),
    }
}

/// Butterworth order selection for digital low-pass filters where the scalar
/// passband is below the stopband.
fn buttord(wp: f64, ws: f64, gpass: f64, gstop: f64, fs: f64) -> Result<(usize, f64)> {
    if gpass <= 0.0 {
        bail!("gpass should be larger than 0.0");
    }
    if gstop <= 0.0 {
        bail!("gstop should be larger than 0.0");
    }
    if gpass > gstop {
        bail!("gpass should be smaller than gstop");
    }

    // fs given: normalize edges to half-cycles/sample.
    let wp = 2.0 * wp / fs;
    let ws = 2.0 * ws / fs;

    // Pre-warp frequencies for digital filter design.
    let passb = (std::f64::consts::PI * wp / 2.0).tan();
    let stopb = (std::f64::consts::PI * ws / 2.0).tan();

    // filter_type == 1 (low): nat = stopb / passb
    let nat = (stopb / passb).abs();

    let gstop_lin = 10.0_f64.powf(0.1 * gstop.abs());
    let gpass_lin = 10.0_f64.powf(0.1 * gpass.abs());
    let ord = (((gstop_lin - 1.0) / (gpass_lin - 1.0)).log10() / (2.0 * nat.log10())).ceil();
    let ord = ord as usize;

    // Find the Butterworth natural frequency WN (the "3dB" frequency)
    // to give exactly gpass at passb.
    let w0 = if ord == 0 {
        1.0
    } else {
        (gpass_lin - 1.0).powf(-1.0 / (2.0 * ord as f64))
    };

    // Convert this frequency back from lowpass prototype to the original
    // analog filter (filter_type 1: low).
    let wn = w0 * passb;

    // postprocess: digital filter with fs specified.
    let wn = wn.atan() * 2.0 / std::f64::consts::PI;
    let wn = wn * fs / 2.0;

    Ok((ord, wn))
}

fn design_filter(
    samp_rate: f64,
    passband: f64,
    stopband: f64,
    order_limit: usize,
) -> Result<(usize, f64)> {
    let max_loss_passband = 3.0; // The maximum loss allowed in the passband
    let min_loss_stopband = 30.0; // The minimum loss allowed in the stopband
    let (mut order, normal_cutoff) = buttord(
        passband,
        stopband,
        max_loss_passband,
        min_loss_stopband,
        samp_rate,
    )?;
    if order > order_limit {
        tracing::warn!(
            "Limiting order of the filter from {} to {}",
            order,
            order_limit
        );
        order = order_limit;
    }
    Ok((order, normal_cutoff))
}

fn iir_lowpass(
    samp_rate: f64,
    cutoff: f64,
    transition_width: f64,
    order_limit: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let stopband = cutoff + transition_width;
    let (order, normal_cutoff) = design_filter(samp_rate, cutoff, stopband, order_limit)?;
    butter_ba(
        order,
        &[normal_cutoff],
        FilterBandType::Lowpass,
        Some(samp_rate),
    )
}

fn iir_highpass(
    samp_rate: f64,
    cutoff: f64,
    transition_width: f64,
    order_limit: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let stopband = cutoff + transition_width;
    let (order, normal_cutoff) = design_filter(samp_rate, cutoff, stopband, order_limit)?;
    butter_ba(
        order,
        &[normal_cutoff],
        FilterBandType::Highpass,
        Some(samp_rate),
    )
}

fn gen_video_lpf_supergauss(
    corner_freq: f64,
    order: usize,
    nyquist_hz: f64,
    block_len: usize,
) -> Vec<f64> {
    let output_len = block_len / 2 + 1;
    let exponent = (order * 2) as i32;
    let half_log_two = f64::ln(2.0) / 2.0;
    let scale = half_log_two.powf(1.0 / exponent as f64);

    (0..output_len)
        .map(|index| {
            let x = if output_len <= 1 {
                0.0
            } else {
                nyquist_hz * index as f64 / (output_len - 1) as f64
            };
            let normalized = (2.0 * x * scale) / corner_freq;
            (-2.0 * normalized.powi(exponent)).exp()
        })
        .collect()
}

fn gen_ramp_filter(
    boost_ramp: &BoostRampFilter,
    nyquist_freq_hz: f64,
    block_len: usize,
) -> Vec<f64> {
    let max_freq_hz = 20e6;
    let half_len = block_len / 2;
    let zero_ratio = ((boost_ramp.start_freq / nyquist_freq_hz) * half_len as f64) as usize;
    let zero_ratio = zero_ratio.min(half_len);
    let ramp_len = half_len.saturating_sub(zero_ratio);

    let mut ramp = vec![0.0f64; half_len];
    if ramp_len > 0 {
        let end_value = boost_ramp.rf_linear_20 * (nyquist_freq_hz / max_freq_hz);
        if ramp_len == 1 {
            ramp[zero_ratio] = boost_ramp.rf_linear_0;
        } else {
            for i in 0..ramp_len {
                let t = i as f64 / (ramp_len - 1) as f64;
                ramp[zero_ratio + i] =
                    boost_ramp.rf_linear_0 + (end_value - boost_ramp.rf_linear_0) * t;
            }
        }
    }

    let mut output = Vec::with_capacity(block_len);
    output.extend_from_slice(&ramp);
    output.extend(ramp.iter().rev().copied());
    output
}

fn sosfiltfft(sos: &[Sos<f64>], block_len: usize) -> Vec<Complex64> {
    use std::f64::consts::TAU;

    let mut output = Vec::with_capacity(block_len);

    for k in 0..block_len {
        let omega = TAU * k as f64 / block_len as f64;
        let z1 = Complex64::new(omega.cos(), -omega.sin());
        let z2 = z1 * z1;
        let mut response = Complex64::new(1.0, 0.0);

        for section in sos {
            let [b0, b1, b2] = section.b;
            let [a0, a1, a2] = section.a;

            let numerator = Complex64::new(b0, 0.0) + z1 * b1 + z2 * b2;
            let denominator = Complex64::new(a0, 0.0) + z1 * a1 + z2 * a2;
            response *= numerator / denominator;
        }

        output.push(response);
    }

    output
}

fn filtfft(b: &[f64], a: &[f64], block_len: usize, whole: bool) -> Vec<Complex64> {
    use std::f64::consts::{PI, TAU};

    assert!(!a.is_empty());
    assert!(!b.is_empty());

    let output_len = if whole {
        block_len
    } else {
        (block_len / 2) + 1
    };
    let mut output = Vec::with_capacity(output_len);

    for k in 0..output_len {
        let omega = if whole {
            TAU * k as f64 / block_len as f64
        } else if output_len > 1 {
            PI * k as f64 / (output_len - 1) as f64
        } else {
            0.0
        };
        let z = Complex64::new(omega.cos(), -omega.sin());
        let mut z_power = Complex64::new(1.0, 0.0);
        let mut numerator = Complex64::new(0.0, 0.0);
        for coefficient in b {
            numerator += z_power * *coefficient;
            z_power *= z;
        }

        z_power = Complex64::new(1.0, 0.0);
        let mut denominator = Complex64::new(0.0, 0.0);
        for coefficient in a {
            denominator += z_power * *coefficient;
            z_power *= z;
        }

        output.push(numerator / denominator);
    }

    output
}

fn ba_to_vec(pair: ([f64; 3], [f64; 3])) -> (Vec<f64>, Vec<f64>) {
    (pair.0.to_vec(), pair.1.to_vec())
}

fn abs_complex_owned(values: Vec<Complex64>) -> Vec<f64> {
    convert_vec_in_place(values, |value| value.norm())
}

fn multiply<A, B, O>(a: &[A], b: &[B]) -> Vec<O>
where
    A: Copy + std::ops::Mul<B, Output = O>,
    B: Copy,
{
    assert_eq!(b.len(), a.len(), "multiply length mismatch");
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

fn gen_video_main_deemp_fft(
    deemph: &DeemphasisParams,
    freq_hz: f64,
    block_len: usize,
) -> Result<Vec<Complex64>> {
    let (ataps, btaps) = gen_shelf(
        deemph.mid,
        deemph.gain,
        ShelfKind::High,
        freq_hz,
        Some(deemph.q),
        None,
        None,
    )?;
    Ok(filtfft(&btaps, &ataps, block_len, false))
}

fn gen_video_lpf(
    corner_freq: f64,
    order: usize,
    nyquist_hz: f64,
    block_len: usize,
) -> Result<Vec<f64>> {
    let sos = butter_sos(order, &[corner_freq / nyquist_hz], FilterBandType::Lowpass)?;
    let mut full = abs_complex_owned(sosfiltfft(&sos, block_len));
    full.truncate(block_len / 2 + 1);
    Ok(full)
}

fn gen_nonlinear_bandpass(
    nonlinear: &NonlinearParams,
    nyquist_hz: f64,
    block_len: usize,
) -> Result<Vec<Complex64>> {
    let (b, a) = if let Some(upper_freq) = nonlinear.bandpass_upper {
        butter_ba(
            nonlinear.bandpass_order,
            &[
                nonlinear.highpass_freq / nyquist_hz,
                upper_freq / nyquist_hz,
            ],
            FilterBandType::Bandpass,
            None,
        )?
    } else {
        butter_ba(
            nonlinear.bandpass_order,
            &[nonlinear.highpass_freq / nyquist_hz],
            FilterBandType::Highpass,
            None,
        )?
    };
    Ok(filtfft(&b, &a, block_len, false))
}

fn gen_fft_notch(
    notch_freq: f64,
    notch_q: f64,
    nyquist_hz: f64,
    block_len: usize,
) -> Result<Vec<Complex64>> {
    let (b, a) = ba_to_vec(iirnotch(notch_freq / nyquist_hz, notch_q)?);
    Ok(filtfft(&b, &a, block_len, true))
}

fn gen_custom_video_filters(
    filter_list: &[VideoLumaFilter],
    freq_hz: f64,
    block_len: usize,
) -> Result<Vec<Complex64>> {
    let mut ret = vec![Complex64::new(1.0, 0.0); block_len / 2 + 1];
    for filter in filter_list {
        match filter {
            VideoLumaFilter::File { filename } => {
                if let Some(values) = embedded_filter_file(filename, freq_hz as i64) {
                    assert_eq!(
                        values.len(),
                        ret.len(),
                        "custom file filter length mismatch"
                    );
                    for (dst, src) in ret.iter_mut().zip(values) {
                        *dst *= src;
                    }
                } else {
                    tracing::warn!(filename, freq_hz, "Cannot load filter from file for samplerate. Output will likely not look correct!");
                }
            }
            VideoLumaFilter::Shelf {
                shelf_kind,
                gain,
                midfreq,
                q,
            } => {
                let (b, a) = ba_to_vec(gen_shelf(
                    *midfreq,
                    *gain,
                    *shelf_kind,
                    freq_hz / 2.0,
                    Some(*q),
                    None,
                    None,
                )?);
                let fft = filtfft(&b, &a, block_len, false);
                ret = multiply(&ret, &fft);
            }
        }
    }
    Ok(ret)
}

fn embedded_filter_file(filename: &str, freq_hz: i64) -> Option<Vec<Complex64>> {
    let text = match (filename, freq_hz) {
        ("svhs-sp-linear-subdeemphasis", 40000000) => {
            include_str!("svhs-sp-linear-subdeemphasis-40000000.txt")
        }
        ("svhs-sp-linear-subdeemphasis", 17734475) => {
            include_str!("svhs-sp-linear-subdeemphasis-17734475.txt")
        }
        _ => return None,
    };
    Some(text.lines().filter_map(parse_complex_line).collect())
}

fn parse_complex_line(line: &str) -> Option<Complex64> {
    let trimmed = line.trim().trim_end_matches('j');
    let mut split = None;
    let bytes = trimmed.as_bytes();
    for i in 1..bytes.len() {
        let ch = bytes[i] as char;
        let prev = bytes[i - 1] as char;
        if (ch == '+' || ch == '-') && prev != 'e' && prev != 'E' {
            split = Some(i);
        }
    }
    let split = split?;
    let (re, im) = trimmed.split_at(split);
    Some(Complex64::new(re.parse().ok()?, im.parse().ok()?))
}

fn firwin_lowpass(numtaps: usize, cutoff_norm_to_nyquist: f64) -> Vec<f64> {
    let m = (numtaps - 1) as f64 / 2.0;
    let fc = cutoff_norm_to_nyquist / 2.0;
    let mut taps = Vec::with_capacity(numtaps);
    for n in 0..numtaps {
        let x = n as f64 - m;
        let sinc = if x == 0.0 {
            2.0 * fc
        } else {
            (2.0 * std::f64::consts::PI * fc * x).sin() / (std::f64::consts::PI * x)
        };
        let window =
            0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / (numtaps - 1) as f64).cos();
        taps.push(sinc * window);
    }
    let sum: f64 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= sum;
    }
    taps
}
