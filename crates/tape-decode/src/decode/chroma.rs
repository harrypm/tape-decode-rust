use super::*;

fn adjust_phase(
    input_data: &[Complex32],
    output_data: &mut [f32],
    input_phase: f32,
    target_phase: f32,
) {
    assert_eq!(input_data.len(), output_data.len());

    let phase_adjustment = target_phase.to_radians() - input_phase.to_radians();
    let rotation_re = phase_adjustment.cos();
    let rotation_im = phase_adjustment.sin();

    for (input, output) in input_data.iter().zip(output_data.iter_mut()) {
        *output = input.re * rotation_re - input.im * rotation_im;
    }
}

fn acc(
    chroma: &[f32],
    burst_abs_ref: f32,
    burststart: usize,
    burstend: usize,
    linelength: usize,
    lines: usize,
) -> Vec<u16> {
    const STARTING_LINE: usize = 16;
    const SIGNED_SAMPLE_MAX: f32 = 32767.0;
    assert!(lines > STARTING_LINE);

    // Burst-normalize each line and encode it straight to the u16 output in one
    // pass. The lead-in lines below STARTING_LINE (and any samples past `lines`)
    // carry no normalization: the old two-step path left them 0.0f32 and then
    // encoded that to the 32767 zero level, so initialize them to that level and
    // overwrite the normalized region in place. This drops the intermediate
    // field-sized f32 buffer and the separate encode pass that re-read it.
    let zero_level = SIGNED_SAMPLE_MAX as u16;
    let mut output = vec![zero_level; chroma.len()];

    for linenumber in STARTING_LINE..lines {
        let linestart = linelength * linenumber;
        let lineend = linestart + linelength;
        let line = &chroma[linestart..lineend];
        let burst_abs_mean = rms(&line[burststart..burstend]);
        let scale = if burst_abs_mean != 0.0 {
            burst_abs_ref / burst_abs_mean as f32
        } else {
            1.0
        };
        for (out, &sample) in output[linestart..lineend].iter_mut().zip(line.iter()) {
            let scaled = sample * scale;
            *out = ((scaled + SIGNED_SAMPLE_MAX) as i64) as u16;
        }
    }

    output
}

fn upconvert_chroma(
    chroma: &[f32],
    field: &DecodedField,
    chroma_heterodyne: &[Vec<f32>],
) -> Result<Vec<f32>> {
    let lineoffset = field.lineoffset + 1;
    let phase_rotation_sequence = field
        .phase_sequence
        .as_ref()
        .context("missing phase sequence")?;
    let mut uphet = vec![0.0f32; chroma.len()];

    for &(linenumber, current_phase, ..) in phase_rotation_sequence {
        let linestart = (linenumber as isize - lineoffset as isize) * field.outlinelen as isize;
        let lineend = linestart + field.outlinelen as isize;
        let start = resolve_slice_bound(chroma.len(), linestart);
        let end = resolve_slice_bound(chroma.len(), lineend);
        if start >= end {
            continue;
        }

        // Pair the line as slices so the loop carries no per-sample bounds
        // checks and vectorizes.
        let heterodyne_row = &chroma_heterodyne[current_phase][start..end];
        for ((out, &sample), &het) in uphet[start..end]
            .iter_mut()
            .zip(&chroma[start..end])
            .zip(heterodyne_row)
        {
            *out = sample * het;
        }
    }

    Ok(uphet)
}

fn burst_deemphasis(chroma: &mut [f32], field: &DecodedField, burstarea_end: usize) {
    let lineoffset = field.lineoffset + 1;
    for line in lineoffset..field.outlinecount + lineoffset {
        let linestart = (line - lineoffset) * field.outlinelen;
        let lineend = linestart + field.outlinelen;
        for sample in &mut chroma[linestart + burstarea_end + 5..lineend] {
            *sample *= 2.0;
        }
    }
}

fn comb_c(data: &mut [f32], line_len: usize, line_distance: usize) {
    let numlines = data.len() / line_len;
    if numlines <= 2 {
        return;
    }

    // The comb writes each line from its own (current) sample, the line
    // `line_distance` ahead, and the line `line_distance` behind. The current
    // and ahead lines are never overwritten before they are read (lines are
    // processed in increasing order), so they read straight from `data`. Only
    // the behind line may already be overwritten, so keep just the last
    // `line_distance` original lines in a small ring instead of copying the
    // whole field as the previous `data.to_vec()` did.
    let mut ring = vec![0.0f32; line_distance * line_len];
    for line_num in 16..numlines - 2 {
        let line_start = line_num * line_len;
        let advanced_start = (line_num + line_distance) * line_len;
        let slot = (line_num % line_distance) * line_len;
        let delayed = line_num - line_distance;
        // The first `line_distance` lines reach back before the processed range,
        // so those originals are still live in `data`; later ones come from the
        // ring slot they last wrote.
        let delayed_from_data = delayed < 16;
        let delayed_start = delayed * line_len;
        for offset in 0..line_len {
            let i = line_start + offset;
            let current = data[i];
            let advanced = data[advanced_start + offset];
            let delayed_sample = if delayed_from_data {
                data[delayed_start + offset]
            } else {
                ring[slot + offset]
            };
            ring[slot + offset] = current;
            data[i] = (current * 2.0 - advanced - delayed_sample) / 4.0;
        }
    }
}

fn process_chroma_internal(
    field: &mut DecodedField,
    spec: &DecoderSpec,
    chroma_afc_state: &mut ChromaAfcState,
) -> Result<Vec<u16>> {
    let chroma_downscaled = downscale_raw_vec(field, None, None, None, true)?;
    let mut chroma: Vec<f32> = if spec.chroma_afc_enabled() {
        let bandpass = chroma_afc_state.get_chroma_bandpass(spec)?;
        let chroma_len = chroma_downscaled.len();
        let chroma = demod_chroma_filt_array(
            &chroma_downscaled,
            spec,
            &bandpass,
            chroma_len,
            Some((10.0 * (spec.sys_outfreq / 40.0)) as isize),
        );
        chroma_afc_state.freq_offset(spec, &chroma, true)?;
        chroma
    } else {
        chroma_downscaled
    };
    let burstarea = padded_burst_area(spec);

    let is_ntsc = spec.color_system == ColorSystem::Ntsc;
    if is_ntsc {
        burst_deemphasis(&mut chroma, field, burstarea.1 as usize)
    }

    let chroma_heterodyne = active_chroma_heterodyne(spec, chroma_afc_state);

    let mut uphet = upconvert_chroma(&chroma, field, chroma_heterodyne)?;

    if is_ntsc && !spec.rf_disable_phase_correction {
        // Rotate the chroma so the measured burst phase lines up with 0 degrees.
        let burst_phase_avg = field.burst_phase_avg.context("missing burst phase avg")?;
        let hilbert = hilbert_f32(
            &uphet,
            spec.fft_field_forward_f32.as_ref(),
            spec.fft_field_inverse_f32.as_ref(),
        );
        adjust_phase(&hilbert, &mut uphet, burst_phase_avg, 0.0);
    }

    uphet = sosfiltfilt_f32(&spec.chroma_filter_final, &uphet);

    if let Some(sos) = spec.chroma_filter_deemphasis.as_ref() {
        uphet = sosfilt_f32(sos, &uphet);
    }

    if !spec.rf_disable_comb {
        let line_distance = if is_ntsc { 1 } else { 2 };
        comb_c(&mut uphet, field.outlinelen, line_distance);
    }

    let burst_abs_ref = spec.sys_burst_abs_ref.context("missing burst_abs_ref")?;
    Ok(acc(
        &uphet,
        burst_abs_ref,
        burstarea.0 as usize,
        burstarea.1 as usize,
        field.outlinelen,
        field.outlinecount,
    ))
}

pub(crate) fn decode_chroma(
    field: &mut DecodedField,
    spec: &DecoderSpec,
    chroma_afc_state: &mut ChromaAfcState,
) -> Result<Option<Vec<u16>>> {
    if !spec.rf_write_chroma || spec.color_system == ColorSystem::Monochrome {
        return Ok(None);
    }
    let upconverted = process_chroma_internal(field, spec, chroma_afc_state)?;
    Ok(Some(upconverted))
}
