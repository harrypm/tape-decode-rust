use sci_rs::signal::filter::design::Sos;
use sci_rs::signal::filter::sosfiltfilt_dyn;

use crate::vec_utils::convert_vec_in_place;

const SOSFILTFILT_STACK_MAX_SECTIONS: usize = 12;

pub(crate) fn sosfiltfilt_f32(sos: &[Sos<f64>], input_array: &[f32]) -> Vec<f32> {
    match sos.len() {
        1 => sosfiltfilt_stack::<1>(sos, input_array),
        2 => sosfiltfilt_stack::<2>(sos, input_array),
        3 => sosfiltfilt_stack::<3>(sos, input_array),
        4 => sosfiltfilt_stack::<4>(sos, input_array),
        5 => sosfiltfilt_stack::<5>(sos, input_array),
        6 => sosfiltfilt_stack::<6>(sos, input_array),
        7 => sosfiltfilt_stack::<7>(sos, input_array),
        8 => sosfiltfilt_stack::<8>(sos, input_array),
        9 => sosfiltfilt_stack::<9>(sos, input_array),
        10 => sosfiltfilt_stack::<10>(sos, input_array),
        11 => sosfiltfilt_stack::<11>(sos, input_array),
        12 => sosfiltfilt_stack::<12>(sos, input_array),
        _ => convert_vec_in_place(
            sosfiltfilt_dyn(input_array.iter().map(|&sample| f64::from(sample)), sos),
            |sample| sample as f32,
        ),
    }
}

#[inline]
fn sosfiltfilt_stack<const SECTIONS: usize>(sos: &[Sos<f64>], input_array: &[f32]) -> Vec<f32> {
    debug_assert!(SECTIONS > 0);
    debug_assert!(SECTIONS <= SOSFILTFILT_STACK_MAX_SECTIONS);
    debug_assert_eq!(sos.len(), SECTIONS);

    let ntaps = sosfiltfilt_ntaps(sos);
    let edge = ntaps * 3;
    assert!(input_array.len() > edge);

    let mut init_sections = [Sos::<f64>::default(); SECTIONS];
    init_sections.copy_from_slice(sos);
    sosfilt_zi_stack(&mut init_sections);

    let left_end = f64::from(input_array[0]);
    let right_end = f64::from(input_array[input_array.len() - 1]);

    let x0 = 2.0 * left_end - f64::from(input_array[edge]);
    let mut forward_sections = init_sections;
    scale_sos_state(&mut forward_sections, x0);

    // Forward filter the left padding only to advance state. The backward pass
    // never reads those samples, so avoid storing them.
    for index in (1..=edge).rev() {
        let sample = 2.0 * left_end - f64::from(input_array[index]);
        sosfilt_sample_stack(sample, &mut forward_sections);
    }

    let n = input_array.len();
    let total = n + edge;
    // Build the forward-filtered buffer with `extend` over exact-size iterators:
    // that reserves `total` up front and writes each slot once via the
    // TrustedLen fast path (no per-element capacity check), so the hot loop is
    // as tight as the old indexed writes while skipping the `vec![0.0; total]`
    // zero-fill that was immediately overwritten in full.
    let mut filtered: Vec<f32> = Vec::with_capacity(total);
    filtered.extend(input_array.iter().map(|&sample| -> f32 {
        sosfilt_sample_stack(f64::from(sample), &mut forward_sections) as f32
    }));
    filtered.extend((1..=edge).map(|index| -> f32 {
        let sample = 2.0 * right_end - f64::from(input_array[n - 1 - index]);
        sosfilt_sample_stack(sample, &mut forward_sections) as f32
    }));

    let y0 = f64::from(filtered[total - 1]);
    let mut backward_sections = init_sections;
    scale_sos_state(&mut backward_sections, y0);

    // Backward filter in-place. The first `edge` reverse samples are the right
    // padding the backward pass never keeps, so advance state over them in a
    // separate branch-free loop; the rest are written back at their final
    // indices.
    let mut index = total;
    for _ in 0..edge {
        index -= 1;
        sosfilt_sample_stack(f64::from(filtered[index]), &mut backward_sections);
    }
    while index > 0 {
        index -= 1;
        debug_assert!(index < n);
        filtered[index] =
            sosfilt_sample_stack(f64::from(filtered[index]), &mut backward_sections) as f32;
    }

    filtered.truncate(n);
    filtered
}

#[inline]
fn sosfiltfilt_ntaps(sos: &[Sos<f64>]) -> usize {
    let mut bzeros = 0;
    let mut azeros = 0;
    for section in sos {
        if section.b[2] == 0.0 {
            bzeros += 1;
        }
        if section.a[2] == 0.0 {
            azeros += 1;
        }
    }
    (2 * sos.len() + 1) - bzeros.min(azeros)
}

#[inline]
fn sosfilt_zi_stack<const SECTIONS: usize>(sections: &mut [Sos<f64>; SECTIONS]) {
    let mut scale = 1.0;
    for section in sections.iter_mut() {
        let (zi0, zi1) = sos_section_lfilter_zi(section);
        section.zi0 = scale * zi0;
        section.zi1 = scale * zi1;
        scale *= sum3(&section.b) / sum3(&section.a);
    }
}

#[inline]
fn sos_section_lfilter_zi(section: &Sos<f64>) -> (f64, f64) {
    // Drop leading zeros in the denominator, then normalize so a[0] == 1.
    // `a0` is the first nonzero coefficient, so dividing by it is always defined.
    let a_start = section
        .a
        .iter()
        .position(|&v| v != 0.0)
        .expect("There must be at least one nonzero `a` coefficient.");
    let a0 = section.a[a_start];

    let b = [section.b[0] / a0, section.b[1] / a0, section.b[2] / a0];
    let mut a = [1.0, 0.0, 0.0];
    for (dst, &src) in a[1..].iter_mut().zip(&section.a[a_start + 1..]) {
        *dst = src / a0;
    }

    let b1_term = b[1] - a[1] * b[0];
    let asum = a[0] + a[1] + a[2];
    let zi0 = (b1_term + (b[2] - a[2] * b[0])) / asum;
    let zi1 = (1.0 + a[1]) * zi0 - b1_term;
    (zi0, zi1)
}

#[inline(always)]
fn sosfilt_sample_stack<const SECTIONS: usize>(
    mut sample: f64,
    sections: &mut [Sos<f64>; SECTIONS],
) -> f64 {
    let mut index = 0;
    while index < SECTIONS {
        let section = &mut sections[index];
        // Fused multiply-adds shorten the recurrence's dependency chain and
        // halve the rounding steps. This trades bit-exactness for speed but
        // stays well within the similarity tolerance.
        let output = section.b[0].mul_add(sample, section.zi0);
        section.zi0 = section.b[1].mul_add(sample, (-section.a[1]).mul_add(output, section.zi1));
        section.zi1 = section.b[2].mul_add(sample, -(section.a[2] * output));
        sample = output;
        index += 1;
    }
    sample
}

#[inline]
fn scale_sos_state<const SECTIONS: usize>(sections: &mut [Sos<f64>; SECTIONS], scale: f64) {
    for section in sections.iter_mut() {
        section.zi0 *= scale;
        section.zi1 *= scale;
    }
}

#[inline(always)]
fn sum3(values: &[f64; 3]) -> f64 {
    values.iter().copied().sum()
}
