use core::iter::Sum;
use core::ops::SubAssign;

use num_traits::Float;
use sci_rs::na::{RealField, Scalar};
use sci_rs::signal::filter::design::Sos;
use sci_rs::signal::filter::sosfiltfilt_dyn;

const SOSFILTFILT_STACK_MAX_SECTIONS: usize = 12;

/// The float types the SOS runtime here is instantiated for. Bundles the
/// `num-traits` numeric surface the stack filter needs (`mul_add`, casts) with
/// the `nalgebra` bounds `Sos<F>` and the sci-rs dynamic fallback require, so
/// the generic functions can carry a single readable bound.
pub(crate) trait SosFloat: Float + RealField + Scalar + Sum + SubAssign + Default {}
impl SosFloat for f32 {}
impl SosFloat for f64 {}

/// Narrow a designed f64 SOS cascade to the native f32 representation used by
/// the filter runtime here. Only the coefficients are carried; the steady-state
/// delay values are re-derived at filter-run time.
pub(crate) fn narrow_sos(sos: &[Sos<f64>]) -> Vec<Sos<f32>> {
    sos.iter()
        .map(|section| {
            Sos::new(
                [
                    section.b[0] as f32,
                    section.b[1] as f32,
                    section.b[2] as f32,
                ],
                [
                    section.a[0] as f32,
                    section.a[1] as f32,
                    section.a[2] as f32,
                ],
            )
        })
        .collect()
}

pub(crate) fn sosfiltfilt_f32(sos: &[Sos<f32>], input_array: &[f32]) -> Vec<f32> {
    #[cfg(nightly_portable_simd)]
    if sos.len() == 1 && sos[0].b[2] == 0.0 && sos[0].a[2] == 0.0 {
        return sosfiltfilt_order1_scan_f32(&sos[0], input_array);
    }
    sosfiltfilt(sos, input_array)
}

#[inline]
fn sosfiltfilt<F: SosFloat>(sos: &[Sos<F>], input_array: &[F]) -> Vec<F> {
    match sos.len() {
        1 if sos[0].b[2].is_zero() && sos[0].a[2].is_zero() => {
            sosfiltfilt_order1(&sos[0], input_array)
        }
        1 => sosfiltfilt_stack::<1, F>(sos, input_array),
        2 => sosfiltfilt_stack::<2, F>(sos, input_array),
        3 => sosfiltfilt_stack::<3, F>(sos, input_array),
        4 => sosfiltfilt_stack::<4, F>(sos, input_array),
        5 => sosfiltfilt_stack::<5, F>(sos, input_array),
        6 => sosfiltfilt_stack::<6, F>(sos, input_array),
        7 => sosfiltfilt_stack::<7, F>(sos, input_array),
        8 => sosfiltfilt_stack::<8, F>(sos, input_array),
        9 => sosfiltfilt_stack::<9, F>(sos, input_array),
        10 => sosfiltfilt_stack::<10, F>(sos, input_array),
        11 => sosfiltfilt_stack::<11, F>(sos, input_array),
        12 => sosfiltfilt_stack::<12, F>(sos, input_array),
        _ => sosfiltfilt_dyn(input_array.iter().copied(), sos),
    }
}

/// A biquad section rewritten for a shorter loop-carried recurrence. The stored
/// transposed-DF2 step computes `out = b0*x + zi0` and then feeds `out` back into
/// both delay updates, so each new delay sits three FMAs downstream of the
/// previous one. Substituting `out` out of the updates,
///
/// ```text
///   out     = b0*x + zi0
///   zi0[t]  = (b1 - a1*b0)*x - a1*zi0[t-1] + zi1[t-1]
///   zi1[t]  = (b2 - a2*b0)*x - a2*zi0[t-1]
/// ```
///
/// leaves the feed-forward terms (`bff1*x`, `bff2*x`) off the carried path, so
/// `zi0[t]` is one FMA past `zi0[t-1]` and `out` is a parallel tap. Same section
/// difference equation; the precomputed `bff` coefficients and regrouping move
/// the result a few ULP.
#[derive(Clone, Copy, Default)]
struct ReducedBiquad<F> {
    b0: F,
    neg_a1: F,
    neg_a2: F,
    bff1: F,
    bff2: F,
    zi0: F,
    zi1: F,
}

impl<F: SosFloat> ReducedBiquad<F> {
    #[inline]
    fn from_sos(section: &Sos<F>) -> Self {
        ReducedBiquad {
            b0: section.b[0],
            neg_a1: -section.a[1],
            neg_a2: -section.a[2],
            bff1: section.b[1] - section.a[1] * section.b[0],
            bff2: section.b[2] - section.a[2] * section.b[0],
            zi0: section.zi0,
            zi1: section.zi1,
        }
    }
}

#[inline(always)]
fn reduced_step<F: SosFloat>(section: &mut ReducedBiquad<F>, sample: F) -> F {
    let zi0 = section.zi0;
    let output = Float::mul_add(section.b0, sample, zi0);
    let feed = Float::mul_add(section.bff1, sample, section.zi1);
    section.zi0 = Float::mul_add(section.neg_a1, zi0, feed);
    section.zi1 = Float::mul_add(section.neg_a2, zi0, section.bff2 * sample);
    output
}

#[inline(always)]
fn reduced_sample_stack<const SECTIONS: usize, F: SosFloat>(
    mut sample: F,
    sections: &mut [ReducedBiquad<F>; SECTIONS],
) -> F {
    let mut index = 0;
    while index < SECTIONS {
        sample = reduced_step(&mut sections[index], sample);
        index += 1;
    }
    sample
}

#[inline]
fn scale_reduced_state<const SECTIONS: usize, F: SosFloat>(
    sections: &mut [ReducedBiquad<F>; SECTIONS],
    scale: F,
) {
    for section in sections.iter_mut() {
        section.zi0 *= scale;
        section.zi1 *= scale;
    }
}

#[inline]
fn sosfiltfilt_stack<const SECTIONS: usize, F: SosFloat>(
    sos: &[Sos<F>],
    input_array: &[F],
) -> Vec<F> {
    debug_assert!(SECTIONS > 0);
    debug_assert!(SECTIONS <= SOSFILTFILT_STACK_MAX_SECTIONS);
    debug_assert_eq!(sos.len(), SECTIONS);

    let ntaps = sosfiltfilt_ntaps(sos);
    let edge = ntaps * 3;
    assert!(input_array.len() > edge);

    // Seed the per-section steady-state initial conditions. The solve runs in
    // f64 internally (it is sensitive to coefficient rounding); the f32 sections
    // carry the narrowed state for the per-sample recurrence. The input is
    // DC-centered to ~[-1, 1] at the source, which keeps that recurrence
    // well-conditioned.
    let mut init_sections = [Sos::<F>::default(); SECTIONS];
    init_sections.copy_from_slice(sos);
    sosfilt_zi(&mut init_sections);
    // Carry the cascade as reduced biquads: the per-sample recurrence is the same
    // section difference equation, but with the output substituted out of the
    // state update so each delay value depends on its predecessor through a
    // single FMA (see `reduced_step`). This shortens the loop-carried chain
    // without adding work, which matters most for the low-section filters whose
    // chain the out-of-order engine cannot hide behind neighbouring sections.
    let init_reduced: [ReducedBiquad<F>; SECTIONS] =
        core::array::from_fn(|i| ReducedBiquad::from_sos(&init_sections[i]));

    let two = F::one() + F::one();
    let left_end = input_array[0];
    let right_end = input_array[input_array.len() - 1];

    let x0 = two * left_end - input_array[edge];
    let mut forward_sections = init_reduced;
    scale_reduced_state(&mut forward_sections, x0);

    // Forward filter the left padding only to advance state. The backward pass
    // never reads those samples, so avoid storing them.
    for index in (1..=edge).rev() {
        let sample = two * left_end - input_array[index];
        reduced_sample_stack(sample, &mut forward_sections);
    }

    let n = input_array.len();
    let total = n + edge;
    // Build the forward-filtered buffer with `extend` over exact-size iterators:
    // that reserves `total` up front and writes each slot once via the
    // TrustedLen fast path (no per-element capacity check), so the hot loop is
    // as tight as the old indexed writes while skipping the `vec![0.0; total]`
    // zero-fill that was immediately overwritten in full.
    let mut filtered: Vec<F> = Vec::with_capacity(total);
    filtered.extend(
        input_array
            .iter()
            .map(|&sample| reduced_sample_stack(sample, &mut forward_sections)),
    );
    filtered.extend((1..=edge).map(|index| -> F {
        let sample = two * right_end - input_array[n - 1 - index];
        reduced_sample_stack(sample, &mut forward_sections)
    }));

    let y0 = filtered[total - 1];
    let mut backward_sections = init_reduced;
    scale_reduced_state(&mut backward_sections, y0);

    // Backward filter in-place. The first `edge` reverse samples are the right
    // padding the backward pass never keeps, so advance state over them in a
    // separate branch-free loop; the rest are written back at their final
    // indices.
    let mut index = total;
    for _ in 0..edge {
        index -= 1;
        reduced_sample_stack(filtered[index], &mut backward_sections);
    }
    while index > 0 {
        index -= 1;
        debug_assert!(index < n);
        filtered[index] = reduced_sample_stack(filtered[index], &mut backward_sections);
    }

    filtered.truncate(n);
    filtered
}

/// Zero-phase filter for a single first-order section (`b[2] == a[2] == 0`, so
/// the second delay `zi1` is identically zero). Mirrors `sosfiltfilt_stack::<1>`
/// — same odd-extension padding, same steady-state seeding — but runs a
/// shortened recurrence.
///
/// The transposed-DF2 step `out = b0·x + zi0; zi0 = b1·x - a1·out` chains
/// `zi0 -> out -> zi0`, a three-FMA loop-carried dependency. With a single
/// section there is no neighbouring section for the out-of-order engine to
/// overlap, so that chain sets the throughput. Substituting `out` into the
/// state update collapses it to `zi0[t] = (b1 - a1·b0)·x[t] - a1·zi0[t-1]`: the
/// `(b1 - a1·b0)·x` term is off the carried path, leaving one FMA between
/// successive `zi0`, and `out` becomes a parallel tap. Mathematically identical
/// to the section recurrence (a precomputed coefficient and regrouped
/// arithmetic shift the result by a few ULP, well inside the similarity bound).
fn sosfiltfilt_order1<F: SosFloat>(section: &Sos<F>, input_array: &[F]) -> Vec<F> {
    // ntaps = (2*1 + 1) - 1 = 2 for a first-order section; edge = ntaps * 3.
    let edge = 6;
    let n = input_array.len();
    assert!(n > edge);

    // Steady-state seed (scipy `sosfilt_zi`); for a first-order section zi1 == 0.
    let mut seed = [*section];
    sosfilt_zi(&mut seed);
    let zi0_base = seed[0].zi0;

    let b0 = section.b[0];
    let b1 = section.b[1];
    let a1 = section.a[1];
    let neg_a1 = -a1;
    // Off-chain feed-forward coefficient `b1 - a1*b0`.
    let bff = b1 - a1 * b0;

    // One shortened step: returns the output and advances `zi0` in a single
    // loop-carried FMA.
    let step = |zi0: &mut F, x: F| -> F {
        let out = Float::mul_add(b0, x, *zi0);
        *zi0 = Float::mul_add(neg_a1, *zi0, bff * x);
        out
    };

    let two = F::one() + F::one();
    let left_end = input_array[0];
    let right_end = input_array[input_array.len() - 1];

    let x0 = two * left_end - input_array[edge];
    let mut zi0 = zi0_base * x0;

    // Forward pass: advance over the left padding (discarded), then emit the
    // input and right padding.
    for index in (1..=edge).rev() {
        step(&mut zi0, two * left_end - input_array[index]);
    }

    let total = n + edge;
    let mut filtered: Vec<F> = Vec::with_capacity(total);
    filtered.extend(input_array.iter().map(|&sample| step(&mut zi0, sample)));
    filtered.extend(
        (1..=edge).map(|index| step(&mut zi0, two * right_end - input_array[n - 1 - index])),
    );

    // Backward pass in place: advance over the right padding (discarded), then
    // write the remaining samples back at their final indices.
    let y0 = filtered[total - 1];
    let mut zi0 = zi0_base * y0;
    let mut index = total;
    for _ in 0..edge {
        index -= 1;
        step(&mut zi0, filtered[index]);
    }
    while index > 0 {
        index -= 1;
        filtered[index] = step(&mut zi0, filtered[index]);
    }

    filtered.truncate(n);
    filtered
}

/// `sosfiltfilt_order1`, with the per-sample recurrence replaced by an 8-wide
/// scan. The first-order state obeys `zi0[t] = A*zi0[t-1] + bff*x[t]`, a linear
/// recurrence whose chunk solution is the decay-weighted prefix sum
/// `zi0[j] = A^(j+1)*zin + sum_k A^(j-k)*bff*x[k]`; three shift/multiply-add
/// steps build the sum for all 8 lanes, so the only chunk-carried value is the
/// leaving delay. The output stays `b0*x[t] + zi0[t-1]`, read from the lanes
/// shifted by one. Padding, seeding, and short tails reuse the scalar step.
#[cfg(nightly_portable_simd)]
fn sosfiltfilt_order1_scan_f32(section: &Sos<f32>, input_array: &[f32]) -> Vec<f32> {
    use std::simd::prelude::*;
    use std::simd::StdFloat;

    const LANES: usize = 16;
    let edge = 6;
    let n = input_array.len();
    assert!(n > edge);

    let mut seed = [*section];
    sosfilt_zi(&mut seed);
    let zi0_base = seed[0].zi0;

    let b0 = section.b[0];
    let neg_a1 = -section.a[1];
    let bff = section.b[1] - section.a[1] * section.b[0];

    let mut apow_arr = [0.0f32; LANES];
    let mut acc = 1.0f32;
    for entry in &mut apow_arr {
        acc *= neg_a1;
        *entry = acc;
    }
    let apow = Simd::from_array(apow_arr);
    let a_last = apow_arr[LANES - 1];
    let zero = Simd::splat(0.0f32);
    let vb0 = Simd::splat(b0);
    let vbff = Simd::splat(bff);
    let va1 = Simd::splat(neg_a1);
    let va2 = Simd::splat(neg_a1 * neg_a1);
    let va4 = Simd::splat(apow_arr[3]);
    let va8 = Simd::splat(apow_arr[7]);

    // One scan chunk: 16 inputs and the entering delay to 16 outputs and the
    // leaving delay. The prefix sum `s` is independent of the entering delay,
    // so the chunk-carried value is one scalar fused multiply-add; everything
    // through the broadcast and output evaluation hangs off that chain.
    let scan = |x: Simd<f32, LANES>, zin: f32| -> (Simd<f32, LANES>, f32) {
        let g = vbff * x;
        let s = va1.mul_add(
            simd_swizzle!(g, zero, [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            g,
        );
        let s = va2.mul_add(
            simd_swizzle!(s, zero, [16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            s,
        );
        let s = va4.mul_add(
            simd_swizzle!(
                s,
                zero,
                [16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            ),
            s,
        );
        let s = va8.mul_add(
            simd_swizzle!(
                s,
                zero,
                [16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
            ),
            s,
        );
        let vzin = Simd::splat(zin);
        let zi = apow.mul_add(vzin, s);
        let zi_prev = simd_swizzle!(
            zi,
            vzin,
            [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        );
        (vb0.mul_add(x, zi_prev), a_last.mul_add(zin, s[LANES - 1]))
    };

    let step = |zi0: &mut f32, x: f32| -> f32 {
        let out = b0.mul_add(x, *zi0);
        *zi0 = neg_a1.mul_add(*zi0, bff * x);
        out
    };

    let left_end = input_array[0];
    let right_end = input_array[n - 1];

    let x0 = 2.0 * left_end - input_array[edge];
    let mut zi0 = zi0_base * x0;
    for index in (1..=edge).rev() {
        step(&mut zi0, 2.0 * left_end - input_array[index]);
    }

    let total = n + edge;
    let mut filtered: Vec<f32> = Vec::with_capacity(total);
    let mut chunks = input_array.chunks_exact(LANES);
    for chunk in &mut chunks {
        let (out, leaving) = scan(Simd::from_slice(chunk), zi0);
        zi0 = leaving;
        filtered.extend_from_slice(&out.to_array());
    }
    for &sample in chunks.remainder() {
        let out = step(&mut zi0, sample);
        filtered.push(out);
    }
    filtered.extend(
        (1..=edge).map(|index| step(&mut zi0, 2.0 * right_end - input_array[n - 1 - index])),
    );

    // Backward pass in place: reverse each chunk's lanes, run the same scan,
    // and store them back reversed.
    let y0 = filtered[total - 1];
    let mut zi0 = zi0_base * y0;
    let mut index = total;
    for _ in 0..edge {
        index -= 1;
        step(&mut zi0, filtered[index]);
    }
    while index >= LANES {
        let x = Simd::<f32, LANES>::from_slice(&filtered[index - LANES..index]).reverse();
        let (out, leaving) = scan(x, zi0);
        zi0 = leaving;
        filtered[index - LANES..index].copy_from_slice(&out.reverse().to_array());
        index -= LANES;
    }
    while index > 0 {
        index -= 1;
        filtered[index] = step(&mut zi0, filtered[index]);
    }

    filtered.truncate(n);
    filtered
}

#[inline]
fn sosfiltfilt_ntaps<F: SosFloat>(sos: &[Sos<F>]) -> usize {
    let mut bzeros = 0;
    let mut azeros = 0;
    for section in sos {
        if section.b[2].is_zero() {
            bzeros += 1;
        }
        if section.a[2].is_zero() {
            azeros += 1;
        }
    }
    (2 * sos.len() + 1) - bzeros.min(azeros)
}

/// Set each section's delay state to the unit-step steady state, scaled by the
/// running cascade gain (scipy's `sosfilt_zi`) — the state the filter would
/// settle into if a constant 1.0 had been applied forever. The per-section
/// solve and the cascade gain run in f64 and the result is narrowed to the f32
/// state.
#[inline]
fn sosfilt_zi<F: SosFloat>(sections: &mut [Sos<F>]) {
    let mut scale = 1.0f64;
    for section in sections.iter_mut() {
        let (zi0, zi1) = sos_section_lfilter_zi(section);
        section.zi0 = F::from(scale * zi0).unwrap();
        section.zi1 = F::from(scale * zi1).unwrap();
        scale *= sum3(&section.b) / sum3(&section.a);
    }
}

#[inline]
fn sos_section_lfilter_zi<F: SosFloat>(section: &Sos<F>) -> (f64, f64) {
    // Drop leading zeros in the denominator, then normalize so a[0] == 1.
    // `a0` is the first nonzero coefficient, so dividing by it is always defined.
    // The solve runs in f64 regardless of `F` (it is sensitive to coefficient
    // rounding); for an f64 cascade the widening is the identity.
    let a_in = [
        section.a[0].to_f64().unwrap(),
        section.a[1].to_f64().unwrap(),
        section.a[2].to_f64().unwrap(),
    ];
    let b_in = [
        section.b[0].to_f64().unwrap(),
        section.b[1].to_f64().unwrap(),
        section.b[2].to_f64().unwrap(),
    ];
    let a_start = a_in
        .iter()
        .position(|&v| v != 0.0)
        .expect("There must be at least one nonzero `a` coefficient.");
    let a0 = a_in[a_start];

    let b = [b_in[0] / a0, b_in[1] / a0, b_in[2] / a0];
    let mut a = [1.0, 0.0, 0.0];
    for (dst, &src) in a[1..].iter_mut().zip(&a_in[a_start + 1..]) {
        *dst = src / a0;
    }

    let b1_term = b[1] - a[1] * b[0];
    let asum = a[0] + a[1] + a[2];
    let zi0 = (b1_term + (b[2] - a[2] * b[0])) / asum;
    let zi1 = (1.0 + a[1]) * zi0 - b1_term;
    (zi0, zi1)
}

/// Advance a single biquad section by one sample, returning its output.
#[inline(always)]
fn sos_step<F: SosFloat>(section: &mut Sos<F>, sample: F) -> F {
    // Fused multiply-adds shorten the recurrence's dependency chain and halve
    // the rounding steps. This trades bit-exactness for speed but stays well
    // within the similarity tolerance.
    // Fully qualify `mul_add`: both `num_traits::Float` and nalgebra's
    // `ComplexField` are in scope via the `SosFloat` bound and supply one.
    let output = Float::mul_add(section.b[0], sample, section.zi0);
    section.zi0 = Float::mul_add(
        section.b[1],
        sample,
        Float::mul_add(-section.a[1], output, section.zi1),
    );
    section.zi1 = Float::mul_add(section.b[2], sample, -(section.a[2] * output));
    output
}

#[inline(always)]
fn sosfilt_sample_stack<const SECTIONS: usize, F: SosFloat>(
    mut sample: F,
    sections: &mut [Sos<F>; SECTIONS],
) -> F {
    let mut index = 0;
    while index < SECTIONS {
        sample = sos_step(&mut sections[index], sample);
        index += 1;
    }
    sample
}

#[inline(always)]
fn sosfilt_sample_slice<F: SosFloat>(mut sample: F, sections: &mut [Sos<F>]) -> F {
    for section in sections.iter_mut() {
        sample = sos_step(section, sample);
    }
    sample
}

/// Forward (single-pass) SOS filter with zero initial conditions, run in f32.
/// Unlike `sosfiltfilt_f32` this is a single forward pass — no boundary padding
/// and no forward/backward symmetry — so it keeps the filter's phase response
/// instead of zeroing it.
pub(crate) fn sosfilt_f32(sos: &[Sos<f32>], input: &[f32]) -> Vec<f32> {
    match sos.len() {
        1 => sosfilt_stack::<1>(sos, input),
        2 => sosfilt_stack::<2>(sos, input),
        3 => sosfilt_stack::<3>(sos, input),
        4 => sosfilt_stack::<4>(sos, input),
        5 => sosfilt_stack::<5>(sos, input),
        6 => sosfilt_stack::<6>(sos, input),
        7 => sosfilt_stack::<7>(sos, input),
        8 => sosfilt_stack::<8>(sos, input),
        9 => sosfilt_stack::<9>(sos, input),
        10 => sosfilt_stack::<10>(sos, input),
        11 => sosfilt_stack::<11>(sos, input),
        12 => sosfilt_stack::<12>(sos, input),
        _ => sosfilt_dynamic(sos, input),
    }
}

#[inline]
fn sosfilt_stack<const SECTIONS: usize>(sos: &[Sos<f32>], input: &[f32]) -> Vec<f32> {
    debug_assert_eq!(sos.len(), SECTIONS);
    // The stored sections carry zero delay state, which is exactly the zero
    // initial condition this single forward pass wants.
    let mut sections = [Sos::<f32>::default(); SECTIONS];
    sections.copy_from_slice(sos);
    input
        .iter()
        .map(|&sample| sosfilt_sample_stack(sample, &mut sections))
        .collect()
}

fn sosfilt_dynamic(sos: &[Sos<f32>], input: &[f32]) -> Vec<f32> {
    let mut sections: Vec<Sos<f32>> = sos.to_vec();
    input
        .iter()
        .map(|&sample| sosfilt_sample_slice(sample, &mut sections))
        .collect()
}

#[inline(always)]
fn sum3<F: SosFloat>(values: &[F; 3]) -> f64 {
    values.iter().map(|&v| v.to_f64().unwrap()).sum()
}
