use anyhow::{bail, Result};

// Full-field wow factors are large; fixed-point keeps the buffer at 4 bytes per
// sample without losing the precision needed by the level-adjust pass.
const WOW_FACTOR_SCALE: f64 = 100_000_000.0;

fn pack_wow_factor(value: f64) -> i32 {
    ((value - 1.0) * WOW_FACTOR_SCALE)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn unpack_wow_factor(value: f64) -> f64 {
    1.0 + value / WOW_FACTOR_SCALE
}

/// Tuning inputs shared by `scale_field` and every `scale_field_k` instantiation.
/// Bundled so the degree dispatch forwards a single value instead of repeating
/// the full argument list once per spline degree.
#[derive(Clone, Copy)]
pub(crate) struct ScaleFieldParams {
    pub eval_scale: f64,
    pub eval_count: usize,
    pub lineoffset: usize,
    pub outwidth: usize,
    pub wow_level_adjust_smoothing: f32,
    pub level_adjust_threshold: f64,
    pub cached_median_mad: Option<(f64, f64)>,
}

pub(crate) fn scale_field(
    buf: &[f32],
    out_len: usize,
    expected_linelocs: &[f64],
    actual_linelocs: &[f64],
    k: usize,
    params: ScaleFieldParams,
) -> Result<(Vec<f32>, f64, f64)> {
    let (interp_t, interp_c) = make_interp_spline_scaled(expected_linelocs, actual_linelocs, k)?;
    let spline = (interp_t.as_slice(), interp_c.as_slice());
    Ok(match k {
        1 => scale_field_linear(buf, out_len, spline, &params),
        2 => scale_field_k::<2>(buf, out_len, spline, &params),
        3 => scale_field_k::<3>(buf, out_len, spline, &params),
        _ => unreachable!("unsupported spline degree"),
    })
}

fn level_adjust_threshold_value(level_adjust_threshold: f64, mad: f64) -> f64 {
    if mad > 0.0 {
        level_adjust_threshold * mad
    } else {
        0.001 * WOW_FACTOR_SCALE
    }
}

fn level_adjusted(packed: i32, median: f64, threshold: f64) -> f64 {
    if (packed as f64 - median).abs() > threshold {
        unpack_wow_factor(median)
    } else {
        unpack_wow_factor(packed as f64)
    }
}

#[inline]
fn catmull_rom4(p0: f32, p1: f32, p2: f32, p3: f32, x: f32) -> f32 {
    let a = p2 - p0;
    let b = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
    let c = 3.0 * (p1 - p2) + p3 - p0;
    // Horner form via fused multiply-add: p1 + 0.5*x*(a + x*(b + x*c)).
    let poly = c.mul_add(x, b).mul_add(x, a);
    (0.5 * x).mul_add(poly, p1)
}

/// One knot span of the degree-1 spline, with everything that is constant
/// across the span hoisted out: the value lerp endpoints and the (constant)
/// derivative.
#[derive(Clone, Copy)]
struct LinSpan {
    t0: f64,
    t1: f64,
    inv_h: f64,
    c0: f64,
    c1: f64,
    deriv: f64,
}

fn lin_span(t: &[f64], c: &[f64], s: usize) -> LinSpan {
    let t0 = t[s];
    let t1 = t[s + 1];
    let inv_h = 1.0 / (t1 - t0);
    let c0 = c[s - 1];
    let c1 = c[s];
    LinSpan {
        t0,
        t1,
        inv_h,
        c0,
        c1,
        deriv: (c1 - c0) * inv_h,
    }
}

/// First eval index in `[lo, hi]` whose sample position `idx * eval_scale`
/// reaches `bound`. The position is monotone in the index, so the fix-up walks
/// land on the exact crossover even when the seeding division rounds.
fn first_index_reaching(bound: f64, eval_scale: f64, lo: usize, hi: usize) -> usize {
    let mut idx = ((bound / eval_scale) as usize).clamp(lo, hi);
    while idx > lo && (idx - 1) as f64 * eval_scale >= bound {
        idx -= 1;
    }
    while idx < hi && (idx as f64) * eval_scale < bound {
        idx += 1;
    }
    idx
}

/// Value holding the given 0-based rank in a run-length encoded sorted list.
fn rank_value<T: Copy>(pairs: &[(T, u32)], rank: usize) -> T {
    let mut before = 0usize;
    for &(value, count) in pairs {
        let next = before + count as usize;
        if next > rank {
            return value;
        }
        before = next;
    }
    unreachable!("rank outside run-length counts")
}

/// Median (mean of the two middle order statistics for an even total) of a
/// run-length encoded multiset, matching `radix_median_by_key` exactly.
fn weighted_median<T: Copy + Ord>(
    pairs: &mut [(T, u32)],
    total: usize,
    decode: impl Fn(T) -> f64,
) -> f64 {
    assert!(total > 0);
    pairs.sort_unstable_by_key(|pair| pair.0);
    let upper_index = total / 2;
    let upper = decode(rank_value(pairs, upper_index));
    if !total.is_multiple_of(2) {
        return upper;
    }
    let lower = decode(rank_value(pairs, upper_index - 1));
    (lower + upper) / 2.0
}

/// Degree-1 (default) specialization of `scale_field_k`. The linear spline has
/// a constant derivative inside each knot span, so the wow factor, its packed
/// form, and the level-adjust threshold decision are all span-invariant: hoist
/// them and walk the output span by span. The median/MAD gather collapses the
/// same way, from one spline evaluation per output sample to one
/// `(value, count)` run per span fed to an exact weighted median.
fn scale_field_linear(
    buf: &[f32],
    out_len: usize,
    spline: (&[f64], &[f64]),
    params: &ScaleFieldParams,
) -> (Vec<f32>, f64, f64) {
    const K: usize = 1;
    let ScaleFieldParams {
        eval_scale,
        eval_count,
        lineoffset,
        outwidth,
        wow_level_adjust_smoothing,
        level_adjust_threshold,
        cached_median_mad,
    } = *params;
    let lineoffset_out_samples = outwidth * (lineoffset + 1);
    let required_eval_count = lineoffset_out_samples + out_len;
    assert!(required_eval_count <= eval_count);

    let (t, c) = spline;
    let nt = t.len() - K - 1;

    let (median, mad) = cached_median_mad.unwrap_or_else(|| {
        assert!(eval_count > 0);
        // Each span contributes one packed wow factor, repeated once per eval
        // index that falls inside it — the same multiset the per-sample walk
        // produced, so the medians are bit-identical.
        let mut pairs: Vec<(i32, u32)> = Vec::with_capacity(nt);
        let mut idx = 0usize;
        for span in K..nt {
            if idx >= eval_count {
                break;
            }
            let sp = lin_span(t, c, span);
            let end = if span + 1 < nt {
                first_index_reaching(sp.t1, eval_scale, idx, eval_count)
            } else {
                eval_count
            };
            if end > idx {
                pairs.push((pack_wow_factor(sp.deriv), (end - idx) as u32));
                idx = end;
            }
        }
        let median = weighted_median(&mut pairs, eval_count, |value| value as f64);
        let mut diff_pairs: Vec<(u32, u32)> = pairs
            .iter()
            .map(|&(value, count)| (wow_factor_abs_diff(value, median), count))
            .collect();
        let mad = weighted_median(&mut diff_pairs, eval_count, f64::from);
        (median, mad)
    });

    let threshold = level_adjust_threshold_value(level_adjust_threshold, mad);
    let smoothing_enabled = wow_level_adjust_smoothing > 0.0;
    let (alpha, one_minus_alpha) = if smoothing_enabled {
        let alpha = 1.0 / (f64::from(wow_level_adjust_smoothing) * outwidth as f64);
        (alpha, 1.0 - alpha)
    } else {
        (0.0, 0.0)
    };

    // Warmup smoothing over the lead-in samples. This walk is a tiny fraction
    // of the field, so it keeps the straightforward per-sample evaluation.
    let mut warmup_span = K;
    let initial = eval_spline_value_deriv_k::<K>(t, c, nt, &mut warmup_span, 0.0).1;
    let mut smoothed_adjust = level_adjusted(pack_wow_factor(initial), median, threshold);
    if smoothing_enabled && lineoffset_out_samples > 1 {
        for index in 1..lineoffset_out_samples {
            let deriv = eval_spline_value_deriv_k::<K>(
                t,
                c,
                nt,
                &mut warmup_span,
                index as f64 * eval_scale,
            )
            .1;
            smoothed_adjust = alpha.mul_add(
                level_adjusted(pack_wow_factor(deriv), median, threshold),
                one_minus_alpha * smoothed_adjust,
            );
        }
    }

    if outwidth == 0 {
        return (Vec::new(), median, mad);
    }

    let mut dsout = Vec::with_capacity(out_len);
    let mut span = K;
    let end_eval = lineoffset_out_samples + out_len;
    let mut idx = lineoffset_out_samples;
    while idx < end_eval {
        // Re-derive the span exactly the way the per-sample evaluation did,
        // then run every sample that stays inside it with the hoisted state.
        let x = idx as f64 * eval_scale;
        if x <= t[K] {
            span = K;
        } else if x >= t[nt] {
            span = nt - 1;
        } else {
            while span + 1 < nt && x >= t[span + 1] {
                span += 1;
            }
        }
        let sp = lin_span(t, c, span);
        let run_end = if span + 1 < nt {
            first_index_reaching(sp.t1, eval_scale, idx + 1, end_eval)
        } else {
            end_eval
        };
        let raw_adjust = level_adjusted(pack_wow_factor(sp.deriv), median, threshold);
        let alpha_raw = alpha * raw_adjust;
        let tail_start = {
            #[cfg(nightly_portable_simd)]
            {
                scale_span_simd(
                    buf,
                    &mut dsout,
                    &sp,
                    idx,
                    run_end,
                    eval_scale,
                    raw_adjust,
                    &mut smoothed_adjust,
                    smoothing_enabled,
                    one_minus_alpha,
                )
            }
            #[cfg(not(nightly_portable_simd))]
            {
                idx
            }
        };
        for eval_index in tail_start..run_end {
            let x = eval_index as f64 * eval_scale;
            let b0 = (sp.t1 - x) * sp.inv_h;
            let b1 = (x - sp.t0) * sp.inv_h;
            let coord = b0 * sp.c0 + b1 * sp.c1;
            let level_adjust = if smoothing_enabled {
                if eval_index == 0 {
                    smoothed_adjust = raw_adjust;
                } else {
                    smoothed_adjust = one_minus_alpha.mul_add(smoothed_adjust, alpha_raw);
                }
                smoothed_adjust
            } else {
                raw_adjust
            };
            let coord_int = coord as usize;
            let w = &buf[coord_int - 1..coord_int + 3];
            let x = (coord - coord_int as f64) as f32;
            dsout.push(level_adjust as f32 * catmull_rom4(w[0], w[1], w[2], w[3], x));
        }
        idx = run_end;
    }
    (dsout, median, mad)
}

/// Vector body of one degree-1 knot-span run: evaluates `[start, end)` in
/// 8-wide chunks and returns the first index left for the scalar tail. The
/// only loop-carried state, the level-adjust EWMA, decays toward the
/// span-constant raw adjust, so each chunk applies the closed form
/// `s[j] = raw + (s_prev - raw) * q^j` and the lanes are independent.
#[cfg(nightly_portable_simd)]
#[allow(clippy::too_many_arguments)]
fn scale_span_simd(
    buf: &[f32],
    dsout: &mut Vec<f32>,
    sp: &LinSpan,
    start: usize,
    end: usize,
    eval_scale: f64,
    raw_adjust: f64,
    smoothed_adjust: &mut f64,
    smoothing_enabled: bool,
    one_minus_alpha: f64,
) -> usize {
    use std::simd::prelude::*;
    use std::simd::StdFloat;

    const LANES: usize = 8;
    if end - start < LANES {
        return start;
    }
    // The eval_index == 0 reset cannot occur here: the main walk starts at
    // lineoffset_out_samples, which is at least one output line.
    debug_assert!(start > 0);

    let iota = Simd::from_array([0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let mut qpow = [0.0f64; LANES];
    if smoothing_enabled {
        let mut q = 1.0;
        for entry in &mut qpow {
            q *= one_minus_alpha;
            *entry = q;
        }
    }
    let qpow = Simd::from_array(qpow);
    let t1 = Simd::splat(sp.t1);
    let t0 = Simd::splat(sp.t0);
    let inv_h = Simd::splat(sp.inv_h);
    let c0 = Simd::splat(sp.c0);
    let c1 = Simd::splat(sp.c1);
    let scale = Simd::splat(eval_scale);
    let raw = Simd::splat(raw_adjust);
    let one = Simd::splat(1usize);

    let mut eval = start;
    while eval + LANES <= end {
        let x = (Simd::splat(eval as f64) + iota) * scale;
        let b0 = (t1 - x) * inv_h;
        let b1 = (x - t0) * inv_h;
        let coord = b0 * c0 + b1 * c1;
        let level_adjust: Simd<f32, LANES> = if smoothing_enabled {
            let s = raw + Simd::splat(*smoothed_adjust - raw_adjust) * qpow;
            *smoothed_adjust = s[LANES - 1];
            s.cast()
        } else {
            raw.cast()
        };
        // Establish the tap window bounds in the float domain (a NaN coordinate
        // fails the comparison and panics here too), which licenses the direct
        // truncating conversion and the unchecked gathers below: the saturating
        // cast and per-lane bounds masks otherwise scalarize the index setup.
        assert!(
            coord.reduce_min() >= 1.0 && coord.reduce_max() + 3.0 <= buf.len() as f64,
            "wow coordinate outside the field buffer"
        );
        // SAFETY: every lane is within [1.0, buf.len() - 3.0] per the assert.
        let ci: Simd<usize, LANES> = unsafe { coord.to_int_unchecked::<isize>() }.cast();
        let frac: Simd<f32, LANES> = (coord - ci.cast::<f64>()).cast();
        let yes = Mask::splat(true);
        let zero = Simd::splat(0.0f32);
        // SAFETY: indices ci - 1 ..= ci + 2 are in bounds per the assert above.
        let (p0, p1, p2, p3) = unsafe {
            (
                Simd::gather_select_unchecked(buf, yes, ci - one, zero),
                Simd::gather_select_unchecked(buf, yes, ci, zero),
                Simd::gather_select_unchecked(buf, yes, ci + one, zero),
                Simd::gather_select_unchecked(buf, yes, ci + one + one, zero),
            )
        };
        let a = p2 - p0;
        let b = Simd::splat(2.0f32) * p0 - Simd::splat(5.0f32) * p1 + Simd::splat(4.0f32) * p2
            - p3;
        let c = Simd::splat(3.0f32) * (p1 - p2) + p3 - p0;
        let poly = c.mul_add(frac, b).mul_add(frac, a);
        let value = (Simd::splat(0.5f32) * frac).mul_add(poly, p1);
        dsout.extend_from_slice(&(level_adjust * value).to_array());
        eval += LANES;
    }
    eval
}

fn scale_field_k<const K: usize>(
    buf: &[f32],
    out_len: usize,
    spline: (&[f64], &[f64]),
    params: &ScaleFieldParams,
) -> (Vec<f32>, f64, f64) {
    let ScaleFieldParams {
        eval_scale,
        eval_count,
        lineoffset,
        outwidth,
        wow_level_adjust_smoothing,
        level_adjust_threshold,
        cached_median_mad,
    } = *params;
    let lineoffset_out_samples = outwidth * (lineoffset + 1);
    let required_eval_count = lineoffset_out_samples + out_len;
    assert!(required_eval_count <= eval_count);

    let (t, c) = spline;
    let nt = t.len() - K - 1;

    // The wow-factor median/MAD depend only on the line-location spline, so the
    // caller caches them across a field's channels. When not cached, evaluate
    // the spline derivative once per index to gather the distribution and take
    // the median and MAD.
    let (median, mad) = cached_median_mad.unwrap_or_else(|| {
        // Collect straight from the spline walk so the buffer is written once,
        // skipping the zero-init pass of `vec![0; eval_count]`.
        let mut span = K;
        let wow_packed: Vec<i32> = (0..eval_count)
            .map(|index| {
                let (_, deriv) =
                    eval_spline_value_deriv_k::<K>(t, c, nt, &mut span, index as f64 * eval_scale);
                pack_wow_factor(deriv)
            })
            .collect();
        let median = median_wow_factors(&wow_packed);
        (median, median_wow_factor_abs_diff(&wow_packed, median))
    });

    let threshold = level_adjust_threshold_value(level_adjust_threshold, mad);
    let adjusted = |value: i32| -> f64 { level_adjusted(value, median, threshold) };
    let smoothing_enabled = wow_level_adjust_smoothing > 0.0;
    let (alpha, one_minus_alpha) = if smoothing_enabled {
        let alpha = 1.0 / (f64::from(wow_level_adjust_smoothing) * outwidth as f64);
        (alpha, 1.0 - alpha)
    } else {
        (0.0, 0.0)
    };

    // Warmup smoothing over the lead-in samples. Walk the spline derivative
    // directly so this does not depend on the (cached-away) packed buffer.
    let mut warmup_span = K;
    let initial = eval_spline_value_deriv_k::<K>(t, c, nt, &mut warmup_span, 0.0).1;
    let mut smoothed_adjust = adjusted(pack_wow_factor(initial));
    if smoothing_enabled && lineoffset_out_samples > 1 {
        for index in 1..lineoffset_out_samples {
            let deriv = eval_spline_value_deriv_k::<K>(
                t,
                c,
                nt,
                &mut warmup_span,
                index as f64 * eval_scale,
            )
            .1;
            smoothed_adjust = alpha.mul_add(
                adjusted(pack_wow_factor(deriv)),
                one_minus_alpha * smoothed_adjust,
            );
        }
    }

    let mut span = K;
    if outwidth == 0 {
        return (Vec::new(), median, mad);
    }

    // Build the output directly instead of zero-initializing a buffer and
    // overwriting every element: every index is written exactly once here, so
    // the `vec![0.0; ...]` memset was pure wasted bandwidth in the MT path.
    // `eval_index`/`smoothed_adjust` advance continuously across the whole
    // field, so a flat walk is identical to the previous per-line chunking.
    let mut dsout = Vec::with_capacity(out_len);
    for output_index in 0..out_len {
        let eval_index = lineoffset_out_samples + output_index;
        let (coord, deriv) =
            eval_spline_value_deriv_k::<K>(t, c, nt, &mut span, eval_index as f64 * eval_scale);
        let raw_adjust = adjusted(pack_wow_factor(deriv));
        let level_adjust = if smoothing_enabled {
            if eval_index == 0 {
                smoothed_adjust = raw_adjust;
            } else {
                smoothed_adjust = alpha.mul_add(raw_adjust, one_minus_alpha * smoothed_adjust);
            }
            smoothed_adjust
        } else {
            raw_adjust
        };
        let coord_int = coord as usize;

        let w = &buf[coord_int - 1..coord_int + 3];
        // `coord` carries the full sample position (up to the field length), so the
        // fractional offset is taken in f64; the Catmull-Rom cubic itself then
        // evaluates in f32 over the already-f32 luma samples.
        let x = (coord - coord_int as f64) as f32;
        dsout.push(level_adjust as f32 * catmull_rom4(w[0], w[1], w[2], w[3], x));
    }
    (dsout, median, mad)
}

const WOW_FACTOR_RADIX_BITS: u32 = 16;
const WOW_FACTOR_RADIX_SIZE: usize = 1 << WOW_FACTOR_RADIX_BITS;
const WOW_FACTOR_RADIX_MASK: u32 = (WOW_FACTOR_RADIX_SIZE as u32) - 1;
const WOW_FACTOR_I32_ORDER_BIAS: u32 = 0x8000_0000;

fn select_count_bucket(counts: &[u32], target_index: usize) -> (usize, usize) {
    let mut before = 0usize;
    for (bucket, &count) in counts.iter().enumerate() {
        let next = before + count as usize;
        if next > target_index {
            return (bucket, before);
        }
        before = next;
    }
    unreachable!("target index outside histogram count")
}

/// Low-radix histogram for the values of `packed` (keyed by `key`) that fall in
/// the given high-radix `high` bucket.
fn low_radix_histogram<F>(packed: &[i32], key: F, high: usize) -> Vec<u32>
where
    F: Fn(i32) -> u32 + Copy,
{
    let mut low_counts = vec![0u32; WOW_FACTOR_RADIX_SIZE];
    for &value in packed {
        let ordered = key(value);
        if (ordered >> WOW_FACTOR_RADIX_BITS) as usize == high {
            low_counts[(ordered & WOW_FACTOR_RADIX_MASK) as usize] += 1;
        }
    }
    low_counts
}

#[inline]
fn radix_value(high: usize, low_counts: &[u32], rank_in_high: usize) -> u32 {
    let (low, _) = select_count_bucket(low_counts, rank_in_high);
    ((high as u32) << WOW_FACTOR_RADIX_BITS) | low as u32
}

/// Median (mean of the two middle order statistics for an even count) of
/// `packed` under the order-preserving `key`, with `decode` mapping an ordered
/// `u32` back to its real value. The high-radix histogram is built once and the
/// low-radix histogram is reused whenever both middle ranks share a bucket, so
/// the buffer is scanned twice (the common case) instead of once per `select`.
fn radix_median_by_key<K, D>(packed: &[i32], key: K, decode: D) -> f64
where
    K: Fn(i32) -> u32 + Copy,
    D: Fn(u32) -> f64,
{
    let count = packed.len();
    assert!(count > 0);
    let upper_index = count / 2;

    let mut high_counts = vec![0u32; WOW_FACTOR_RADIX_SIZE];
    for &value in packed {
        high_counts[(key(value) >> WOW_FACTOR_RADIX_BITS) as usize] += 1;
    }

    let (high_u, before_u) = select_count_bucket(&high_counts, upper_index);
    let low_u = low_radix_histogram(packed, key, high_u);
    let upper = radix_value(high_u, &low_u, upper_index - before_u);

    if !count.is_multiple_of(2) {
        return decode(upper);
    }

    let lower_index = upper_index - 1;
    let (high_l, before_l) = select_count_bucket(&high_counts, lower_index);
    let lower = if high_l == high_u {
        radix_value(high_l, &low_u, lower_index - before_l)
    } else {
        let low_l = low_radix_histogram(packed, key, high_l);
        radix_value(high_l, &low_l, lower_index - before_l)
    };
    (decode(lower) + decode(upper)) / 2.0
}

fn median_wow_factors(packed: &[i32]) -> f64 {
    radix_median_by_key(
        packed,
        |value| (value as u32) ^ WOW_FACTOR_I32_ORDER_BIAS,
        |ordered| ((ordered ^ WOW_FACTOR_I32_ORDER_BIAS) as i32) as f64,
    )
}

fn wow_factor_abs_diff(value: i32, median: f64) -> u32 {
    (value as f64 - median).abs().round() as u32
}

fn median_wow_factor_abs_diff(packed: &[i32], median: f64) -> f64 {
    radix_median_by_key(
        packed,
        |value| wow_factor_abs_diff(value, median),
        |ordered| ordered as f64,
    )
}

fn eval_spline_value_deriv_k<const K: usize>(
    t: &[f64],
    c: &[f64],
    nt: usize,
    span: &mut usize,
    x: f64,
) -> (f64, f64) {
    if x <= t[K] {
        *span = K;
    } else if x >= t[nt] {
        *span = nt - 1;
    } else {
        while *span + 1 < nt && x >= t[*span + 1] {
            *span += 1;
        }
    }
    // The degree-1 (linear) spline is the default wow-interpolation mode and the
    // hot caller here. Its two basis functions are straight lines, so evaluate
    // them and their derivatives in closed form rather than running the general
    // `bspline_ders_basis` recurrence (with its 4x4 scratch arrays and factor
    // passes) for what reduces to a lerp. `K` is a const generic, so the branch
    // is resolved at compile time.
    if K == 1 {
        let s = *span;
        let inv_h = 1.0 / (t[s + 1] - t[s]);
        let b0 = (t[s + 1] - x) * inv_h;
        let b1 = (x - t[s]) * inv_h;
        let c0 = c[s - 1];
        let c1 = c[s];
        return (b0 * c0 + b1 * c1, (c1 - c0) * inv_h);
    }
    let ders = bspline_ders_basis::<K>(t, *span, x, 1);
    let cofs = &c[*span - K..=*span];
    let mut value = 0.0;
    let mut deriv = 0.0;
    for j in 0..=K {
        let cj = cofs[j];
        value += ders[0][j] * cj;
        deriv += ders[1][j] * cj;
    }
    (value, deriv)
}

/// Build an interpolating spline of degree `k` over (x, y) for the wow
/// interpolation modes (k=1 linear, k=2 quadratic, k=3 natural cubic).
fn make_interp_spline_scaled(x: &[f64], y: &[f64], k: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    // Dispatch once on the runtime degree so the per-point inner loops are
    // monomorphized over a compile-time `K` (collapses the general NURBS recurrence,
    // especially for the common linear k=1 case).
    match k {
        1 => make_interp_spline::<1>(x, y, false),
        2 => make_interp_spline::<2>(x, y, false),
        3 => make_interp_spline::<3>(x, y, true),
        _ => bail!("unsupported spline degree"),
    }
}

/// Construct an interpolating B-spline (knots + coefficients) for degree `k`.
/// `natural` selects the zero-second-derivative boundary condition at both
/// ends, used for the cubic case; otherwise not-a-knot knots are used.
fn make_interp_spline<const K: usize>(
    x: &[f64],
    y: &[f64],
    natural: bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = x.len();
    if n != y.len() {
        bail!("make_interp_spline: x and y length mismatch");
    }
    if n < 2 {
        bail!("make_interp_spline: need at least two points");
    }

    // special-case k=1: t = [x0, x, x_{-1}], c = y (Lyche and Morken, Eq.(2.16)).
    if K == 1 {
        let mut t = Vec::with_capacity(n + 2);
        t.push(x[0]);
        t.extend_from_slice(x);
        t.push(x[n - 1]);
        return Ok((t, y.to_vec()));
    }

    // Construct the knot vector.
    let t: Vec<f64> = if natural {
        // _augknt(x, k): k copies of x0, x, k copies of x_{-1}.
        let mut t = Vec::with_capacity(n + 2 * K);
        for _ in 0..K {
            t.push(x[0]);
        }
        t.extend_from_slice(x);
        for _ in 0..K {
            t.push(x[n - 1]);
        }
        t
    } else {
        // _not_a_knot(x, k).
        let interior: Vec<f64> = if K % 2 == 1 {
            let k2 = K.div_ceil(2);
            x[k2..n - k2].to_vec()
        } else {
            let k2 = K / 2;
            let mids: Vec<f64> = (0..n - 1).map(|i| (x[i + 1] + x[i]) / 2.0).collect();
            mids[k2..mids.len() - k2].to_vec()
        };
        let mut t = Vec::with_capacity(2 * (K + 1) + interior.len());
        for _ in 0..=K {
            t.push(x[0]);
        }
        t.extend_from_slice(&interior);
        for _ in 0..=K {
            t.push(x[n - 1]);
        }
        t
    };

    let nt = t.len() - K - 1;
    let (nleft, nright) = if natural {
        (1usize, 1usize)
    } else {
        (0usize, 0usize)
    };
    if nt != n + nleft + nright {
        bail!("make_interp_spline: knot/condition count mismatch");
    }

    // Build the collocation matrix with boundary derivative rows.
    let mut a = vec![vec![0.0; nt]; nt];
    let mut rhs = vec![0.0; nt];

    // Left boundary derivative rows (natural: 2nd derivative == 0 at x[0]).
    if nleft > 0 {
        let span = bspline_find_span(&t, K, nt, x[0]);
        let ders = bspline_ders_basis::<K>(&t, span, x[0], 2);
        for j in 0..=K {
            a[0][span - K + j] = ders[2][j];
        }
        rhs[0] = 0.0;
    }

    // Collocation rows: spline value at each data point equals y.
    for i in 0..n {
        let span = bspline_find_span(&t, K, nt, x[i]);
        let ders = bspline_ders_basis::<K>(&t, span, x[i], 0);
        let row = nleft + i;
        for j in 0..=K {
            a[row][span - K + j] = ders[0][j];
        }
        rhs[row] = y[i];
    }

    // Right boundary derivative rows.
    if nright > 0 {
        let span = bspline_find_span(&t, K, nt, x[n - 1]);
        let ders = bspline_ders_basis::<K>(&t, span, x[n - 1], 2);
        let row = nt - nright;
        for j in 0..=K {
            a[row][span - K + j] = ders[2][j];
        }
        rhs[row] = 0.0;
    }

    let c = solve_dense(a, rhs)?;
    Ok((t, c))
}

/// Evaluate the k+1 nonzero B-spline basis functions and their derivatives up to
/// order `d` at `x` in knot span `span`. Returns ders[order][j], j = 0..=k indexing
/// the basis functions B_{span-k+j}. Algorithm A2.3 from "The NURBS Book".
///
/// Degree `k` is bounded by 3 (cubic) in this codebase, so all scratch storage uses
/// fixed-size stack arrays of width 4 to avoid per-call heap allocation in the hot path.
fn bspline_ders_basis<const K: usize>(t: &[f64], span: usize, x: f64, d: usize) -> [[f64; 4]; 3] {
    let p = K;
    debug_assert!(p <= 3 && d <= 2);
    let mut ndu = [[0.0f64; 4]; 4];
    let mut left = [0.0f64; 4];
    let mut right = [0.0f64; 4];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = x - t[span + 1 - j];
        right[j] = t[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    let mut ders = [[0.0f64; 4]; 3];
    for j in 0..=p {
        ders[0][j] = ndu[j][p];
    }

    let mut a = [[0.0f64; 4]; 2];
    for r in 0..=p {
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        a[0][0] = 1.0;
        for kk in 1..=d {
            let mut acc = 0.0;
            let rk = r as isize - kk as isize;
            let pk = p as isize - kk as isize;
            if r >= kk {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                acc = a[s2][0] * ndu[rk as usize][pk as usize];
            }
            let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
            let j2 = if (r as isize - 1) <= pk {
                kk - 1
            } else {
                p - r
            };
            for j in j1..=j2 {
                a[s2][j] =
                    (a[s1][j] - a[s1][j - 1]) / ndu[(pk + 1) as usize][(rk + j as isize) as usize];
                acc += a[s2][j] * ndu[(rk + j as isize) as usize][pk as usize];
            }
            if (r as isize) <= pk {
                a[s2][kk] = -a[s1][kk - 1] / ndu[(pk + 1) as usize][r];
                acc += a[s2][kk] * ndu[r][pk as usize];
            }
            ders[kk][r] = acc;
            std::mem::swap(&mut s1, &mut s2);
        }
    }

    // Multiply through by the correct factors (Eq. [2.9] in "The NURBS Book").
    let mut r = p;
    for kk in 1..=d {
        for j in 0..=p {
            ders[kk][j] *= r as f64;
        }
        r *= p - kk;
    }
    ders
}

/// Find the knot span index `i` such that t[i] <= x < t[i+1], clamped to the
/// valid evaluation range [k, nt-1] so out-of-bounds x uses the edge polynomial.
fn bspline_find_span(t: &[f64], k: usize, nt: usize, x: f64) -> usize {
    if x <= t[k] {
        return k;
    }
    if x >= t[nt] {
        return nt - 1;
    }
    let mut lo = k;
    let mut hi = nt;
    let mut mid = (lo + hi) / 2;
    while x < t[mid] || x >= t[mid + 1] {
        if x < t[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    mid
}

/// Solve a dense linear system A x = b in place using Gaussian elimination with
/// partial pivoting. `a` is row-major n x n; returns the solution vector.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best == 0.0 {
            bail!("Colocation matrix is singular.");
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let inv = 1.0 / a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] * inv;
            if factor != 0.0 {
                for c in col..n {
                    a[row][c] -= factor * a[col][c];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    // Back-substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for c in (i + 1)..n {
            sum -= a[i][c] * x[c];
        }
        x[i] = sum / a[i][i];
    }
    Ok(x)
}
