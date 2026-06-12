//! Vectorizable math kernels: explicit-SIMD and branch-free scalar routines
//! for the hot per-sample loops.

/// Sum evaluated 16 lanes wide instead of as one serial carried add. The
/// vector accumulator is explicit `Simd` (or strict per-lane adds on stable):
/// expressing the lanes through reassociation-licensed adds let the compiler
/// legally fold them back into a serial chain, which it did.
#[inline]
pub(crate) fn sum_algebraic(values: &[f32]) -> f32 {
    const LANES: usize = 16;
    let mut chunks = values.chunks_exact(LANES);
    let lanes_total = {
        #[cfg(nightly_portable_simd)]
        {
            use std::simd::num::SimdFloat;
            use std::simd::Simd;
            let mut acc = Simd::<f32, LANES>::splat(0.0);
            for chunk in &mut chunks {
                acc += Simd::from_slice(chunk);
            }
            acc.reduce_sum()
        }
        #[cfg(not(nightly_portable_simd))]
        {
            let mut acc = [0.0f32; LANES];
            for chunk in &mut chunks {
                for (lane, &value) in acc.iter_mut().zip(chunk) {
                    *lane += value;
                }
            }
            acc.iter().sum()
        }
    };
    let tail: f32 = chunks.remainder().iter().sum();
    lanes_total + tail
}

/// exp2(y) for `y` within the normal-exponent range: split into an integer
/// scale assembled in the exponent bits and a short series over `|f| <= 1/2`.
#[inline(always)]
fn exp2_fast(y: f32) -> f32 {
    let n = y.round();
    let f = y - n;
    const LN_2: f32 = core::f32::consts::LN_2;
    let z = f * LN_2;
    let p = z.mul_add(1.0 / 5040.0, 1.0 / 720.0);
    let p = z.mul_add(p, 1.0 / 120.0);
    let p = z.mul_add(p, 1.0 / 24.0);
    let p = z.mul_add(p, 1.0 / 6.0);
    let p = z.mul_add(p, 0.5);
    let p = z.mul_add(p, 1.0);
    let p = z.mul_add(p, 1.0);
    let scale = f32::from_bits((((n as i32) + 127) << 23) as u32);
    p * scale
}

/// `x.powf(c)` for finite `x >= 0`, evaluated as `exp2(c * log2(x))` with
/// branch-free range reduction so callers' loops vectorize instead of calling
/// scalar libm per sample. Relative error stays within a few ulp over the
/// normal range; subnormal `x` (including zero) comes out as a tiny positive
/// value rather than exactly `x.powf(c)`.
#[inline(always)]
pub(crate) fn powf_fast_nonneg(x: f32, c: f32) -> f32 {
    // log2(x): reduce to m in [sqrt(2)/2, sqrt(2)) with integer exponent e.
    let bits = x.to_bits();
    const SQRT2_BITS: u32 = 0x3FB5_04F3;
    let mbits = (bits & 0x007F_FFFF) | 0x3F80_0000;
    let adjust = (mbits >= SQRT2_BITS) as u32;
    let e = ((bits >> 23) as i32) - 127 + adjust as i32;
    let m = f32::from_bits(mbits - (adjust << 23));
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    // ln(m) = 2*atanh(s); on the reduced range the series truncates below f32
    // rounding after the s^9 term.
    let t = s2.mul_add(1.0 / 9.0, 1.0 / 7.0);
    let t = s2.mul_add(t, 1.0 / 5.0);
    let t = s2.mul_add(t, 1.0 / 3.0);
    let ln_m = 2.0 * s * s2.mul_add(t, 1.0);
    const LOG2_E: f32 = core::f32::consts::LOG2_E;
    // Fold the exponent into y after the multiply by `c` so the rounding
    // happens at magnitude |c*e| rather than |e|.
    let y = c.mul_add(e as f32, c * (ln_m * LOG2_E));
    exp2_fast(y)
}

/// `z.exp()` for finite `z`, branch-free so callers' loops vectorize. The
/// exponent is clamped to the normal f32 range, so saturating inputs come out
/// as ~2^127 or ~2^-126 rather than infinity or zero; relative error stays
/// within a few ulp in between.
#[inline(always)]
pub(crate) fn exp_fast(z: f32) -> f32 {
    const LOG2_E: f32 = core::f32::consts::LOG2_E;
    exp2_fast((z * LOG2_E).clamp(-126.0, 127.0))
}
