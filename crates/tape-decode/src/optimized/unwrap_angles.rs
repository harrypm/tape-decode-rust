use rustfft::num_complex::Complex32;

// https://mazzo.li/posts/vectorized-atan2.html
#[inline(always)]
fn atan_approx(x: f32) -> f32 {
    const A1: f32 = 0.99997726f32;
    const A3: f32 = -0.33262347f32;
    const A5: f32 = 0.19354346f32;
    const A7: f32 = -0.11643287f32;
    const A9: f32 = 0.05265332f32;
    const A11: f32 = -0.011_721_2_f32;

    let x_sq = x * x;
    x * (A1 + x_sq * (A3 + x_sq * (A5 + x_sq * (A7 + x_sq * (A9 + x_sq * A11)))))
}

#[inline(always)]
fn atan2_fast(y: f32, x: f32) -> f32 {
    use std::f32::consts::FRAC_PI_2;
    use std::f32::consts::PI;

    let x = x + f32::MIN_POSITIVE.copysign(x);
    let swap = x.abs() < y.abs();
    let atan_input = if swap { x } else { y } / if swap { y } else { x };
    let mut res = atan_approx(atan_input);
    let tmp = if atan_input >= 0.0f32 {
        FRAC_PI_2
    } else {
        -FRAC_PI_2
    };
    res = if swap { tmp - res } else { res };
    match (x >= 0f32, y >= 0f32) {
        (true, _) => res,
        (false, true) => PI + res,
        (false, false) => -PI + res,
    }
}

#[inline(always)]
fn unwrap_two(a: Complex32, b: Complex32, freq: f32, offset: f32) -> f32 {
    use std::f32::consts::TAU;

    let a = atan2_fast(a.im, a.re);
    let b = atan2_fast(b.im, b.re);
    let diff = b - a;
    let diff = diff - (diff / TAU).floor() * TAU;
    // `offset` recenters the instantaneous-frequency output (e.g. by the
    // carrier's 0-IRE frequency) so the demod sits near zero instead of on a
    // multi-MHz DC pedestal; it folds into the existing per-sample math with no
    // extra pass.
    (diff * freq / TAU) - offset
}

#[inline(always)]
fn unwrap_more(a: &[Complex32; 8], b: &[Complex32; 8], out: &mut [f32; 8], freq: f32, offset: f32) {
    for i in 0..8 {
        out[i] = unwrap_two(a[i], b[i], freq, offset);
    }
}

#[inline(never)]
pub(crate) fn unwrap_angles(
    input_slice: &[Complex32],
    output_slice: &mut [f32],
    freq: f32,
    offset: f32,
) {
    let len = input_slice.len();
    assert_ne!(len, 0);

    let big_chunks = (len - 1) / 8;
    for i in 0..big_chunks {
        let prevs_slice = &input_slice[i * 8..(i + 1) * 8];
        let currs_slice = &input_slice[i * 8 + 1..(i + 1) * 8 + 1];
        let outs_slice = &mut output_slice[i * 8 + 1..(i + 1) * 8 + 1];
        let prevs = <&[Complex32; 8]>::try_from(prevs_slice).unwrap();
        let currs = <&[Complex32; 8]>::try_from(currs_slice).unwrap();
        let outs = <&mut [f32; 8]>::try_from(outs_slice).unwrap();
        unwrap_more(prevs, currs, outs, freq, offset);
    }
    for i in big_chunks * 8..len - 1 {
        output_slice[i + 1] = unwrap_two(input_slice[i], input_slice[i + 1], freq, offset);
    }
}
