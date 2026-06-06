use num_traits::{cast, Float};

/// Maximum direct-form IIR delay order dispatched to stack-backed monomorphized code.
const LFILTER_STACK_MAX_DELAY_ORDER: usize = 12;

#[inline(always)]
fn normalize_coefficients_stack<const TAPS: usize>(coeffs: &[f64], inv_a0: f64) -> [f64; TAPS] {
    let mut normalized = [0.0; TAPS];
    for (dst, &c) in normalized.iter_mut().zip(coeffs) {
        *dst = c * inv_a0;
    }
    normalized
}

#[inline]
fn normalize_coefficients_vec(coeffs: &[f64], inv_a0: f64, taps: usize) -> Vec<f64> {
    let mut normalized = vec![0.0; taps];
    for (dst, &c) in normalized.iter_mut().zip(coeffs) {
        *dst = c * inv_a0;
    }
    normalized
}

/// Stack-backed lfilter implementation for a fixed number of BA coefficients.
///
/// `TAPS` is the normalized coefficient length (`max(b.len(), a.len())`), so the
/// actual IIR delay order is `TAPS - 1`.
#[inline(always)]
fn lfilter_stack<const TAPS: usize, const RETURN_STATE: bool, T, O>(
    b: &[f64],
    a: &[f64],
    input: &[T],
    zi: &[f64],
    inv_a0: f64,
) -> (Vec<O>, Vec<f64>)
where
    T: Float,
    O: Float,
{
    debug_assert!(TAPS > 0);

    if TAPS == 1 {
        let gain = b[0] / a[0];
        // `map`/`collect` over a slice preallocates exactly and writes without
        // a per-element capacity check.
        let output = input
            .iter()
            .map(|&sample| cast(sample.to_f64().unwrap() * gain).unwrap())
            .collect();
        return (output, Vec::new());
    }

    let b_norm = normalize_coefficients_stack::<TAPS>(b, inv_a0);

    let a_norm = normalize_coefficients_stack::<TAPS>(a, inv_a0);
    let state_len = TAPS - 1;
    let mut state = [0.0; TAPS];
    let zi_len = zi.len().min(state_len);
    state[..zi_len].copy_from_slice(&zi[..zi_len]);

    // Write directly into a pre-sized buffer through a zipped iterator. This
    // keeps the recurrence loop free of `Vec::push` capacity checks so the
    // small constant-bound state updates stay in registers.
    let mut output = Vec::with_capacity(input.len());
    for &sample in input.iter() {
        let sample = sample.to_f64().unwrap();
        let y = b_norm[0].mul_add(sample, state[0]);
        let mut index = 1;
        while index < state_len {
            let t = (-a_norm[index]).mul_add(y, state[index]);
            state[index - 1] = b_norm[index].mul_add(sample, t);
            index += 1;
        }
        state[state_len - 1] = b_norm[state_len].mul_add(sample, -(a_norm[state_len] * y));
        output.push(cast(y).unwrap());
    }
    debug_assert_eq!(output.len(), input.len());

    let final_state = if RETURN_STATE {
        state[..state_len].to_vec()
    } else {
        Vec::new()
    };
    (output, final_state)
}

fn lfilter_dynamic<const RETURN_STATE: bool, T, O>(
    b: &[f64],
    a: &[f64],
    input: &[T],
    zi: &[f64],
    taps: usize,
    inv_a0: f64,
) -> (Vec<O>, Vec<f64>)
where
    T: Float,
    O: Float,
{
    debug_assert!(taps > LFILTER_STACK_MAX_DELAY_ORDER + 1);

    let b_norm = normalize_coefficients_vec(b, inv_a0, taps);
    let a_norm = normalize_coefficients_vec(a, inv_a0, taps);
    let state_len = taps - 1;
    let mut state = vec![0.0; state_len];
    let zi_len = zi.len().min(state_len);
    state[..zi_len].copy_from_slice(&zi[..zi_len]);

    let mut output = Vec::with_capacity(input.len());
    for &sample in input.iter() {
        let sample = sample.to_f64().unwrap();
        let y = b_norm[0].mul_add(sample, state[0]);
        for index in 1..state_len {
            let t = (-a_norm[index]).mul_add(y, state[index]);
            state[index - 1] = b_norm[index].mul_add(sample, t);
        }
        state[state_len - 1] = b_norm[state_len].mul_add(sample, -(a_norm[state_len] * y));
        output.push(cast(y).unwrap());
    }
    debug_assert_eq!(output.len(), input.len());

    let final_state = if RETURN_STATE { state } else { Vec::new() };
    (output, final_state)
}

#[inline]
pub(crate) fn lfilter<const RETURN_STATE: bool, T>(
    b: &[f64],
    a: &[f64],
    input: &[T],
    zi: &[f64],
) -> (Vec<f64>, Vec<f64>)
where
    T: Float,
{
    lfilter_with::<RETURN_STATE, _, f64>(b, a, input, zi)
}

pub(crate) fn lfilter_f32<const RETURN_STATE: bool>(
    b: &[f64],
    a: &[f64],
    input: &[f32],
    zi: &[f64],
) -> (Vec<f32>, Vec<f64>) {
    lfilter_with::<RETURN_STATE, f32, f32>(b, a, input, zi)
}

fn lfilter_with<const RETURN_STATE: bool, T, O>(
    b: &[f64],
    a: &[f64],
    input: &[T],
    zi: &[f64],
) -> (Vec<O>, Vec<f64>)
where
    T: Float,
    O: Float,
{
    assert!(!b.is_empty());
    assert!(!a.is_empty());
    assert_ne!(a[0], 0.0);

    let taps = b.len().max(a.len());
    let inv_a0 = 1.0 / a[0];
    match taps {
        1 => lfilter_stack::<1, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        2 => lfilter_stack::<2, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        3 => lfilter_stack::<3, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        4 => lfilter_stack::<4, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        5 => lfilter_stack::<5, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        6 => lfilter_stack::<6, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        7 => lfilter_stack::<7, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        8 => lfilter_stack::<8, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        9 => lfilter_stack::<9, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        10 => lfilter_stack::<10, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        11 => lfilter_stack::<11, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        12 => lfilter_stack::<12, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        13 => lfilter_stack::<13, RETURN_STATE, _, _>(b, a, input, zi, inv_a0),
        _ => lfilter_dynamic::<RETURN_STATE, _, _>(b, a, input, zi, taps, inv_a0),
    }
}
