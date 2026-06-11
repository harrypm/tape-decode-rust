//! Per-field similarity metric shared by the multithreaded stitching logic and
//! the `compare` subcommand.

const U16_MODULUS: i32 = 65536;

/// Mean of `squared` after discarding its largest `trim_fraction` of values.
fn trimmed_mean_of_squares(mut squared: Vec<f64>, trim_fraction: f64) -> f64 {
    let count = squared.len();
    if count == 0 {
        return 0.0;
    }
    let trimmed = (count as f64 * trim_fraction) as usize;
    let keep = count - trimmed;
    if keep >= count {
        return squared.iter().sum::<f64>() / count as f64;
    }
    let keep = keep.max(1);
    // Partition so the smallest `keep` squared deviations occupy `[..keep]`
    squared.select_nth_unstable_by(keep, |a, b| a.total_cmp(b));
    squared[..keep].iter().sum::<f64>() / keep as f64
}

fn trimmed_mean_of_squares_u32(mut squared: Vec<u32>, trim_fraction: f64) -> f64 {
    let count = squared.len();
    if count == 0 {
        return 0.0;
    }
    let trimmed = (count as f64 * trim_fraction) as usize;
    let keep = count - trimmed;
    if keep >= count {
        return squared.iter().map(|&value| u64::from(value)).sum::<u64>() as f64 / count as f64;
    }
    let keep = keep.max(1);
    squared.select_nth_unstable(keep);
    squared[..keep]
        .iter()
        .map(|&value| u64::from(value))
        .sum::<u64>() as f64
        / keep as f64
}

/// Per-field trimmed MSRE over wrapped u16 sample differences. The metric used
/// for encoded (u16) luma and for chroma.
pub fn wrapped_u16_msre(reference: &[u16], candidate: &[u16], trim_fraction: f64) -> f64 {
    let squared = reference
        .iter()
        .zip(candidate)
        .map(|(&r, &c)| {
            let diff = (i32::from(r) - i32::from(c)).abs();
            let wrapped = diff.min(U16_MODULUS - diff) as u32;
            wrapped * wrapped
        })
        .collect();
    trimmed_mean_of_squares_u32(squared, trim_fraction).sqrt()
}

/// Per-field trimmed MSRE over raw luma, compared at the f32 precision the raw
/// TBC export is written with.
pub fn f32_msre(reference: &[f32], candidate: &[f32], trim_fraction: f64) -> f64 {
    let squared = reference
        .iter()
        .zip(candidate)
        .map(|(&r, &c)| {
            let diff = f64::from(r) - f64::from(c);
            diff * diff
        })
        .collect();
    trimmed_mean_of_squares(squared, trim_fraction).sqrt()
}
