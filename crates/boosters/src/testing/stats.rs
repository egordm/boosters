/// Pearson correlation coefficient between two slices.
///
/// Returns a value between -1 and 1:
/// - 1 indicates perfect positive correlation
/// - 0 indicates no linear correlation
/// - -1 indicates perfect negative correlation
///
/// Returns 0 if either slice has zero variance.
///
/// # Panics
///
/// Panics if the slices have different lengths.
pub fn pearson_correlation(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "slices must have equal length");
    let n = a.len() as f64;

    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;

    for i in 0..a.len() {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a == 0.0 || var_b == 0.0 {
        return 0.0;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}
