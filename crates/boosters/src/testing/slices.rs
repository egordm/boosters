/// Format a git-style diff between two f32 slices.
#[doc(hidden)]
pub fn format_slice_diff(actual: &[f32], expected: &[f32], epsilon: f32) -> String {
    let mut result = String::new();
    result.push_str(&format!(
        "Length: actual={}, expected={}\n\n",
        actual.len(),
        expected.len()
    ));

    let max_len = actual.len().max(expected.len());
    let mut diff_count = 0;

    for i in 0..max_len {
        let act = actual.get(i);
        let exp = expected.get(i);

        match (act, exp) {
            (Some(&a), Some(&e)) if (a - e).abs() > epsilon => {
                result.push_str(&format!("[{i:4}] - {e:>14.6}  (expected)\n"));
                result.push_str(&format!("       + {a:>14.6}  (actual)\n"));
                diff_count += 1;
            }
            (Some(&a), None) => {
                result.push_str(&format!("[{i:4}] + {a:>14.6}  (extra in actual)\n"));
                diff_count += 1;
            }
            (None, Some(&e)) => {
                result.push_str(&format!("[{i:4}] - {e:>14.6}  (missing in actual)\n"));
                diff_count += 1;
            }
            _ => {} // Equal or both missing
        }
    }

    if diff_count > 0 {
        result.insert_str(0, &format!("{diff_count} values differ:\n\n"));
    }
    result
}
/// Format a diff between f32 actual and f64 expected slices.
#[doc(hidden)]
pub fn format_slice_diff_f64(actual: &[f32], expected: &[f64], epsilon: f64) -> String {
    let mut result = String::new();
    result.push_str(&format!(
        "Length: actual={}, expected={}\n\n",
        actual.len(),
        expected.len()
    ));

    let max_len = actual.len().max(expected.len());
    let mut diff_count = 0;

    for i in 0..max_len {
        let act = actual.get(i);
        let exp = expected.get(i);

        match (act, exp) {
            (Some(&a), Some(&e)) if ((a as f64) - e).abs() > epsilon => {
                result.push_str(&format!("[{i:4}] - {e:>14.6}  (expected)\n"));
                result.push_str(&format!("       + {a:>14.6}  (actual)\n"));
                diff_count += 1;
            }
            (Some(&a), None) => {
                result.push_str(&format!("[{i:4}] + {a:>14.6}  (extra in actual)\n"));
                diff_count += 1;
            }
            (None, Some(&e)) => {
                result.push_str(&format!("[{i:4}] - {e:>14.6}  (missing in actual)\n"));
                diff_count += 1;
            }
            _ => {}
        }
    }

    if diff_count > 0 {
        result.insert_str(0, &format!("{diff_count} values differ:\n\n"));
    }
    result
}

/// Assert that two f32 slices are approximately equal with git-style diff on failure.
///
/// This macro provides better error output than element-wise assertions,
/// showing a diff of all differing elements at once.
///
/// # Example
///
/// ```
/// use boosters::testing::{assert_slices_approx_eq, DEFAULT_TOLERANCE};
///
/// let actual = &[1.0f32, 2.0, 3.0];
/// let expected = &[1.0f32, 2.0, 3.0];
/// assert_slices_approx_eq!(actual, expected, DEFAULT_TOLERANCE);
/// ```
#[macro_export]
macro_rules! assert_slices_approx_eq {
    ($actual:expr, $expected:expr, $epsilon:expr) => {{
        let actual_slice: &[f32] = $actual;
        let expected_slice: &[f32] = $expected;
        let eps: f32 = $epsilon;

        let differs = actual_slice.len() != expected_slice.len()
            || actual_slice
                .iter()
                .zip(expected_slice.iter())
                .any(|(a, e)| (a - e).abs() > eps);

        if differs {
            let diff = $crate::testing::format_slice_diff(actual_slice, expected_slice, eps);
            panic!(
                "assertion failed: slices not approximately equal (epsilon = {:.0e})\n\n{}",
                eps, diff
            );
        }
    }};
    ($actual:expr, $expected:expr, $epsilon:expr, $($arg:tt)+) => {{
        let actual_slice: &[f32] = $actual;
        let expected_slice: &[f32] = $expected;
        let eps: f32 = $epsilon;

        let differs = actual_slice.len() != expected_slice.len()
            || actual_slice
                .iter()
                .zip(expected_slice.iter())
                .any(|(a, e)| (a - e).abs() > eps);

        if differs {
            let diff = $crate::testing::format_slice_diff(actual_slice, expected_slice, eps);
            panic!(
                "assertion failed: slices not approximately equal - {}\n(epsilon = {:.0e})\n\n{}",
                format_args!($($arg)+), eps, diff
            );
        }
    }};
}
/// Assert f32 actual slice approximately equals f64 expected slice.
///
/// Useful when expected values come from test data stored as f64.
#[macro_export]
macro_rules! assert_slices_approx_eq_f64 {
    ($actual:expr, $expected:expr, $epsilon:expr) => {{
        let actual_slice: &[f32] = $actual;
        let expected_slice: &[f64] = $expected;
        let eps: f64 = $epsilon;

        let differs = actual_slice.len() != expected_slice.len()
            || actual_slice
                .iter()
                .zip(expected_slice.iter())
                .any(|(a, e)| ((*a as f64) - *e).abs() > eps);

        if differs {
            let diff = $crate::testing::format_slice_diff_f64(actual_slice, expected_slice, eps);
            panic!(
                "assertion failed: slices not approximately equal (epsilon = {:.0e})\n\n{}",
                eps, diff
            );
        }
    }};
    ($actual:expr, $expected:expr, $epsilon:expr, $($arg:tt)+) => {{
        let actual_slice: &[f32] = $actual;
        let expected_slice: &[f64] = $expected;
        let eps: f64 = $epsilon;

        let differs = actual_slice.len() != expected_slice.len()
            || actual_slice
                .iter()
                .zip(expected_slice.iter())
                .any(|(a, e)| ((*a as f64) - *e).abs() > eps);

        if differs {
            let diff = $crate::testing::format_slice_diff_f64(actual_slice, expected_slice, eps);
            panic!(
                "assertion failed: slices not approximately equal - {}\n(epsilon = {:.0e})\n\n{}",
                format_args!($($arg)+), eps, diff
            );
        }
    }};
}
