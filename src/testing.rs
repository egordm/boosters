//! Testing utilities for booste-rs.
//!
//! This module provides common assertion helpers and testing utilities
//! that can be used in both unit tests and integration tests.
//!
//! # Usage
//!
//! The module is only available when testing:
//!
//! ```ignore
//! #[cfg(test)]
//! use booste_rs::testing::{assert_approx_eq, DEFAULT_TOLERANCE};
//! ```
//!
//! For integration tests:
//!
//! ```ignore
//! use booste_rs::testing::{assert_approx_eq, assert_predictions_eq};
//! ```

use crate::predict::PredictionOutput;
use approx::AbsDiffEq;

// =============================================================================
// Constants
// =============================================================================

/// Default tolerance for floating point comparisons.
/// This is appropriate for most predictions where values are O(1).
pub const DEFAULT_TOLERANCE: f32 = 1e-5;

/// Same tolerance as f64 for compatibility with test expected values.
pub const DEFAULT_TOLERANCE_F64: f64 = 1e-5;

// =============================================================================
// Floating Point Assertions
// =============================================================================

/// Assert that two f32 values are approximately equal.
///
/// Uses absolute difference comparison with the given tolerance.
///
/// # Examples
///
/// ```
/// # use booste_rs::assert_approx_eq;
/// assert_approx_eq!(1.0f32, 1.0001f32, 0.001);
/// ```
///
/// # Panics
///
/// Panics if the absolute difference exceeds tolerance.
#[macro_export]
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr, $tolerance:expr) => {{
        let left_val = $left;
        let right_val = $right;
        let tol = $tolerance;
        let diff = (left_val - right_val).abs();
        if diff > tol {
            panic!(
                "assertion failed: `(left ≈ right)`\n  left: `{:?}`\n right: `{:?}`\n  diff: `{:?}` > tolerance `{:?}`",
                left_val, right_val, diff, tol
            );
        }
    }};
    ($left:expr, $right:expr, $tolerance:expr, $($arg:tt)+) => {{
        let left_val = $left;
        let right_val = $right;
        let tol = $tolerance;
        let diff = (left_val - right_val).abs();
        if diff > tol {
            panic!(
                "assertion failed: `(left ≈ right)` - {}\n  left: `{:?}`\n right: `{:?}`\n  diff: `{:?}` > tolerance `{:?}`",
                format_args!($($arg)+), left_val, right_val, diff, tol
            );
        }
    }};
}

/// Assert that two f64 values are approximately equal.
///
/// Uses absolute difference comparison with the given tolerance.
///
/// # Examples
///
/// ```
/// # use booste_rs::assert_approx_eq_f64;
/// assert_approx_eq_f64!(1.0f64, 1.0001f64, 0.001);
/// ```
#[macro_export]
macro_rules! assert_approx_eq_f64 {
    ($left:expr, $right:expr, $tolerance:expr) => {{
        let left_val: f64 = $left;
        let right_val: f64 = $right;
        let tol: f64 = $tolerance;
        let diff = (left_val - right_val).abs();
        if diff > tol {
            panic!(
                "assertion failed: `(left ≈ right)`\n  left: `{:?}`\n right: `{:?}`\n  diff: `{:?}` > tolerance `{:?}`",
                left_val, right_val, diff, tol
            );
        }
    }};
    ($left:expr, $right:expr, $tolerance:expr, $($arg:tt)+) => {{
        let left_val: f64 = $left;
        let right_val: f64 = $right;
        let tol: f64 = $tolerance;
        let diff = (left_val - right_val).abs();
        if diff > tol {
            panic!(
                "assertion failed: `(left ≈ right)` - {}\n  left: `{:?}`\n right: `{:?}`\n  diff: `{:?}` > tolerance `{:?}`",
                format_args!($($arg)+), left_val, right_val, diff, tol
            );
        }
    }};
}

/// Assert that two slices of f32 values are approximately equal element-wise.
///
/// # Panics
///
/// Panics if lengths differ or any element differs by more than tolerance.
pub fn assert_slice_approx_eq(actual: &[f32], expected: &[f32], tolerance: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch - got {}, expected {}",
        actual.len(),
        expected.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tolerance,
            "{context}[{i}]: {a} ≠ {e} (diff={diff}, tolerance={tolerance})"
        );
    }
}

/// Assert that two slices are approximately equal, with f64 expected values.
///
/// Useful when comparing against test data stored as f64.
pub fn assert_slice_approx_eq_f64(
    actual: &[f32],
    expected: &[f64],
    tolerance: f64,
    context: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch - got {}, expected {}",
        actual.len(),
        expected.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (*a as f64 - *e).abs();
        assert!(
            diff <= tolerance,
            "{context}[{i}]: {a} ≠ {e} (diff={diff}, tolerance={tolerance})"
        );
    }
}

// =============================================================================
// Prediction Assertions
// =============================================================================

/// Generate a git-style diff between expected and actual predictions.
///
/// Shows `-` lines for expected and `+` lines for actual, only for rows that differ.
fn diff_predictions(
    actual: &PredictionOutput,
    expected: &PredictionOutput,
    epsilon: f32,
) -> String {
    let mut result = String::new();
    let (rows, cols) = actual.shape();

    // Header for context
    result.push_str(&format!("Shape: ({rows}, {cols})\n"));
    result.push_str(&format!("Epsilon: {epsilon:.0e}\n\n"));

    if cols == 1 {
        // 1D output - show differing rows
        for (i, (act_row, exp_row)) in actual.rows().zip(expected.rows()).enumerate() {
            if !act_row[0].abs_diff_eq(&exp_row[0], epsilon) {
                let diff = act_row[0] - exp_row[0];
                result.push_str(&format!(
                    "[{i:3}] - {:>12.6}  (expected)\n",
                    exp_row[0]
                ));
                result.push_str(&format!(
                    "      + {:>12.6}  (actual, Δ={diff:+.2e})\n",
                    act_row[0]
                ));
            }
        }
    } else {
        // 2D output - show entire row if any column differs
        for (i, (act_row, exp_row)) in actual.rows().zip(expected.rows()).enumerate() {
            let row_differs = act_row
                .iter()
                .zip(exp_row.iter())
                .any(|(a, e)| !a.abs_diff_eq(e, epsilon));

            if row_differs {
                // Show expected row
                result.push_str(&format!("[{i:3}] -"));
                for val in exp_row {
                    result.push_str(&format!(" {val:>12.6}"));
                }
                result.push_str("  (expected)\n");

                // Show actual row
                result.push_str("      +");
                for val in act_row {
                    result.push_str(&format!(" {val:>12.6}"));
                }
                result.push_str("  (actual)\n");

                // Show deltas for differing columns
                result.push_str("      Δ");
                for (a, e) in act_row.iter().zip(exp_row.iter()) {
                    let delta = a - e;
                    if !a.abs_diff_eq(e, epsilon) {
                        result.push_str(&format!(" {delta:>+12.2e}"));
                    } else {
                        result.push_str(&format!(" {:>12}", "-"));
                    }
                }
                result.push('\n');
            }
        }
    }

    result
}

/// Assert that two [`PredictionOutput`]s are approximately equal.
///
/// Uses the `approx` crate's `AbsDiffEq` trait for comparison.
/// On failure, shows a git-style diff of differing values.
///
/// # Panics
///
/// Panics if shapes differ or if any value differs by more than the default epsilon.
pub fn assert_predictions_eq(
    actual: &PredictionOutput,
    expected: &PredictionOutput,
    context: &str,
) {
    assert_predictions_eq_eps(actual, expected, DEFAULT_TOLERANCE, context);
}

/// Assert that two [`PredictionOutput`]s are approximately equal with custom epsilon.
///
/// # Panics
///
/// Panics if shapes differ or if any value differs by more than epsilon.
pub fn assert_predictions_eq_eps(
    actual: &PredictionOutput,
    expected: &PredictionOutput,
    epsilon: f32,
    context: &str,
) {
    // Check shape first
    if actual.shape() != expected.shape() {
        panic!(
            "\n{context}: shape mismatch\n- {:?}  (expected)\n+ {:?}  (actual)\n",
            expected.shape(),
            actual.shape()
        );
    }

    // Use approx trait for comparison
    if !actual.abs_diff_eq(expected, epsilon) {
        let diff_count = actual
            .as_slice()
            .iter()
            .zip(expected.as_slice().iter())
            .filter(|(a, e)| !a.abs_diff_eq(e, epsilon))
            .count();

        let total = actual.as_slice().len();
        let diff_output = diff_predictions(actual, expected, epsilon);

        panic!("\n{context}: {diff_count}/{total} values differ\n\n{diff_output}");
    }
}

/// Assert that a slice of f32 predictions matches expected f64 values.
///
/// Convenience wrapper that converts to [`PredictionOutput`] and uses git-style diff.
///
/// # Panics
///
/// Panics if lengths differ or if any value differs by more than tolerance.
pub fn assert_predictions_match(actual: &[f32], expected: &[f64], tolerance: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch - got {}, expected {}",
        actual.len(),
        expected.len()
    );

    let actual_output = PredictionOutput::new(actual.to_vec(), actual.len(), 1);
    let expected_output = PredictionOutput::new(
        expected.iter().map(|&x| x as f32).collect(),
        expected.len(),
        1,
    );

    assert_predictions_eq_eps(&actual_output, &expected_output, tolerance, context);
}

/// Assert that a 2D prediction output matches expected values.
///
/// Convenience wrapper that converts to [`PredictionOutput`] and uses git-style diff.
///
/// # Panics
///
/// Panics if shapes differ or if any value differs by more than tolerance.
pub fn assert_batch_predictions_match(
    actual: &[f32],
    expected: &[Vec<f64>],
    num_cols: usize,
    tolerance: f32,
    context: &str,
) {
    let num_rows = expected.len();
    assert_eq!(
        actual.len(),
        num_rows * num_cols,
        "{context}: total size mismatch - got {}, expected {}",
        actual.len(),
        num_rows * num_cols
    );

    let actual_output = PredictionOutput::new(actual.to_vec(), num_rows, num_cols);
    let expected_flat: Vec<f32> = expected
        .iter()
        .flat_map(|row| row.iter().map(|&x| x as f32))
        .collect();
    let expected_output = PredictionOutput::new(expected_flat, num_rows, num_cols);

    assert_predictions_eq_eps(&actual_output, &expected_output, tolerance, context);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_approx_eq_macro() {
        assert_approx_eq!(1.0f32, 1.0001f32, 0.001);
        assert_approx_eq!(0.0f32, 0.0f32, 1e-10);
        assert_approx_eq!(-1.5f32, -1.5001f32, 0.001);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_assert_approx_eq_fails() {
        assert_approx_eq!(1.0f32, 2.0f32, 0.1);
    }

    #[test]
    fn test_assert_approx_eq_with_message() {
        assert_approx_eq!(1.0f32, 1.0001f32, 0.001, "testing value");
    }

    #[test]
    fn test_slice_approx_eq() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0001f32, 2.0001, 3.0001];
        assert_slice_approx_eq(&a, &b, 0.001, "test");
    }

    #[test]
    fn test_slice_approx_eq_f64() {
        let actual = [1.0f32, 2.0, 3.0];
        let expected = [1.0f64, 2.0, 3.0];
        assert_slice_approx_eq_f64(&actual, &expected, 1e-5, "test");
    }

    #[test]
    fn test_predictions_eq() {
        let a = PredictionOutput::new(vec![1.0, 2.0, 3.0], 3, 1);
        let b = PredictionOutput::new(vec![1.000005, 2.000005, 3.000005], 3, 1);
        assert_predictions_eq(&a, &b, "test");
    }

    #[test]
    fn test_predictions_match() {
        let actual = vec![1.0f32, 2.0, 3.0];
        let expected = vec![1.0f64, 2.0, 3.0];
        assert_predictions_match(&actual, &expected, 1e-5, "test");
    }
}
