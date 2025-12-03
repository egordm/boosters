//! Testing utilities for booste-rs.
//!
//! This module provides common assertion helpers and testing utilities
//! that can be used in both unit tests and integration tests.
//!
//! # Approach
//!
//! For scalar floating-point comparisons, use the macros:
//! ```ignore
//! use booste_rs::{assert_approx_eq, assert_approx_eq_f64};
//!
//! assert_approx_eq!(actual, expected, tolerance);
//! assert_approx_eq_f64!(actual, expected, tolerance);
//! ```
//!
//! For slice comparisons with nice diff output:
//! ```ignore
//! use booste_rs::testing::{assert_slices_approx_eq, DEFAULT_TOLERANCE};
//!
//! let actual = &[1.0f32, 2.0, 3.0];
//! let expected = &[1.0f32, 2.0, 3.0];
//! assert_slices_approx_eq!(actual, expected, DEFAULT_TOLERANCE);
//! ```
//!
//! For `PredictionOutput`, use the `approx` crate directly:
//! ```ignore
//! use booste_rs::predict::PredictionOutput;
//! use approx::assert_abs_diff_eq;
//!
//! let actual = PredictionOutput::new(vec![1.0, 2.0], 2, 1);
//! let expected = PredictionOutput::new(vec![1.0, 2.0], 2, 1);
//! assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
//! ```

use approx::AbsDiffEq;

// =============================================================================
// Constants
// =============================================================================

/// Default tolerance for floating point comparisons (f32).
/// This is appropriate for most predictions where values are O(1).
pub const DEFAULT_TOLERANCE: f32 = 1e-5;

/// Same tolerance as f64 for compatibility with test expected values.
pub const DEFAULT_TOLERANCE_F64: f64 = 1e-5;

// =============================================================================
// Scalar Floating Point Assertions
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

// =============================================================================
// Slice Diff Formatting
// =============================================================================

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
            (Some(&a), Some(&e)) if !a.abs_diff_eq(&e, epsilon) => {
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
            (Some(&a), Some(&e)) if !(a as f64).abs_diff_eq(&e, epsilon) => {
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

// =============================================================================
// Slice Assertion Macros
// =============================================================================

/// Assert that two f32 slices are approximately equal with git-style diff on failure.
///
/// This macro provides better error output than element-wise assertions,
/// showing a diff of all differing elements at once.
///
/// # Example
///
/// ```
/// use booste_rs::testing::{assert_slices_approx_eq, DEFAULT_TOLERANCE};
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

// Re-export the macros at testing module level
pub use crate::assert_slices_approx_eq;
pub use crate::assert_slices_approx_eq_f64;

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
    fn test_slices_approx_eq_macro() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0001f32, 2.0001, 3.0001];
        assert_slices_approx_eq!(&a, &b, 0.001);
    }

    #[test]
    fn test_slices_approx_eq_f64_macro() {
        let actual = [1.0f32, 2.0, 3.0];
        let expected = [1.0f64, 2.0, 3.0];
        assert_slices_approx_eq_f64!(&actual, &expected, 1e-5);
    }

    #[test]
    #[should_panic(expected = "slices not approximately equal")]
    fn test_slices_macro_fails() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 5.0, 3.0]; // different
        assert_slices_approx_eq!(&a, &b, 1e-5);
    }

    #[test]
    fn test_format_slice_diff() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 5.0, 3.0];
        let diff = format_slice_diff(&a, &b, 1e-5);
        assert!(diff.contains("1 values differ"));
        assert!(diff.contains("[   1]"));
    }
}
