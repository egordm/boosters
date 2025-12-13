//! Testing utilities for booste-rs.
//!
//! This module provides testing utilities for both unit and integration tests.
//!
//! # Approach
//!
//! For scalar floating-point comparisons, use `approx` crate directly:
//! ```ignore
//! use approx::assert_abs_diff_eq;
//!
//! assert_abs_diff_eq!(1.0f32, 1.0001f32, epsilon = 0.001);
//! ```
//!
//! For `PredictionOutput`, use `approx` directly - it implements `AbsDiffEq`
//! which checks both shape and values:
//! ```ignore
//! use approx::assert_abs_diff_eq;
//! use booste_rs::inference::PredictionOutput;
//!
//! let actual = PredictionOutput::new(vec![1.0, 2.0], 2, 1);
//! let expected = PredictionOutput::new(vec![1.0, 2.0], 2, 1);
//! assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
//! ```
//!
//! For slice comparisons with nice diff output on failure:
//! ```ignore
//! use booste_rs::testing::assert_slices_approx_eq;
//!
//! let actual = &[1.0f32, 2.0, 3.0];
//! let expected = &[1.0f32, 2.0, 3.0];
//! assert_slices_approx_eq!(actual, expected, 1e-5);
//! ```
//!
//! # Test Data Structures
//!
//! For loading test cases from JSON files, use [`TestInput`] and [`TestExpected`]:
//! ```ignore
//! use booste_rs::testing::{TestInput, TestExpected};
//!
//! let input: TestInput = serde_json::from_str(json).unwrap();
//! let features = input.to_f32_rows();
//! ```

use approx::AbsDiffEq;

#[cfg(feature = "xgboost-compat")]
use serde::Deserialize;

// =============================================================================
// Constants
// =============================================================================

/// Default tolerance for floating point comparisons (f32).
/// This is appropriate for most predictions where values are O(1).
pub const DEFAULT_TOLERANCE: f32 = 1e-5;

/// Same tolerance as f64 for compatibility with test expected values.
pub const DEFAULT_TOLERANCE_F64: f64 = 1e-5;

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

// =============================================================================
// Statistical Utilities
// =============================================================================

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

// =============================================================================
// Test Data Structures
// =============================================================================

/// Input features for a test case, loaded from JSON.
///
/// Expects JSON format:
/// ```json
/// {
///   "features": [[1.0, 2.0, null], [3.0, 4.0, 5.0]],
///   "num_rows": 2,
///   "num_features": 3
/// }
/// ```
///
/// Use `None` (JSON `null`) to represent NaN/missing values.
#[cfg(feature = "xgboost-compat")]
#[derive(Debug, Deserialize)]
pub struct TestInput {
    /// Features matrix, where None represents NaN (missing value)
    pub features: Vec<Vec<Option<f64>>>,
    pub num_rows: usize,
    pub num_features: usize,
    /// Optional feature types (for categorical features)
    #[serde(default)]
    pub feature_types: Vec<String>,
}

#[cfg(feature = "xgboost-compat")]
impl TestInput {
    /// Convert input features to f32 row vectors, mapping None to NaN.
    pub fn to_f32_rows(&self) -> Vec<Vec<f32>> {
        self.features
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| x.map(|v| v as f32).unwrap_or(f32::NAN))
                    .collect()
            })
            .collect()
    }

    /// Convert to flat f32 slice for RowMatrix.
    pub fn to_flat_f32(&self) -> Vec<f32> {
        self.features
            .iter()
            .flat_map(|row| {
                row.iter()
                    .map(|&x| x.map(|v| v as f32).unwrap_or(f32::NAN))
            })
            .collect()
    }
}

/// Expected predictions for a test case, loaded from JSON.
///
/// Supports both scalar predictions (regression/binary) and
/// nested predictions (multiclass).
#[cfg(feature = "xgboost-compat")]
#[derive(Debug, Deserialize)]
pub struct TestExpected {
    /// Raw predictions (margin scores). Can be `Vec<f64>` or `Vec<Vec<f64>>`.
    pub predictions: serde_json::Value,
    /// Transformed predictions (after sigmoid/softmax)
    #[serde(default)]
    pub predictions_transformed: Option<serde_json::Value>,
    /// Objective function name
    #[serde(default)]
    pub objective: Option<String>,
    /// Number of classes (for multiclass)
    #[serde(default)]
    pub num_class: Option<u32>,
}

#[cfg(feature = "xgboost-compat")]
impl TestExpected {
    /// Parse predictions as flat `Vec<f64>` (for regression/binary).
    pub fn as_flat(&self) -> Vec<f64> {
        serde_json::from_value(self.predictions.clone())
            .expect("Failed to parse predictions as Vec<f64>")
    }

    /// Parse predictions as `Vec<Vec<f64>>` (for multiclass).
    pub fn as_nested(&self) -> Vec<Vec<f64>> {
        serde_json::from_value(self.predictions.clone())
            .expect("Failed to parse predictions as Vec<Vec<f64>>")
    }

    /// Parse transformed predictions as flat `Vec<f64>`.
    pub fn transformed_as_flat(&self) -> Option<Vec<f64>> {
        self.predictions_transformed
            .as_ref()
            .map(|v| serde_json::from_value(v.clone()).expect("Failed to parse transformed"))
    }

    /// Parse transformed predictions as nested `Vec<Vec<f64>>`.
    pub fn transformed_as_nested(&self) -> Option<Vec<Vec<f64>>> {
        self.predictions_transformed
            .as_ref()
            .map(|v| serde_json::from_value(v.clone()).expect("Failed to parse transformed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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

    #[test]
    fn test_pearson_correlation_perfect() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 4.0, 6.0, 8.0, 10.0]; // Perfect linear: b = 2*a
        let corr = pearson_correlation(&a, &b);
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0f32, 4.0, 3.0, 2.0, 1.0]; // Perfect negative
        let corr = pearson_correlation(&a, &b);
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_correlation_zero_variance() {
        let a = [1.0f32, 1.0, 1.0];
        let b = [2.0f32, 3.0, 4.0];
        let corr = pearson_correlation(&a, &b);
        assert_eq!(corr, 0.0, "Zero variance should return 0");
    }
}
