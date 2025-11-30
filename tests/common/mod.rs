//! Shared test utilities for integration tests.
//!
//! This module provides common assertion helpers and test case loading utilities
//! that can be used across different test files.

// Allow unused items - these are library functions for test use
#![allow(dead_code)]

use std::fs::File;
use std::path::PathBuf;

use approx::AbsDiffEq;
use serde::de::DeserializeOwned;
use serde::Deserialize;

use booste_rs::predict::PredictionOutput;

// =============================================================================
// Constants
// =============================================================================

/// Default tolerance for floating point comparisons.
/// This is appropriate for most predictions where values are O(1).
pub const DEFAULT_TOLERANCE: f32 = 1e-5;

/// Same tolerance as f64 for compatibility with test expected values.
pub const DEFAULT_TOLERANCE_F64: f64 = 1e-5;

// =============================================================================
// Assertion Helpers
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
    result.push_str(&format!("Shape: ({}, {})\n", rows, cols));
    result.push_str(&format!("Epsilon: {:.0e}\n\n", epsilon));

    if cols == 1 {
        // 1D output - show differing rows
        for (i, (act_row, exp_row)) in actual.rows().zip(expected.rows()).enumerate() {
            if !act_row[0].abs_diff_eq(&exp_row[0], epsilon) {
                let diff = act_row[0] - exp_row[0];
                result.push_str(&format!(
                    "[{:3}] - {:>12.6}  (expected)\n",
                    i, exp_row[0]
                ));
                result.push_str(&format!(
                    "      + {:>12.6}  (actual, Δ={:+.2e})\n",
                    act_row[0], diff
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
                result.push_str(&format!("[{:3}] -", i));
                for val in exp_row {
                    result.push_str(&format!(" {:>12.6}", val));
                }
                result.push_str("  (expected)\n");

                // Show actual row
                result.push_str(&format!("      +"));
                for val in act_row {
                    result.push_str(&format!(" {:>12.6}", val));
                }
                result.push_str("  (actual)\n");

                // Show deltas for differing columns
                result.push_str(&format!("      Δ"));
                for (a, e) in act_row.iter().zip(exp_row.iter()) {
                    let delta = a - e;
                    if !a.abs_diff_eq(e, epsilon) {
                        result.push_str(&format!(" {:>+12.2e}", delta));
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

/// Assert that two PredictionOutputs are approximately equal.
///
/// Uses the `approx` crate's `AbsDiffEq` trait for comparison.
/// On failure, shows a git-style diff of differing values.
///
/// # Panics
/// Panics if shapes differ or if any value differs by more than epsilon.
pub fn assert_predictions_eq(actual: &PredictionOutput, expected: &PredictionOutput, context: &str) {
    assert_predictions_eq_eps(actual, expected, DEFAULT_TOLERANCE, context);
}

/// Assert that two PredictionOutputs are approximately equal with custom epsilon.
///
/// # Panics
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

        panic!(
            "\n{context}: {diff_count}/{total} values differ\n\n{diff_output}"
        );
    }
}

/// Assert that a slice of f32 predictions matches expected f64 values.
///
/// Convenience wrapper that converts to PredictionOutput and uses git-style diff.
///
/// # Panics
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
/// Convenience wrapper that converts to PredictionOutput and uses git-style diff.
///
/// # Panics
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

// =============================================================================
// Test Case Loading
// =============================================================================

/// Base directory for test cases.
pub fn test_cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases")
}

/// Directory for XGBoost test cases.
pub fn xgboost_test_cases_dir() -> PathBuf {
    test_cases_dir().join("xgboost")
}

/// Load a JSON file and deserialize it.
pub fn load_json<T: DeserializeOwned>(path: &std::path::Path) -> T {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    serde_json::from_reader(file)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()))
}

// =============================================================================
// Common Test Data Structures
// =============================================================================

/// Input features for a test case.
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

impl TestInput {
    /// Convert input features to f32, mapping None to NaN.
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

    /// Convert to flat f32 slice for DenseMatrix.
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

/// Expected predictions for a test case.
#[derive(Debug, Deserialize)]
pub struct TestExpected {
    /// Raw predictions (margin scores)
    pub predictions: serde_json::Value, // Can be Vec<f64> or Vec<Vec<f64>>
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

impl TestExpected {
    /// Parse predictions as flat Vec<f64> (for regression/binary).
    pub fn as_flat(&self) -> Vec<f64> {
        serde_json::from_value(self.predictions.clone())
            .expect("Failed to parse predictions as Vec<f64>")
    }

    /// Parse predictions as Vec<Vec<f64>> (for multiclass).
    pub fn as_nested(&self) -> Vec<Vec<f64>> {
        serde_json::from_value(self.predictions.clone())
            .expect("Failed to parse predictions as Vec<Vec<f64>>")
    }

    /// Parse transformed predictions as flat Vec<f64>.
    pub fn transformed_as_flat(&self) -> Option<Vec<f64>> {
        self.predictions_transformed
            .as_ref()
            .map(|v| serde_json::from_value(v.clone()).expect("Failed to parse transformed"))
    }

    /// Parse transformed predictions as nested Vec<Vec<f64>>.
    pub fn transformed_as_nested(&self) -> Option<Vec<Vec<f64>>> {
        self.predictions_transformed
            .as_ref()
            .map(|v| serde_json::from_value(v.clone()).expect("Failed to parse transformed"))
    }
}

/// A complete test case with model path, input, and expected output.
pub struct TestCase<M> {
    pub name: String,
    pub model: M,
    pub input: TestInput,
    pub expected: TestExpected,
}
