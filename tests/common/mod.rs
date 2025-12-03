//! Test case loading utilities for integration tests.
//!
//! This module provides helpers for loading test cases from JSON files.
//! For assertion helpers, use `booste_rs::testing`.

#![allow(dead_code)]

use std::fs::File;
use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::Deserialize;

// Re-export testing utilities for convenience
#[allow(unused_imports)]
pub use booste_rs::testing::{
    assert_slices_approx_eq, assert_slices_approx_eq_f64, format_slice_diff,
    format_slice_diff_f64, DEFAULT_TOLERANCE, DEFAULT_TOLERANCE_F64,
};
#[allow(unused_imports)]
pub use booste_rs::{assert_approx_eq, assert_approx_eq_f64};

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
    let file =
        File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
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
