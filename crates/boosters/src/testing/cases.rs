// Note: This module is only compiled with the xgboost-compat feature.
// The feature gate is already applied in testing/mod.rs.

use serde::Deserialize;

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
#[derive(Debug, Deserialize)]
pub struct TestInput {
    /// Features matrix, where None represents NaN (missing value)
    pub features: Vec<Vec<Option<f64>>>,
    #[serde(rename = "num_rows")]
    pub n_rows: usize,
    #[serde(rename = "num_features")]
    pub n_features: usize,
    /// Optional feature types (for categorical features)
    #[serde(default)]
    pub feature_types: Vec<String>,
}

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
    pub n_class: Option<u32>,
}

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
