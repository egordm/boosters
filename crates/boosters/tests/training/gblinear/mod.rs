//! Integration tests for linear model training.
//!
//! Compares booste-rs trained models to XGBoost reference:
//! - Weight similarity (correlation > 0.95)
//! - Held-out test set predictions (RMSE < threshold)
//! - Metrics within tolerance of XGBoost
//!
//! Tests are split into modules by category:
//! - `regression`: Basic regression training tests
//! - `classification`: Binary and multiclass classification tests
//! - `quantile`: Quantile regression tests
//! - `selectors`: Feature selector tests
//! - `loss_functions`: Alternative loss function tests

// Allow needless range loops in test code where index clarity is preferred.
#![allow(clippy::needless_range_loop)]

mod classification;
mod loss_functions;
mod quantile;
mod regression;
mod selectors;

use boosters::data::transpose_to_c_order;
use boosters::training::Rmse;
use ndarray::{Array2, ArrayView1, ArrayView2};
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub const TEST_CASES_DIR: &str = "tests/test-cases/xgboost/gblinear/training";

// =============================================================================
// Shared Types
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct TrainData {
    pub num_rows: usize,
    pub num_features: usize,
    pub data: Vec<Option<f32>>,
}

#[derive(Debug, Deserialize)]
pub struct TrainLabels {
    pub labels: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct XgbWeights {
    pub weights: Vec<f32>,
    pub num_features: usize,
    pub num_groups: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct TrainConfig {
    pub objective: String,
    pub booster: String,
    pub eta: f32,
    pub lambda: f32,
    pub alpha: f32,
    #[serde(default)]
    pub base_score: Option<f32>,
    pub num_boost_round: usize,
    #[serde(default)]
    pub quantile_alpha: Option<f32>,
}

// =============================================================================
// Data Loading Functions
// =============================================================================

/// Load training data, returning owned Array2 in feature-major layout and labels.
pub fn load_train_data(name: &str) -> (Array2<f32>, Vec<f32>) {
    let data_path = Path::new(TEST_CASES_DIR).join(format!("{}.train_data.json", name));
    let labels_path = Path::new(TEST_CASES_DIR).join(format!("{}.train_labels.json", name));

    let data_json = fs::read_to_string(&data_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", data_path.display()));
    let labels_json = fs::read_to_string(&labels_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", labels_path.display()));

    let data: TrainData = serde_json::from_str(&data_json).expect("Failed to parse train data");
    let labels: TrainLabels =
        serde_json::from_str(&labels_json).expect("Failed to parse train labels");

    // Convert Option<f32> to f32 (None -> NaN)
    let features: Vec<f32> = data
        .data
        .into_iter()
        .map(|v| v.unwrap_or(f32::NAN))
        .collect();

    // Data is row-major [n_samples, n_features], convert to feature-major [n_features, n_samples]
    let row_view = ArrayView2::from_shape((data.num_rows, data.num_features), &features).unwrap();
    let col_matrix = transpose_to_c_order(row_view.view());
    (col_matrix, labels.labels)
}

pub fn load_xgb_weights(name: &str) -> XgbWeights {
    let path = Path::new(TEST_CASES_DIR).join(format!("{}.xgb_weights.json", name));
    let json =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read {}", path.display()));
    serde_json::from_str(&json).expect("Failed to parse xgb weights")
}

pub fn load_config(name: &str) -> TrainConfig {
    let path = Path::new(TEST_CASES_DIR).join(format!("{}.config.json", name));
    let json =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read {}", path.display()));
    serde_json::from_str(&json).expect("Failed to parse config")
}

/// Load test data, returning owned Array2 in sample-major layout and labels.
pub fn load_test_data(name: &str) -> Option<(Array2<f32>, Vec<f32>)> {
    let data_path = Path::new(TEST_CASES_DIR).join(format!("{}.test_data.json", name));
    let labels_path = Path::new(TEST_CASES_DIR).join(format!("{}.test_labels.json", name));

    if !data_path.exists() {
        return None;
    }

    let data_json = fs::read_to_string(&data_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", data_path.display()));
    let labels_json = fs::read_to_string(&labels_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", labels_path.display()));

    let data: TrainData = serde_json::from_str(&data_json).expect("Failed to parse test data");
    let labels: TrainLabels =
        serde_json::from_str(&labels_json).expect("Failed to parse test labels");

    let features: Vec<f32> = data
        .data
        .into_iter()
        .map(|v| v.unwrap_or(f32::NAN))
        .collect();

    // Test data returned as sample-major [n_samples, n_features] for prediction
    let samples_matrix = ArrayView2::from_shape((data.num_rows, data.num_features), &features)
        .unwrap()
        .to_owned();
    Some((samples_matrix, labels.labels))
}

/// XGBoost predictions wrapper for JSON deserialization.
#[derive(Debug, Deserialize)]
struct XgbPredictions {
    predictions: Vec<f32>,
}

pub fn load_xgb_predictions(name: &str) -> Option<Vec<f32>> {
    let path = Path::new(TEST_CASES_DIR).join(format!("{}.xgb_predictions.json", name));
    if !path.exists() {
        return None;
    }
    let json = fs::read_to_string(&path).expect("Failed to read XGBoost predictions");
    let data: XgbPredictions =
        serde_json::from_str(&json).expect("Failed to parse XGBoost predictions");
    Some(data.predictions)
}

// =============================================================================
// Statistical Utilities
// =============================================================================

// Re-export from library
pub use boosters::testing::pearson_correlation;

/// Transpose a feature-major Array2 to sample-major Array2, ensuring C-order.
///
/// Input: [n_features, n_samples] (feature-major)
/// Output: [n_samples, n_features] (sample-major, C-order)
#[inline]
pub fn transpose_to_samples(features: &Array2<f32>) -> Array2<f32> {
    transpose_to_c_order(features.view())
}

/// Root mean squared error - uses library Rmse metric.
pub fn rmse(predictions: &[f32], labels: &[f32]) -> f64 {
    use boosters::training::MetricFn;
    let n = labels.len();
    let pred_arr = Array2::from_shape_vec((1, n), predictions.to_vec()).unwrap();
    let targets_arr = ArrayView1::from(labels);
    Rmse.compute(pred_arr.view(), targets_arr, None)
}
