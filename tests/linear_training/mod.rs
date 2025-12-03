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

use booste_rs::data::{ColMatrix, DataMatrix, RowMatrix};
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub const TEST_CASES_DIR: &str = "tests/test-cases/xgboost/training";

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

pub fn load_train_data(name: &str) -> (ColMatrix<f32>, Vec<f32>) {
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

    let row_matrix = RowMatrix::from_vec(features, data.num_rows, data.num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();
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

pub fn load_test_data(name: &str) -> Option<(ColMatrix<f32>, Vec<f32>)> {
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

    let row_matrix = RowMatrix::from_vec(features, data.num_rows, data.num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();
    Some((col_matrix, labels.labels))
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

/// Pearson correlation coefficient between two slices.
pub fn pearson_correlation(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
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

/// Root mean squared error.
pub fn rmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai as f64 - bi as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    mse.sqrt()
}

// =============================================================================
// Prediction Utilities
// =============================================================================

pub fn get_predictions(
    model: &booste_rs::linear::LinearModel,
    data: &ColMatrix<f32>,
    base_scores: &[f32],
) -> Vec<f32> {
    let num_rows = data.num_rows();
    let num_features = data.num_features();

    (0..num_rows)
        .map(|row_idx| {
            let features: Vec<f32> = (0..num_features)
                .map(|col| data.get(row_idx, col).copied().unwrap_or(0.0))
                .collect();
            model.predict_row(&features, base_scores)[0]
        })
        .collect()
}

pub fn compute_test_rmse(
    model: &booste_rs::linear::LinearModel,
    data: &ColMatrix<f32>,
    labels: &[f32],
    base_scores: &[f32],
) -> f64 {
    let preds = get_predictions(model, data, base_scores);
    rmse(&preds, labels)
}
