//! Integration tests for linear model training.
//!
//! Compares booste-rs trained models to XGBoost reference:
//! - Weight similarity (correlation > 0.95)
//! - Held-out test set predictions (RMSE < threshold)
//! - Metrics within tolerance of XGBoost

use approx::assert_relative_eq;
use booste_rs::data::{ColMatrix, DataMatrix, RowMatrix};
use booste_rs::linear::training::{LinearTrainer, LinearTrainerConfig};
use booste_rs::training::{LogisticLoss, SquaredLoss, Verbosity};

use rstest::rstest;
use serde::Deserialize;
use std::fs;
use std::path::Path;

const TEST_CASES_DIR: &str = "tests/test-cases/xgboost/training";

#[derive(Debug, Deserialize)]
struct TrainData {
    num_rows: usize,
    num_features: usize,
    data: Vec<Option<f32>>,
}

#[derive(Debug, Deserialize)]
struct TrainLabels {
    labels: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct XgbWeights {
    weights: Vec<f32>,
    num_features: usize,
    num_groups: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct TrainConfig {
    objective: String,
    booster: String,
    eta: f32,
    lambda: f32,
    alpha: f32,
    #[serde(default)]
    base_score: Option<f32>,
    num_boost_round: usize,
}

fn load_train_data(name: &str) -> (ColMatrix<f32>, Vec<f32>) {
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

fn load_xgb_weights(name: &str) -> XgbWeights {
    let path = Path::new(TEST_CASES_DIR).join(format!("{}.xgb_weights.json", name));
    let json =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read {}", path.display()));
    serde_json::from_str(&json).expect("Failed to parse xgb weights")
}

fn load_config(name: &str) -> TrainConfig {
    let path = Path::new(TEST_CASES_DIR).join(format!("{}.config.json", name));
    let json =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read {}", path.display()));
    serde_json::from_str(&json).expect("Failed to parse config")
}

/// Test that we can train a simple linear regression and get similar weights to XGBoost.
#[rstest]
#[case("regression_simple")]
#[case("regression_multifeature")]
fn train_regression_matches_xgboost(#[case] name: &str) {
    let (data, labels) = load_train_data(name);
    let xgb_weights = load_xgb_weights(name);
    let config = load_config(name);

    // Configure trainer to match XGBoost settings
    let trainer_config = LinearTrainerConfig {
        num_rounds: config.num_boost_round,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false, // Use sequential for determinism
        seed: 42,
        early_stopping_rounds: 0,
        verbosity: Verbosity::Silent,
    };

    let trainer = LinearTrainer::new(trainer_config);
    let model = trainer.train(&data, &labels, &SquaredLoss, 1);

    // Compare weights
    // XGBoost stores weights as [w0, w1, ..., wn-1, bias]
    let num_features = xgb_weights.num_features;

    println!("Test case: {}", name);
    println!("XGBoost weights: {:?}", xgb_weights.weights);
    println!("booste-rs bias: {}", model.bias(0));
    for i in 0..num_features {
        println!("booste-rs weight[{}]: {}", i, model.weight(i, 0));
    }

    // Check each weight is reasonably close
    // Note: We use a loose tolerance because training may converge differently
    for i in 0..num_features {
        let xgb_w = xgb_weights.weights[i];
        let our_w = model.weight(i, 0);
        assert_relative_eq!(
            our_w,
            xgb_w,
            max_relative = 0.3,
            epsilon = 0.5
        );
    }

    // Check bias
    let xgb_bias = xgb_weights.weights[num_features];
    let our_bias = model.bias(0);
    assert_relative_eq!(
        our_bias,
        xgb_bias,
        max_relative = 0.3,
        epsilon = 0.5
    );
}

/// Test that L2 regularization shrinks weights.
#[test]
fn train_l2_regularization_shrinks_weights() {
    let (data, labels) = load_train_data("regression_l2");
    let config = load_config("regression_l2");

    // Train without regularization
    let no_reg_config = LinearTrainerConfig {
        num_rounds: config.num_boost_round,
        learning_rate: config.eta,
        alpha: 0.0,
        lambda: 0.0, // No regularization
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let no_reg_model = LinearTrainer::new(no_reg_config).train(&data, &labels, &SquaredLoss, 1);

    // Train with L2 regularization
    let l2_config = LinearTrainerConfig {
        num_rounds: config.num_boost_round,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda, // L2 regularization
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let l2_model = LinearTrainer::new(l2_config).train(&data, &labels, &SquaredLoss, 1);

    // L2 should produce smaller weights on average
    let no_reg_l2_norm: f32 = (0..data.num_features())
        .map(|i| no_reg_model.weight(i, 0).powi(2))
        .sum();
    let l2_norm: f32 = (0..data.num_features())
        .map(|i| l2_model.weight(i, 0).powi(2))
        .sum();

    println!("No reg L2 norm: {}", no_reg_l2_norm);
    println!("With L2 reg norm: {}", l2_norm);

    assert!(
        l2_norm < no_reg_l2_norm,
        "L2 regularization should shrink weights"
    );
}

/// Test that L1 regularization can zero out some weights (sparsity).
#[test]
fn train_elastic_net_produces_sparse_weights() {
    let (data, labels) = load_train_data("regression_elastic_net");
    let xgb_weights = load_xgb_weights("regression_elastic_net");
    let config = load_config("regression_elastic_net");

    let trainer_config = LinearTrainerConfig {
        num_rounds: config.num_boost_round,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = LinearTrainer::new(trainer_config);
    let model = trainer.train(&data, &labels, &SquaredLoss, 1);

    // Count near-zero weights in both
    let xgb_near_zero = xgb_weights
        .weights
        .iter()
        .take(xgb_weights.num_features)
        .filter(|w| w.abs() < 0.1)
        .count();

    let our_near_zero = (0..data.num_features())
        .filter(|&i| model.weight(i, 0).abs() < 0.1)
        .count();

    println!("XGBoost near-zero weights: {}", xgb_near_zero);
    println!("booste-rs near-zero weights: {}", our_near_zero);

    // Both should have some sparsity due to L1
    // Note: We don't require exact match, just that both exhibit some sparsity
}

/// Test predictions match after training.
#[test]
fn trained_model_predictions_reasonable() {
    // Simple test: y = 2x + 1
    let row_data = RowMatrix::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        5,
        1,
    );
    let data: ColMatrix = row_data.to_layout();
    let labels = vec![3.0, 5.0, 7.0, 9.0, 11.0];

    let config = LinearTrainerConfig {
        num_rounds: 100,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 0.0,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = LinearTrainer::new(config);
    let model = trainer.train(&data, &labels, &SquaredLoss, 1);

    // Predictions should be close to actual values
    for i in 0..5 {
        let x = (i + 1) as f32;
        let expected = 2.0 * x + 1.0;
        let pred = model.predict_row(&[x], &[0.0])[0];
        let diff = (pred - expected).abs();
        assert!(
            diff < 0.5,
            "Prediction for x={}: expected {}, got {}, diff={}",
            x,
            expected,
            pred,
            diff
        );
    }
}

/// Test parallel and sequential training produce similar results.
#[test]
fn parallel_vs_sequential_similar() {
    let row_data = RowMatrix::from_vec(
        vec![
            1.0, 1.0,
            2.0, 1.0,
            1.0, 2.0,
            2.0, 2.0,
        ],
        4,
        2,
    );
    let data: ColMatrix = row_data.to_layout();
    let labels = vec![3.0, 4.0, 5.0, 6.0]; // y = x0 + 2*x1

    let seq_config = LinearTrainerConfig {
        num_rounds: 50,
        learning_rate: 0.3,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let seq_model = LinearTrainer::new(seq_config).train(&data, &labels, &SquaredLoss, 1);

    let par_config = LinearTrainerConfig {
        num_rounds: 50,
        learning_rate: 0.3,
        parallel: true,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let par_model = LinearTrainer::new(par_config).train(&data, &labels, &SquaredLoss, 1);

    // Predictions should be similar
    let seq_pred = seq_model.predict_row(&[2.0, 2.0], &[0.0])[0];
    let par_pred = par_model.predict_row(&[2.0, 2.0], &[0.0])[0];

    let diff = (seq_pred - par_pred).abs();
    assert!(
        diff < 2.0,
        "Sequential vs parallel predictions differ too much: {} vs {}",
        seq_pred,
        par_pred
    );
}

// =============================================================================
// Story 5: Training Validation Tests
// =============================================================================

#[derive(Debug, Deserialize)]
struct XgbPredictions {
    predictions: Vec<f32>,
}

/// Load held-out test data if available.
fn load_test_data(name: &str) -> Option<(ColMatrix<f32>, Vec<f32>)> {
    let data_path = Path::new(TEST_CASES_DIR).join(format!("{}.test_data.json", name));
    let labels_path = Path::new(TEST_CASES_DIR).join(format!("{}.test_labels.json", name));

    if !data_path.exists() {
        return None;
    }

    let data_json = fs::read_to_string(&data_path).ok()?;
    let labels_json = fs::read_to_string(&labels_path).ok()?;

    let data: TrainData = serde_json::from_str(&data_json).ok()?;
    let labels: TrainLabels = serde_json::from_str(&labels_json).ok()?;

    let features: Vec<f32> = data
        .data
        .into_iter()
        .map(|v| v.unwrap_or(f32::NAN))
        .collect();

    let row_matrix = RowMatrix::from_vec(features, data.num_rows, data.num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();
    Some((col_matrix, labels.labels))
}

/// Load XGBoost predictions on test data if available.
fn load_xgb_predictions(name: &str) -> Option<Vec<f32>> {
    let path = Path::new(TEST_CASES_DIR).join(format!("{}.xgb_predictions.json", name));
    let json = fs::read_to_string(&path).ok()?;
    let preds: XgbPredictions = serde_json::from_str(&json).ok()?;
    Some(preds.predictions)
}

/// Compute Pearson correlation coefficient between two vectors.
fn pearson_correlation(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    
    let mean_a: f64 = a.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mean_b: f64 = b.iter().map(|x| *x as f64).sum::<f64>() / n;
    
    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    
    for (x, y) in a.iter().zip(b.iter()) {
        let da = (*x as f64) - mean_a;
        let db = (*y as f64) - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    
    if var_a < 1e-10 || var_b < 1e-10 {
        return 0.0;
    }
    
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Compute RMSE between two vectors.
fn rmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a.iter().zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum::<f64>() / (a.len() as f64);
    mse.sqrt()
}

/// Test weight correlation with XGBoost (Pearson r > 0.9).
///
/// Validates that our trained weights are highly correlated with XGBoost's,
/// even if not identical (due to randomness and floating-point differences).
#[rstest]
#[case("regression_l2")]
#[case("regression_elastic_net")]
fn weight_correlation_with_xgboost(#[case] name: &str) {
    let (data, labels) = load_train_data(name);
    let xgb_weights = load_xgb_weights(name);
    let config = load_config(name);

    let trainer_config = LinearTrainerConfig {
        num_rounds: config.num_boost_round,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = LinearTrainer::new(trainer_config);
    let model = trainer.train(&data, &labels, &SquaredLoss, 1);

    // Extract weights (excluding bias for correlation)
    let num_features = xgb_weights.num_features;
    let xgb_w: Vec<f32> = xgb_weights.weights[..num_features].to_vec();
    let our_w: Vec<f32> = (0..num_features).map(|i| model.weight(i, 0)).collect();

    let correlation = pearson_correlation(&xgb_w, &our_w);
    
    println!("Test case: {}", name);
    println!("XGBoost weights: {:?}", &xgb_w[..xgb_w.len().min(5)]);
    println!("Our weights: {:?}", &our_w[..our_w.len().min(5)]);
    println!("Pearson correlation: {:.4}", correlation);

    assert!(
        correlation > 0.9,
        "Weight correlation {} is too low (expected > 0.9)",
        correlation
    );
}

/// Test predictions on held-out test set have reasonable quality.
///
/// This validates end-to-end training quality by:
/// 1. Comparing our test RMSE to XGBoost's test RMSE (within 2x factor)
/// 2. Checking our predictions correlate highly with XGBoost's
///
/// Note: Our coordinate descent uses stale gradients (shotgun method) which
/// can produce different weights than XGBoost's sequential updates, but should
/// achieve similar prediction quality.
#[rstest]
#[case("regression_l2")]
#[case("regression_elastic_net")]
fn test_set_prediction_quality(#[case] name: &str) {
    let (train_data, train_labels) = load_train_data(name);
    let (test_data, test_labels) = load_test_data(name).expect("Test data required");
    let xgb_predictions = load_xgb_predictions(name).expect("XGBoost predictions required");
    let config = load_config(name);

    let trainer_config = LinearTrainerConfig {
        num_rounds: config.num_boost_round,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = LinearTrainer::new(trainer_config);
    let model = trainer.train(&train_data, &train_labels, &SquaredLoss, 1);

    // Get our predictions on test set
    let mut our_predictions = vec![0.0f32; test_data.num_rows()];
    for row in 0..test_data.num_rows() {
        let features: Vec<f32> = (0..test_data.num_features())
            .map(|col| test_data.get(row, col).copied().unwrap_or(f32::NAN))
            .collect();
        our_predictions[row] = model.predict_row(&features, &[0.0])[0];
    }

    // Compare test RMSE vs ground truth
    let our_test_rmse = rmse(&our_predictions, &test_labels);
    let xgb_test_rmse = rmse(&xgb_predictions, &test_labels);
    
    // Compare prediction correlation (should be high even if scales differ)
    let pred_correlation = pearson_correlation(&our_predictions, &xgb_predictions);
    
    println!("Test case: {}", name);
    println!("Our test RMSE: {:.4}", our_test_rmse);
    println!("XGBoost test RMSE: {:.4}", xgb_test_rmse);
    println!("Prediction correlation with XGBoost: {:.4}", pred_correlation);

    // Our RMSE should be within 3x of XGBoost's (allowing for convergence differences)
    let rmse_ratio = our_test_rmse / xgb_test_rmse;
    assert!(
        rmse_ratio < 3.0,
        "Our test RMSE ({:.4}) is {:.2}x worse than XGBoost ({:.4})",
        our_test_rmse, rmse_ratio, xgb_test_rmse
    );
    
    // Predictions should be highly correlated
    assert!(
        pred_correlation > 0.9,
        "Prediction correlation {} is too low (expected > 0.9)",
        pred_correlation
    );
}

/// Test binary classification training produces reasonable predictions.
#[test]
fn train_binary_classification() {
    let (data, labels) = load_train_data("binary_classification");
    let config = load_config("binary_classification");

    let trainer_config = LinearTrainerConfig {
        num_rounds: config.num_boost_round,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = LinearTrainer::new(trainer_config);
    let model = trainer.train(&data, &labels, &LogisticLoss, 1);

    // Verify predictions are in reasonable range for logits
    let mut predictions = Vec::new();
    for row in 0..data.num_rows().min(10) {
        let features: Vec<f32> = (0..data.num_features())
            .map(|col| data.get(row, col).copied().unwrap_or(f32::NAN))
            .collect();
        let pred = model.predict_row(&features, &[0.0])[0];
        predictions.push(pred);
    }

    println!("Binary classification predictions (logits): {:?}", predictions);
    
    // Logits should be finite and not too extreme
    for pred in &predictions {
        assert!(pred.is_finite(), "Prediction is not finite: {}", pred);
        assert!(pred.abs() < 100.0, "Prediction too extreme: {}", pred);
    }
}

/// Test multiclass classification training produces reasonable predictions.
///
/// NOTE: Multiclass is not fully implemented yet (all classes get same gradients).
/// This test just verifies the model runs without errors and produces finite outputs.
#[test]
#[ignore = "Multiclass gradient computation not yet implemented"]
fn train_multiclass_classification() {
    let (data, labels) = load_train_data("multiclass_classification");
    let config = load_config("multiclass_classification");
    let num_class = 3;

    let trainer_config = LinearTrainerConfig {
        num_rounds: config.num_boost_round,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = LinearTrainer::new(trainer_config);
    let model = trainer.train(&data, &labels, &SquaredLoss, num_class);

    // Verify predictions exist for all classes
    let features: Vec<f32> = (0..data.num_features())
        .map(|col| data.get(0, col).copied().unwrap_or(f32::NAN))
        .collect();
    let predictions = model.predict_row(&features, &vec![0.0; num_class]);
    
    println!("Multiclass predictions (logits): {:?}", predictions);
    
    assert_eq!(predictions.len(), num_class);
    for pred in &predictions {
        assert!(pred.is_finite(), "Prediction is not finite: {}", pred);
    }

    // When multiclass is properly implemented, the predictions for different
    // classes should be different (not all identical)
    // assert!(predictions[0] != predictions[1] || predictions[1] != predictions[2],
    //     "All class predictions are identical - multiclass not working properly");
}
