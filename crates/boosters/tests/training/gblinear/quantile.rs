//! Quantile regression tests.
//!
//! Tests for single and multi-quantile regression including:
//! - Single quantile regression (0.1, 0.5, 0.9)
//! - Multi-quantile regression (joint training)
//! - Quantile ordering validation

use super::{load_config, load_test_data, load_train_data, pearson_correlation, TEST_CASES_DIR};
use approx::assert_relative_eq;
use boosters::data::Dataset;
use boosters::inference::LinearModelPredict;
use boosters::training::{GBLinearParams, GBLinearTrainer, PinballLoss, Rmse, Verbosity};
use rstest::rstest;
use serde::Deserialize;
use std::fs;
use std::path::Path;

// =============================================================================
// Single Quantile Regression Tests
// =============================================================================

/// Test quantile regression training produces reasonable predictions.
///
/// Tests median regression (α=0.5), low quantile (α=0.1), and high quantile (α=0.9).
#[rstest]
#[case("quantile_regression", 0.5)]
#[case("quantile_low", 0.1)]
#[case("quantile_high", 0.9)]
fn train_quantile_regression(#[case] name: &str, #[case] expected_alpha: f32) {
    let (data, labels) = load_train_data(name);
    let config = load_config(name);
    let (test_data, test_labels) = load_test_data(name).expect("Test data should exist");
    let train = Dataset::from_numeric(&data, labels).unwrap();

    // Verify config has correct quantile alpha
    let alpha = config.quantile_alpha.expect("Config should have quantile_alpha");
    assert_relative_eq!(alpha, expected_alpha, epsilon = 0.01);

    let params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(PinballLoss::new(alpha), Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();

    // Compute predictions on test set
    let output = model.predict(&test_data, &[0.0]);
    let test_preds: Vec<f32> = output.column(0).to_vec();

    // Compute pinball loss on test set
    let pinball_loss: f64 = test_preds
        .iter()
        .zip(test_labels.iter())
        .map(|(&pred, &label)| {
            let residual = label - pred;
            if residual >= 0.0 {
                alpha as f64 * residual as f64
            } else {
                (1.0 - alpha as f64) * (-residual as f64)
            }
        })
        .sum::<f64>()
        / test_labels.len() as f64;

    // Pinball loss should be reasonable (not NaN or infinite)
    assert!(pinball_loss.is_finite(), "Pinball loss is not finite");

    // For quantile regression, check that predictions respect the quantile
    let under_predictions = test_preds
        .iter()
        .zip(test_labels.iter())
        .filter(|(pred, label)| **pred < **label)
        .count();
    let fraction_under = under_predictions as f64 / test_labels.len() as f64;

    let expected_fraction = 1.0 - alpha as f64;

    // Verify predictions are in a reasonable range around the expected quantile
    // For α=0.1: expected_fraction ≈ 0.9, for α=0.5: ≈ 0.5, for α=0.9: ≈ 0.1
    // Allow tolerance of ±0.15 around the expected fraction
    let tolerance = 0.20;
    assert!(
        (fraction_under - expected_fraction).abs() < tolerance,
        "Under-prediction fraction {} differs from expected {} by more than {}",
        fraction_under,
        expected_fraction,
        tolerance
    );
}

/// Test that different quantiles produce different predictions.
#[test]
fn quantile_regression_predictions_differ() {
    let (data, labels) = load_train_data("quantile_regression");
    let config = load_config("quantile_regression");
    let train = Dataset::from_numeric(&data, labels).unwrap();

    let base_params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    // Train three models with different quantiles
    let trainer_low = GBLinearTrainer::new(PinballLoss::new(0.1), Rmse, base_params.clone());
    let trainer_med = GBLinearTrainer::new(PinballLoss::new(0.5), Rmse, base_params.clone());
    let trainer_high = GBLinearTrainer::new(PinballLoss::new(0.9), Rmse, base_params);

    let model_low = trainer_low.train(&train, &[]).unwrap();
    let model_med = trainer_med.train(&train, &[]).unwrap();
    let model_high = trainer_high.train(&train, &[]).unwrap();

    // Get predictions for first sample
    let pred_low = model_low.predict(&data, &[0.0]).get(0, 0);
    let pred_med = model_med.predict(&data, &[0.0]).get(0, 0);
    let pred_high = model_high.predict(&data, &[0.0]).get(0, 0);

    // Lower quantile should produce lower predictions.
    assert!(
        pred_low <= pred_med,
        "Low quantile ({:.2}) should be <= median ({:.2})",
        pred_low,
        pred_med
    );
    assert!(
        pred_med <= pred_high,
        "Median ({:.2}) should be <= high quantile ({:.2})",
        pred_med,
        pred_high
    );
}

// =============================================================================
// Multi-Quantile Regression Tests
// =============================================================================

/// Config for multi-quantile test case.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MultiQuantileConfig {
    objective: String,
    quantile_alpha: Vec<f32>,
    booster: String,
    eta: f32,
    lambda: f32,
    alpha: f32,
    num_boost_round: usize,
    num_quantiles: usize,
}

/// XGBoost predictions for multi-quantile (list of lists, one per quantile).
#[derive(Debug, Deserialize)]
struct MultiQuantilePredictions {
    predictions: Vec<Vec<f32>>,
    #[allow(dead_code)]
    quantile_alphas: Vec<f32>,
}

fn load_multi_quantile_config() -> MultiQuantileConfig {
    let path = Path::new(TEST_CASES_DIR).join("multi_quantile.config.json");
    let json = fs::read_to_string(&path).expect("Failed to read multi_quantile config");
    serde_json::from_str(&json).expect("Failed to parse multi_quantile config")
}

fn load_multi_quantile_xgb_predictions() -> MultiQuantilePredictions {
    let path = Path::new(TEST_CASES_DIR).join("multi_quantile.xgb_predictions.json");
    let json = fs::read_to_string(&path).expect("Failed to read multi_quantile predictions");
    serde_json::from_str(&json).expect("Failed to parse multi_quantile predictions")
}

/// Test multi-quantile regression training with a single model.
#[test]
fn train_multi_quantile_regression() {
    let (data, labels) = load_train_data("multi_quantile");
    let (test_data, _test_labels) = load_test_data("multi_quantile").expect("Test data required");
    let config = load_multi_quantile_config();
    let xgb_preds = load_multi_quantile_xgb_predictions();
    let train = Dataset::from_numeric(&data, labels).unwrap();

    let quantile_alphas = config.quantile_alpha.clone();
    let num_quantiles = quantile_alphas.len();
    assert_eq!(num_quantiles, 3);

    let params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer =
        GBLinearTrainer::new(PinballLoss::with_quantiles(quantile_alphas.clone()), Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();

    // Verify model has correct number of output groups
    assert_eq!(model.num_groups(), num_quantiles);

    // Get predictions on test set
    let output = model.predict(&test_data, &vec![0.0; num_quantiles]);
    let mut our_predictions: Vec<Vec<f32>> = vec![vec![0.0; test_data.num_rows()]; num_quantiles];
    for i in 0..test_data.num_rows() {
        for q in 0..num_quantiles {
            our_predictions[q][i] = output.get(i, q);
        }
    }

    // Compare with XGBoost's separate-model predictions
    for q in 0..num_quantiles {
        let alpha = quantile_alphas[q];
        let our_preds = &our_predictions[q];
        let xgb_preds_q = &xgb_preds.predictions[q];

        let correlation = pearson_correlation(our_preds, xgb_preds_q);

        assert!(
            correlation > 0.7,
            "Quantile {} prediction correlation {} too low (expected > 0.7)",
            alpha,
            correlation
        );
    }

    // Verify quantile ordering: q0.1 < q0.5 < q0.9 for most samples
    let mut ordered_count = 0;
    for i in 0..test_data.num_rows() {
        let low = our_predictions[0][i];
        let med = our_predictions[1][i];
        let high = our_predictions[2][i];
        if low <= med && med <= high {
            ordered_count += 1;
        }
    }
    let ordered_fraction = ordered_count as f64 / test_data.num_rows() as f64;
    assert!(
        ordered_fraction > 0.7,
        "Quantile predictions should be ordered for most samples (got {:.2})",
        ordered_fraction
    );
}

/// Test multi-quantile vs separate models comparison.
#[test]
fn multi_quantile_vs_separate_models() {
    let (data, labels) = load_train_data("multi_quantile");
    let config = load_multi_quantile_config();
    let quantile_alphas = config.quantile_alpha.clone();
    let train = Dataset::from_numeric(&data, labels).unwrap();

    let base_params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    // Train single multi-quantile model
    let multi_trainer = GBLinearTrainer::new(
        PinballLoss::with_quantiles(quantile_alphas.clone()),
        Rmse,
        base_params.clone(),
    );
    let multi_model = multi_trainer.train(&train, &[]).unwrap();

    // Train 3 separate single-quantile models
    let single_models: Vec<_> = quantile_alphas
        .iter()
        .map(|&alpha| {
            let trainer = GBLinearTrainer::new(PinballLoss::new(alpha), Rmse, base_params.clone());
            trainer.train(&train, &[]).unwrap()
        })
        .collect();

    // Compare predictions on training set (used later for correlation checks)

    // Predictions should be correlated
    for (q, single_model) in single_models.iter().enumerate() {
        let multi_output = multi_model.predict(&data, &[0.0; 3]);
        let single_output = single_model.predict(&data, &[0.0]);

        let mut multi_preds = Vec::with_capacity(data.num_rows());
        let mut single_preds = Vec::with_capacity(data.num_rows());
        for i in 0..data.num_rows() {
            multi_preds.push(multi_output.get(i, q));
            single_preds.push(single_output.get(i, 0));
        }

        let correlation = pearson_correlation(&multi_preds, &single_preds);

        assert!(
            correlation > 0.5,
            "Quantile {} multi vs single correlation {} too low",
            quantile_alphas[q],
            correlation
        );
    }
}
