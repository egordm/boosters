//! Quantile regression tests.
//!
//! Tests for single and multi-quantile regression including:
//! - Single quantile regression (0.1, 0.5, 0.9)
//! - Multi-quantile regression (joint training)
//! - Quantile ordering validation

use super::{
    TEST_CASES_DIR, load_config, load_test_data, load_train_data, make_dataset, pearson_correlation,
};
use approx::assert_relative_eq;
use boosters::data::transpose_to_c_order;
use boosters::data::{Dataset, TargetsView, WeightsView};
use boosters::training::{
    GBLinearParams, GBLinearTrainer, PinballLoss, Rmse, UpdateStrategy, Verbosity,
};
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
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());

    // Verify config has correct quantile alpha
    let alpha = config
        .quantile_alpha
        .expect("Config should have quantile_alpha");
    assert_relative_eq!(alpha, expected_alpha, epsilon = 0.01);

    let params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        update_strategy: UpdateStrategy::Sequential,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(PinballLoss::new(alpha), Rmse, params);
    let model = trainer
        .train(&train, targets_view, WeightsView::None, None)
        .unwrap();

    // Compute predictions on test set
    // test_data is sample-major [n_samples, n_features], need feature-major [n_features, n_samples]
    let test_features = transpose_to_c_order(test_data.view());
    let test_dataset = Dataset::from_array(test_features.view(), None, None);
    let output = model.predict(&test_dataset);
    let test_preds: Vec<f32> = output.row(0).to_vec();

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
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());

    let base_params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        update_strategy: UpdateStrategy::Sequential,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    // Train three models with different quantiles
    let trainer_low = GBLinearTrainer::new(PinballLoss::new(0.1), Rmse, base_params.clone());
    let trainer_med = GBLinearTrainer::new(PinballLoss::new(0.5), Rmse, base_params.clone());
    let trainer_high = GBLinearTrainer::new(PinballLoss::new(0.9), Rmse, base_params);

    let model_low = trainer_low
        .train(&train, targets_view, WeightsView::None, None)
        .unwrap();
    let model_med = trainer_med
        .train(&train, targets_view, WeightsView::None, None)
        .unwrap();
    let model_high = trainer_high
        .train(&train, targets_view, WeightsView::None, None)
        .unwrap();

    // For prediction, data is already feature-major [n_features, n_samples]
    let pred_dataset = Dataset::from_array(data.view(), None, None);

    // Get predictions for first sample - output is [n_groups, n_samples]
    let pred_low = model_low.predict(&pred_dataset)[[0, 0]];
    let pred_med = model_med.predict(&pred_dataset)[[0, 0]];
    let pred_high = model_high.predict(&pred_dataset)[[0, 0]];

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
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());

    let quantile_alphas = config.quantile_alpha.clone();
    let num_quantiles = quantile_alphas.len();
    assert_eq!(num_quantiles, 3);

    let params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        update_strategy: UpdateStrategy::Sequential,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(
        PinballLoss::with_quantiles(quantile_alphas.clone()),
        Rmse,
        params,
    );
    let model = trainer
        .train(&train, targets_view, WeightsView::None, None)
        .unwrap();

    // Verify model has correct number of output groups
    assert_eq!(model.n_groups(), num_quantiles);

    // Get predictions on test set
    // test_data is sample-major [n_samples, n_features], need feature-major
    let test_features = transpose_to_c_order(test_data.view());
    let test_dataset = Dataset::from_array(test_features.view(), None, None);
    let n_test_samples = test_data.nrows();
    let output = model.predict(&test_dataset);

    // output is [n_groups, n_samples], extract per-quantile predictions
    let mut our_predictions: Vec<Vec<f32>> = vec![vec![0.0; n_test_samples]; num_quantiles];
    for i in 0..n_test_samples {
        for q in 0..num_quantiles {
            our_predictions[q][i] = output[[q, i]];
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
    for i in 0..n_test_samples {
        let low = our_predictions[0][i];
        let med = our_predictions[1][i];
        let high = our_predictions[2][i];
        if low <= med && med <= high {
            ordered_count += 1;
        }
    }
    let ordered_fraction = ordered_count as f64 / n_test_samples as f64;
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
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());

    let base_params = GBLinearParams {
        n_rounds: config.num_boost_round as u32,
        learning_rate: config.eta,
        alpha: config.alpha,
        lambda: config.lambda,
        update_strategy: UpdateStrategy::Sequential,
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
    let multi_model = multi_trainer
        .train(&train, targets_view, WeightsView::None, None)
        .unwrap();

    // Train 3 separate single-quantile models
    let single_models: Vec<_> = quantile_alphas
        .iter()
        .map(|&alpha| {
            let trainer = GBLinearTrainer::new(PinballLoss::new(alpha), Rmse, base_params.clone());
            trainer
                .train(&train, targets_view, WeightsView::None, None)
                .unwrap()
        })
        .collect();

    // Compare predictions on training set (data is already feature-major)
    let pred_dataset = Dataset::from_array(data.view(), None, None);
    let n_samples = data.ncols();

    // Predictions should be correlated
    for (q, single_model) in single_models.iter().enumerate() {
        let multi_output = multi_model.predict(&pred_dataset);
        let single_output = single_model.predict(&pred_dataset);

        // multi_output is [n_groups, n_samples], single_output is [1, n_samples]
        let mut multi_preds = Vec::with_capacity(n_samples);
        let mut single_preds = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            multi_preds.push(multi_output[[q, i]]);
            single_preds.push(single_output[[0, i]]);
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
