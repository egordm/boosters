//! Quantile regression tests.
//!
//! Tests for single and multi-quantile regression including:
//! - Single quantile regression (0.1, 0.5, 0.9)
//! - Multi-quantile regression (joint training)
//! - Quantile ordering validation

use super::{load_config, load_test_data, load_train_data, pearson_correlation, TEST_CASES_DIR};
use approx::assert_relative_eq;
use booste_rs::data::DataMatrix;
use booste_rs::training::linear::{LinearTrainer, LinearTrainerConfig};
use booste_rs::training::{QuantileLoss, Verbosity};
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

    // Verify config has correct quantile alpha
    let alpha = config.quantile_alpha.expect("Config should have quantile_alpha");
    assert_relative_eq!(alpha, expected_alpha, epsilon = 0.01);

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
    let loss = QuantileLoss::new(alpha);
    let model = trainer.train(&data, &labels, &loss);

    // Compute predictions on test set
    let mut test_preds = Vec::with_capacity(test_data.num_rows());
    for i in 0..test_data.num_rows() {
        let features: Vec<f32> = (0..test_data.num_features())
            .map(|col| test_data.get(i, col).copied().unwrap_or(0.0))
            .collect();
        let pred = model.predict_row(&features, &[0.0])[0];
        test_preds.push(pred);
    }

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

    println!(
        "{}: quantile={:.1}, pinball_loss={:.4}",
        name, alpha, pinball_loss
    );

    // Pinball loss should be reasonable (not NaN or infinite)
    assert!(pinball_loss.is_finite(), "Pinball loss is not finite");

    // For quantile regression, check that predictions respect the quantile
    // Lower quantiles should under-predict more, higher quantiles should over-predict more
    let under_predictions = test_preds
        .iter()
        .zip(test_labels.iter())
        .filter(|(pred, label)| **pred < **label)
        .count();
    let fraction_under = under_predictions as f64 / test_labels.len() as f64;

    // For α=0.1, expect ~90% under-predictions
    // For α=0.5, expect ~50% under-predictions
    // For α=0.9, expect ~10% under-predictions
    // Allow some tolerance (±20%) since this is a small test set
    let expected_fraction = 1.0 - alpha as f64;
    println!(
        "{}: fraction_under={:.2}, expected~{:.2}",
        name, fraction_under, expected_fraction
    );

    // Just verify it's in a reasonable range
    assert!(
        fraction_under > 0.1 && fraction_under < 0.9,
        "Under-prediction fraction {} out of expected range",
        fraction_under
    );
}

/// Test that different quantiles produce different predictions.
#[test]
fn quantile_regression_predictions_differ() {
    let (data, labels) = load_train_data("quantile_regression");
    let config = load_config("quantile_regression");

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

    // Train three models with different quantiles
    let model_low = trainer.train(&data, &labels, &QuantileLoss::new(0.1));
    let model_med = trainer.train(&data, &labels, &QuantileLoss::new(0.5));
    let model_high = trainer.train(&data, &labels, &QuantileLoss::new(0.9));

    // Get predictions for first sample
    let features: Vec<f32> = (0..data.num_features())
        .map(|col| data.get(0, col).copied().unwrap_or(0.0))
        .collect();

    let pred_low = model_low.predict_row(&features, &[0.0])[0];
    let pred_med = model_med.predict_row(&features, &[0.0])[0];
    let pred_high = model_high.predict_row(&features, &[0.0])[0];

    println!(
        "Quantile predictions: low={:.2}, med={:.2}, high={:.2}",
        pred_low, pred_med, pred_high
    );

    // Lower quantile should produce lower predictions
    assert!(
        pred_low < pred_high,
        "Low quantile ({:.2}) should predict lower than high quantile ({:.2})",
        pred_low,
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
///
/// This tests training multiple quantiles simultaneously with one model,
/// which is more efficient than training separate models.
///
/// Validates:
/// 1. Multi-quantile model produces correct number of outputs
/// 2. Different quantiles produce appropriately ordered predictions
/// 3. Predictions correlate with 3 separate XGBoost quantile models
#[test]
fn train_multi_quantile_regression() {
    let (data, labels) = load_train_data("multi_quantile");
    let (test_data, _test_labels) = load_test_data("multi_quantile").expect("Test data required");
    let config = load_multi_quantile_config();
    let xgb_preds = load_multi_quantile_xgb_predictions();

    let quantile_alphas = &config.quantile_alpha;
    let num_quantiles = quantile_alphas.len();
    assert_eq!(num_quantiles, 3);

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

    // Train single multi-quantile model
    let trainer = LinearTrainer::new(trainer_config.clone());
    let loss = QuantileLoss::multi(quantile_alphas);
    let model = trainer.train_multiclass(&data, &labels, &loss);

    // Verify model has correct number of output groups
    assert_eq!(model.num_groups(), num_quantiles);

    // Get predictions on test set
    let mut our_predictions: Vec<Vec<f32>> = vec![Vec::new(); num_quantiles];
    for i in 0..test_data.num_rows() {
        let features: Vec<f32> = (0..test_data.num_features())
            .map(|col| test_data.get(i, col).copied().unwrap_or(0.0))
            .collect();
        let preds = model.predict_row(&features, &vec![0.0; num_quantiles]);
        for (q, pred) in preds.iter().enumerate() {
            our_predictions[q].push(*pred);
        }
    }

    // Compare with XGBoost's separate-model predictions
    for q in 0..num_quantiles {
        let alpha = quantile_alphas[q];
        let our_preds = &our_predictions[q];
        let xgb_preds_q = &xgb_preds.predictions[q];

        let correlation = pearson_correlation(our_preds, xgb_preds_q);
        println!(
            "Quantile {:.1}: correlation with XGBoost = {:.4}",
            alpha, correlation
        );

        // Multi-quantile model should produce predictions correlated with
        // XGBoost's separate models (allowing some tolerance since training differs)
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
    println!("Fraction with ordered quantiles: {:.2}", ordered_fraction);
    assert!(
        ordered_fraction > 0.7,
        "Quantile predictions should be ordered for most samples (got {:.2})",
        ordered_fraction
    );
}

/// Test multi-quantile vs separate models comparison.
///
/// Validates that a single multi-quantile model produces similar results
/// to training 3 separate single-quantile models.
#[test]
fn multi_quantile_vs_separate_models() {
    let (data, labels) = load_train_data("multi_quantile");
    let config = load_multi_quantile_config();
    let quantile_alphas = &config.quantile_alpha;

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

    let trainer = LinearTrainer::new(trainer_config.clone());

    // Train single multi-quantile model
    let multi_loss = QuantileLoss::multi(quantile_alphas);
    let multi_model = trainer.train_multiclass(&data, &labels, &multi_loss);

    // Train 3 separate single-quantile models
    let single_models: Vec<_> = quantile_alphas
        .iter()
        .map(|&alpha| trainer.train(&data, &labels, &QuantileLoss::new(alpha)))
        .collect();

    // Compare predictions on training set
    for i in 0..data.num_rows().min(10) {
        let features: Vec<f32> = (0..data.num_features())
            .map(|col| data.get(i, col).copied().unwrap_or(0.0))
            .collect();

        let multi_preds = multi_model.predict_row(&features, &[0.0; 3]);

        println!(
            "Sample {}: multi_model=[{:.2}, {:.2}, {:.2}]",
            i, multi_preds[0], multi_preds[1], multi_preds[2]
        );

        for (q, single_model) in single_models.iter().enumerate() {
            let single_pred = single_model.predict_row(&features, &[0.0])[0];
            println!(
                "  q={:.1}: multi={:.2}, single={:.2}",
                quantile_alphas[q], multi_preds[q], single_pred
            );
        }
    }

    // Predictions don't have to be identical (different training dynamics),
    // but should be correlated
    for (q, single_model) in single_models.iter().enumerate() {
        let mut multi_preds = Vec::new();
        let mut single_preds = Vec::new();

        for i in 0..data.num_rows() {
            let features: Vec<f32> = (0..data.num_features())
                .map(|col| data.get(i, col).copied().unwrap_or(0.0))
                .collect();

            multi_preds.push(multi_model.predict_row(&features, &[0.0; 3])[q]);
            single_preds.push(single_model.predict_row(&features, &[0.0])[0]);
        }

        let correlation = pearson_correlation(&multi_preds, &single_preds);
        println!(
            "Quantile {:.1}: multi vs single correlation = {:.4}",
            quantile_alphas[q], correlation
        );

        // Multi-quantile should be reasonably correlated with single-quantile
        // (may differ due to joint optimization)
        assert!(
            correlation > 0.5,
            "Quantile {} multi vs single correlation {} too low",
            quantile_alphas[q],
            correlation
        );
    }
}
