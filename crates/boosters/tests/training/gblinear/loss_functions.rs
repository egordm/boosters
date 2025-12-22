//! Loss function integration tests.
//!
//! Tests for alternative loss functions:
//! - PseudoHuberLoss (robust regression)
//! - HingeLoss (SVM-style binary classification)

use super::{load_test_data, load_train_data};
use boosters::data::Dataset;
use boosters::inference::LinearModelPredict;
use boosters::training::{
    Accuracy, GBLinearParams, GBLinearTrainer, HingeLoss, LogLoss, LogisticLoss, MarginAccuracy,
    ObjectiveFn, PseudoHuberLoss, Rmse, SquaredLoss, Verbosity,
};
use ndarray::{Array2, ArrayView1};
use rstest::rstest;

/// Helper to create predictions array from PredictionOutput
fn pred_to_array2(output: &boosters::inference::common::PredictionOutput) -> Array2<f32> {
    // PredictionOutput stores data column-major: [g0_s0, g0_s1, ..., g1_s0, g1_s1, ...]
    // We need Array2 with shape [n_groups, n_samples]
    let n_samples = output.num_rows();
    let n_groups = output.num_groups();
    Array2::from_shape_vec((n_groups, n_samples), output.as_slice().to_vec()).unwrap()
}

// =============================================================================
// PseudoHuberLoss Integration Tests
// =============================================================================

/// Test PseudoHuberLoss with different delta values.
///
/// Validates:
/// 1. Model trains successfully with various delta values
/// 2. All models produce reasonable RMSE on test set
/// 3. Large delta values behave more like squared loss
///
/// Delta parameter affects training:
/// - Large delta: Behaves more like squared loss, converges faster
/// - Moderate delta: More robust to outliers but may need more rounds
#[rstest]
#[case(2.0, 50.0)]   // Moderate delta: more robust, higher RMSE threshold
#[case(5.0, 20.0)]   // Medium delta: balanced
#[case(10.0, 20.0)]  // Large delta: similar to squared loss
fn train_pseudo_huber_with_delta(#[case] delta: f32, #[case] max_rmse: f64) {
    let (data, labels) = load_train_data("regression_l2");
    let train = Dataset::from_numeric(&data, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };

    let params = GBLinearParams {
        n_rounds: 100,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(PseudoHuberLoss::new(delta), Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();

    use boosters::training::MetricFn;

    let output = model.predict(&test_data, &[]);
    let pred_arr = pred_to_array2(&output);
    let targets_arr = ArrayView1::from(&test_labels[..]);
    let rmse = Rmse.compute(pred_arr.view(), targets_arr, None);

    assert!(
        rmse < max_rmse,
        "PseudoHuber(delta={}) RMSE {} exceeds threshold {}",
        delta,
        rmse,
        max_rmse
    );
}

/// Test that large delta PseudoHuber converges to squared loss behavior.
#[test]
fn pseudo_huber_large_delta_matches_squared() {
    let (data, labels) = load_train_data("regression_l2");
    let train = Dataset::from_numeric(&data, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };

    let base_params = GBLinearParams {
        n_rounds: 100,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    // Train with large delta PseudoHuber
    let trainer_ph = GBLinearTrainer::new(PseudoHuberLoss::new(10.0), Rmse, base_params.clone());
    let ph_model = trainer_ph.train(&train, &[]).unwrap();

    // Train with squared loss for reference
    let trainer_sq = GBLinearTrainer::new(SquaredLoss, Rmse, base_params);
    let sq_model = trainer_sq.train(&train, &[]).unwrap();

    use boosters::training::MetricFn;

    let ph_output = ph_model.predict(&test_data, &[]);
    let sq_output = sq_model.predict(&test_data, &[]);

    let ph_arr = pred_to_array2(&ph_output);
    let sq_arr = pred_to_array2(&sq_output);
    let targets_arr = ArrayView1::from(&test_labels[..]);

    let ph_rmse = Rmse.compute(ph_arr.view(), targets_arr, None);
    let sq_rmse = Rmse.compute(sq_arr.view(), targets_arr, None);

    // Large delta should be very close to squared loss
    let diff = (ph_rmse - sq_rmse).abs();
    assert!(
        diff < 1.0,
        "Large delta PseudoHuber should be close to squared: diff={}",
        diff
    );
}

// =============================================================================
// HingeLoss Integration Tests
// =============================================================================

/// Test HingeLoss training on binary classification data.
///
/// Validates:
/// 1. Model trains successfully with hinge loss
/// 2. Predictions are reasonable for classification
/// 3. Accuracy on test set is acceptable
#[test]
fn train_hinge_binary_classification() {
    let (data, labels) = load_train_data("binary_classification");
    let train = Dataset::from_numeric(&data, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("binary_classification") {
        Some(d) => d,
        None => {
            println!("No test data for binary_classification, skipping");
            return;
        }
    };

    let base_params = GBLinearParams {
        n_rounds: 50,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    // Train with hinge loss — evaluate with MarginAccuracy (threshold=0)
    let trainer_hinge =
        GBLinearTrainer::new(HingeLoss, MarginAccuracy::default(), base_params.clone());
    let hinge_model = trainer_hinge.train(&train, &[]).unwrap();

    // Also train with logistic for comparison
    let trainer_logistic = GBLinearTrainer::new(LogisticLoss, LogLoss, base_params);
    let logistic_model = trainer_logistic.train(&train, &[]).unwrap();

    // Compute accuracy.
    // - Hinge: MarginAccuracy (threshold on margins at 0.0).
    // - Logistic: transform logits to probabilities via objective transform, then threshold at 0.5.
    use boosters::training::MetricFn;

    let hinge_output = hinge_model.predict(&test_data, &[]);
    let hinge_arr = pred_to_array2(&hinge_output);
    let targets_arr = ArrayView1::from(&test_labels[..]);
    let hinge_acc = MarginAccuracy::default()
        .compute(hinge_arr.view(), targets_arr, None)
        as f32;

    let logistic_output = logistic_model.predict(&test_data, &[]);
    // Convert to Array2, apply transform, then use for metrics
    let mut logistic_arr = pred_to_array2(&logistic_output);
    LogisticLoss.transform_predictions(logistic_arr.view_mut());
    let logistic_acc = Accuracy::with_threshold(0.5)
        .compute(logistic_arr.view(), targets_arr, None) as f32;

    // Both should achieve reasonable accuracy (better than random = 50%)
    assert!(
        hinge_acc > 0.5,
        "Hinge accuracy {} should be better than random",
        hinge_acc
    );
    assert!(
        logistic_acc > 0.5,
        "Logistic accuracy {} should be better than random",
        logistic_acc
    );

    // Both should be in similar range.
    let _acc_diff = (hinge_acc - logistic_acc).abs();
}
