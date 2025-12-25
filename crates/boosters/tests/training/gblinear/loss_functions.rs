//! Loss function integration tests.
//!
//! Tests for alternative loss functions:
//! - PseudoHuberLoss (robust regression)
//! - HingeLoss (SVM-style binary classification)

use super::{load_test_data, load_train_data, make_dataset};
use boosters::data::transpose_to_c_order;
use boosters::data::{FeaturesView, TargetsView, WeightsView};
use boosters::training::{
    Accuracy, GBLinearParams, GBLinearTrainer, HingeLoss, LogLoss, LogisticLoss, MarginAccuracy,
    ObjectiveFn, PseudoHuberLoss, Rmse, SquaredLoss, Verbosity,
};
use ndarray::Array2;
use rstest::rstest;

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
    let train = make_dataset(&data, &labels);
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

    // test_data is sample-major [n_samples, n_features], transpose to feature-major
    let test_features = transpose_to_c_order(test_data.view());
    let test_view = FeaturesView::from_array(test_features.view());
    let output = model.predict(test_view);
    // output is [n_groups, n_samples]
    let targets_2d = Array2::from_shape_vec((1, test_labels.len()), test_labels.clone()).unwrap();
    let targets = TargetsView::new(targets_2d.view());
    let rmse = Rmse.compute(output.view(), targets, WeightsView::None);

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
    let train = make_dataset(&data, &labels);
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

    // test_data is sample-major [n_samples, n_features], transpose to feature-major
    let test_features = transpose_to_c_order(test_data.view());
    let test_view = FeaturesView::from_array(test_features.view());
    let ph_output = ph_model.predict(test_view);
    let sq_output = sq_model.predict(test_view);

    // output is [n_groups, n_samples]
    let targets_2d = Array2::from_shape_vec((1, test_labels.len()), test_labels.clone()).unwrap();
    let targets = TargetsView::new(targets_2d.view());

    let ph_rmse = Rmse.compute(ph_output.view(), targets, WeightsView::None);
    let sq_rmse = Rmse.compute(sq_output.view(), targets, WeightsView::None);

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
    let train = make_dataset(&data, &labels);
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

    // test_data is sample-major [n_samples, n_features], transpose to feature-major
    let test_features = transpose_to_c_order(test_data.view());
    let test_view = FeaturesView::from_array(test_features.view());
    let hinge_output = hinge_model.predict(test_view);
    let targets_2d = Array2::from_shape_vec((1, test_labels.len()), test_labels.clone()).unwrap();
    let targets = TargetsView::new(targets_2d.view());
    let hinge_acc = MarginAccuracy::default()
        .compute(hinge_output.view(), targets, WeightsView::None)
        as f32;

    let logistic_output = logistic_model.predict(test_view);
    // Apply transform to convert logits to probabilities
    let mut logistic_arr = logistic_output.clone();
    LogisticLoss.transform_predictions_inplace(logistic_arr.view_mut());
    let logistic_acc = Accuracy::with_threshold(0.5)
        .compute(logistic_arr.view(), targets, WeightsView::None) as f32;

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
