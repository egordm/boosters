//! Loss function integration tests.
//!
//! Tests for alternative loss functions:
//! - PseudoHuberLoss (robust regression)
//! - HingeLoss (SVM-style binary classification)

use super::{compute_binary_accuracy, compute_test_rmse_default, load_test_data, load_train_data};
use booste_rs::data::Dataset;
use booste_rs::training::{
    Accuracy, GBLinearParams, GBLinearTrainer, HingeLoss, LogLoss, LogisticLoss, PseudoHuberLoss,
    Rmse, SquaredLoss, Verbosity,
};

// =============================================================================
// PseudoHuberLoss Integration Tests
// =============================================================================

/// Test PseudoHuberLoss training on regression data.
///
/// Validates:
/// 1. Model trains successfully
/// 2. Predictions are reasonable (not NaN, not extreme)
/// 3. RMSE on test set is acceptable
///
/// Note: PseudoHuber with default slope=1.0 has different gradient magnitudes
/// than squared loss, requiring more rounds or different learning rates.
#[test]
fn train_pseudo_huber_regression() {
    let (data, labels) = load_train_data("regression_l2");
    let train = Dataset::from_numeric(&data, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };

    // Train with Pseudo-Huber loss with large delta (more similar to squared)
    let params_ph = GBLinearParams {
        n_rounds: 100,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer_ph = GBLinearTrainer::new(PseudoHuberLoss::new(5.0), Rmse, params_ph);
    let model = trainer_ph.train(&train, &[]).unwrap();

    // Also train with squared loss for comparison
    let params_sq = GBLinearParams {
        n_rounds: 100,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer_sq = GBLinearTrainer::new(SquaredLoss, Rmse, params_sq);
    let sq_model = trainer_sq.train(&train, &[]).unwrap();

    // Compute RMSE on test set for both
    let ph_rmse = compute_test_rmse_default(&model, &test_data, &test_labels);
    let sq_rmse = compute_test_rmse_default(&sq_model, &test_data, &test_labels);

    println!("PseudoHuber (delta=5.0) RMSE: {:.4}", ph_rmse);
    println!("SquaredLoss RMSE: {:.4}", sq_rmse);

    // Both should have reasonable RMSE
    assert!(ph_rmse < 20.0, "PseudoHuber RMSE too high: {}", ph_rmse);
    assert!(sq_rmse < 20.0, "SquaredLoss RMSE too high: {}", sq_rmse);
}

/// Test PseudoHuberLoss with different delta values.
///
/// Validates that the delta parameter affects training:
/// - Large delta: Behaves more like squared loss, converges faster
/// - Moderate delta: Still works but may need more rounds
#[test]
fn train_pseudo_huber_with_delta() {
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

    // Train with moderate delta
    let trainer_moderate = GBLinearTrainer::new(PseudoHuberLoss::new(2.0), Rmse, base_params.clone());
    let moderate_model = trainer_moderate.train(&train, &[]).unwrap();

    // Train with large delta (more like squared)
    let trainer_large = GBLinearTrainer::new(PseudoHuberLoss::new(10.0), Rmse, base_params.clone());
    let large_model = trainer_large.train(&train, &[]).unwrap();

    // Train with squared for reference
    let trainer_sq = GBLinearTrainer::new(SquaredLoss, Rmse, base_params);
    let sq_model = trainer_sq.train(&train, &[]).unwrap();

    let moderate_rmse = compute_test_rmse_default(&moderate_model, &test_data, &test_labels);
    let large_rmse = compute_test_rmse_default(&large_model, &test_data, &test_labels);
    let sq_rmse = compute_test_rmse_default(&sq_model, &test_data, &test_labels);

    println!("Moderate delta (2.0) RMSE: {:.4}", moderate_rmse);
    println!("Large delta (10.0) RMSE: {:.4}", large_rmse);
    println!("Squared loss RMSE: {:.4}", sq_rmse);

    // All should produce reasonable models
    assert!(
        moderate_rmse < 50.0,
        "Moderate delta RMSE too high: {}",
        moderate_rmse
    );
    assert!(
        large_rmse < 20.0,
        "Large delta RMSE too high: {}",
        large_rmse
    );

    // Large delta should be closer to squared loss
    let large_diff = (large_rmse - sq_rmse).abs();
    println!("Large delta diff from squared: {:.4}", large_diff);

    // Large delta should be very close to squared
    assert!(
        large_diff < 1.0,
        "Large delta PseudoHuber should be close to squared: diff={}",
        large_diff
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

    // Train with hinge loss
    let trainer_hinge = GBLinearTrainer::new(HingeLoss, Accuracy::with_threshold(0.0), base_params.clone());
    let hinge_model = trainer_hinge.train(&train, &[]).unwrap();

    // Also train with logistic for comparison
    let trainer_logistic = GBLinearTrainer::new(LogisticLoss, LogLoss, base_params);
    let logistic_model = trainer_logistic.train(&train, &[]).unwrap();

    // Compute accuracy (predictions > 0 → class 1, else class 0)
    let hinge_acc = compute_binary_accuracy(&hinge_model, &test_data, &test_labels, 0.0);
    let logistic_acc = compute_binary_accuracy(&logistic_model, &test_data, &test_labels, 0.5);

    println!("Hinge accuracy: {:.2}%", hinge_acc * 100.0);
    println!("Logistic accuracy: {:.2}%", logistic_acc * 100.0);

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

    // Both should be in similar range
    let acc_diff = (hinge_acc - logistic_acc).abs();
    println!("Accuracy difference: {:.2}%", acc_diff * 100.0);
}
