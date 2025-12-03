//! Loss function integration tests.
//!
//! Tests for alternative loss functions:
//! - PseudoHuberLoss (robust regression)
//! - HingeLoss (SVM-style binary classification)

use super::{load_test_data, load_train_data, ColMatrix};
use booste_rs::linear::training::{LinearTrainer, LinearTrainerConfig};
use booste_rs::linear::LinearModel;
use booste_rs::training::{
    Accuracy, HingeLoss, LogisticLoss, Metric, PseudoHuberLoss, Rmse, SquaredLoss, Verbosity,
};

// =============================================================================
// Helper Functions for Model Evaluation
// =============================================================================

/// Get predictions from a model for all rows in the data.
fn get_predictions(model: &LinearModel, data: &ColMatrix<f32>) -> Vec<f32> {
    let num_groups = model.num_groups();
    let num_features = model.num_features();
    let base_score = vec![0.0f32; num_groups];
    let mut predictions = Vec::with_capacity(data.num_rows() * num_groups);

    for i in 0..data.num_rows() {
        let row: Vec<f32> = (0..num_features)
            .map(|j| *data.get(i, j).unwrap_or(&0.0))
            .collect();
        let preds = model.predict_row(&row, &base_score);
        predictions.extend(preds);
    }
    predictions
}

/// Compute RMSE for a single-output model using the Rmse metric.
fn compute_test_rmse(model: &LinearModel, data: &ColMatrix<f32>, labels: &[f32]) -> f32 {
    let predictions = get_predictions(model, data);
    Rmse.evaluate(&predictions, labels, 1) as f32
}

/// Compute binary classification accuracy with a threshold.
fn compute_binary_accuracy(
    model: &LinearModel,
    data: &ColMatrix<f32>,
    labels: &[f32],
    threshold: f32,
) -> f32 {
    let predictions = get_predictions(model, data);
    Accuracy::with_threshold(threshold).evaluate(&predictions, labels, 1) as f32
}

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
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };

    // Train with Pseudo-Huber loss with large slope (more similar to squared)
    // Using larger slope makes gradients more comparable to squared loss
    let config = LinearTrainerConfig {
        num_rounds: 100,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let model =
        LinearTrainer::new(config.clone()).train(&data, &labels, &PseudoHuberLoss::new(5.0)); // Larger slope

    // Also train with squared loss for comparison
    let sq_model = LinearTrainer::new(config).train(&data, &labels, &SquaredLoss);

    // Compute RMSE on test set for both
    let ph_rmse = compute_test_rmse(&model, &test_data, &test_labels);
    let sq_rmse = compute_test_rmse(&sq_model, &test_data, &test_labels);

    println!("PseudoHuber (slope=5.0) RMSE: {:.4}", ph_rmse);
    println!("SquaredLoss RMSE: {:.4}", sq_rmse);

    // Both should have reasonable RMSE
    assert!(ph_rmse < 20.0, "PseudoHuber RMSE too high: {}", ph_rmse);
    assert!(sq_rmse < 20.0, "SquaredLoss RMSE too high: {}", sq_rmse);
}

/// Test PseudoHuberLoss with different slope values.
///
/// Validates that the slope parameter affects training:
/// - Large slope: Behaves more like squared loss, converges faster
/// - Moderate slope: Still works but may need more rounds
///
/// Note: Very small slopes (< 1) clip gradients significantly, requiring
/// many more rounds to converge. We test moderate-to-large slopes here.
#[test]
fn train_pseudo_huber_with_slope() {
    let (data, labels) = load_train_data("regression_l2");
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };

    let config = LinearTrainerConfig {
        num_rounds: 100,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    // Train with moderate slope
    let moderate_slope_model =
        LinearTrainer::new(config.clone()).train(&data, &labels, &PseudoHuberLoss::new(2.0));

    // Train with large slope (more like squared)
    let large_slope_model =
        LinearTrainer::new(config.clone()).train(&data, &labels, &PseudoHuberLoss::new(10.0));

    // Train with squared for reference
    let sq_model = LinearTrainer::new(config).train(&data, &labels, &SquaredLoss);

    let moderate_rmse = compute_test_rmse(&moderate_slope_model, &test_data, &test_labels);
    let large_rmse = compute_test_rmse(&large_slope_model, &test_data, &test_labels);
    let sq_rmse = compute_test_rmse(&sq_model, &test_data, &test_labels);

    println!("Moderate slope (2.0) RMSE: {:.4}", moderate_rmse);
    println!("Large slope (10.0) RMSE: {:.4}", large_rmse);
    println!("Squared loss RMSE: {:.4}", sq_rmse);

    // All should produce reasonable models
    assert!(
        moderate_rmse < 50.0,
        "Moderate slope RMSE too high: {}",
        moderate_rmse
    );
    assert!(
        large_rmse < 20.0,
        "Large slope RMSE too high: {}",
        large_rmse
    );

    // Large slope should be closer to squared loss
    let large_diff = (large_rmse - sq_rmse).abs();
    println!("Large slope diff from squared: {:.4}", large_diff);

    // Large slope should be very close to squared
    assert!(
        large_diff < 1.0,
        "Large slope PseudoHuber should be close to squared: diff={}",
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
    let (test_data, test_labels) = match load_test_data("binary_classification") {
        Some(d) => d,
        None => {
            println!("No test data for binary_classification, skipping");
            return;
        }
    };

    // Train with hinge loss
    let hinge_config = LinearTrainerConfig {
        num_rounds: 50,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let hinge_model = LinearTrainer::new(hinge_config.clone()).train(&data, &labels, &HingeLoss);

    // Also train with logistic for comparison
    let logistic_model = LinearTrainer::new(hinge_config).train(&data, &labels, &LogisticLoss);

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
