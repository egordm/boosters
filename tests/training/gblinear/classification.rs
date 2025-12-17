//! Classification training tests.
//!
//! Tests for binary and multiclass classification including:
//! - Binary classification with logistic loss
//! - Multiclass classification with softmax loss

use super::load_config;
use super::load_train_data;
use booste_rs::data::Dataset;
use booste_rs::inference::common::PredictionOutput;
use booste_rs::training::{
    GBLinearParams, GBLinearTrainer, LogLoss, LogisticLoss, MulticlassLogLoss, SoftmaxLoss,
    Verbosity,
};

/// Test binary classification training produces reasonable predictions.
#[test]
fn train_binary_classification() {
    let (data, labels) = load_train_data("binary_classification");
    let config = load_config("binary_classification");
    let train = Dataset::from_numeric(&data, labels.clone()).unwrap();

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

    let trainer = GBLinearTrainer::new(LogisticLoss, LogLoss, params);
    let model = trainer.train(&train, &[]).unwrap();

    // Verify predictions are in reasonable range for logits.
    let output: PredictionOutput = model.predict(&data, &[0.0]);
    let predictions: Vec<f32> = output
        .rows()
        .take(10)
        .map(|row| row[0])
        .collect();

    // Logits should be finite and not too extreme
    for pred in &predictions {
        assert!(pred.is_finite(), "Prediction is not finite: {}", pred);
        assert!(pred.abs() < 100.0, "Prediction too extreme: {}", pred);
    }
}

/// Test multiclass classification training produces reasonable predictions.
///
/// Uses proper SoftmaxLoss which computes per-class gradients correctly.
#[test]
fn train_multioutput_classification() {
    let (data, labels) = load_train_data("multiclass_classification");
    let config = load_config("multiclass_classification");
    let num_class = 3;
    let train = Dataset::from_numeric(&data, labels.clone()).unwrap();

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

    let trainer = GBLinearTrainer::new(SoftmaxLoss::new(num_class), MulticlassLogLoss, params);
    let model = trainer.train(&train, &[]).unwrap();

    // Verify model has correct number of output groups
    assert_eq!(model.num_groups(), num_class);

    // Verify predictions exist for all classes
    let output: PredictionOutput = model.predict(&data, &vec![0.0; num_class]);
    let predictions = output.row(0);

    assert_eq!(predictions.len(), num_class);
    for pred in predictions {
        assert!(pred.is_finite(), "Prediction is not finite: {}", pred);
    }

    // With proper multiclass gradients, predictions for different classes should differ
    let all_same = predictions.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
    assert!(
        !all_same,
        "All class predictions are identical - multiclass not working properly"
    );

    // Compute training accuracy (argmax over logits).
    use booste_rs::training::{Metric, MulticlassAccuracy};

    let output = model.predict(&data, &vec![0.0; num_class]);
    let n_rows = output.num_rows();
    let pred_classes: Vec<f32> = (0..n_rows)
        .map(|row| {
            let row_preds = output.row(row);
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in row_preds.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            best_idx as f32
        })
        .collect();

    let accuracy = MulticlassAccuracy.compute(n_rows, 1, &pred_classes, &labels, &[]);

    // Training accuracy should be reasonable for linear model
    // (better than random = 33.3% for 3 classes)
    assert!(accuracy > 0.4, "Training accuracy {} should be > 40%", accuracy);
}
