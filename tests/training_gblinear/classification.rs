//! Classification training tests.
//!
//! Tests for binary and multiclass classification including:
//! - Binary classification with logistic loss
//! - Multiclass classification with softmax loss

use super::load_train_data;
use super::load_config;
use booste_rs::data::DataMatrix;
use booste_rs::training::linear::{LinearTrainer, LinearTrainerConfig};
use booste_rs::training::{LogisticLoss, SoftmaxLoss, Verbosity};

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
    let model = trainer.train(&data, &labels, &LogisticLoss);

    // Verify predictions are in reasonable range for logits
    let mut predictions = Vec::new();
    for row in 0..data.num_rows().min(10) {
        let features: Vec<f32> = (0..data.num_features())
            .map(|col| data.get(row, col).copied().unwrap_or(f32::NAN))
            .collect();
        let pred = model.predict_row(&features, &[0.0])[0];
        predictions.push(pred);
    }

    println!(
        "Binary classification predictions (logits): {:?}",
        predictions
    );

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
    let loss = SoftmaxLoss::new(num_class);
    let model = trainer.train_multiclass(&data, &labels, &loss);

    // Verify model has correct number of output groups
    assert_eq!(model.num_groups(), num_class);

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

    // With proper multiclass gradients, predictions for different classes should differ
    let all_same = predictions.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
    assert!(
        !all_same,
        "All class predictions are identical - multiclass not working properly"
    );

    // Compute training accuracy
    let mut correct = 0;
    for i in 0..data.num_rows() {
        let features: Vec<f32> = (0..data.num_features())
            .map(|col| data.get(i, col).copied().unwrap_or(0.0))
            .collect();
        let preds = model.predict_row(&features, &vec![0.0; num_class]);

        // Find predicted class (argmax)
        let predicted = preds
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let true_class = labels[i] as usize;
        if predicted == true_class {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / data.num_rows() as f64;
    println!("Multiclass training accuracy: {:.2}%", accuracy * 100.0);

    // Training accuracy should be reasonable for linear model
    // (better than random = 33.3% for 3 classes)
    assert!(
        accuracy > 0.4,
        "Training accuracy {} should be > 40%",
        accuracy
    );
}
