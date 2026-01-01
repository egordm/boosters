//! Classification training tests.
//!
//! Tests for binary and multiclass classification including:
//! - Binary classification with logistic loss
//! - Multiclass classification with softmax loss

use super::{load_config, load_train_data, make_dataset};
use boosters::data::{Dataset, TargetsView, WeightsView};
use boosters::training::{
    GBLinearParams, GBLinearTrainer, LogLoss, LogisticLoss, MulticlassLogLoss, SoftmaxLoss,
    UpdateStrategy, Verbosity,
};
use ndarray::Array2;

/// Test binary classification training produces reasonable predictions.
#[test]
fn train_binary_classification() {
    let (data, labels) = load_train_data("binary_classification");
    let config = load_config("binary_classification");
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());

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

    let trainer = GBLinearTrainer::new(LogisticLoss, LogLoss, params);
    let model = trainer
        .train(&train, targets_view, WeightsView::None, None)
        .unwrap();

    // Verify predictions are in reasonable range for logits.
    // Data is feature-major [n_features, n_samples]
    let pred_dataset = Dataset::from_array(data.view(), None, None);
    let output = model.predict(&pred_dataset);

    // output is Array2<f32> with shape [n_groups, n_samples]
    let predictions: Vec<f32> = output.row(0).iter().copied().take(10).collect();

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
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());

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

    let trainer = GBLinearTrainer::new(SoftmaxLoss::new(num_class), MulticlassLogLoss, params);
    let model = trainer
        .train(&train, targets_view, WeightsView::None, None)
        .unwrap();

    // Verify model has correct number of output groups
    assert_eq!(model.n_groups(), num_class);

    // Verify predictions exist for all classes
    // Data is feature-major [n_features, n_samples]
    let pred_dataset = Dataset::from_array(data.view(), None, None);
    let output = model.predict(&pred_dataset);

    // output is Array2<f32> with shape [n_groups, n_samples]
    // Get first sample's predictions (column 0)
    let predictions: Vec<f32> = output.column(0).iter().copied().collect();

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

    // Compute training accuracy (argmax over logits).
    use boosters::training::{MetricFn, MulticlassAccuracy};

    let pred_dataset = Dataset::from_array(data.view(), None, None);
    let output = model.predict(&pred_dataset);
    let n_samples = output.ncols();
    let pred_classes: Vec<f32> = (0..n_samples)
        .map(|sample| {
            let sample_preds = output.column(sample);
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in sample_preds.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            best_idx as f32
        })
        .collect();

    // Metrics expect shape [n_groups, n_samples] - for class predictions, n_groups=1
    let pred_arr = Array2::from_shape_vec((1, n_samples), pred_classes).unwrap();
    let targets_2d = Array2::from_shape_vec((1, labels.len()), labels.clone()).unwrap();
    let metric_targets = TargetsView::new(targets_2d.view());
    let accuracy = MulticlassAccuracy.compute(pred_arr.view(), metric_targets, WeightsView::None);

    // Training accuracy should be reasonable for linear model
    // (better than random = 33.3% for 3 classes)
    assert!(
        accuracy > 0.4,
        "Training accuracy {} should be > 40%",
        accuracy
    );
}
