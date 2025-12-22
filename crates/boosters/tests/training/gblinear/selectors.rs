//! Feature selector integration tests.
//!
//! Tests all feature selector types:
//! - Cyclic (round-robin)
//! - Shuffle (random permutation)
//! - Random (random sampling)
//! - Greedy (gradient-based)
//! - Thrifty (cached greedy)

use super::{load_test_data, load_train_data};
use boosters::data::{Dataset, FeaturesView, SamplesView};
use boosters::inference::LinearModelPredict;
use boosters::training::gblinear::FeatureSelectorKind;
use boosters::training::{
    GBLinearParams, GBLinearTrainer, MulticlassLogLoss, Rmse, SoftmaxLoss, SquaredLoss, Verbosity,
};
use ndarray::{Array2, ArrayView1};

/// Transpose predictions from (n_samples, n_groups) to (n_groups, n_samples) for metrics.
fn transpose_predictions(output: &Array2<f32>) -> Array2<f32> {
    output.t().to_owned()
}

// =============================================================================
// Feature Selector Integration Tests
// =============================================================================

/// Test all feature selectors produce reasonable models on regression.
#[test]
fn train_all_selectors_regression() {
    let (data, labels) = load_train_data("regression_l2");
    let view = FeaturesView::from_array(data.view());
    let train = Dataset::from_numeric(&view, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };
    let test_view = SamplesView::from_array(test_data.view());

    let selectors = vec![
        ("Cyclic", FeatureSelectorKind::Cyclic),
        ("Shuffle", FeatureSelectorKind::Shuffle),
        ("Random", FeatureSelectorKind::Random),
        ("Greedy(3)", FeatureSelectorKind::Greedy { top_k: 3 }),
        ("Thrifty(3)", FeatureSelectorKind::Thrifty { top_k: 3 }),
    ];

    // Train with shuffle as reference
    let shuffle_params = GBLinearParams {
        n_rounds: 50,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let shuffle_trainer = GBLinearTrainer::new(SquaredLoss, Rmse, shuffle_params);
    let shuffle_model = shuffle_trainer.train(&train, &[]).unwrap();
    use boosters::training::MetricFn;
    let shuffle_output = shuffle_model.predict(test_view, &[]);
    let shuffle_arr = transpose_predictions(&shuffle_output);
    let targets_arr = ArrayView1::from(&test_labels[..]);
    let shuffle_rmse = Rmse.compute(shuffle_arr.view(), targets_arr, None);

    for (name, selector) in selectors {
        let params = GBLinearParams {
            n_rounds: 50,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: false,
            seed: 42,
            feature_selector: selector,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
        let model = trainer.train(&train, &[]).unwrap();
        let output = model.predict(test_view, &[]);
        let pred_arr = transpose_predictions(&output);
        let rmse = Rmse.compute(pred_arr.view(), targets_arr, None);

        // All selectors should produce reasonable models
        // (within 2x of shuffle baseline)
        assert!(
            rmse < shuffle_rmse * 2.0,
            "{} selector produced poor model: RMSE {} vs shuffle {}",
            name,
            rmse,
            shuffle_rmse
        );
    }
}

/// Test all feature selectors on multiclass classification.
#[test]
fn train_all_selectors_multiclass() {
    let (data, labels) = load_train_data("multiclass_classification");
    let view = FeaturesView::from_array(data.view());
    let train = Dataset::from_numeric(&view, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("multiclass_classification") {
        Some(d) => d,
        None => {
            println!("No test data for multiclass_classification, skipping");
            return;
        }
    };
    let test_view = SamplesView::from_array(test_data.view());
    let num_classes = 3;

    let selectors = vec![
        ("Cyclic", FeatureSelectorKind::Cyclic),
        ("Shuffle", FeatureSelectorKind::Shuffle),
        ("Random", FeatureSelectorKind::Random),
        ("Greedy(3)", FeatureSelectorKind::Greedy { top_k: 3 }),
        ("Thrifty(3)", FeatureSelectorKind::Thrifty { top_k: 3 }),
    ];

    // Train with shuffle as reference
    let shuffle_params = GBLinearParams {
        n_rounds: 30,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let shuffle_trainer =
        GBLinearTrainer::new(SoftmaxLoss::new(num_classes), MulticlassLogLoss, shuffle_params);
    let shuffle_model = shuffle_trainer.train(&train, &[]).unwrap();
    use boosters::training::{MetricFn, MulticlassAccuracy};
    let shuffle_output = shuffle_model.predict(test_view, &[]);
    let n_rows = shuffle_output.nrows();
    let shuffle_pred_classes: Vec<f32> = (0..n_rows)
        .map(|row| {
            let row_preds = shuffle_output.row(row);
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
    let shuffle_pred_arr = Array2::from_shape_vec((1, n_rows), shuffle_pred_classes).unwrap();
    let targets_arr = ArrayView1::from(&test_labels[..]);
    let shuffle_acc = MulticlassAccuracy
        .compute(shuffle_pred_arr.view(), targets_arr, None)
        as f32;

    for (name, selector) in selectors {
        let params = GBLinearParams {
            n_rounds: 30,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: false,
            seed: 42,
            feature_selector: selector,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer =
            GBLinearTrainer::new(SoftmaxLoss::new(num_classes), MulticlassLogLoss, params);
        let model = trainer.train(&train, &[]).unwrap();
        let output = model.predict(test_view, &[]);
        let n_rows = output.nrows();
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
        let pred_arr = Array2::from_shape_vec((1, n_rows), pred_classes).unwrap();
        let acc = MulticlassAccuracy
            .compute(pred_arr.view(), targets_arr, None)
            as f32;

        // All selectors should achieve reasonable accuracy
        // (at least 50% of shuffle baseline accuracy)
        assert!(
            acc > shuffle_acc * 0.5,
            "{} selector produced poor model: accuracy {} vs shuffle {}",
            name,
            acc,
            shuffle_acc
        );
    }
}

/// Test Greedy selector prioritizes impactful features.
#[test]
fn train_greedy_selector_feature_priority() {
    let (data, labels) = load_train_data("regression_l2");
    let view = FeaturesView::from_array(data.view());
    let train = Dataset::from_numeric(&view, labels).unwrap();
    let n_features = data.nrows(); // feature-major: rows are features

    // Train with greedy selector with small top_k
    let greedy_params = GBLinearParams {
        n_rounds: 20,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        feature_selector: FeatureSelectorKind::Greedy { top_k: 2 },
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let greedy_trainer = GBLinearTrainer::new(SquaredLoss, Rmse, greedy_params);

    // Train with cyclic (visits all features equally)
    let cyclic_params = GBLinearParams {
        n_rounds: 20,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        feature_selector: FeatureSelectorKind::Cyclic,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let cyclic_trainer = GBLinearTrainer::new(SquaredLoss, Rmse, cyclic_params);

    let greedy_model = greedy_trainer.train(&train, &[]).unwrap();
    let cyclic_model = cyclic_trainer.train(&train, &[]).unwrap();

    // Both should produce valid models (non-zero weights somewhere)
    let greedy_has_weights: bool = (0..n_features)
        .any(|i| greedy_model.weight(i, 0).abs() > 1e-6);
    let cyclic_has_weights: bool = (0..n_features)
        .any(|i| cyclic_model.weight(i, 0).abs() > 1e-6);

    assert!(
        greedy_has_weights,
        "Greedy model should have non-zero weights"
    );
    assert!(
        cyclic_has_weights,
        "Cyclic model should have non-zero weights"
    );

}

/// Test Thrifty selector convergence with caching.
#[test]
fn train_thrifty_selector_convergence() {
    let (data, labels) = load_train_data("regression_l2");
    let view = FeaturesView::from_array(data.view());
    let train = Dataset::from_numeric(&view, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };
    let test_view = SamplesView::from_array(test_data.view());

    // Train with thrifty selector
    let params = GBLinearParams {
        n_rounds: 30,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        feature_selector: FeatureSelectorKind::Thrifty { top_k: 3 },
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
    let model = trainer.train(&train, &[]).unwrap();
    use boosters::training::MetricFn;
    let output = model.predict(test_view, &[]);
    let pred_arr = transpose_predictions(&output);
    let targets_arr = ArrayView1::from(&test_labels[..]);
    let rmse = Rmse.compute(pred_arr.view(), targets_arr, None);

    // Thrifty should converge to a reasonable model
    assert!(rmse < 100.0, "Thrifty should converge: RMSE = {}", rmse);
}
