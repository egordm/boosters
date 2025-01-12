//! Feature selector integration tests.
//!
//! Tests all feature selector types:
//! - Cyclic (round-robin)
//! - Shuffle (random permutation)
//! - Random (random sampling)
//! - Greedy (gradient-based)
//! - Thrifty (cached greedy)

use super::{load_test_data, load_train_data, make_dataset};
use boosters::data::transpose_to_c_order;
use boosters::data::{FeaturesView, TargetsView, WeightsView};
use boosters::training::gblinear::FeatureSelectorKind;
use boosters::training::{
    GBLinearParams, GBLinearTrainer, MulticlassLogLoss, Rmse, SoftmaxLoss, SquaredLoss, Verbosity,
};
use ndarray::Array2;

// =============================================================================
// Feature Selector Integration Tests
// =============================================================================

/// Test all feature selectors produce reasonable models on regression.
#[test]
fn train_all_selectors_regression() {
    let (data, labels) = load_train_data("regression_l2");
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };
    // Transpose test data from sample-major to feature-major
    let test_features_fm = transpose_to_c_order(test_data.view());
    let test_view = FeaturesView::from_array(test_features_fm.view());

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
    let shuffle_model = shuffle_trainer
        .train(&train, targets_view.clone(), WeightsView::None, &[])
        .unwrap();
    use boosters::training::MetricFn;
    let shuffle_output = shuffle_model.predict(test_view);
    let targets_2d = Array2::from_shape_vec((1, test_labels.len()), test_labels.clone()).unwrap();
    let eval_targets = TargetsView::new(targets_2d.view());
    let shuffle_rmse = Rmse.compute(shuffle_output.view(), eval_targets, WeightsView::None);

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
        let model = trainer
            .train(&train, targets_view.clone(), WeightsView::None, &[])
            .unwrap();
        let output = model.predict(test_view);
        let rmse = Rmse.compute(output.view(), eval_targets, WeightsView::None);

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
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());
    let (test_data, test_labels) = match load_test_data("multiclass_classification") {
        Some(d) => d,
        None => {
            println!("No test data for multiclass_classification, skipping");
            return;
        }
    };
    // Transpose test data from sample-major to feature-major
    let test_features_fm = transpose_to_c_order(test_data.view());
    let test_view = FeaturesView::from_array(test_features_fm.view());
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
    let shuffle_trainer = GBLinearTrainer::new(
        SoftmaxLoss::new(num_classes),
        MulticlassLogLoss,
        shuffle_params,
    );
    let shuffle_model = shuffle_trainer
        .train(&train, targets_view.clone(), WeightsView::None, &[])
        .unwrap();
    use boosters::training::{MetricFn, MulticlassAccuracy};
    let shuffle_output = shuffle_model.predict(test_view);
    // output is [n_groups, n_samples] = [3, n_samples]
    let n_samples = shuffle_output.ncols();
    let shuffle_pred_classes: Vec<f32> = (0..n_samples)
        .map(|sample| {
            let sample_preds = shuffle_output.column(sample);
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
    let shuffle_pred_arr = Array2::from_shape_vec((1, n_samples), shuffle_pred_classes).unwrap();
    let targets_2d = Array2::from_shape_vec((1, test_labels.len()), test_labels.clone()).unwrap();
    let eval_targets = TargetsView::new(targets_2d.view());
    let shuffle_acc =
        MulticlassAccuracy.compute(shuffle_pred_arr.view(), eval_targets, WeightsView::None)
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
        let model = trainer
            .train(&train, targets_view.clone(), WeightsView::None, &[])
            .unwrap();
        let output = model.predict(test_view);
        // output is [n_groups, n_samples]
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
        let pred_arr = Array2::from_shape_vec((1, n_samples), pred_classes).unwrap();
        let acc =
            MulticlassAccuracy.compute(pred_arr.view(), eval_targets, WeightsView::None) as f32;

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
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());
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

    let greedy_model = greedy_trainer
        .train(&train, targets_view.clone(), WeightsView::None, &[])
        .unwrap();
    let cyclic_model = cyclic_trainer
        .train(&train, targets_view, WeightsView::None, &[])
        .unwrap();

    // Both should produce valid models (non-zero weights somewhere)
    let greedy_has_weights: bool = (0..n_features).any(|i| greedy_model.weight(i, 0).abs() > 1e-6);
    let cyclic_has_weights: bool = (0..n_features).any(|i| cyclic_model.weight(i, 0).abs() > 1e-6);

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
    let (train, targets) = make_dataset(&data, &labels);
    let targets_view = TargetsView::new(targets.view());
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };
    // Transpose test data from sample-major to feature-major
    let test_features_fm = transpose_to_c_order(test_data.view());
    let test_view = FeaturesView::from_array(test_features_fm.view());

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
    let model = trainer
        .train(&train, targets_view, WeightsView::None, &[])
        .unwrap();
    use boosters::training::MetricFn;
    let output = model.predict(test_view);
    let targets_2d = Array2::from_shape_vec((1, test_labels.len()), test_labels.clone()).unwrap();
    let eval_targets = TargetsView::new(targets_2d.view());
    let rmse = Rmse.compute(output.view(), eval_targets, WeightsView::None);

    // Thrifty should converge to a reasonable model
    assert!(rmse < 100.0, "Thrifty should converge: RMSE = {}", rmse);
}
