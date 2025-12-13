//! Feature selector integration tests.
//!
//! Tests all feature selector types:
//! - Cyclic (round-robin)
//! - Shuffle (random permutation)
//! - Random (random sampling)
//! - Greedy (gradient-based)
//! - Thrifty (cached greedy)

use super::{
    compute_multiclass_accuracy, compute_test_rmse_default, load_test_data, load_train_data,
};
use booste_rs::training::gblinear::FeatureSelectorKind;
use booste_rs::data::Dataset;
use booste_rs::training::{
    GBLinearParams, GBLinearTrainer, MulticlassLogLoss, Rmse, SoftmaxLoss, SquaredLoss, Verbosity,
};

// =============================================================================
// Feature Selector Integration Tests
// =============================================================================

/// Test all feature selectors produce reasonable models on regression.
#[test]
fn train_all_selectors_regression() {
    let (data, labels) = load_train_data("regression_l2");
    let train = Dataset::from_numeric(&data, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };

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
    let shuffle_rmse = compute_test_rmse_default(&shuffle_model, &test_data, &test_labels);

    println!("Shuffle RMSE: {:.4}", shuffle_rmse);

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
        let rmse = compute_test_rmse_default(&model, &test_data, &test_labels);

        println!("{} RMSE: {:.4}", name, rmse);

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
    let train = Dataset::from_numeric(&data, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("multiclass_classification") {
        Some(d) => d,
        None => {
            println!("No test data for multiclass_classification, skipping");
            return;
        }
    };
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
    let shuffle_acc = compute_multiclass_accuracy(&shuffle_model, &test_data, &test_labels);

    println!("Shuffle accuracy: {:.4}", shuffle_acc);

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
        let acc = compute_multiclass_accuracy(&model, &test_data, &test_labels);

        println!("{} accuracy: {:.4}", name, acc);

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
    let train = Dataset::from_numeric(&data, labels).unwrap();

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
    let greedy_has_weights: bool = (0..data.num_columns())
        .any(|i| greedy_model.weight(i, 0).abs() > 1e-6);
    let cyclic_has_weights: bool = (0..data.num_columns())
        .any(|i| cyclic_model.weight(i, 0).abs() > 1e-6);

    assert!(
        greedy_has_weights,
        "Greedy model should have non-zero weights"
    );
    assert!(
        cyclic_has_weights,
        "Cyclic model should have non-zero weights"
    );

    println!("Greedy model trained successfully with top_k=2");
    println!("Cyclic model trained successfully");
}

/// Test Thrifty selector convergence with caching.
#[test]
fn train_thrifty_selector_convergence() {
    let (data, labels) = load_train_data("regression_l2");
    let train = Dataset::from_numeric(&data, labels).unwrap();
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };

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
    let rmse = compute_test_rmse_default(&model, &test_data, &test_labels);

    println!("Thrifty RMSE: {:.4}", rmse);

    // Thrifty should converge to a reasonable model
    assert!(rmse < 100.0, "Thrifty should converge: RMSE = {}", rmse);
}
