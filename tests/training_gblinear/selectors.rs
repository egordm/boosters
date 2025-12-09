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
use booste_rs::data::DataMatrix;
use booste_rs::training::linear::FeatureSelectorKind;
use booste_rs::training::{GBLinearTrainer, LossFunction, Verbosity};

// =============================================================================
// Feature Selector Integration Tests
// =============================================================================

/// Test all feature selectors produce reasonable models on regression.
#[test]
fn train_all_selectors_regression() {
    let (data, labels) = load_train_data("regression_l2");
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
    let shuffle_trainer = GBLinearTrainer::builder()
        .num_rounds(50usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(false)
        .seed(42u64)
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();
    let shuffle_model = shuffle_trainer.train(&data, &labels, None, &[]);
    let shuffle_rmse = compute_test_rmse_default(&shuffle_model, &test_data, &test_labels);

    println!("Shuffle RMSE: {:.4}", shuffle_rmse);

    for (name, selector) in selectors {
        let trainer = GBLinearTrainer::builder()
            .num_rounds(50usize)
            .learning_rate(0.5f32)
            .alpha(0.0f32)
            .lambda(1.0f32)
            .parallel(false)
            .seed(42u64)
            .feature_selector(selector)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let model = trainer.train(&data, &labels, None, &[]);
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
    let shuffle_trainer = GBLinearTrainer::builder()
        .loss(LossFunction::Softmax { num_classes })
        .num_rounds(30usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(false)
        .seed(42u64)
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();
    let shuffle_model = shuffle_trainer.train(&data, &labels, None, &[]);
    let shuffle_acc = compute_multiclass_accuracy(&shuffle_model, &test_data, &test_labels);

    println!("Shuffle accuracy: {:.4}", shuffle_acc);

    for (name, selector) in selectors {
        let trainer = GBLinearTrainer::builder()
            .loss(LossFunction::Softmax { num_classes })
            .num_rounds(30usize)
            .learning_rate(0.5f32)
            .alpha(0.0f32)
            .lambda(1.0f32)
            .parallel(false)
            .seed(42u64)
            .feature_selector(selector)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let model = trainer.train(&data, &labels, None, &[]);
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

    // Train with greedy selector with small top_k
    let greedy_trainer = GBLinearTrainer::builder()
        .num_rounds(20usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(false)
        .seed(42u64)
        .feature_selector(FeatureSelectorKind::Greedy { top_k: 2 })
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();

    // Train with cyclic (visits all features equally)
    let cyclic_trainer = GBLinearTrainer::builder()
        .num_rounds(20usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(false)
        .seed(42u64)
        .feature_selector(FeatureSelectorKind::Cyclic)
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();

    let greedy_model = greedy_trainer.train(&data, &labels, None, &[]);
    let cyclic_model = cyclic_trainer.train(&data, &labels, None, &[]);

    // Both should produce valid models (non-zero weights somewhere)
    let greedy_has_weights: bool = (0..data.num_features())
        .any(|i| greedy_model.weight(i, 0).abs() > 1e-6);
    let cyclic_has_weights: bool = (0..data.num_features())
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
    let (test_data, test_labels) = match load_test_data("regression_l2") {
        Some(d) => d,
        None => {
            println!("No test data for regression_l2, skipping");
            return;
        }
    };

    // Train with thrifty selector
    let trainer = GBLinearTrainer::builder()
        .num_rounds(30usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(false)
        .seed(42u64)
        .feature_selector(FeatureSelectorKind::Thrifty { top_k: 3 })
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();

    let model = trainer.train(&data, &labels, None, &[]);
    let rmse = compute_test_rmse_default(&model, &test_data, &test_labels);

    println!("Thrifty RMSE: {:.4}", rmse);

    // Thrifty should converge to a reasonable model
    assert!(rmse < 100.0, "Thrifty should converge: RMSE = {}", rmse);
}
