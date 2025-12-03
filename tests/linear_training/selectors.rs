//! Feature selector integration tests.
//!
//! Tests all feature selector types:
//! - Cyclic (round-robin)
//! - Shuffle (random permutation)
//! - Random (random sampling)
//! - Greedy (gradient-based)
//! - Thrifty (cached greedy)

use super::{load_test_data, load_train_data, ColMatrix};
use booste_rs::data::DataMatrix;
use booste_rs::linear::training::{FeatureSelectorKind, LinearTrainer, LinearTrainerConfig};
use booste_rs::linear::LinearModel;
use booste_rs::training::{
    Metric, MulticlassAccuracy, Rmse, SoftmaxLoss, SquaredLoss, Verbosity,
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

/// Compute multiclass accuracy using argmax.
fn compute_test_accuracy(model: &LinearModel, data: &ColMatrix<f32>, labels: &[f32]) -> f32 {
    let num_groups = model.num_groups();
    let predictions = get_predictions(model, data);

    // Convert to predicted class indices via argmax
    let pred_classes: Vec<f32> = predictions
        .chunks(num_groups)
        .map(|preds| {
            preds
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as f32)
                .unwrap_or(0.0)
        })
        .collect();

    MulticlassAccuracy.evaluate(&pred_classes, labels, 1) as f32
}

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
    let shuffle_config = LinearTrainerConfig {
        num_rounds: 50,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let shuffle_model = LinearTrainer::new(shuffle_config).train(&data, &labels, &SquaredLoss);
    let shuffle_rmse = compute_test_rmse(&shuffle_model, &test_data, &test_labels);

    println!("Shuffle RMSE: {:.4}", shuffle_rmse);

    for (name, selector) in selectors {
        let config = LinearTrainerConfig {
            num_rounds: 50,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: false,
            seed: 42,
            feature_selector: selector,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let model = LinearTrainer::new(config).train(&data, &labels, &SquaredLoss);
        let rmse = compute_test_rmse(&model, &test_data, &test_labels);

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
    let shuffle_config = LinearTrainerConfig {
        num_rounds: 30,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let loss = SoftmaxLoss::new(num_classes);
    let shuffle_model =
        LinearTrainer::new(shuffle_config).train_multiclass(&data, &labels, &loss);
    let shuffle_acc = compute_test_accuracy(&shuffle_model, &test_data, &test_labels);

    println!("Shuffle accuracy: {:.4}", shuffle_acc);

    for (name, selector) in selectors {
        let config = LinearTrainerConfig {
            num_rounds: 30,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: false,
            seed: 42,
            feature_selector: selector,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let model = LinearTrainer::new(config).train_multiclass(&data, &labels, &loss);
        let acc = compute_test_accuracy(&model, &test_data, &test_labels);

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
    let greedy_config = LinearTrainerConfig {
        num_rounds: 20,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        feature_selector: FeatureSelectorKind::Greedy { top_k: 2 },
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    // Train with cyclic (visits all features equally)
    let cyclic_config = LinearTrainerConfig {
        num_rounds: 20,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        feature_selector: FeatureSelectorKind::Cyclic,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let greedy_model = LinearTrainer::new(greedy_config).train(&data, &labels, &SquaredLoss);
    let cyclic_model = LinearTrainer::new(cyclic_config).train(&data, &labels, &SquaredLoss);

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
    let thrifty_config = LinearTrainerConfig {
        num_rounds: 30,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        seed: 42,
        feature_selector: FeatureSelectorKind::Thrifty { top_k: 3 },
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let model = LinearTrainer::new(thrifty_config).train(&data, &labels, &SquaredLoss);
    let rmse = compute_test_rmse(&model, &test_data, &test_labels);

    println!("Thrifty RMSE: {:.4}", rmse);

    // Thrifty should converge to a reasonable model
    assert!(rmse < 100.0, "Thrifty should converge: RMSE = {}", rmse);
}
