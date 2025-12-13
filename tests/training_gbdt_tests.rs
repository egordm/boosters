//! Integration tests for GBDT training.
//!
//! These tests validate that training produces reasonable results.

use booste_rs::data::{BinnedDatasetBuilder, ColMatrix};
use booste_rs::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, SquaredLoss};

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_gbdt_params_default() {
    let params = GBDTParams::default();

    assert_eq!(params.n_trees, 100);
    assert!((params.learning_rate - 0.3).abs() < 1e-6);
    assert_eq!(
        params.growth_strategy,
        GrowthStrategy::DepthWise { max_depth: 6 }
    );
}

#[test]
fn test_gbdt_train_single_tree() {
    let features = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // feature 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // feature 1
    ];
    let col_matrix = ColMatrix::from_vec(features, 8, 2);
    let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Build binned dataset
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 16)
        .build()
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 1,
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let forest = trainer.train(&dataset, &targets, &[]).unwrap();

    assert_eq!(forest.n_trees(), 1);
    assert_eq!(forest.n_groups(), 1);
}

#[test]
fn test_gbdt_train_multiple_trees() {
    let features = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // feature 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // feature 1
    ];
    let col_matrix = ColMatrix::from_vec(features, 8, 2);
    let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 16)
        .build()
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 10,
        learning_rate: 0.1,
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let forest = trainer.train(&dataset, &targets, &[]).unwrap();

    assert_eq!(forest.n_trees(), 10);
}

#[test]
fn test_gbdt_train_with_regularization() {
    let features = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    let col_matrix = ColMatrix::from_vec(features, 8, 2);
    let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 16)
        .build()
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 5,
        gain: GainParams {
            reg_lambda: 10.0,
            min_gain: 0.5,
            ..Default::default()
        },
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let forest = trainer.train(&dataset, &targets, &[]).unwrap();

    assert_eq!(forest.n_trees(), 5);
}

#[test]
fn test_gbdt_train_with_weights() {
    let features = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    let col_matrix = ColMatrix::from_vec(features, 8, 2);
    let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weights: Vec<f32> = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 16)
        .build()
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 5,
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let forest = trainer.train(&dataset, &targets, &weights).unwrap();

    assert_eq!(forest.n_trees(), 5);
}

#[test]
fn test_gbdt_leaf_wise_growth() {
    let features = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    let col_matrix = ColMatrix::from_vec(features, 8, 2);
    let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 16)
        .build()
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 3,
        growth_strategy: GrowthStrategy::LeafWise { max_leaves: 8 },
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let forest = trainer.train(&dataset, &targets, &[]).unwrap();

    assert_eq!(forest.n_trees(), 3);
}

#[test]
fn test_gbdt_train_invalid_targets() {
    let features = vec![0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0];
    let col_matrix = ColMatrix::from_vec(features, 4, 2);
    let targets: Vec<f32> = vec![1.0, 2.0]; // Too few targets

    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 16)
        .build()
        .expect("Failed to build binned dataset");

    let params = GBDTParams::default();

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let result = trainer.train(&dataset, &targets, &[]);

    assert!(result.is_none());
}

// =============================================================================
// Prediction Tests (using binned row views)
// =============================================================================

#[test]
fn test_gbdt_predictions_reasonable() {
    // Create a simple linear relationship: y ≈ x
    let n_samples = 100;
    let features: Vec<f32> = (0..n_samples).map(|i| i as f32 / 10.0).collect();
    let targets: Vec<f32> = features.iter().map(|&x| x + 0.5).collect();

    let col_matrix = ColMatrix::from_vec(features.clone(), n_samples, 1);
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 64)
        .build()
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 50,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let forest = trainer.train(&dataset, &targets, &[]).unwrap();

    // Predict on training data using the inference forest
    let mut predictions = Vec::with_capacity(n_samples);
    for row in 0..n_samples {
        let x = col_matrix.col_slice(0)[row];
        let pred = forest.predict_row(&[x])[0];
        predictions.push(pred);
    }

    // Compute RMSE - should be reasonably low for this simple problem
    let mse: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / n_samples as f32;
    let rmse = mse.sqrt();

    // RMSE should be < 1.0 for this simple linear problem
    assert!(
        rmse < 1.0,
        "RMSE {} is too high for simple linear problem",
        rmse
    );
}

// =============================================================================
// Larger Scale Tests
// =============================================================================

#[test]
fn test_gbdt_larger_dataset() {
    // Generate synthetic regression data
    let n_samples = 200;
    let n_features = 3;

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            features.push(((i * (j + 1)) % 100) as f32 / 10.0);
        }
        // Simple target function
        let x0 = ((i * 1) % 100) as f32 / 10.0;
        let x1 = ((i * 2) % 100) as f32 / 10.0;
        targets.push(x0 + 0.5 * x1);
    }

    let col_matrix = ColMatrix::from_vec(features, n_samples, n_features);
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 64)
        .build()
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 5,
        learning_rate: 0.3,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
        cache_size: 32, // Larger cache to avoid eviction issues
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let forest = trainer.train(&dataset, &targets, &[]).unwrap();

    assert_eq!(forest.n_trees(), 5);

    // Verify predictions improve over base score
    let base = forest.base_score()[0];

    let mut base_error_sum = 0.0f32;
    let mut pred_error_sum = 0.0f32;

    for row in 0..n_samples {
        let mut row_features = Vec::with_capacity(n_features);
        for f in 0..n_features {
            row_features.push(col_matrix.col_slice(f)[row]);
        }

        let pred = forest.predict_row(&row_features)[0];
        let target = targets[row];

        base_error_sum += (base - target).powi(2);
        pred_error_sum += (pred - target).powi(2);
    }

    // Model predictions should be better than just using base score
    assert!(
        pred_error_sum < base_error_sum,
        "Model should improve over base score"
    );
}
