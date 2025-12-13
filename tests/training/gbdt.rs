//! GBDT training integration tests.
//!
//! Focused on behavior and invariants (not default params or superficial shapes).

use booste_rs::data::{BinnedDatasetBuilder, ColMatrix};
use booste_rs::training::{GBDTParams, GBDTTrainer, GrowthStrategy, SquaredLoss};

#[test]
fn train_rejects_invalid_targets_len() {
    let features = vec![0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0];
    let col_matrix = ColMatrix::from_vec(features, 4, 2);
    let targets: Vec<f32> = vec![1.0, 2.0]; // Too few targets

    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 16)
        .build()
        .expect("Failed to build binned dataset");

    let trainer = GBDTTrainer::new(SquaredLoss, GBDTParams::default());
    let result = trainer.train(&dataset, &targets, &[]);

    assert!(result.is_none());
}

#[test]
fn trained_model_improves_over_base_score_on_simple_problem() {
    // Simple linear relationship: y = x + 0.5
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

    forest
        .validate()
        .expect("trained forest should be structurally valid");

    let base = forest.base_score()[0];

    let mut base_error_sum = 0.0f32;
    let mut pred_error_sum = 0.0f32;

    for row in 0..n_samples {
        let x = col_matrix.col_slice(0)[row];
        let pred = forest.predict_row(&[x])[0];
        let target = targets[row];

        base_error_sum += (base - target).powi(2);
        pred_error_sum += (pred - target).powi(2);
    }

    assert!(
        pred_error_sum < base_error_sum,
        "Model should improve over base score"
    );
}

#[test]
fn trained_model_improves_over_base_score_on_medium_problem() {
    // Synthetic regression data
    let n_samples = 200;
    let n_features = 3;

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            features.push(((i * (j + 1)) % 100) as f32 / 10.0);
        }
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
        cache_size: 32,
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, params);
    let forest = trainer.train(&dataset, &targets, &[]).unwrap();

    forest
        .validate()
        .expect("trained forest should be structurally valid");

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

    assert!(
        pred_error_sum < base_error_sum,
        "Model should improve over base score"
    );
}
