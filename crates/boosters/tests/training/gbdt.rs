//! GBDT training integration tests.
//!
//! Focused on behavior and invariants (not default params or superficial shapes).

use boosters::data::{BinMapper, BinnedDatasetBuilder, ColMatrix, GroupLayout, GroupStrategy, MissingType};
use boosters::repr::gbdt::{TreeView, SplitType};
use boosters::training::{GBDTParams, GBDTTrainer, GrowthStrategy, Rmse, SquaredLoss};
use boosters::Parallelism;
use ndarray::ArrayView1;

#[test]
fn train_rejects_invalid_targets_len() {
    let features = vec![0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0];
    let col_matrix = ColMatrix::from_vec(features, 4, 2);
    let targets: Vec<f32> = vec![1.0, 2.0]; // Too few targets

    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 16)
        .build()
        .expect("Failed to build binned dataset");

    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, GBDTParams::default());
    let targets_view = ArrayView1::from(&targets[..]);
    let empty_weights: ArrayView1<f32> = ArrayView1::from(&[][..]);
    let result = trainer.train(&dataset, targets_view, empty_weights, &[], Parallelism::Sequential);

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

    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
    let targets_view = ArrayView1::from(&targets[..]);
    let empty_weights: ArrayView1<f32> = ArrayView1::from(&[][..]);
    let forest = trainer.train(&dataset, targets_view, empty_weights, &[], Parallelism::Sequential).unwrap();

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

    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
    let targets_view = ArrayView1::from(&targets[..]);
    let empty_weights: ArrayView1<f32> = ArrayView1::from(&[][..]);
    let forest = trainer.train(&dataset, targets_view, empty_weights, &[], Parallelism::Sequential).unwrap();

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

/// Test that training with categorical features produces categorical splits.
///
/// This is an end-to-end verification of the categorical feature pipeline:
/// 1. Dataset with categorical column → BinnedDataset with is_categorical flag
/// 2. TreeGrower receives feature_types and dispatches to categorical split finder
/// 3. Resulting tree has SplitType::Categorical nodes
/// 4. Inference correctly handles categorical splits
#[test]
fn train_with_categorical_features_produces_categorical_splits() {
    // Create a dataset where a categorical feature is the only useful predictor.
    // 4 categories: 0, 1, 2, 3
    // Target: category 0 or 2 → low value (1.0), category 1 or 3 → high value (10.0)
    //
    // This forces the model to use categorical splits since there's no
    // numeric threshold that separates the groups.
    
    let n_samples = 40;
    let categories: Vec<u32> = (0..n_samples).map(|i| (i % 4) as u32).collect();
    let targets: Vec<f32> = categories
        .iter()
        .map(|&c| if c == 0 || c == 2 { 1.0 } else { 10.0 })
        .collect();

    // Build binned dataset with categorical feature
    let bins: Vec<u32> = categories.iter().map(|&c| c as u32).collect();
    let mapper = BinMapper::categorical(
        vec![0, 1, 2, 3],  // category values
        MissingType::None,
        0,   // missing bin
        0,   // zero bin
        0.0, // zero value
    );

    let dataset = BinnedDatasetBuilder::new()
        .add_binned(bins, mapper)
        .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::ColumnMajor })
        .build()
        .expect("Failed to build binned dataset");

    // Verify the dataset reports the feature as categorical
    assert!(dataset.is_categorical(0), "Feature 0 should be categorical");

    let params = GBDTParams {
        n_trees: 5,
        learning_rate: 0.3,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
        max_onehot_cats: 4,  // Allow one-hot splits for this test
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
    let targets_view = ArrayView1::from(&targets[..]);
    let empty_weights: ArrayView1<f32> = ArrayView1::from(&[][..]);
    let forest = trainer.train(&dataset, targets_view, empty_weights, &[], Parallelism::Sequential).unwrap();

    forest
        .validate()
        .expect("trained forest should be structurally valid");

    // Check that at least one tree has a categorical split
    let has_categorical_split = forest.trees().any(|tree| {
        tree.has_categorical()
    });

    assert!(
        has_categorical_split,
        "At least one tree should have a categorical split"
    );

    // Verify the first tree has categorical split nodes
    let first_tree = forest.trees().next().expect("should have at least one tree");
    let mut found_categorical = false;
    for node_idx in 0..first_tree.n_nodes() as u32 {
        if !first_tree.is_leaf(node_idx) {
            if matches!(first_tree.split_type(node_idx), SplitType::Categorical) {
                found_categorical = true;
                break;
            }
        }
    }
    assert!(
        found_categorical,
        "First tree should contain at least one categorical split node"
    );

    // Verify predictions are reasonable
    // Categories 0, 2 should predict low, categories 1, 3 should predict high
    let pred_cat0 = forest.predict_row(&[0.0])[0];
    let pred_cat1 = forest.predict_row(&[1.0])[0];
    let pred_cat2 = forest.predict_row(&[2.0])[0];
    let pred_cat3 = forest.predict_row(&[3.0])[0];

    // Low predictions (categories 0, 2) should be < 5.5 (midpoint)
    // High predictions (categories 1, 3) should be > 5.5
    assert!(
        pred_cat0 < 5.5 && pred_cat2 < 5.5,
        "Categories 0 and 2 should predict low values, got {} and {}",
        pred_cat0, pred_cat2
    );
    assert!(
        pred_cat1 > 5.5 && pred_cat3 > 5.5,
        "Categories 1 and 3 should predict high values, got {} and {}",
        pred_cat1, pred_cat3
    );
}
