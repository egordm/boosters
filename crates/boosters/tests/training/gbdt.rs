//! GBDT training integration tests.
//!
//! Focused on behavior and invariants (not default params or superficial shapes).

use boosters::Parallelism;
use boosters::data::{BinnedDataset, BinningConfig, transpose_to_c_order};
use boosters::data::{Dataset, TargetsView, WeightsView};
use boosters::inference::gbdt::SimplePredictor;
use boosters::model::gbdt::{GBDTConfig, GBDTModel};
use boosters::repr::gbdt::{Forest, SplitType, TreeView};
use boosters::training::{GBDTParams, GBDTTrainer, GrowthStrategy, Metric, Objective};
use ndarray::{Array2, ArrayView2};

/// Predict a single row using the predictor.
fn predict_row(forest: &Forest, features: &[f32]) -> Vec<f32> {
    let predictor = SimplePredictor::new(forest);
    let mut output = vec![0.0; predictor.n_groups()];
    let sample = ndarray::ArrayView1::from(features);
    predictor.predict_row_into(sample, None, &mut output);
    output
}

#[test]
fn train_rejects_invalid_targets_len() {
    // Data is feature-major: [n_features, n_samples]
    // 2 features, 4 samples
    let features = Array2::from_shape_vec(
        (2, 4),
        vec![
            0.0, 1.0, 2.0, 3.0, // feature 0
            1.0, 2.0, 3.0, 4.0, // feature 1
        ],
    )
    .unwrap();
    let features_dataset = Dataset::from_array(features.view(), None, None);
    let targets: Vec<f32> = vec![1.0, 2.0]; // Too few targets

    let binning_config = BinningConfig::builder().max_bins(64).build();
    let dataset = BinnedDataset::from_dataset(&features_dataset, &binning_config)
        .expect("Failed to build binned dataset");

    let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, GBDTParams::default());
    let targets_2d = Array2::from_shape_vec((1, targets.len()), targets).unwrap();
    let targets_view = TargetsView::new(targets_2d.view());
    let result = trainer.train(
        &features_dataset,
        &dataset,
        targets_view,
        WeightsView::None,
        None,
        Parallelism::Sequential,
    );

    assert!(result.is_none());
}

#[test]
fn trained_model_improves_over_base_score_on_simple_problem() {
    // Simple linear relationship: y = x + 0.5
    let n_samples = 100;
    let features_raw: Vec<f32> = (0..n_samples).map(|i| i as f32 / 10.0).collect();
    let targets: Vec<f32> = features_raw.iter().map(|&x| x + 0.5).collect();

    // Feature-major: shape [1, n_samples]
    let features = Array2::from_shape_vec((1, n_samples), features_raw.clone()).unwrap();
    let features_dataset = Dataset::from_array(features.view(), None, None);
    let binning_config = BinningConfig::builder().max_bins(64).build();
    let dataset = BinnedDataset::from_dataset(&features_dataset, &binning_config)
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 50,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
    let targets_2d = Array2::from_shape_vec((1, targets.len()), targets.clone()).unwrap();
    let targets_view = TargetsView::new(targets_2d.view());
    let forest = trainer
        .train(
            &features_dataset,
            &dataset,
            targets_view,
            WeightsView::None,
            None,
            Parallelism::Sequential,
        )
        .unwrap();

    forest
        .validate()
        .expect("trained forest should be structurally valid");

    let base = forest.base_score()[0];

    let mut base_error_sum = 0.0f32;
    let mut pred_error_sum = 0.0f32;

    for row in 0..n_samples {
        let x = features_raw[row];
        let pred = predict_row(&forest, &[x])[0];
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

    // Build row-major feature data first
    let mut features_row_major = Vec::with_capacity(n_samples * n_features);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            features_row_major.push(((i * (j + 1)) % 100) as f32 / 10.0);
        }
        let x0 = (i % 100) as f32 / 10.0;
        let x1 = ((i * 2) % 100) as f32 / 10.0;
        targets.push(x0 + 0.5 * x1);
    }

    // Convert to feature-major: [n_features, n_samples]
    let row_view = ArrayView2::from_shape((n_samples, n_features), &features_row_major).unwrap();
    let features = transpose_to_c_order(row_view.view());
    let features_dataset = Dataset::from_array(features.view(), None, None);
    let binning_config = BinningConfig::builder().max_bins(64).build();
    let dataset = BinnedDataset::from_dataset(&features_dataset, &binning_config)
        .expect("Failed to build binned dataset");

    let params = GBDTParams {
        n_trees: 5,
        learning_rate: 0.3,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
        cache_size: 32,
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
    let targets_2d = Array2::from_shape_vec((1, targets.len()), targets.clone()).unwrap();
    let targets_view = TargetsView::new(targets_2d.view());
    let forest = trainer
        .train(
            &features_dataset,
            &dataset,
            targets_view,
            WeightsView::None,
            None,
            Parallelism::Sequential,
        )
        .unwrap();

    forest
        .validate()
        .expect("trained forest should be structurally valid");

    let base = forest.base_score()[0];

    let mut base_error_sum = 0.0f32;
    let mut pred_error_sum = 0.0f32;

    for row in 0..n_samples {
        let mut row_features = Vec::with_capacity(n_features);
        for f in 0..n_features {
            row_features.push(features[[f, row]]);
        }

        let pred = predict_row(&forest, &row_features)[0];
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
    let categories: Vec<f32> = (0..n_samples).map(|i| (i % 4) as f32).collect();
    let targets: Vec<f32> = categories
        .iter()
        .map(|&c| if c == 0.0 || c == 2.0 { 1.0 } else { 10.0 })
        .collect();

    let categories = ndarray::Array1::from_vec(categories);
    let features_dataset = Dataset::builder()
        .add_categorical_unnamed(categories)
        .build()
        .unwrap();

    // Build binned dataset - should detect categorical from schema
    let dataset = BinnedDataset::from_dataset(&features_dataset, &BinningConfig::default())
        .expect("Failed to build binned dataset");

    // Verify the dataset reports the feature as categorical
    assert!(dataset.is_categorical(0), "Feature 0 should be categorical");

    let params = GBDTParams {
        n_trees: 5,
        learning_rate: 0.3,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
        max_onehot_cats: 4, // Allow one-hot splits for this test
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
    let targets_2d = Array2::from_shape_vec((1, targets.len()), targets.clone()).unwrap();
    let targets_view = TargetsView::new(targets_2d.view());
    let forest = trainer
        .train(
            &features_dataset,
            &dataset,
            targets_view,
            WeightsView::None,
            None,
            Parallelism::Sequential,
        )
        .unwrap();

    forest
        .validate()
        .expect("trained forest should be structurally valid");

    // Check that at least one tree has a categorical split
    let has_categorical_split = forest.trees().any(|tree| tree.has_categorical());

    assert!(
        has_categorical_split,
        "At least one tree should have a categorical split"
    );

    // Verify the first tree has categorical split nodes
    let first_tree = forest
        .trees()
        .next()
        .expect("should have at least one tree");
    let mut found_categorical = false;
    for node_idx in 0..first_tree.n_nodes() as u32 {
        if !first_tree.is_leaf(node_idx)
            && matches!(first_tree.split_type(node_idx), SplitType::Categorical)
        {
            found_categorical = true;
            break;
        }
    }
    assert!(
        found_categorical,
        "First tree should contain at least one categorical split node"
    );

    // Verify predictions are reasonable
    // Categories 0, 2 should predict low, categories 1, 3 should predict high
    let pred_cat0 = predict_row(&forest, &[0.0])[0];
    let pred_cat1 = predict_row(&forest, &[1.0])[0];
    let pred_cat2 = predict_row(&forest, &[2.0])[0];
    let pred_cat3 = predict_row(&forest, &[3.0])[0];

    // Low predictions (categories 0, 2) should be < 5.5 (midpoint)
    // High predictions (categories 1, 3) should be > 5.5
    assert!(
        pred_cat0 < 5.5 && pred_cat2 < 5.5,
        "Categories 0 and 2 should predict low values, got {} and {}",
        pred_cat0,
        pred_cat2
    );
    assert!(
        pred_cat1 > 5.5 && pred_cat3 > 5.5,
        "Categories 1 and 3 should predict high values, got {} and {}",
        pred_cat1,
        pred_cat3
    );
}

/// Test GBDTModel::train() using the new Dataset type.
///
/// This is an end-to-end test of the high-level training API that accepts
/// a Dataset and returns a trained model without manual binning.
#[test]
fn train_from_dataset_api() {
    // Simple linear relationship: y = x0 + 0.5*x1
    let n_samples = 100;

    // Build row-major features: [n_samples, n_features]
    let mut features_data = Vec::with_capacity(n_samples * 2);
    let mut targets_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x0 = i as f32 / 10.0;
        let x1 = (i as f32 * 2.0) % 10.0;
        features_data.push(x0);
        features_data.push(x1);
        targets_data.push(x0 + 0.5 * x1);
    }

    // Create Dataset using the new API (feature-major: [n_features, n_samples])
    let features = Array2::from_shape_vec((n_samples, 2), features_data).unwrap();
    let features_fm = transpose_to_c_order(features.view());
    let targets_fm = Array2::from_shape_vec((1, n_samples), targets_data.clone()).unwrap();
    let dataset = Dataset::from_array(features_fm.view(), Some(targets_fm.clone()), None);

    assert_eq!(dataset.n_samples(), n_samples);
    assert_eq!(dataset.n_features(), 2);

    // Train using high-level API
    let config = GBDTConfig::builder()
        .n_trees(20)
        .learning_rate(0.2)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
        .build()
        .unwrap();

    let model = GBDTModel::train(&dataset, None, config, 1).expect("training should succeed");

    // Verify model produces reasonable predictions
    let forest = model.forest();
    forest.validate().expect("trained forest should be valid");

    // Compute error
    let base = forest.base_score()[0];
    let mut base_error = 0.0f32;
    let mut pred_error = 0.0f32;

    for (row, &target) in targets_data.iter().enumerate().take(n_samples) {
        let x0 = row as f32 / 10.0;
        let x1 = (row as f32 * 2.0) % 10.0;
        let pred = predict_row(forest, &[x0, x1])[0];

        base_error += (base - target).powi(2);
        pred_error += (pred - target).powi(2);
    }

    assert!(
        pred_error < base_error,
        "Model should improve over base score: pred_error={}, base_error={}",
        pred_error,
        base_error
    );
}

/// Test GBDTModel::train_with_eval() using EvalSet.
#[test]
fn train_from_dataset_with_eval_set() {
    // Create training data
    let n_train = 80;
    let n_eval = 20;

    let mut train_features = Vec::with_capacity(n_train * 2);
    let mut train_targets = Vec::with_capacity(n_train);
    let mut eval_features = Vec::with_capacity(n_eval * 2);
    let mut eval_targets = Vec::with_capacity(n_eval);

    for i in 0..n_train {
        let x0 = i as f32 / 10.0;
        let x1 = (i as f32 * 2.0) % 10.0;
        train_features.push(x0);
        train_features.push(x1);
        train_targets.push(x0 + 0.5 * x1);
    }

    for i in 0..n_eval {
        let x0 = (n_train + i) as f32 / 10.0;
        let x1 = ((n_train + i) as f32 * 2.0) % 10.0;
        eval_features.push(x0);
        eval_features.push(x1);
        eval_targets.push(x0 + 0.5 * x1);
    }

    // Create training dataset (convert to feature-major)
    let train_feat = Array2::from_shape_vec((n_train, 2), train_features).unwrap();
    let train_feat_fm = transpose_to_c_order(train_feat.view());
    let train_targ_fm = Array2::from_shape_vec((1, n_train), train_targets).unwrap();
    let train_ds = Dataset::from_array(train_feat_fm.view(), Some(train_targ_fm.clone()), None);

    // Create eval set using binned data (existing API)
    // For now, just test train() works. EvalSet integration will be story 3.3+
    let config = GBDTConfig::builder()
        .n_trees(10)
        .learning_rate(0.2)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
        .build()
        .unwrap();

    // Train without validation set first (validation set integration is future work)
    let model = GBDTModel::train(&train_ds, None, config, 1).expect("training should succeed");

    // Verify predictions are reasonable
    let forest = model.forest();
    let base = forest.base_score()[0];
    let mut pred_error = 0.0f32;
    let mut base_error = 0.0f32;

    for (i, &target) in eval_targets.iter().enumerate().take(n_eval) {
        let x0 = (n_train + i) as f32 / 10.0;
        let x1 = ((n_train + i) as f32 * 2.0) % 10.0;
        let pred = predict_row(forest, &[x0, x1])[0];

        pred_error += (pred - target).powi(2);
        base_error += (base - target).powi(2);
    }

    assert!(
        pred_error < base_error,
        "Model should generalize to unseen data: pred_error={}, base_error={}",
        pred_error,
        base_error
    );
}

/// Test GBDTModel::predict() with feature-major storage.
#[test]
fn predict_from_dataset_with_block_buffering() {
    // Train a model first
    let n_samples = 100;
    let mut features_data = Vec::with_capacity(n_samples * 2);
    let mut targets_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x0 = i as f32 / 10.0;
        let x1 = (i as f32 * 2.0) % 10.0;
        features_data.push(x0);
        features_data.push(x1);
        targets_data.push(x0 + 0.5 * x1);
    }

    let features = Array2::from_shape_vec((n_samples, 2), features_data.clone()).unwrap();
    let features_fm = transpose_to_c_order(features.view());
    let targets_fm = Array2::from_shape_vec((1, n_samples), targets_data.clone()).unwrap();
    let train_dataset = Dataset::from_array(features_fm.view(), Some(targets_fm.clone()), None);

    let config = GBDTConfig::builder()
        .n_trees(10)
        .learning_rate(0.3)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
        .build()
        .unwrap();

    let model = GBDTModel::train(&train_dataset, None, config, 1).expect("training should succeed");

    // Create a prediction dataset (same data)
    let pred_dataset = Dataset::from_array(features_fm.view(), Some(targets_fm.clone()), None);

    // Predict using Dataset API (feature-major with block buffering)
    let output = model.predict(&pred_dataset, 1);

    // Verify predictions are reasonable
    let mut pred_error = 0.0f32;
    let mut base_error = 0.0f32;
    let base = model.forest().base_score()[0];

    for i in 0..n_samples {
        let pred = output[[0, i]];
        let target = targets_data[i];
        pred_error += (pred - target).powi(2);
        base_error += (base - target).powi(2);
    }

    assert!(
        pred_error < base_error,
        "Predictions should be better than base: pred_error={}, base_error={}",
        pred_error,
        base_error
    );
}

/// Test that parallel predict_dataset matches sequential.
#[test]
fn predict_from_dataset_parallel_matches_sequential() {
    // Train a model
    let n_samples = 200;
    let mut features_data = Vec::with_capacity(n_samples * 3);
    let mut targets_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x0 = i as f32 / 20.0;
        let x1 = (i as f32 * 2.0) % 10.0;
        let x2 = (i as f32 * 0.5) % 5.0;
        features_data.push(x0);
        features_data.push(x1);
        features_data.push(x2);
        targets_data.push(x0 + 0.5 * x1 + 0.2 * x2);
    }

    let features = Array2::from_shape_vec((n_samples, 3), features_data).unwrap();
    let features_fm = transpose_to_c_order(features.view());
    let targets_fm = Array2::from_shape_vec((1, n_samples), targets_data).unwrap();
    let dataset = Dataset::from_array(features_fm.view(), Some(targets_fm.clone()), None);

    let config = GBDTConfig::builder()
        .n_trees(10)
        .learning_rate(0.3)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
        .build()
        .unwrap();

    let model = GBDTModel::train(&dataset, None, config, 1).expect("training should succeed");

    // Sequential prediction (n_threads=1)
    let seq_output = model.predict(&dataset, 1);

    // Parallel prediction (n_threads=0 = auto)
    let par_output = model.predict(&dataset, 0);

    // Results should match
    assert_eq!(seq_output.shape(), par_output.shape());

    for i in 0..n_samples {
        let seq = seq_output[[0, i]];
        let par = par_output[[0, i]];
        assert!(
            (seq - par).abs() < 1e-5,
            "Sequential and parallel should match at sample {}: seq={}, par={}",
            i,
            seq,
            par
        );
    }
}
