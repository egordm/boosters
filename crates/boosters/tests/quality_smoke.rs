use boosters::data::{Dataset, TargetsView, WeightsView, transpose_to_c_order};
use boosters::model::gbdt::{GBDTConfig, GBDTModel};
use boosters::testing::synthetic_datasets::{
    features_row_major, random_features_array, split_indices, synthetic_binary,
    synthetic_multiclass, synthetic_regression,
};
use boosters::training::{GrowthStrategy, LinearLeafConfig, Metric, Objective};
use ndarray::Array2;

/// Select samples from Array2 by indices, returns sample-major Array2.
fn select_samples(features: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let n_features = features.ncols();
    let mut out = Array2::zeros((indices.len(), n_features));
    for (i, &idx) in indices.iter().enumerate() {
        out.row_mut(i).assign(&features.row(idx));
    }
    out
}

/// Select targets by indices.
fn select_targets(targets: &[f32], row_indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(row_indices.len());
    for &r in row_indices {
        out.push(targets[r]);
    }
    out
}

fn run_synthetic_regression(
    rows: usize,
    cols: usize,
    trees: u32,
    depth: u32,
    seed: u64,
) -> (f64, f64) {
    let dataset = synthetic_regression(rows, cols, seed, 0.05);
    let x_all = features_row_major(&dataset);
    let y_all = dataset
        .targets()
        .expect("synthetic datasets have targets")
        .as_single_output()
        .to_vec();
    let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

    let x_train = select_samples(&x_all, &train_idx);
    let y_train = select_targets(&y_all, &train_idx);
    let x_valid = select_samples(&x_all, &valid_idx);
    let y_valid = select_targets(&y_all, &valid_idx);

    // Transpose to feature-major for training
    let col_train = transpose_to_c_order(x_train.view());
    let y_train_2d = Array2::from_shape_vec((1, y_train.len()), y_train.clone()).unwrap();
    let dataset_train = Dataset::from_array(col_train.view(), Some(y_train_2d), None);
    let row_valid = x_valid;

    let config = GBDTConfig::builder()
        .objective(Objective::SquaredLoss)
        .n_trees(trees)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: depth })
        .lambda(1.0)
        .cache_size(64)
        .seed(seed)
        .build()
        .unwrap();

    let model = GBDTModel::train(&dataset_train, None, config, 1).unwrap();
    // Transpose validation to feature-major for prediction
    let col_valid = transpose_to_c_order(row_valid.view());
    let dataset_valid = Dataset::from_array(col_valid.view(), None, None);
    let pred = model.predict(&dataset_valid, 1);
    let targets_2d = Array2::from_shape_vec((1, y_valid.len()), y_valid).unwrap();
    let targets = TargetsView::new(targets_2d.view());

    let rmse = Metric::Rmse.compute(pred.view(), targets, WeightsView::None);
    let mae = Metric::Mae.compute(pred.view(), targets, WeightsView::None);
    (rmse, mae)
}

fn run_synthetic_binary(rows: usize, cols: usize, trees: u32, depth: u32, seed: u64) -> (f64, f64) {
    let dataset = synthetic_binary(rows, cols, seed, 0.2);
    let x_all = features_row_major(&dataset);
    let y_all = dataset
        .targets()
        .expect("synthetic datasets have targets")
        .as_single_output()
        .to_vec();
    let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

    let x_train = select_samples(&x_all, &train_idx);
    let y_train = select_targets(&y_all, &train_idx);
    let x_valid = select_samples(&x_all, &valid_idx);
    let y_valid = select_targets(&y_all, &valid_idx);

    // Transpose to feature-major for training
    let col_train = transpose_to_c_order(x_train.view());
    let y_train_2d = Array2::from_shape_vec((1, y_train.len()), y_train.clone()).unwrap();
    let dataset_train = Dataset::from_array(col_train.view(), Some(y_train_2d), None);
    let row_valid = x_valid;

    let config = GBDTConfig::builder()
        .objective(Objective::LogisticLoss)
        .n_trees(trees)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: depth })
        .lambda(1.0)
        .cache_size(64)
        .seed(seed)
        .build()
        .unwrap();

    let model = GBDTModel::train(&dataset_train, None, config, 1).unwrap();
    // Transpose validation to feature-major for prediction
    let col_valid = transpose_to_c_order(row_valid.view());
    let dataset_valid = Dataset::from_array(col_valid.view(), None, None);
    let pred = model.predict(&dataset_valid, 1);
    let targets_2d = Array2::from_shape_vec((1, y_valid.len()), y_valid).unwrap();
    let targets = TargetsView::new(targets_2d.view());

    let ll = Metric::LogLoss.compute(pred.view(), targets, WeightsView::None);
    let acc = Metric::Accuracy { threshold: 0.5 }.compute(pred.view(), targets, WeightsView::None);
    (ll, acc)
}

fn run_synthetic_multiclass(
    rows: usize,
    cols: usize,
    classes: usize,
    trees: u32,
    depth: u32,
    seed: u64,
) -> (f64, f64) {
    let dataset = synthetic_multiclass(rows, cols, classes, seed, 0.1);
    let x_all = features_row_major(&dataset);
    let y_all = dataset
        .targets()
        .expect("synthetic datasets have targets")
        .as_single_output()
        .to_vec();
    let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

    let x_train = select_samples(&x_all, &train_idx);
    let y_train = select_targets(&y_all, &train_idx);
    let x_valid = select_samples(&x_all, &valid_idx);
    let y_valid = select_targets(&y_all, &valid_idx);

    // Transpose to feature-major for training
    let col_train = transpose_to_c_order(x_train.view());
    let y_train_2d = Array2::from_shape_vec((1, y_train.len()), y_train.clone()).unwrap();
    let dataset_train = Dataset::from_array(col_train.view(), Some(y_train_2d), None);
    let row_valid = x_valid;

    let config = GBDTConfig::builder()
        .objective(Objective::SoftmaxLoss { n_classes: classes })
        .n_trees(trees)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: depth })
        .lambda(1.0)
        .cache_size(64)
        .seed(seed)
        .build()
        .unwrap();

    let model = GBDTModel::train(&dataset_train, None, config, 1).unwrap();
    // Transpose validation to feature-major for prediction
    let col_valid = transpose_to_c_order(row_valid.view());
    let dataset_valid = Dataset::from_array(col_valid.view(), None, None);
    let pred = model.predict(&dataset_valid, 1);
    let targets_2d = Array2::from_shape_vec((1, y_valid.len()), y_valid).unwrap();
    let targets = TargetsView::new(targets_2d.view());

    let ll = Metric::MulticlassLogLoss.compute(pred.view(), targets, WeightsView::None);
    let acc = Metric::MulticlassAccuracy.compute(pred.view(), targets, WeightsView::None);
    (ll, acc)
}

fn run_quality_report_suite() -> bool {
    std::env::var_os("BOOSTERS_RUN_QUALITY").is_some()
}

#[test]
fn quality_smoke_synthetic_regression() {
    let (rmse, mae) = run_synthetic_regression(4_000, 30, 25, 6, 42);
    assert!(rmse < 2.0, "rmse too high: {rmse}");
    assert!(mae < 1.7, "mae too high: {mae}");
}

#[test]
fn quality_smoke_synthetic_binary() {
    let (ll, acc) = run_synthetic_binary(4_000, 30, 25, 6, 42);
    assert!(ll < 0.65, "logloss too high: {ll}");
    assert!(acc > 0.72, "accuracy too low: {acc}");
}

#[test]
fn quality_smoke_synthetic_multiclass() {
    let (ll, acc) = run_synthetic_multiclass(4_000, 30, 5, 30, 6, 42);
    assert!(ll < 1.4, "mlogloss too high: {ll}");
    assert!(acc > 0.30, "accuracy too low: {acc}");
}

/// Optional (but CI-friendly) quality suite aligned with the benchmark report.
///
/// Enable with: `BOOSTERS_RUN_QUALITY=1 cargo test --test quality_smoke`.
#[test]
fn quality_report_synthetic_targets() {
    if !run_quality_report_suite() {
        return;
    }

    let (rmse, mae) = run_synthetic_regression(20_000, 50, 50, 6, 42);
    assert!(rmse < 1.40, "rmse too high: {rmse}");
    assert!(mae < 1.15, "mae too high: {mae}");

    let (ll, acc) = run_synthetic_binary(20_000, 50, 50, 6, 42);
    assert!(ll < 0.45, "logloss too high: {ll}");
    assert!(acc > 0.82, "accuracy too low: {acc}");

    let (ll, acc) = run_synthetic_multiclass(20_000, 50, 5, 50, 6, 42);
    assert!(ll < 0.90, "mlogloss too high: {ll}");
    assert!(acc > 0.70, "accuracy too low: {acc}");
}

/// Test that linear leaves improve quality on a synthetic linear dataset.
///
/// Dataset: y = 3*x1 + 2*x2 + uniform_noise(σ≈0.1)
/// n = 10000, seed = 42
///
/// Expectation: Linear leaves should capture the linear relationship better
/// than constant leaves, achieving ≥5% RMSE improvement.
///
/// Note: With 16 leaves (depth=4), constant leaves already approximate the linear
/// function reasonably well. Linear leaves provide incremental improvement.
#[test]
fn test_quality_improvement_linear_leaves() {
    use rand::prelude::*;

    const N_SAMPLES: usize = 10_000;
    const SEED: u64 = 42;
    const NOISE_AMPLITUDE: f32 = 0.17; // Uniform[-0.17, 0.17] has variance ≈ σ²=0.01
    const N_TREES: u32 = 20;
    const MAX_DEPTH: u32 = 4;
    const VALID_FRACTION: f32 = 0.2;

    // Generate synthetic data: y = 3*x1 + 2*x2 + noise
    // Using random_features_array for consistent API
    let features = random_features_array(N_SAMPLES, 2, SEED, 0.0, 1.0);

    // Generate targets with known linear relationship
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xDEAD);
    let targets: Vec<f32> = (0..N_SAMPLES)
        .map(|i| {
            let x1 = features[[i, 0]];
            let x2 = features[[i, 1]];
            let noise = (rng.r#gen::<f32>() * 2.0 - 1.0) * NOISE_AMPLITUDE;
            3.0 * x1 + 2.0 * x2 + noise
        })
        .collect();

    // Train/valid split
    let (train_idx, valid_idx) = split_indices(N_SAMPLES, VALID_FRACTION, SEED ^ 0x51EED);

    let x_train = select_samples(&features, &train_idx);
    let y_train = select_targets(&targets, &train_idx);
    let x_valid = select_samples(&features, &valid_idx);
    let y_valid = select_targets(&targets, &valid_idx);

    // Convert to matrices - transpose to feature-major for training
    let col_train = transpose_to_c_order(x_train.view());
    let y_train_2d = Array2::from_shape_vec((1, y_train.len()), y_train.clone()).unwrap();
    let dataset_train = Dataset::from_array(col_train.view(), Some(y_train_2d), None);
    let row_valid = x_valid;

    // --- Train without linear leaves ---
    let base_config = GBDTConfig::builder()
        .objective(Objective::SquaredLoss)
        .n_trees(N_TREES)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise {
            max_depth: MAX_DEPTH,
        })
        .lambda(1.0)
        .seed(SEED)
        .build()
        .unwrap();

    let model_baseline = GBDTModel::train(&dataset_train, None, base_config, 1).unwrap();
    // Transpose validation to feature-major for prediction
    let col_valid = transpose_to_c_order(row_valid.view());
    let dataset_valid = Dataset::from_array(col_valid.view(), None, None);
    let pred_baseline = model_baseline.predict(&dataset_valid, 1);
    let targets_2d = Array2::from_shape_vec((1, y_valid.len()), y_valid).unwrap();
    let targets = TargetsView::new(targets_2d.view());
    let rmse_baseline = Metric::Rmse.compute(pred_baseline.view(), targets, WeightsView::None);

    // --- Train with linear leaves ---
    let linear_config = GBDTConfig::builder()
        .objective(Objective::SquaredLoss)
        .n_trees(N_TREES)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise {
            max_depth: MAX_DEPTH,
        })
        .lambda(1.0)
        .linear_leaves(LinearLeafConfig::builder().min_samples(10).build())
        .seed(SEED)
        .build()
        .unwrap();

    eprintln!("Training with linear leaves...");
    let model_linear = GBDTModel::train(&dataset_train, None, linear_config, 1).unwrap();
    let pred_linear = model_linear.predict(&dataset_valid, 1);
    let rmse_linear = Metric::Rmse.compute(pred_linear.view(), targets, WeightsView::None);

    // Assert: linear leaves should improve RMSE by at least 5%
    let improvement = (rmse_baseline - rmse_linear) / rmse_baseline;

    assert!(
        improvement >= 0.05,
        "Linear leaves should improve RMSE by ≥5%, got {:.2}% improvement \
		 (baseline RMSE={:.4}, linear RMSE={:.4})",
        improvement * 100.0,
        rmse_baseline,
        rmse_linear
    );
}
