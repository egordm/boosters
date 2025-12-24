use boosters::data::{binned::BinnedDatasetBuilder, transpose_to_c_order, FeaturesView};
use boosters::model::gbdt::{GBDTConfig, GBDTModel, RegularizationParams, TreeParams};
use boosters::testing::data::{
	random_features_array, split_indices, synthetic_binary, synthetic_multiclass,
	synthetic_regression,
};
use boosters::training::{
	Accuracy, LinearLeafConfig, LogLoss, Mae, MetricFn, MulticlassAccuracy, MulticlassLogLoss,
	Objective, Rmse,
};
use ndarray::{Array2, ArrayView1};

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
	let x_all = random_features_array(rows, cols, seed, -1.0, 1.0);
	let y_all = dataset.targets_slice();
	let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

	let x_train = select_samples(&x_all, &train_idx);
	let y_train = select_targets(y_all, &train_idx);
	let x_valid = select_samples(&x_all, &valid_idx);
	let y_valid = select_targets(y_all, &valid_idx);

	// Transpose to feature-major for training
	let col_train = transpose_to_c_order(x_train.view());
	let view_train = FeaturesView::from_array(col_train.view());
	let binned_train = BinnedDatasetBuilder::from_matrix(&view_train, 256)
		.build()
		.unwrap();
	let row_valid = x_valid;

	let config = GBDTConfig::builder()
		.objective(Objective::squared())
		.n_trees(trees)
		.learning_rate(0.1)
		.tree(TreeParams::depth_wise(depth))
		.regularization(RegularizationParams {
			lambda: 1.0,
			..Default::default()
		})
		.cache_size(64)
		.seed(seed)
		.build()
		.unwrap();

	let model = GBDTModel::train_binned(
		&binned_train,
		ArrayView1::from(&y_train[..]),
		None,
		&[],
		config,
		1,
	)
	.unwrap();
	let pred = model.predict_array(row_valid.view(), 1);
	let targets_arr = ArrayView1::from(&y_valid[..]);

	let rmse = Rmse.compute(pred.view(), targets_arr, None);
	let mae = Mae.compute(pred.view(), targets_arr, None);
	(rmse, mae)
}

fn run_synthetic_binary(
	rows: usize,
	cols: usize,
	trees: u32,
	depth: u32,
	seed: u64,
) -> (f64, f64) {
	let dataset = synthetic_binary(rows, cols, seed, 0.2);
	let x_all = random_features_array(rows, cols, seed, -1.0, 1.0);
	let y_all = dataset.targets_slice();
	let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

	let x_train = select_samples(&x_all, &train_idx);
	let y_train = select_targets(y_all, &train_idx);
	let x_valid = select_samples(&x_all, &valid_idx);
	let y_valid = select_targets(y_all, &valid_idx);

	// Transpose to feature-major for training
	let col_train = transpose_to_c_order(x_train.view());
	let view_train = FeaturesView::from_array(col_train.view());
	let binned_train = BinnedDatasetBuilder::from_matrix(&view_train, 256)
		.build()
		.unwrap();
	let row_valid = x_valid;

	let config = GBDTConfig::builder()
		.objective(Objective::logistic())
		.n_trees(trees)
		.learning_rate(0.1)
		.tree(TreeParams::depth_wise(depth))
		.regularization(RegularizationParams {
			lambda: 1.0,
			..Default::default()
		})
		.cache_size(64)
		.seed(seed)
		.build()
		.unwrap();

	let model = GBDTModel::train_binned(
		&binned_train,
		ArrayView1::from(&y_train[..]),
		None,
		&[],
		config,
		1,
	)
	.unwrap();
	// predict_array() returns probabilities automatically
	let pred = model.predict_array(row_valid.view(), 1);
	let targets_arr = ArrayView1::from(&y_valid[..]);

	let ll = LogLoss.compute(pred.view(), targets_arr, None);
	let acc = Accuracy::default().compute(pred.view(), targets_arr, None);
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
	let x_all = random_features_array(rows, cols, seed, -1.0, 1.0);
	let y_all = dataset.targets_slice();
	let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

	let x_train = select_samples(&x_all, &train_idx);
	let y_train = select_targets(y_all, &train_idx);
	let x_valid = select_samples(&x_all, &valid_idx);
	let y_valid = select_targets(y_all, &valid_idx);

	// Transpose to feature-major for training
	let col_train = transpose_to_c_order(x_train.view());
	let view_train = FeaturesView::from_array(col_train.view());
	let binned_train = BinnedDatasetBuilder::from_matrix(&view_train, 256)
		.build()
		.unwrap();
	let row_valid = x_valid;

	let config = GBDTConfig::builder()
		.objective(Objective::softmax(classes))
		.n_trees(trees)
		.learning_rate(0.1)
		.tree(TreeParams::depth_wise(depth))
		.regularization(RegularizationParams {
			lambda: 1.0,
			..Default::default()
		})
		.cache_size(64)
		.seed(seed)
		.build()
		.unwrap();

	let model = GBDTModel::train_binned(
		&binned_train,
		ArrayView1::from(&y_train[..]),
		None,
		&[],
		config,
		1,
	)
	.unwrap();
	// predict_array() returns softmax probabilities
	let pred = model.predict_array(row_valid.view(), 1);
	let targets_arr = ArrayView1::from(&y_valid[..]);

	let ll = MulticlassLogLoss.compute(pred.view(), targets_arr, None);
	let acc = MulticlassAccuracy.compute(pred.view(), targets_arr, None);
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
	let view_train = FeaturesView::from_array(col_train.view());
	let binned_train = BinnedDatasetBuilder::from_matrix(&view_train, 256)
		.build()
		.unwrap();
	let row_valid = x_valid;

	// --- Train without linear leaves ---
	let base_config = GBDTConfig::builder()
		.objective(Objective::squared())
		.n_trees(N_TREES)
		.learning_rate(0.1)
		.tree(TreeParams::depth_wise(MAX_DEPTH))
		.regularization(RegularizationParams {
			lambda: 1.0,
			..Default::default()
		})
		.seed(SEED)
		.build()
		.unwrap();

	let model_baseline = GBDTModel::train_binned(
		&binned_train,
		ArrayView1::from(&y_train[..]),
		None,
		&[],
		base_config,
		1,
	)
	.unwrap();
	let pred_baseline = model_baseline.predict_array(row_valid.view(), 1);
	let targets_arr = ArrayView1::from(&y_valid[..]);
	let rmse_baseline = Rmse.compute(pred_baseline.view(), targets_arr, None);

	// --- Train with linear leaves ---
	let linear_config = GBDTConfig::builder()
		.objective(Objective::squared())
		.n_trees(N_TREES)
		.learning_rate(0.1)
		.tree(TreeParams::depth_wise(MAX_DEPTH))
		.regularization(RegularizationParams {
			lambda: 1.0,
			..Default::default()
		})
		.linear_leaves(LinearLeafConfig::default().with_min_samples(10))
		.seed(SEED)
		.build()
		.unwrap();

	eprintln!("Training with linear leaves...");
	let model_linear = GBDTModel::train_binned(
		&binned_train,
		ArrayView1::from(&y_train[..]),
		None,
		&[],
		linear_config,
		1,
	)
	.unwrap();
	let pred_linear = model_linear.predict_array(row_valid.view(), 1);
	let rmse_linear = Rmse.compute(pred_linear.view(), targets_arr, None);

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
