use boosters::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor, RowMatrix};
use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
use boosters::testing::data::{
	random_dense_f32, split_indices, synthetic_binary_targets_from_linear_score,
	synthetic_multiclass_targets_from_linear_scores, synthetic_regression_targets_linear,
};
use boosters::training::{
	Accuracy, GainParams, GBDTParams, GBDTTrainer, GrowthStrategy, LogLoss, Mae, Metric,
	MulticlassAccuracy, MulticlassLogLoss, ObjectiveFn, Rmse, LogisticLoss, SoftmaxLoss, SquaredLoss,
	LinearLeafConfig,
};

fn select_rows_row_major(features_row_major: &[f32], rows: usize, cols: usize, row_indices: &[usize]) -> Vec<f32> {
	assert_eq!(features_row_major.len(), rows * cols);
	let mut out = Vec::with_capacity(row_indices.len() * cols);
	for &r in row_indices {
		assert!(r < rows);
		let start = r * cols;
		out.extend_from_slice(&features_row_major[start..start + cols]);
	}
	out
}

fn select_targets(targets: &[f32], row_indices: &[usize]) -> Vec<f32> {
	let mut out = Vec::with_capacity(row_indices.len());
	for &r in row_indices {
		out.push(targets[r]);
	}
	out
}

fn default_params(trees: u32, growth_strategy: GrowthStrategy, seed: u64) -> GBDTParams {
	GBDTParams {
		n_trees: trees,
		learning_rate: 0.1,
		growth_strategy,
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		n_threads: 0,
		cache_size: 64,
		seed,
		..Default::default()
	}
}

fn run_synthetic_regression(rows: usize, cols: usize, trees: u32, depth: u32, seed: u64) -> (f64, f64) {
	let x_all = random_dense_f32(rows, cols, seed, -1.0, 1.0);
	let y_all = synthetic_regression_targets_linear(&x_all, rows, cols, seed ^ 0x0BAD_5EED, 0.05).0;
	let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

	let x_train = select_rows_row_major(&x_all, rows, cols, &train_idx);
	let y_train = select_targets(&y_all, &train_idx);
	let x_valid = select_rows_row_major(&x_all, rows, cols, &valid_idx);
	let y_valid = select_targets(&y_all, &valid_idx);

	let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train, train_idx.len(), cols);
	let col_train: ColMatrix<f32> = row_train.to_layout();
	let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();
	let row_valid: RowMatrix<f32> = RowMatrix::from_vec(x_valid, valid_idx.len(), cols);

	let params = default_params(trees, GrowthStrategy::DepthWise { max_depth: depth }, seed);
	let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
	let forest = trainer.train(&binned_train, &y_train, &[], &[]).unwrap();
	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
	let pred = predictor.predict(&row_valid);
	let pred0: Vec<f32> = pred.column(0).to_vec();

	let n_rows = y_valid.len();
	let rmse = Rmse.compute(n_rows, 1, &pred0, &y_valid, &[]);
	let mae = Mae.compute(n_rows, 1, &pred0, &y_valid, &[]);
	(rmse, mae)
}

fn run_synthetic_binary(rows: usize, cols: usize, trees: u32, depth: u32, seed: u64) -> (f64, f64) {
	let x_all = random_dense_f32(rows, cols, seed, -1.0, 1.0);
	let y_all = synthetic_binary_targets_from_linear_score(&x_all, rows, cols, seed ^ 0xB1A2_0001, 0.2);
	let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

	let x_train = select_rows_row_major(&x_all, rows, cols, &train_idx);
	let y_train = select_targets(&y_all, &train_idx);
	let x_valid = select_rows_row_major(&x_all, rows, cols, &valid_idx);
	let y_valid = select_targets(&y_all, &valid_idx);

	let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train, train_idx.len(), cols);
	let col_train: ColMatrix<f32> = row_train.to_layout();
	let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();
	let row_valid: RowMatrix<f32> = RowMatrix::from_vec(x_valid, valid_idx.len(), cols);

	let params = default_params(trees, GrowthStrategy::DepthWise { max_depth: depth }, seed);
	let objective = LogisticLoss;
	let trainer = GBDTTrainer::new(objective, LogLoss, params);
	let forest = trainer.train(&binned_train, &y_train, &[], &[]).unwrap();
	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
	let mut raw = predictor.predict(&row_valid);
	objective.transform_prediction_inplace(&mut raw);
	let prob = raw.column(0);

	let n_rows = y_valid.len();
	let ll = LogLoss.compute(n_rows, 1, prob, &y_valid, &[]);
	let acc = Accuracy::default().compute(n_rows, 1, prob, &y_valid, &[]);
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
	let x_all = random_dense_f32(rows, cols, seed, -1.0, 1.0);
	let y_all = synthetic_multiclass_targets_from_linear_scores(&x_all, rows, cols, classes, seed ^ 0x00C1_A550, 0.1);
	let (train_idx, valid_idx) = split_indices(rows, 0.2, seed ^ 0x51EED);

	let x_train = select_rows_row_major(&x_all, rows, cols, &train_idx);
	let y_train = select_targets(&y_all, &train_idx);
	let x_valid = select_rows_row_major(&x_all, rows, cols, &valid_idx);
	let y_valid = select_targets(&y_all, &valid_idx);

	let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train, train_idx.len(), cols);
	let col_train: ColMatrix<f32> = row_train.to_layout();
	let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();
	let row_valid: RowMatrix<f32> = RowMatrix::from_vec(x_valid, valid_idx.len(), cols);

	let params = default_params(trees, GrowthStrategy::DepthWise { max_depth: depth }, seed);
	let objective = SoftmaxLoss::new(classes);
	let trainer = GBDTTrainer::new(objective, MulticlassLogLoss, params);
	let forest = trainer.train(&binned_train, &y_train, &[], &[]).unwrap();
	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);

	let n_rows = row_valid.num_rows();
	// Apply softmax directly to column-major output
	let mut raw = predictor.predict(&row_valid);
	objective.transform_prediction_inplace(&mut raw);
	// raw is already column-major: predictions[class * n_rows + row]
	let prob_col_major = raw.as_slice();

	let ll = MulticlassLogLoss.compute(n_rows, classes, prob_col_major, &y_valid, &[]);
	let acc = MulticlassAccuracy.compute(n_rows, classes, &prob_col_major, &y_valid, &[]);
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
	let mut rng = StdRng::seed_from_u64(SEED);
	let mut features: Vec<f32> = Vec::with_capacity(N_SAMPLES * 2);
	let mut targets: Vec<f32> = Vec::with_capacity(N_SAMPLES);

	for _ in 0..N_SAMPLES {
		let x1: f32 = rng.r#gen(); // uniform [0, 1]
		let x2: f32 = rng.r#gen(); // uniform [0, 1]
		let noise: f32 = (rng.r#gen::<f32>() * 2.0 - 1.0) * NOISE_AMPLITUDE;
		let y = 3.0 * x1 + 2.0 * x2 + noise;

		features.push(x1);
		features.push(x2);
		targets.push(y);
	}

	// Train/valid split
	let (train_idx, valid_idx) = split_indices(N_SAMPLES, VALID_FRACTION, SEED ^ 0x51EED);

	let x_train = select_rows_row_major(&features, N_SAMPLES, 2, &train_idx);
	let y_train = select_targets(&targets, &train_idx);
	let x_valid = select_rows_row_major(&features, N_SAMPLES, 2, &valid_idx);
	let y_valid = select_targets(&targets, &valid_idx);

	// Convert to matrices
	let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train, train_idx.len(), 2);
	let col_train: ColMatrix<f32> = row_train.to_layout();
	let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();
	let row_valid: RowMatrix<f32> = RowMatrix::from_vec(x_valid, valid_idx.len(), 2);

	// --- Train without linear leaves ---
	let base_params = GBDTParams {
		n_trees: N_TREES,
		learning_rate: 0.1,
		growth_strategy: GrowthStrategy::DepthWise { max_depth: MAX_DEPTH },
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		seed: SEED,
		linear_leaves: None,
		..Default::default()
	};

	let trainer = GBDTTrainer::new(SquaredLoss, Rmse, base_params.clone());
	let forest_baseline = trainer.train(&binned_train, &y_train, &[], &[]).unwrap();
	let predictor = Predictor::<UnrolledTraversal6>::new(&forest_baseline);
	let pred_baseline = predictor.predict(&row_valid);
	let pred_baseline_slice: Vec<f32> = pred_baseline.column(0).to_vec();
	let rmse_baseline = Rmse.compute(y_valid.len(), 1, &pred_baseline_slice, &y_valid, &[]);

	// --- Train with linear leaves ---
	let linear_params = GBDTParams {
		linear_leaves: Some(LinearLeafConfig::default().with_min_samples(10)),
		..base_params
	};

	eprintln!("Training with linear leaves...");
	let trainer = GBDTTrainer::new(SquaredLoss, Rmse, linear_params);
	let forest_linear = trainer.train(&binned_train, &y_train, &[], &[]).unwrap();

	let predictor = Predictor::<UnrolledTraversal6>::new(&forest_linear);
	let pred_linear = predictor.predict(&row_valid);
	let pred_linear_slice: Vec<f32> = pred_linear.column(0).to_vec();
	let rmse_linear = Rmse.compute(y_valid.len(), 1, &pred_linear_slice, &y_valid, &[]);

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
