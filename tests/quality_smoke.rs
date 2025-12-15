use booste_rs::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor, RowMatrix};
use booste_rs::inference::common::{sigmoid_inplace, softmax_inplace};
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
use booste_rs::testing::data::{
	random_dense_f32, split_indices, synthetic_binary_targets_from_linear_score,
	synthetic_multiclass_targets_from_linear_scores, synthetic_regression_targets_linear,
};
use booste_rs::training::{
	Accuracy, GainParams, GBDTParams, GBDTTrainer, GrowthStrategy, LogLoss, Mae, Metric,
	MulticlassAccuracy, MulticlassLogLoss, Rmse, LogisticLoss, SoftmaxLoss, SquaredLoss,
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
	let trainer = GBDTTrainer::new(SquaredLoss, params);
	let forest = trainer.train(&binned_train, &y_train, &[]).unwrap();
	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
	let pred = predictor.predict(&row_valid);
	let pred0: Vec<f32> = (0..row_valid.num_rows()).map(|r| pred.row(r)[0]).collect();

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
	let trainer = GBDTTrainer::new(LogisticLoss, params);
	let forest = trainer.train(&binned_train, &y_train, &[]).unwrap();
	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
	let raw = predictor.predict(&row_valid);
	let mut prob: Vec<f32> = (0..row_valid.num_rows()).map(|r| raw.row(r)[0]).collect();
	sigmoid_inplace(&mut prob);

	let n_rows = y_valid.len();
	let ll = LogLoss.compute(n_rows, 1, &prob, &y_valid, &[]);
	let acc = Accuracy::default().compute(n_rows, 1, &prob, &y_valid, &[]);
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
	let trainer = GBDTTrainer::new(SoftmaxLoss::new(classes), params);
	let forest = trainer.train(&binned_train, &y_train, &[]).unwrap();
	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
	let raw = predictor.predict(&row_valid);

	let n_rows = row_valid.num_rows();
	let mut prob_row_major = vec![0.0f32; n_rows * classes];
	for r in 0..n_rows {
		let mut logits = raw.row(r).to_vec();
		softmax_inplace(&mut logits);
		prob_row_major[r * classes..(r + 1) * classes].copy_from_slice(&logits);
	}

	let ll = MulticlassLogLoss.compute(n_rows, classes, &prob_row_major, &y_valid, &[]);
	let acc = MulticlassAccuracy.compute(n_rows, classes, &prob_row_major, &y_valid, &[]);
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
