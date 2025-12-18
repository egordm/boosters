//! End-to-end benchmarks: GBDT training + prediction.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use booste_rs::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor, RowMatrix};
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
use booste_rs::testing::data::{random_dense_f32, split_indices, synthetic_regression_targets_linear};
use booste_rs::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};

use common::select::{select_rows_row_major, select_targets};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_train_then_predict_regression(c: &mut Criterion) {
	let mut group = c.benchmark_group("e2e/train_predict/gbdt/regression");
	group.sample_size(10);

	let (rows, cols) = (50_000usize, 100usize);
	let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
	let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);
	let (train_idx, valid_idx) = split_indices(rows, 0.2, 999);

	let x_train = select_rows_row_major(&features, rows, cols, &train_idx);
	let y_train = select_targets(&targets, &train_idx);
	let x_valid = select_rows_row_major(&features, rows, cols, &valid_idx);

	let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train, train_idx.len(), cols);
	let col_train: ColMatrix<f32> = row_train.to_layout();
	let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();

	let row_valid: RowMatrix<f32> = RowMatrix::from_vec(x_valid, valid_idx.len(), cols);

	let params = GBDTParams {
		n_trees: 50,
		learning_rate: 0.1,
		growth_strategy: GrowthStrategy::DepthWise { max_depth: 6 },
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		n_threads: 1,
		cache_size: 256,
		..Default::default()
	};
	let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

	group.bench_function("train_then_predict", |b| {
		b.iter(|| {
			let forest = trainer.train(black_box(&binned_train), black_box(&y_train), &[], &[]).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let preds = predictor.predict(black_box(&row_valid));
			black_box(preds)
		})
	});

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_train_then_predict_regression
}
criterion_main!(benches);
