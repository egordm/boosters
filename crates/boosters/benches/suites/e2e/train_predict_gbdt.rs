//! End-to-end benchmarks: GBDT training + prediction.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{binned::BinnedDatasetBuilder, FeaturesView};
use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
use boosters::testing::data::{select_rows, select_targets, split_indices, synthetic_regression};
use boosters::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};
use boosters::Parallelism;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_train_then_predict_regression(c: &mut Criterion) {
	let mut group = c.benchmark_group("e2e/train_predict/gbdt/regression");
	group.sample_size(10);

	let (rows, cols) = (50_000usize, 100usize);
	let dataset = synthetic_regression(rows, cols, 42, 0.05);
	let (train_idx, valid_idx) = split_indices(rows, 0.2, 999);

	// Select training and validation data using ndarray helpers
	let x_train = select_rows(dataset.features.view(), &train_idx);
	let y_train = select_targets(dataset.targets.view(), &train_idx);
	let x_valid = select_rows(dataset.features.view(), &valid_idx);

	// Build binned dataset for training (x_train is already feature-major)
	let features_view = FeaturesView::from_array(x_train.view());
	let binned_train = BinnedDatasetBuilder::from_matrix(&features_view, 256).build().unwrap();

	// Transpose validation features to row-major for prediction
	let valid_row_major = x_valid.t().to_owned();

	let params = GBDTParams {
		n_trees: 50,
		learning_rate: 0.1,
		growth_strategy: GrowthStrategy::DepthWise { max_depth: 6 },
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		cache_size: 256,
		..Default::default()
	};
	let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

	group.bench_function("train_then_predict", |b| {
		b.iter(|| {
			let forest = trainer.train(black_box(&binned_train), black_box(y_train.view()), None, &[], Parallelism::Sequential).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let preds = predictor.predict(black_box(valid_row_major.view()), Parallelism::Sequential);
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
