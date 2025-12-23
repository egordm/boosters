//! End-to-end benchmarks: GBDT training + prediction.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{binned::BinnedDatasetBuilder, transpose_to_c_order, FeaturesView};
use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
use boosters::testing::data::{split_indices, synthetic_regression};
use boosters::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};
use boosters::Parallelism;

use ndarray::{Array2, ArrayView1, ArrayView2};

use common::select::{select_rows_row_major, select_targets};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_train_then_predict_regression(c: &mut Criterion) {
	let mut group = c.benchmark_group("e2e/train_predict/gbdt/regression");
	group.sample_size(10);

	let (rows, cols) = (50_000usize, 100usize);
	let dataset = synthetic_regression(rows, cols, 42, 0.05);
	let targets: Vec<f32> = dataset.targets.to_vec();
	// Convert feature-major to row-major for helper functions
	let features_fm = dataset.features.view();
	let features: Vec<f32> = {
		let mut v = Vec::with_capacity(rows * cols);
		for r in 0..rows {
			for f in 0..cols {
				v.push(features_fm[(f, r)]);
			}
		}
		v
	};
	let (train_idx, valid_idx) = split_indices(rows, 0.2, 999);

	let x_train = select_rows_row_major(&features, rows, cols, &train_idx);
	let y_train = select_targets(&targets, &train_idx);
	let x_valid = select_rows_row_major(&features, rows, cols, &valid_idx);

	// Build binned dataset for training
	let sample_major_view = ArrayView2::from_shape((train_idx.len(), cols), &x_train).unwrap();
	let features_fm = transpose_to_c_order(sample_major_view);
	let features_view = FeaturesView::from_array(features_fm.view());
	let binned_train = BinnedDatasetBuilder::from_matrix(&features_view, 256).build().unwrap();

	let valid_array = Array2::from_shape_vec((valid_idx.len(), cols), x_valid).unwrap();

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
			let forest = trainer.train(black_box(&binned_train), ArrayView1::from(black_box(&y_train[..])), None, &[], Parallelism::Sequential).unwrap();
			let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
			let preds = predictor.predict(black_box(valid_array.view()), Parallelism::Sequential);
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
