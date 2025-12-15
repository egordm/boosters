//! Comparison benchmarks: booste-rs vs XGBoost training.
//!
//! Run with: `cargo bench --features bench-xgboost --bench training_xgboost`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use booste_rs::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor};
use booste_rs::testing::data::{random_dense_f32, synthetic_regression_targets_linear};
use booste_rs::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, SquaredLoss};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "bench-xgboost")]
use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::{learning::LearningTaskParametersBuilder, learning::Objective, BoosterParametersBuilder, TrainingParametersBuilder};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::BoosterType;
#[cfg(feature = "bench-xgboost")]
use xgb::{Booster, DMatrix};

fn build_col_matrix(features_row_major: Vec<f32>, rows: usize, cols: usize) -> ColMatrix<f32> {
	let row: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features_row_major, rows, cols);
	row.to_layout()
}

fn bench_train_regression(c: &mut Criterion) {
	let mut group = c.benchmark_group("compare/train/xgboost/regression");
	group.sample_size(10);

	let configs = [("small", 10_000usize, 50usize), ("medium", 50_000, 100)];
	let (n_trees, max_depth) = (50u32, 6u32);

	for (name, rows, cols) in configs {
		let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
		let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);

		// booste-rs: cold full pipeline (row->col->bin + train)
		let params = GBDTParams {
			n_trees,
			learning_rate: 0.3,
			growth_strategy: GrowthStrategy::DepthWise { max_depth },
			gain: GainParams { reg_lambda: 1.0, ..Default::default() },
			n_threads: 1,
			cache_size: 32,
			..Default::default()
		};
		let trainer = GBDTTrainer::new(SquaredLoss, params);

		group.throughput(Throughput::Elements((rows * cols) as u64));
		// booste-rs: cold full pipeline (row->col->bin + train)
		group.bench_function(BenchmarkId::new("boosters/cold_full", name), |b| {
			b.iter(|| {
				let col_matrix = build_col_matrix(features.clone(), rows, cols);
				let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build().unwrap();
				black_box(trainer.train(black_box(&binned), black_box(&targets), &[]).unwrap())
			})
		});

		#[cfg(feature = "bench-xgboost")]
		{
			let tree_params = TreeBoosterParametersBuilder::default()
				.eta(0.3)
				.max_depth(max_depth)
				.lambda(1.0)
				.alpha(0.0)
				.gamma(0.0)
				.min_child_weight(1.0)
				.tree_method(TreeMethod::Hist)
				.max_bin(256u32)
				.build()
				.unwrap();

			let learning_params = LearningTaskParametersBuilder::default()
				.objective(Objective::RegLinear)
				.build()
				.unwrap();

			let booster_params = BoosterParametersBuilder::default()
				.booster_type(BoosterType::Tree(tree_params))
				.learning_params(learning_params)
				.verbose(false)
				.threads(Some(1))
				.build()
				.unwrap();

			// cold DMatrix
			group.bench_function(BenchmarkId::new("xgboost/cold_dmatrix", name), |b| {
				b.iter(|| {
					let mut dtrain = DMatrix::from_dense(&features, rows).unwrap();
					dtrain.set_labels(&targets).unwrap();

					let training_params = TrainingParametersBuilder::default()
						.dtrain(&dtrain)
						.boost_rounds(n_trees)
						.booster_params(booster_params.clone())
						.evaluation_sets(None)
						.build()
						.unwrap();

					black_box(Booster::train(&training_params).unwrap())
				})
			});
		}
	}

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_train_regression
}
criterion_main!(benches);
