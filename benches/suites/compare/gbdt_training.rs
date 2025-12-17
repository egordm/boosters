//! Unified comparison benchmarks: booste-rs vs XGBoost vs LightGBM training.
//!
//! Run with: `cargo bench --features "bench-xgboost,bench-lightgbm" --bench compare_training`
//! Or single library: `cargo bench --features bench-xgboost --bench compare_training`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use booste_rs::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor};
use booste_rs::testing::data::{random_dense_f32, synthetic_regression_targets_linear};
use booste_rs::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "bench-xgboost")]
use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::{learning::LearningTaskParametersBuilder, learning::Objective, BoosterParametersBuilder, TrainingParametersBuilder};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::BoosterType;
#[cfg(feature = "bench-xgboost")]
use xgb::{Booster, DMatrix};

#[cfg(feature = "bench-lightgbm")]
use serde_json::json;

// =============================================================================
// Standardized Dataset Sizes
// =============================================================================

/// Small dataset: quick iteration, sanity checks
const SMALL: (usize, usize) = (5_000, 50);
/// Medium dataset: primary comparison point
const MEDIUM: (usize, usize) = (50_000, 100);

// =============================================================================
// Helpers
// =============================================================================

fn build_col_matrix(features_row_major: Vec<f32>, rows: usize, cols: usize) -> ColMatrix<f32> {
	let row: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features_row_major, rows, cols);
	row.to_layout()
}

// =============================================================================
// Training Comparison Benchmarks
// =============================================================================

fn bench_train_regression(c: &mut Criterion) {
	let mut group = c.benchmark_group("compare/train/regression");
	group.sample_size(10);

	let configs = [("small", SMALL.0, SMALL.1), ("medium", MEDIUM.0, MEDIUM.1)];
	let (n_trees, max_depth) = (50u32, 6u32);

	for (name, rows, cols) in configs {
		let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
		let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);

		group.throughput(Throughput::Elements((rows * cols) as u64));

		// =====================================================================
		// booste-rs
		// =====================================================================
		let params = GBDTParams {
			n_trees,
			learning_rate: 0.1,
			growth_strategy: GrowthStrategy::DepthWise { max_depth },
			gain: GainParams { reg_lambda: 1.0, ..Default::default() },
			n_threads: 1,
			cache_size: 32,
			..Default::default()
		};
		let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

		group.bench_function(BenchmarkId::new("boosters/cold_full", name), |b| {
			b.iter(|| {
				let col_matrix = build_col_matrix(features.clone(), rows, cols);
				let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build().unwrap();
				black_box(trainer.train(black_box(&binned), black_box(&targets), &[], &[]).unwrap())
			})
		});

		// =====================================================================
		// XGBoost
		// =====================================================================
		#[cfg(feature = "bench-xgboost")]
		{
			let tree_params = TreeBoosterParametersBuilder::default()
				.eta(0.1)
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

		// =====================================================================
		// LightGBM
		// =====================================================================
		#[cfg(feature = "bench-lightgbm")]
		{
			let features_f64: Vec<f64> = features.iter().map(|&x| x as f64).collect();
			let labels_f32: Vec<f32> = targets.clone();
			let num_features = cols as i32;

			group.bench_function(BenchmarkId::new("lightgbm/cold_full", name), |b| {
				b.iter(|| {
					let dataset = lightgbm3::Dataset::from_slice(
						black_box(&features_f64),
						black_box(&labels_f32),
						num_features,
						true,
					)
					.unwrap();
					let params = json!({
						"objective": "regression",
						"metric": "l2",
						"num_iterations": n_trees,
						"learning_rate": 0.1,
						"max_depth": max_depth as i32,
						"num_leaves": 64,
						"min_data_in_leaf": 1,
						"lambda_l2": 1.0,
						"feature_fraction": 1.0,
						"bagging_fraction": 1.0,
						"bagging_freq": 0,
						"verbosity": -1,
						"num_threads": 1
					});
					black_box(lightgbm3::Booster::train(dataset, &params).unwrap())
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
