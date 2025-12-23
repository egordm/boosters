//! Dataset scaling benchmark: how training time grows with dataset size.
//!
//! Tests booste-rs vs XGBoost vs LightGBM across increasing row counts.
//! All libraries use depth-wise expansion.
//!
//! Run with: `cargo bench --features bench-compare --bench scaling`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::testing::data::synthetic_regression;
use boosters::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};
use boosters::Parallelism;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "bench-xgboost")]
use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::{
    learning::LearningTaskParametersBuilder, learning::Objective, BoosterParametersBuilder,
    BoosterType, TrainingParametersBuilder,
};
#[cfg(feature = "bench-xgboost")]
use xgb::{Booster, DMatrix};

#[cfg(feature = "bench-lightgbm")]
use serde_json::json;

// =============================================================================
// Scaling Configurations
// =============================================================================

/// Row counts to test scaling behavior
const ROW_COUNTS: &[usize] = &[10_000, 50_000, 100_000, 200_000];
const N_FEATURES: usize = 100;
const N_TREES: u32 = 50;
const MAX_DEPTH: u32 = 6;

// =============================================================================
// Row Scaling Benchmark
// =============================================================================

fn bench_row_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature/scaling_rows");
    group.sample_size(10);

    for &rows in ROW_COUNTS {
        let cols = N_FEATURES;
        let row_label = format!("{}k_rows", rows / 1000);

        let dataset = synthetic_regression(rows, cols, 42, 0.05);
        // Get row-major features for XGBoost/LightGBM compatibility
        let features = dataset.features_row_major_slice();
        let targets = dataset.targets.to_vec();

        group.throughput(Throughput::Elements((rows * cols) as u64));

        // Pre-build binned dataset
        let binned = dataset.to_binned(256);

        // =====================================================================
        // booste-rs (depth-wise, single-threaded for fair comparison)
        // =====================================================================
        let params = GBDTParams {
            n_trees: N_TREES,
            learning_rate: 0.1,
            growth_strategy: GrowthStrategy::DepthWise {
                max_depth: MAX_DEPTH,
            },
            gain: GainParams {
                reg_lambda: 1.0,
                ..Default::default()
            },
            cache_size: 32,
            ..Default::default()
        };
        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

        group.bench_function(BenchmarkId::new("boosters", &row_label), |b| {
            b.iter(|| {
                black_box(
                    trainer
                        .train(black_box(&binned), black_box(dataset.targets.view()), None, &[], Parallelism::Sequential)
                        .unwrap(),
                )
            })
        });

        // =====================================================================
        // XGBoost (depth-wise, single-threaded)
        // =====================================================================
        #[cfg(feature = "bench-xgboost")]
        {
            let tree_params = TreeBoosterParametersBuilder::default()
                .eta(0.1)
                .max_depth(MAX_DEPTH)
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

            // Pre-build DMatrix
            let mut dtrain = DMatrix::from_dense(&features, rows).unwrap();
            dtrain.set_labels(&targets).unwrap();

            group.bench_function(BenchmarkId::new("xgboost", &row_label), |b| {
                b.iter(|| {
                    let training_params = TrainingParametersBuilder::default()
                        .dtrain(&dtrain)
                        .boost_rounds(N_TREES)
                        .booster_params(booster_params.clone())
                        .evaluation_sets(None)
                        .build()
                        .unwrap();

                    black_box(Booster::train(&training_params).unwrap())
                })
            });
        }

        // =====================================================================
        // LightGBM (depth-wise, single-threaded)
        // =====================================================================
        #[cfg(feature = "bench-lightgbm")]
        {
            let features_f64: Vec<f64> = features.iter().map(|&x| x as f64).collect();
            let labels_f32: Vec<f32> = targets.clone();
            let num_features = cols as i32;

            group.bench_function(BenchmarkId::new("lightgbm", &row_label), |b| {
                b.iter(|| {
                    let dataset =
                        lightgbm3::Dataset::from_slice(&features_f64, &labels_f32, num_features, true)
                            .unwrap();
                    let params = json!({
                        "objective": "regression",
                        "metric": "l2",
                        "num_iterations": N_TREES,
                        "learning_rate": 0.1,
                        "max_depth": MAX_DEPTH as i32,
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

// =============================================================================
// Feature Scaling Benchmark
// =============================================================================

/// Feature counts to test
const FEATURE_COUNTS: &[usize] = &[50, 100, 200, 500];
const SCALING_ROWS: usize = 50_000;

fn bench_feature_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature/scaling_features");
    group.sample_size(10);

    for &cols in FEATURE_COUNTS {
        let rows = SCALING_ROWS;
        let feat_label = format!("{}_features", cols);

        let dataset = synthetic_regression(rows, cols, 42, 0.05);
        // Get row-major features for XGBoost/LightGBM compatibility
        let features = dataset.features_row_major_slice();
        let targets = dataset.targets.to_vec();

        group.throughput(Throughput::Elements((rows * cols) as u64));

        // Pre-build binned dataset
        let binned = dataset.to_binned(256);

        // =====================================================================
        // booste-rs
        // =====================================================================
        let params = GBDTParams {
            n_trees: N_TREES,
            learning_rate: 0.1,
            growth_strategy: GrowthStrategy::DepthWise {
                max_depth: MAX_DEPTH,
            },
            gain: GainParams {
                reg_lambda: 1.0,
                ..Default::default()
            },
            cache_size: 32,
            ..Default::default()
        };
        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

        group.bench_function(BenchmarkId::new("boosters", &feat_label), |b| {
            b.iter(|| {
                black_box(
                    trainer
                        .train(black_box(&binned), black_box(dataset.targets.view()), None, &[], Parallelism::Sequential)
                        .unwrap(),
                )
            })
        });

        // =====================================================================
        // XGBoost
        // =====================================================================
        #[cfg(feature = "bench-xgboost")]
        {
            let tree_params = TreeBoosterParametersBuilder::default()
                .eta(0.1)
                .max_depth(MAX_DEPTH)
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

            let mut dtrain = DMatrix::from_dense(&features, rows).unwrap();
            dtrain.set_labels(&targets).unwrap();

            group.bench_function(BenchmarkId::new("xgboost", &feat_label), |b| {
                b.iter(|| {
                    let training_params = TrainingParametersBuilder::default()
                        .dtrain(&dtrain)
                        .boost_rounds(N_TREES)
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

            group.bench_function(BenchmarkId::new("lightgbm", &feat_label), |b| {
                b.iter(|| {
                    let dataset =
                        lightgbm3::Dataset::from_slice(&features_f64, &labels_f32, num_features, true)
                            .unwrap();
                    let params = json!({
                        "objective": "regression",
                        "metric": "l2",
                        "num_iterations": N_TREES,
                        "learning_rate": 0.1,
                        "max_depth": MAX_DEPTH as i32,
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
    targets = bench_row_scaling, bench_feature_scaling
}
criterion_main!(benches);
