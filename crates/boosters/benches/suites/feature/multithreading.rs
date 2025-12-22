//! Multi-threading scalability benchmark: booste-rs vs XGBoost vs LightGBM.
//!
//! Compares training speedup across different thread counts (1, 2, 4, 8).
//! All libraries use depth-wise expansion.
//!
//! Run with: `cargo bench --features bench-compare --bench multithreading`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::matrix::THREAD_COUNTS;
use common::threading::with_rayon_threads;

use boosters::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor};
use boosters::testing::data::{random_dense_f32, synthetic_regression_targets_linear};
use boosters::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};
use boosters::Parallelism;

use ndarray::ArrayView1;

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
// Dataset Configuration
// =============================================================================

/// Medium dataset for multi-threading comparison
const DATASET: (usize, usize) = (50_000, 100);
const N_TREES: u32 = 50;
const MAX_DEPTH: u32 = 6;

// =============================================================================
// Helpers
// =============================================================================

fn build_col_matrix(features_row_major: Vec<f32>, rows: usize, cols: usize) -> ColMatrix<f32> {
    let row: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features_row_major, rows, cols);
    row.to_layout()
}

// =============================================================================
// Multi-threading Benchmark
// =============================================================================

fn bench_multithreading(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature/multithreading");
    group.sample_size(10);

    let (rows, cols) = DATASET;
    let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
    let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);

    group.throughput(Throughput::Elements((rows * cols) as u64));

    // Pre-build binned dataset (we're benchmarking training, not binning)
    let col_matrix = build_col_matrix(features.clone(), rows, cols);
    let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .unwrap();

    for &n_threads in THREAD_COUNTS {
        let thread_label = format!("{}_threads", n_threads);

        // =====================================================================
        // booste-rs (depth-wise)
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

        group.bench_function(BenchmarkId::new("boosters", &thread_label), |b| {
            b.iter(|| {
                with_rayon_threads(n_threads, || {
                    black_box(
                        trainer
                            .train(black_box(&binned), ArrayView1::from(black_box(&targets[..])), None, &[], Parallelism::Parallel)
                            .unwrap(),
                    )
                })
            })
        });

        // =====================================================================
        // XGBoost (depth-wise via hist)
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
                .threads(Some(n_threads as u32))
                .build()
                .unwrap();

            // Pre-build DMatrix
            let mut dtrain = DMatrix::from_dense(&features, rows).unwrap();
            dtrain.set_labels(&targets).unwrap();

            group.bench_function(BenchmarkId::new("xgboost", &thread_label), |b| {
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
        // LightGBM (depth-wise via max_depth)
        // =====================================================================
        #[cfg(feature = "bench-lightgbm")]
        {
            let features_f64: Vec<f64> = features.iter().map(|&x| x as f64).collect();
            let labels_f32: Vec<f32> = targets.clone();
            let num_features = cols as i32;

            group.bench_function(BenchmarkId::new("lightgbm", &thread_label), |b| {
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
                        "num_threads": n_threads as i32
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
    targets = bench_multithreading
}
criterion_main!(benches);
