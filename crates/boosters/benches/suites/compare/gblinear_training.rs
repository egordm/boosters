//! Unified comparison benchmarks: booste-rs GBLinear vs XGBoost GBLinear.
//!
//! Run with: `cargo bench --features bench-xgboost --bench gblinear_training`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{ColMatrix, Dataset, DenseMatrix, RowMajor};
use boosters::testing::data::{random_dense_f32, synthetic_regression_targets_linear};
use boosters::training::{GBLinearParams, GBLinearTrainer, Rmse, SquaredLoss, Verbosity};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "bench-xgboost")]
use xgb::parameters::linear::LinearBoosterParametersBuilder;
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::{
    learning::LearningTaskParametersBuilder, learning::Objective, BoosterParametersBuilder,
    BoosterType, TrainingParametersBuilder,
};
#[cfg(feature = "bench-xgboost")]
use xgb::{Booster, DMatrix};

// =============================================================================
// Standardized Dataset Sizes
// =============================================================================

/// Small dataset: quick iteration
const SMALL: (usize, usize) = (5_000, 50);
/// Medium dataset: primary comparison point
const MEDIUM: (usize, usize) = (50_000, 100);
/// Large dataset: stress test
const LARGE: (usize, usize) = (100_000, 200);

// =============================================================================
// Helpers
// =============================================================================

fn build_col_matrix(features_row_major: Vec<f32>, rows: usize, cols: usize) -> ColMatrix<f32> {
    let row: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features_row_major, rows, cols);
    row.to_layout()
}

// =============================================================================
// GBLinear Training Comparison
// =============================================================================

fn bench_gblinear_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/train/gblinear");
    group.sample_size(10);

    let configs = [
        ("small", SMALL.0, SMALL.1),
        ("medium", MEDIUM.0, MEDIUM.1),
        ("large", LARGE.0, LARGE.1),
    ];
    let n_rounds = 100u32;

    for (name, rows, cols) in configs {
        let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
        let (targets, _w, _b) =
            synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);

        group.throughput(Throughput::Elements((rows * cols) as u64));

        // =====================================================================
        // booste-rs GBLinear
        // =====================================================================
        let col_matrix = build_col_matrix(features.clone(), rows, cols);
        let dataset = Dataset::from_numeric(&col_matrix, targets.clone()).unwrap();

        let params = GBLinearParams {
            n_rounds,
            learning_rate: 0.5,
            lambda: 0.0,
            alpha: 0.0,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);

        group.bench_function(BenchmarkId::new("boosters", name), |b| {
            b.iter(|| black_box(trainer.train(black_box(&dataset), &[]).unwrap()))
        });

        // =====================================================================
        // XGBoost GBLinear
        // =====================================================================
        #[cfg(feature = "bench-xgboost")]
        {
            let linear_params = LinearBoosterParametersBuilder::default()
                .lambda(0.0)
                .alpha(0.0)
                .build()
                .unwrap();

            let learning_params = LearningTaskParametersBuilder::default()
                .objective(Objective::RegLinear)
                .build()
                .unwrap();

            let booster_params = BoosterParametersBuilder::default()
                .booster_type(BoosterType::Linear(linear_params))
                .learning_params(learning_params)
                .verbose(false)
                .build()
                .unwrap();

            group.bench_function(BenchmarkId::new("xgboost", name), |b| {
                b.iter(|| {
                    let mut dmat = DMatrix::from_dense(&features, rows).unwrap();
                    dmat.set_labels(&targets).unwrap();

                    let training_params = TrainingParametersBuilder::default()
                        .dtrain(&dmat)
                        .boost_rounds(n_rounds)
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

fn bench_gblinear_classification(c: &mut Criterion) {
    use boosters::training::{LogLoss, LogisticLoss};

    let mut group = c.benchmark_group("compare/train/gblinear_binary");
    group.sample_size(10);

    let (rows, cols) = MEDIUM;
    let n_rounds = 100u32;

    let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);

    // Generate binary labels based on linear combination
    let mut targets = vec![0.0f32; rows];
    for i in 0..rows {
        let mut sum = 0.0f32;
        for j in 0..cols.min(10) {
            sum += features[i * cols + j];
        }
        targets[i] = if sum > 0.0 { 1.0 } else { 0.0 };
    }

    group.throughput(Throughput::Elements((rows * cols) as u64));

    // =========================================================================
    // booste-rs GBLinear (binary classification)
    // =========================================================================
    let col_matrix = build_col_matrix(features.clone(), rows, cols);
    let dataset = Dataset::from_numeric(&col_matrix, targets.clone()).unwrap();

    let params = GBLinearParams {
        n_rounds,
        learning_rate: 0.5,
        lambda: 0.0,
        alpha: 0.0,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };

    let trainer = GBLinearTrainer::new(LogisticLoss, LogLoss, params);

    group.bench_function("boosters", |b| {
        b.iter(|| black_box(trainer.train(black_box(&dataset), &[]).unwrap()))
    });

    // =========================================================================
    // XGBoost GBLinear (binary classification)
    // =========================================================================
    #[cfg(feature = "bench-xgboost")]
    {
        let linear_params = LinearBoosterParametersBuilder::default()
            .lambda(0.0)
            .alpha(0.0)
            .build()
            .unwrap();

        let learning_params = LearningTaskParametersBuilder::default()
            .objective(Objective::BinaryLogistic)
            .build()
            .unwrap();

        let booster_params = BoosterParametersBuilder::default()
            .booster_type(BoosterType::Linear(linear_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();

        group.bench_function("xgboost", |b| {
            b.iter(|| {
                let mut dmat = DMatrix::from_dense(&features, rows).unwrap();
                dmat.set_labels(&targets).unwrap();

                let training_params = TrainingParametersBuilder::default()
                    .dtrain(&dmat)
                    .boost_rounds(n_rounds)
                    .booster_params(booster_params.clone())
                    .evaluation_sets(None)
                    .build()
                    .unwrap();

                black_box(Booster::train(&training_params).unwrap())
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_gblinear_regression, bench_gblinear_classification
}

criterion_main!(benches);
