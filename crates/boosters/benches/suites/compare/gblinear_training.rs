//! Unified comparison benchmarks: booste-rs GBLinear vs XGBoost GBLinear.
//!
//! Run with: `cargo bench --features bench-xgboost --bench gblinear_training`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{Dataset, Feature, WeightsView};
use boosters::testing::synthetic_datasets::{features_row_major_slice, synthetic_regression};
use boosters::training::{GBLinearParams, GBLinearTrainer, Rmse, SquaredLoss, Verbosity};

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

#[cfg(feature = "bench-xgboost")]
use xgb::parameters::linear::LinearBoosterParametersBuilder;
#[cfg(feature = "bench-xgboost")]
use xgb::parameters::{
    BoosterParametersBuilder, BoosterType, TrainingParametersBuilder,
    learning::LearningTaskParametersBuilder, learning::Objective,
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
        let dataset = synthetic_regression(rows, cols, 42, 0.05);

        group.throughput(Throughput::Elements((rows * cols) as u64));

        // =====================================================================
        // booste-rs GBLinear
        // =====================================================================
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
            b.iter(|| {
                black_box(
                    trainer
                        .train(
                            black_box(&dataset),
                            dataset.targets().expect("dataset has targets"),
                            WeightsView::None,
                            None,
                        )
                        .unwrap(),
                )
            })
        });

        // =====================================================================
        // XGBoost GBLinear
        // =====================================================================
        #[cfg(feature = "bench-xgboost")]
        {
            let targets: Vec<f32> = dataset
                .targets()
                .expect("synthetic datasets have targets")
                .as_single_output()
                .to_vec();
            let features: Vec<f32> = features_row_major_slice(&dataset);

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

    let dataset = synthetic_regression(rows, cols, 42, 0.0);
    let features: Vec<f32> = features_row_major_slice(&dataset);

    // Generate binary labels based on linear combination
    let mut targets = vec![0.0f32; rows];
    for (row_idx, row) in features.chunks_exact(cols).enumerate() {
        let sum: f32 = row.iter().take(cols.min(10)).copied().sum();
        targets[row_idx] = if sum > 0.0 { 1.0 } else { 0.0 };
    }

    group.throughput(Throughput::Elements((rows * cols) as u64));

    // =========================================================================
    // booste-rs GBLinear (binary classification)
    // =========================================================================
    let targets_2d =
        ndarray::Array2::from_shape_vec((1, rows), targets.clone()).expect("shape mismatch");
    let mut builder = Dataset::builder();
    for (feature_idx, feature) in dataset.feature_columns().iter().enumerate() {
        let name = format!("f{feature_idx}");
        builder = match feature {
            Feature::Dense(values) => builder.add_feature(&name, values.clone()),
            Feature::Sparse {
                indices,
                values,
                n_samples,
                default,
            } => builder.add_sparse(&name, indices.clone(), values.clone(), *n_samples, *default),
        };
    }
    let dataset = builder.targets(targets_2d.view()).build().unwrap();

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
        b.iter(|| {
            black_box(
                trainer
                    .train(
                        black_box(&dataset),
                        dataset.targets().expect("dataset has targets"),
                        WeightsView::None,
                        None,
                    )
                    .unwrap(),
            )
        })
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
