//! End-to-end benchmarks: GBDT training with linear leaves overhead measurement.
//!
//! Compares training time with and without linear leaves to measure overhead.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{binned::BinnedDatasetBuilder, transpose_to_c_order, FeaturesView};
use boosters::testing::data::{split_indices, synthetic_regression};
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LinearLeafConfig, Rmse, SquaredLoss,
};
use boosters::Parallelism;

use ndarray::{ArrayView1, ArrayView2};

use common::select::{select_rows_row_major, select_targets};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_linear_training_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e/train/linear_leaves");
    group.sample_size(10);

    // Generate synthetic linear data where linear leaves should help
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
    let (train_idx, _valid_idx) = split_indices(rows, 0.2, 999);

    let x_train = select_rows_row_major(&features, rows, cols, &train_idx);
    let y_train = select_targets(&targets, &train_idx);

    // Convert to feature-major format for binning
    let sample_major_view = ArrayView2::from_shape((train_idx.len(), cols), &x_train).unwrap();
    let features_fm = transpose_to_c_order(sample_major_view);
    let features_view = FeaturesView::from_array(features_fm.view());
    let binned_train = BinnedDatasetBuilder::from_matrix(&features_view, 256).build().unwrap();

    let base_params = GBDTParams {
        n_trees: 20,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
        gain: GainParams { reg_lambda: 1.0, ..Default::default() },
        cache_size: 256,
        linear_leaves: None,
        ..Default::default()
    };

    // Benchmark baseline (no linear leaves)
    let trainer_baseline = GBDTTrainer::new(SquaredLoss, Rmse, base_params.clone());
    group.bench_function("baseline", |b| {
        b.iter(|| {
            let forest = trainer_baseline
                .train(black_box(&binned_train), ArrayView1::from(black_box(&y_train[..])), None, &[], Parallelism::Sequential)
                .unwrap();
            black_box(forest)
        })
    });

    // Benchmark with linear leaves
    let linear_params = GBDTParams {
        linear_leaves: Some(LinearLeafConfig::default().with_min_samples(10)),
        ..base_params
    };
    let trainer_linear = GBDTTrainer::new(SquaredLoss, Rmse, linear_params);
    group.bench_function("linear_leaves", |b| {
        b.iter(|| {
            let forest = trainer_linear
                .train(black_box(&binned_train), ArrayView1::from(black_box(&y_train[..])), None, &[], Parallelism::Sequential)
                .unwrap();
            black_box(forest)
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_linear_training_overhead
}
criterion_main!(benches);
