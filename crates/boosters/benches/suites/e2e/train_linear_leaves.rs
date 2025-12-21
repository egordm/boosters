//! End-to-end benchmarks: GBDT training with linear leaves overhead measurement.
//!
//! Compares training time with and without linear leaves to measure overhead.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor};
use boosters::testing::data::{random_dense_f32, split_indices, synthetic_regression_targets_linear};
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LinearLeafConfig, Rmse, SquaredLoss,
};
use boosters::Parallelism;

use common::select::{select_rows_row_major, select_targets};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_linear_training_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e/train/linear_leaves");
    group.sample_size(10);

    // Generate synthetic linear data where linear leaves should help
    let (rows, cols) = (50_000usize, 100usize);
    let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
    let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);
    let (train_idx, _valid_idx) = split_indices(rows, 0.2, 999);

    let x_train = select_rows_row_major(&features, rows, cols, &train_idx);
    let y_train = select_targets(&targets, &train_idx);

    let row_train: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(x_train, train_idx.len(), cols);
    let col_train: ColMatrix<f32> = row_train.to_layout();
    let binned_train = BinnedDatasetBuilder::from_matrix(&col_train, 256).build().unwrap();

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
                .train(black_box(&binned_train), black_box(&y_train), &[], &[], Parallelism::SEQUENTIAL)
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
                .train(black_box(&binned_train), black_box(&y_train), &[], &[], Parallelism::SEQUENTIAL)
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
