//! End-to-end benchmarks: GBDT training with linear leaves overhead measurement.
//!
//! Compares training time with and without linear leaves to measure overhead.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::Parallelism;
use boosters::data::{BinningConfig, binned::BinnedDatasetBuilder};
use boosters::data::{Dataset, TargetsView, WeightsView};
use boosters::testing::synthetic_datasets::{
    select_rows, select_targets, split_indices, synthetic_regression,
};
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LinearLeafConfig, Rmse, SquaredLoss,
};

use ndarray::Array2;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_linear_training_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e/train/linear_leaves");
    group.sample_size(10);

    // Generate synthetic linear data where linear leaves should help
    let (rows, cols) = (50_000usize, 100usize);
    let dataset = synthetic_regression(rows, cols, 42, 0.05);
    let (train_idx, _valid_idx) = split_indices(rows, 0.2, 999);

    // Select training data using ndarray helpers
    let x_train = select_rows(dataset.features.view(), &train_idx);
    let y_train = select_targets(dataset.targets.view(), &train_idx);

    // Build binned dataset for training (x_train is already feature-major)
    let x_train_dataset = Dataset::new(x_train.view(), None, None);
    let binned_train = BinnedDatasetBuilder::with_config(BinningConfig::builder().max_bins(256).build())
        .add_features(x_train_dataset.features(), Parallelism::Parallel)
        .build()
        .unwrap();

    let base_params = GBDTParams {
        n_trees: 20,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        cache_size: 256,
        linear_leaves: None,
        ..Default::default()
    };

    // Benchmark baseline (no linear leaves)
    let trainer_baseline = GBDTTrainer::new(SquaredLoss, Rmse, base_params.clone());
    let y_train_2d =
        Array2::from_shape_vec((1, y_train.len()), y_train.iter().cloned().collect()).unwrap();
    group.bench_function("baseline", |b| {
        b.iter(|| {
            let targets = TargetsView::new(y_train_2d.view());
            let forest = trainer_baseline
                .train(
                    black_box(&binned_train),
                    targets,
                    WeightsView::None,
                    &[],
                    Parallelism::Sequential,
                )
                .unwrap();
            black_box(forest)
        })
    });

    // Benchmark with linear leaves
    let linear_params = GBDTParams {
        linear_leaves: Some(LinearLeafConfig::builder().min_samples(10).build()),
        ..base_params
    };
    let trainer_linear = GBDTTrainer::new(SquaredLoss, Rmse, linear_params);
    group.bench_function("linear_leaves", |b| {
        b.iter(|| {
            let targets = TargetsView::new(y_train_2d.view());
            let forest = trainer_linear
                .train(
                    black_box(&binned_train),
                    targets,
                    WeightsView::None,
                    &[],
                    Parallelism::Sequential,
                )
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
