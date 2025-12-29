//! Component benchmarks: GBDT (tree) training throughput and scaling.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::threading::with_rayon_threads;

use boosters::Parallelism;
use boosters::data::{BinningConfig, binned::BinnedDatasetBuilder, transpose_to_c_order};
use boosters::data::{Dataset, TargetsView, WeightsView};
use boosters::testing::synthetic_datasets::{
    random_features_array, synthetic_binary, synthetic_multiclass, synthetic_regression,
};
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LogLoss, LogisticLoss, MulticlassLogLoss,
    Rmse, SoftmaxLoss, SquaredLoss,
};

use ndarray::Array2;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

fn build_binned_dataset(
    rows: usize,
    cols: usize,
    seed: u64,
) -> (boosters::data::binned::BinnedDataset, Array2<f32>) {
    let synth_dataset = synthetic_regression(rows, cols, seed, 0.05);
    let features_dataset = Dataset::new(synth_dataset.features.view(), None, None);
    let binned = BinnedDatasetBuilder::with_config(BinningConfig::builder().max_bins(256).build())
        .add_features(features_dataset.features(), Parallelism::Parallel)
        .build()
        .unwrap();
    let targets = Array2::from_shape_vec(
        (1, synth_dataset.targets.len()),
        synth_dataset.targets.to_vec(),
    )
    .unwrap();
    (binned, targets)
}

fn bench_gbdt_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/quantize");
    group.sample_size(20);

    for (rows, cols) in [(10_000usize, 50usize), (50_000, 100), (100_000, 20)] {
        let features = random_features_array(rows, cols, 42, -1.0, 1.0);
        let features_fm = transpose_to_c_order(features.view());
        let features_dataset = Dataset::new(features_fm.view(), None, None);

        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::new("to_binned/max_bins=256", format!("{rows}x{cols}")),
            &features_dataset,
            |b, ds| {
                b.iter(|| {
                    black_box(
                        BinnedDatasetBuilder::with_config(BinningConfig::builder().max_bins(256).build())
                            .add_features(black_box(ds).features(), Parallelism::Parallel)
                            .build()
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_gbdt_train_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/train_regression");
    group.sample_size(10);

    let configs = [
        ("small", 10_000usize, 50usize, 50u32, 6u32),
        ("medium", 50_000usize, 100usize, 50u32, 6u32),
        ("narrow", 100_000usize, 20usize, 50u32, 6u32),
    ];

    for (name, rows, cols, n_trees, max_depth) in configs {
        let (binned, targets) = build_binned_dataset(rows, cols, 42);

        let params = GBDTParams {
            n_trees,
            learning_rate: 0.3,
            growth_strategy: GrowthStrategy::DepthWise { max_depth },
            gain: GainParams {
                reg_lambda: 1.0,
                ..Default::default()
            },
            cache_size: 256,
            ..Default::default()
        };
        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("train", name), |b| {
            let targets_view = TargetsView::new(targets.view());
            b.iter(|| {
                black_box(
                    trainer
                        .train(
                            black_box(&binned),
                            targets_view,
                            WeightsView::None,
                            &[],
                            Parallelism::Sequential,
                        )
                        .unwrap(),
                )
            })
        });
    }

    group.finish();
}

fn bench_gbdt_train_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/train_binary");
    group.sample_size(10);

    let (rows, cols, n_trees, max_depth) = (50_000usize, 100usize, 50u32, 6u32);
    let synth_dataset = synthetic_binary(rows, cols, 42, 0.2);
    let targets: Array2<f32> = Array2::from_shape_vec(
        (1, synth_dataset.targets.len()),
        synth_dataset.targets.to_vec(),
    )
    .unwrap();
    let features_dataset = Dataset::new(synth_dataset.features.view(), None, None);
    let binned = BinnedDatasetBuilder::with_config(BinningConfig::builder().max_bins(256).build())
        .add_features(features_dataset.features(), Parallelism::Parallel)
        .build()
        .unwrap();

    let params = GBDTParams {
        n_trees,
        learning_rate: 0.3,
        growth_strategy: GrowthStrategy::DepthWise { max_depth },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        cache_size: 256,
        ..Default::default()
    };
    let trainer = GBDTTrainer::new(LogisticLoss, LogLoss, params);

    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("train_binary", |b| {
        let targets_view = TargetsView::new(targets.view());
        b.iter(|| {
            black_box(
                trainer
                    .train(
                        black_box(&binned),
                        targets_view,
                        WeightsView::None,
                        &[],
                        Parallelism::Sequential,
                    )
                    .unwrap(),
            )
        })
    });
    group.finish();
}

fn bench_gbdt_train_multiclass(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/train_multiclass");
    group.sample_size(10);

    let (rows, cols, n_trees, max_depth, n_classes) = (20_000usize, 50usize, 30u32, 6u32, 10usize);
    let synth_dataset = synthetic_multiclass(rows, cols, n_classes, 42, 0.1);
    let targets: Array2<f32> = Array2::from_shape_vec(
        (1, synth_dataset.targets.len()),
        synth_dataset.targets.to_vec(),
    )
    .unwrap();
    let features_dataset = Dataset::new(synth_dataset.features.view(), None, None);
    let binned = BinnedDatasetBuilder::with_config(BinningConfig::builder().max_bins(256).build())
        .add_features(features_dataset.features(), Parallelism::Parallel)
        .build()
        .unwrap();

    let params = GBDTParams {
        n_trees,
        learning_rate: 0.3,
        growth_strategy: GrowthStrategy::DepthWise { max_depth },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        cache_size: 256,
        ..Default::default()
    };
    let trainer = GBDTTrainer::new(SoftmaxLoss::new(n_classes), MulticlassLogLoss, params);

    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("train_multiclass", |b| {
        let targets_view = TargetsView::new(targets.view());
        b.iter(|| {
            black_box(
                trainer
                    .train(
                        black_box(&binned),
                        targets_view,
                        WeightsView::None,
                        &[],
                        Parallelism::Sequential,
                    )
                    .unwrap(),
            )
        })
    });
    group.finish();
}

fn bench_gbdt_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/thread_scaling");
    group.sample_size(10);

    let (rows, cols, n_trees, max_depth) = (50_000usize, 100usize, 30u32, 6u32);
    let (binned, targets) = build_binned_dataset(rows, cols, 42);

    for &n_threads in common::matrix::THREAD_COUNTS {
        let params = GBDTParams {
            n_trees,
            learning_rate: 0.3,
            growth_strategy: GrowthStrategy::DepthWise { max_depth },
            gain: GainParams {
                reg_lambda: 1.0,
                ..Default::default()
            },
            cache_size: 256,
            ..Default::default()
        };
        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("train", n_threads), |b| {
            b.iter(|| {
                with_rayon_threads(n_threads, || {
                    let targets_view = TargetsView::new(targets.view());
                    black_box(
                        trainer
                            .train(
                                black_box(&binned),
                                targets_view,
                                WeightsView::None,
                                &[],
                                Parallelism::Parallel,
                            )
                            .unwrap(),
                    )
                })
            })
        });
    }

    group.finish();
}

fn bench_gbdt_growth_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/growth_strategy");
    group.sample_size(10);

    let (rows, cols, n_trees, max_depth) = (50_000usize, 100usize, 50u32, 6u32);
    let max_leaves = 64u32;

    let (binned, targets) = build_binned_dataset(rows, cols, 42);

    let common = GBDTParams {
        n_trees,
        learning_rate: 0.1,
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        cache_size: 256,
        ..Default::default()
    };

    let depthwise = GBDTTrainer::new(
        SquaredLoss,
        Rmse,
        GBDTParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth },
            ..common.clone()
        },
    );
    let leafwise = GBDTTrainer::new(
        SquaredLoss,
        Rmse,
        GBDTParams {
            growth_strategy: GrowthStrategy::LeafWise { max_leaves },
            ..common
        },
    );

    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function(
        BenchmarkId::new("depthwise", format!("{rows}x{cols}")),
        |b| {
            let targets_view = TargetsView::new(targets.view());
            b.iter(|| {
                black_box(
                    depthwise
                        .train(
                            black_box(&binned),
                            targets_view,
                            WeightsView::None,
                            &[],
                            Parallelism::Sequential,
                        )
                        .unwrap(),
                )
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("leafwise", format!("{rows}x{cols}")),
        |b| {
            let targets_view = TargetsView::new(targets.view());
            b.iter(|| {
                black_box(
                    leafwise
                        .train(
                            black_box(&binned),
                            targets_view,
                            WeightsView::None,
                            &[],
                            Parallelism::Sequential,
                        )
                        .unwrap(),
                )
            })
        },
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_gbdt_quantize,
        bench_gbdt_train_regression,
        bench_gbdt_train_binary,
        bench_gbdt_train_multiclass,
        bench_gbdt_thread_scaling,
        bench_gbdt_growth_strategy
}
criterion_main!(benches);
