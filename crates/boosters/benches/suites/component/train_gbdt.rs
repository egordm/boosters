//! Component benchmarks: GBDT (tree) training throughput and scaling.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::threading::with_rayon_threads;

use boosters::Parallelism;
use boosters::data::{BinningConfig, binned::BinnedDataset, transpose_to_c_order};
use boosters::data::{Dataset, WeightsView};
use boosters::testing::synthetic_datasets::{
    random_features_array, synthetic_binary, synthetic_multiclass, synthetic_regression,
};
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LogLoss, LogisticLoss, MulticlassLogLoss,
    Rmse, SoftmaxLoss, SquaredLoss,
};

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

fn build_binned_dataset(
    rows: usize,
    cols: usize,
    seed: u64,
) -> (Dataset, boosters::data::binned::BinnedDataset) {
    let dataset = synthetic_regression(rows, cols, seed, 0.05);
    let binning_config = BinningConfig::builder().max_bins(256).build();
    let binned = BinnedDataset::from_dataset(&dataset, &binning_config).unwrap();
    (dataset, binned)
}

fn bench_gbdt_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/quantize");
    group.sample_size(20);

    for (rows, cols) in [(10_000usize, 50usize), (50_000, 100), (100_000, 20)] {
        let features = random_features_array(rows, cols, 42, -1.0, 1.0);
        let features_fm = transpose_to_c_order(features.view());
        let features_dataset = Dataset::from_array(features_fm.view(), None, None);

        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::new("to_binned/max_bins=256", format!("{rows}x{cols}")),
            &features_dataset,
            |b, ds| {
                let binning_config = BinningConfig::builder().max_bins(256).build();
                b.iter(|| {
                    black_box(BinnedDataset::from_dataset(black_box(ds), &binning_config).unwrap())
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
        let (dataset, binned) = build_binned_dataset(rows, cols, 42);

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
            b.iter(|| {
                let targets_view = dataset.targets().unwrap();
                black_box(
                    trainer
                        .train(
                            black_box(&dataset),
                            black_box(&binned),
                            targets_view,
                            WeightsView::None,
                            None,
                            Parallelism::Sequential,
                        )
                        .expect("training should succeed"),
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
    let dataset = synthetic_binary(rows, cols, 42, 0.2);
    let binning_config = BinningConfig::builder().max_bins(256).build();
    let binned = BinnedDataset::from_dataset(&dataset, &binning_config).unwrap();

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
        b.iter(|| {
            let targets_view = dataset.targets().unwrap();
            black_box(
                trainer
                    .train(
                        black_box(&dataset),
                        black_box(&binned),
                        targets_view,
                        WeightsView::None,
                        None,
                        Parallelism::Sequential,
                    )
                    .expect("training should succeed"),
            )
        })
    });
    group.finish();
}

fn bench_gbdt_train_multiclass(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/train_multiclass");
    group.sample_size(10);

    let (rows, cols, n_trees, max_depth, n_classes) = (20_000usize, 50usize, 30u32, 6u32, 10usize);
    let dataset = synthetic_multiclass(rows, cols, n_classes, 42, 0.1);
    let binning_config = BinningConfig::builder().max_bins(256).build();
    let binned = BinnedDataset::from_dataset(&dataset, &binning_config).unwrap();

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
        b.iter(|| {
            let targets_view = dataset.targets().unwrap();
            black_box(
                trainer
                    .train(
                        black_box(&dataset),
                        black_box(&binned),
                        targets_view,
                        WeightsView::None,
                        None,
                        Parallelism::Sequential,
                    )
                    .expect("training should succeed"),
            )
        })
    });
    group.finish();
}

fn bench_gbdt_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/train/gbdt/thread_scaling");
    group.sample_size(10);

    let (rows, cols, n_trees, max_depth) = (50_000usize, 100usize, 30u32, 6u32);
    let (dataset, binned) = build_binned_dataset(rows, cols, 42);

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
                    let targets_view = dataset.targets().unwrap();
                    black_box(
                        trainer
                            .train(
                                black_box(&dataset),
                                black_box(&binned),
                                targets_view,
                                WeightsView::None,
                                None,
                                Parallelism::Parallel,
                            )
                            .expect("training should succeed"),
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

    let (dataset, binned) = build_binned_dataset(rows, cols, 42);

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
            b.iter(|| {
                let targets_view = dataset.targets().unwrap();
                black_box(
                    depthwise
                        .train(
                            black_box(&dataset),
                            black_box(&binned),
                            targets_view,
                            WeightsView::None,
                            None,
                            Parallelism::Sequential,
                        )
                        .expect("training should succeed"),
                )
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("leafwise", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                let targets_view = dataset.targets().unwrap();
                black_box(
                    leafwise
                        .train(
                            black_box(&dataset),
                            black_box(&binned),
                            targets_view,
                            WeightsView::None,
                            None,
                            Parallelism::Sequential,
                        )
                        .expect("training should succeed"),
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
