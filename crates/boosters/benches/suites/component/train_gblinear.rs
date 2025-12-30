//! Component benchmarks: GBLinear training throughput.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::binned::{BinningConfig, DatasetBuilder, BinnedDataset};
use boosters::data::{TargetsView, WeightsView};
use boosters::data::Dataset;
use boosters::testing::synthetic_datasets::synthetic_regression;
use boosters::training::{
    GBLinearParams, GBLinearTrainer, MulticlassLogLoss, Rmse, SoftmaxLoss, SquaredLoss, Verbosity,
};

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::prelude::*;

/// Helper to build a Dataset from synthetic data
fn make_dataset(features: ndarray::ArrayView2<'_, f32>, targets: &[f32]) -> Dataset {
    let targets_2d = ndarray::Array2::from_shape_vec((1, targets.len()), targets.to_vec()).unwrap();
    Dataset::new(features, Some(targets_2d.view()), None)
}

fn bench_gblinear_regression_train(c: &mut Criterion) {
    let n_features = 100;
    let mut group = c.benchmark_group("component/train/gblinear/regression");

    for n_rows in [1_000usize, 10_000, 50_000] {
        let syn_dataset = synthetic_regression(n_rows, n_features, 42, 0.05);
        // SyntheticDataset.features is already feature-major [n_features, n_samples]
        let labels: Vec<f32> = syn_dataset.targets.to_vec();
        let dataset = make_dataset(syn_dataset.features.view(), &labels);

        let params = GBLinearParams {
            n_rounds: 10,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: true,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);

        group.throughput(Throughput::Elements((n_rows * n_features) as u64));
        group.bench_with_input(BenchmarkId::new("train", n_rows), &dataset, |b, ds| {
            b.iter(|| black_box(trainer.train(black_box(ds), &[])).unwrap())
        });
    }

    group.finish();
}

fn bench_gblinear_updater(c: &mut Criterion) {
    let n_features = 100;
    let n_rows = 10_000usize;

    let syn_dataset = synthetic_regression(n_rows, n_features, 42, 0.05);
    let labels: Vec<f32> = syn_dataset.targets.to_vec();
    let dataset = make_dataset(syn_dataset.features.view(), &labels);

    let mut group = c.benchmark_group("component/train/gblinear/updater");
    group.throughput(Throughput::Elements((n_rows * n_features) as u64));

    let params_seq = GBLinearParams {
        n_rounds: 10,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: false,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer_seq = GBLinearTrainer::new(SquaredLoss, Rmse, params_seq);

    group.bench_with_input(BenchmarkId::new("sequential", n_rows), &dataset, |b, ds| {
        b.iter(|| black_box(trainer_seq.train(black_box(ds), &[])).unwrap())
    });

    let params_par = GBLinearParams {
        n_rounds: 10,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: true,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer_par = GBLinearTrainer::new(SquaredLoss, Rmse, params_par);

    group.bench_with_input(BenchmarkId::new("parallel", n_rows), &dataset, |b, ds| {
        b.iter(|| black_box(trainer_par.train(black_box(ds), &[])).unwrap())
    });

    group.finish();
}

fn bench_gblinear_feature_scaling(c: &mut Criterion) {
    let n_rows = 10_000usize;
    let mut group = c.benchmark_group("component/train/gblinear/feature_scaling");

    let params = GBLinearParams {
        n_rounds: 10,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel: true,
        verbosity: Verbosity::Silent,
        ..Default::default()
    };
    let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);

    for n_features in [10usize, 50, 100, 500, 1_000] {
        let syn_dataset = synthetic_regression(n_rows, n_features, 42, 0.05);
        let labels: Vec<f32> = syn_dataset.targets.to_vec();
        let dataset = make_dataset(syn_dataset.features.view(), &labels);

        group.throughput(Throughput::Elements((n_rows * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::new("features", n_features),
            &dataset,
            |b, ds| b.iter(|| black_box(trainer.train(black_box(ds), &[])).unwrap()),
        );
    }

    group.finish();
}

fn bench_gblinear_multiclass(c: &mut Criterion) {
    let n_features = 50;
    let n_classes = 10;
    let mut group = c.benchmark_group("component/train/gblinear/multiclass");

    for n_rows in [1_000usize, 5_000, 10_000] {
        let syn_dataset = synthetic_regression(n_rows, n_features, 42, 0.0);
        // SyntheticDataset.features is already feature-major [n_features, n_samples]
        let mut rng = StdRng::seed_from_u64(1337);
        let labels: Vec<f32> = (0..n_rows)
            .map(|_| (rng.r#gen::<u32>() % n_classes as u32) as f32)
            .collect();

        let dataset = make_dataset(syn_dataset.features.view(), &labels);

        let params_seq = GBLinearParams {
            n_rounds: 10,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer_seq =
            GBLinearTrainer::new(SoftmaxLoss::new(n_classes), MulticlassLogLoss, params_seq);

        let params_par = GBLinearParams {
            n_rounds: 10,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: true,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer_par =
            GBLinearTrainer::new(SoftmaxLoss::new(n_classes), MulticlassLogLoss, params_par);

        group.throughput(Throughput::Elements((n_rows * n_features) as u64));
        group.bench_with_input(BenchmarkId::new("sequential", n_rows), &dataset, |b, ds| {
            b.iter(|| black_box(trainer_seq.train(black_box(ds), &[])).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("parallel", n_rows), &dataset, |b, ds| {
            b.iter(|| black_box(trainer_par.train(black_box(ds), &[])).unwrap())
        });
    }

    group.finish();
}

/// Benchmark: train() with Dataset vs train_binned() with BinnedDataset
///
/// This measures the overhead of using BinnedDataset for GBLinear training.
/// Story 3.2: Key question is whether reusing BinnedDataset for GBLinear
/// incurs significant overhead compared to using Dataset directly.
fn bench_gblinear_binned_overhead(c: &mut Criterion) {
    let n_features = 100;
    let mut group = c.benchmark_group("component/train/gblinear/binned_overhead");
    group.sample_size(10);

    for n_rows in [1_000usize, 10_000, 50_000] {
        let syn_dataset = synthetic_regression(n_rows, n_features, 42, 0.05);
        let labels: Vec<f32> = syn_dataset.targets.to_vec();

        // Build Dataset (what train() uses)
        let dataset = make_dataset(syn_dataset.features.view(), &labels);

        // Build BinnedDataset (what train_binned() uses)
        // Need to transpose from feature-major [n_features, n_samples] to sample-major [n_samples, n_features]
        let features_sample_major: ndarray::Array2<f32> = {
            let fm = syn_dataset.features.view();
            let mut arr = ndarray::Array2::zeros((n_rows, n_features));
            for f in 0..n_features {
                for s in 0..n_rows {
                    arr[[s, f]] = fm[[f, s]];
                }
            }
            arr
        };
        let config = BinningConfig::default();
        let built = DatasetBuilder::from_array(features_sample_major.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();
        let binned_dataset = BinnedDataset::from_built_groups(built);

        // TargetsView: [n_outputs, n_samples]
        let targets_2d =
            ndarray::Array2::from_shape_vec((1, labels.len()), labels.clone()).unwrap();
        let targets_view = TargetsView::new(targets_2d.view());

        let params = GBLinearParams {
            n_rounds: 10,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: true,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);

        group.throughput(Throughput::Elements((n_rows * n_features) as u64));

        // Benchmark train() with Dataset
        group.bench_with_input(
            BenchmarkId::new("Dataset_train", n_rows),
            &dataset,
            |b, ds| b.iter(|| black_box(trainer.train(black_box(ds), &[])).unwrap()),
        );

        // Benchmark train_binned() with BinnedDataset
        group.bench_function(BenchmarkId::new("BinnedDataset_train_binned", n_rows), |b| {
            b.iter(|| {
                black_box(
                    trainer
                        .train_binned(
                            black_box(&binned_dataset),
                            targets_view,
                            WeightsView::None,
                        )
                        .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_gblinear_regression_train,
        bench_gblinear_updater,
        bench_gblinear_feature_scaling,
        bench_gblinear_multiclass,
        bench_gblinear_binned_overhead
}
criterion_main!(benches);
