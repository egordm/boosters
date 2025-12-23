//! Component benchmarks: GBLinear training throughput.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{Dataset, FeaturesView};
use boosters::testing::data::synthetic_regression;
use boosters::training::{GBLinearParams, GBLinearTrainer, MulticlassLogLoss, Rmse, SoftmaxLoss, SquaredLoss, Verbosity};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

fn bench_gblinear_regression_train(c: &mut Criterion) {
	let num_features = 100;
	let mut group = c.benchmark_group("component/train/gblinear/regression");

	for num_rows in [1_000usize, 10_000, 50_000] {
		let syn_dataset = synthetic_regression(num_rows, num_features, 42, 0.05);
		// SyntheticDataset.features is already feature-major [n_features, n_samples]
		let features_view = FeaturesView::from_array(syn_dataset.features.view());
		let labels: Vec<f32> = syn_dataset.targets.to_vec();
		let dataset = Dataset::from_numeric(&features_view, labels).unwrap();

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

		group.throughput(Throughput::Elements((num_rows * num_features) as u64));
		group.bench_with_input(BenchmarkId::new("train", num_rows), &dataset, |b, ds| {
			b.iter(|| black_box(trainer.train(black_box(ds), &[])).unwrap())
		});
	}

	group.finish();
}

fn bench_gblinear_conversion_overhead(c: &mut Criterion) {
	let num_features = 100;
	let mut group = c.benchmark_group("component/train/gblinear/conversion");

	for num_rows in [1_000usize, 10_000, 50_000] {
		let syn_dataset = synthetic_regression(num_rows, num_features, 42, 0.05);
		let labels: Vec<f32> = syn_dataset.targets.to_vec();

		group.throughput(Throughput::Elements((num_rows * num_features) as u64));

		// Benchmark creating FeaturesView from Array2
		group.bench_with_input(BenchmarkId::new("array_to_features_view", num_rows), &syn_dataset.features, |b, features| {
			b.iter(|| black_box(FeaturesView::from_array(black_box(features.view()))))
		});

		// Benchmark creating Dataset from FeaturesView
		let features_view = FeaturesView::from_array(syn_dataset.features.view());
		group.bench_with_input(BenchmarkId::new("features_view_to_dataset", num_rows), &features_view, |b, fv| {
			let labels = labels.clone();
			b.iter(|| black_box(Dataset::from_numeric(black_box(fv), labels.clone())).unwrap())
		});
	}

	group.finish();
}

fn bench_gblinear_updater(c: &mut Criterion) {
	let num_features = 100;
	let num_rows = 10_000usize;

	let syn_dataset = synthetic_regression(num_rows, num_features, 42, 0.05);
	// SyntheticDataset.features is already feature-major [n_features, n_samples]
	let features_view = FeaturesView::from_array(syn_dataset.features.view());
	let labels: Vec<f32> = syn_dataset.targets.to_vec();
	let dataset = Dataset::from_numeric(&features_view, labels).unwrap();

	let mut group = c.benchmark_group("component/train/gblinear/updater");
	group.throughput(Throughput::Elements((num_rows * num_features) as u64));

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

	group.bench_with_input(BenchmarkId::new("sequential", num_rows), &dataset, |b, ds| {
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

	group.bench_with_input(BenchmarkId::new("parallel", num_rows), &dataset, |b, ds| {
		b.iter(|| black_box(trainer_par.train(black_box(ds), &[])).unwrap())
	});

	group.finish();
}

fn bench_gblinear_feature_scaling(c: &mut Criterion) {
	let num_rows = 10_000usize;
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

	for num_features in [10usize, 50, 100, 500, 1_000] {
		let dataset = synthetic_regression(num_rows, num_features, 42, 0.05);
		// Convert feature-major to row-major for RowMatrix
		let features_fm = dataset.features.view();
		let features: Vec<f32> = {
			let mut v = Vec::with_capacity(num_rows * num_features);
			for r in 0..num_rows {
				for f in 0..num_features {
					v.push(features_fm[(f, r)]);
				}
			}
			v
		};
		let labels: Vec<f32> = dataset.targets.to_vec();
		let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
		let col_matrix: ColMatrix = ColMatrix::from_data_matrix(&row_matrix);
		let dataset = Dataset::from_numeric(&col_matrix, labels).unwrap();

		group.throughput(Throughput::Elements((num_rows * num_features) as u64));
		group.bench_with_input(BenchmarkId::new("features", num_features), &dataset, |b, ds| {
			b.iter(|| black_box(trainer.train(black_box(ds), &[])).unwrap())
		});
	}

	group.finish();
}

fn bench_gblinear_multiclass(c: &mut Criterion) {
	let num_features = 50;
	let num_classes = 10;
	let mut group = c.benchmark_group("component/train/gblinear/multiclass");

	for num_rows in [1_000usize, 5_000, 10_000] {
		let dataset = synthetic_regression(num_rows, num_features, 42, 0.0);
		// Convert feature-major to row-major for RowMatrix
		let features_fm = dataset.features.view();
		let features: Vec<f32> = {
			let mut v = Vec::with_capacity(num_rows * num_features);
			for r in 0..num_rows {
				for f in 0..num_features {
					v.push(features_fm[(f, r)]);
				}
			}
			v
		};
		let mut rng = StdRng::seed_from_u64(1337);
		let labels: Vec<f32> = (0..num_rows)
			.map(|_| (rng.r#gen::<u32>() % num_classes as u32) as f32)
			.collect();

		let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
		let col_matrix: ColMatrix = ColMatrix::from_data_matrix(&row_matrix);
		let dataset = Dataset::from_numeric(&col_matrix, labels).unwrap();

		let params_seq = GBLinearParams {
			n_rounds: 10,
			learning_rate: 0.5,
			alpha: 0.0,
			lambda: 1.0,
			parallel: false,
			verbosity: Verbosity::Silent,
			..Default::default()
		};
		let trainer_seq = GBLinearTrainer::new(SoftmaxLoss::new(num_classes), MulticlassLogLoss, params_seq);

		let params_par = GBLinearParams {
			n_rounds: 10,
			learning_rate: 0.5,
			alpha: 0.0,
			lambda: 1.0,
			parallel: true,
			verbosity: Verbosity::Silent,
			..Default::default()
		};
		let trainer_par = GBLinearTrainer::new(SoftmaxLoss::new(num_classes), MulticlassLogLoss, params_par);

		group.throughput(Throughput::Elements((num_rows * num_features) as u64));
		group.bench_with_input(BenchmarkId::new("sequential", num_rows), &dataset, |b, ds| {
			b.iter(|| black_box(trainer_seq.train(black_box(ds), &[])).unwrap())
		});
		group.bench_with_input(BenchmarkId::new("parallel", num_rows), &dataset, |b, ds| {
			b.iter(|| black_box(trainer_par.train(black_box(ds), &[])).unwrap())
		});
	}

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_gblinear_regression_train,
		bench_gblinear_conversion_overhead,
		bench_gblinear_updater,
		bench_gblinear_feature_scaling,
		bench_gblinear_multiclass
}
criterion_main!(benches);
