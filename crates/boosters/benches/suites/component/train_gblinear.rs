//! Component benchmarks: GBLinear training throughput.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{ColMatrix, Dataset, RowMatrix};
use boosters::testing::data::{random_dense_f32, synthetic_regression_targets_linear};
use boosters::training::{GBLinearParams, GBLinearTrainer, MulticlassLogLoss, Rmse, SoftmaxLoss, SquaredLoss, Verbosity};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

fn bench_gblinear_regression_train(c: &mut Criterion) {
	let num_features = 100;
	let mut group = c.benchmark_group("component/train/gblinear/regression");

	for num_rows in [1_000usize, 10_000, 50_000] {
		let features = random_dense_f32(num_rows, num_features, 42, -1.0, 1.0);
		let (labels, _w, _b) = synthetic_regression_targets_linear(&features, num_rows, num_features, 1337, 0.05);

		let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
		let col_matrix: ColMatrix = ColMatrix::from_data_matrix(&row_matrix);
		let dataset = Dataset::from_numeric(&col_matrix, labels).unwrap();

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
		let features = random_dense_f32(num_rows, num_features, 42, -1.0, 1.0);
		let (labels, _w, _b) = synthetic_regression_targets_linear(&features, num_rows, num_features, 1337, 0.05);
		let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);

		group.throughput(Throughput::Elements((num_rows * num_features) as u64));

		group.bench_with_input(BenchmarkId::new("row_to_col", num_rows), &row_matrix, |b, m| {
			b.iter(|| black_box(ColMatrix::from_data_matrix(black_box(m))))
		});

		let col_matrix: ColMatrix = ColMatrix::from_data_matrix(&row_matrix);
		group.bench_with_input(BenchmarkId::new("col_to_dataset", num_rows), &col_matrix, |b, m| {
			let labels = labels.clone();
			b.iter(|| black_box(Dataset::from_numeric(black_box(m), labels.clone())).unwrap())
		});
	}

	group.finish();
}

fn bench_gblinear_updater(c: &mut Criterion) {
	let num_features = 100;
	let num_rows = 10_000usize;

	let features = random_dense_f32(num_rows, num_features, 42, -1.0, 1.0);
	let (labels, _w, _b) = synthetic_regression_targets_linear(&features, num_rows, num_features, 1337, 0.05);
	let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
	let col_matrix: ColMatrix = ColMatrix::from_data_matrix(&row_matrix);
	let dataset = Dataset::from_numeric(&col_matrix, labels).unwrap();

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
		let features = random_dense_f32(num_rows, num_features, 42, -1.0, 1.0);
		let (labels, _w, _b) = synthetic_regression_targets_linear(&features, num_rows, num_features, 1337, 0.05);
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
		let features = random_dense_f32(num_rows, num_features, 42, -1.0, 1.0);
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
