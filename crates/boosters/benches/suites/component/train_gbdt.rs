//! Component benchmarks: GBDT (tree) training throughput and scaling.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::threading::with_rayon_threads;

use boosters::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor};
use boosters::testing::data::{
	random_dense_f32, synthetic_binary_targets_from_linear_score,
	synthetic_multiclass_targets_from_linear_scores, synthetic_regression_targets_linear,
};
use boosters::training::{
	GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, LogLoss, LogisticLoss, MulticlassLogLoss,
	Rmse, SoftmaxLoss, SquaredLoss,
};
use boosters::Parallelism;

use ndarray::ArrayView1;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn build_col_matrix(features_row_major: Vec<f32>, rows: usize, cols: usize) -> ColMatrix<f32> {
	let row: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features_row_major, rows, cols);
	row.to_layout()
}

fn bench_gbdt_quantize(c: &mut Criterion) {
	let mut group = c.benchmark_group("component/train/gbdt/quantize");
	group.sample_size(20);

	for (rows, cols) in [(10_000usize, 50usize), (50_000, 100), (100_000, 20)] {
		let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
		let col_matrix = build_col_matrix(features, rows, cols);

		group.throughput(Throughput::Elements((rows * cols) as u64));
		group.bench_with_input(BenchmarkId::new("to_binned/max_bins=256", format!("{rows}x{cols}")), &col_matrix, |b, m| {
			b.iter(|| black_box(BinnedDatasetBuilder::from_matrix(black_box(m), 256).build().unwrap()))
		});
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
		let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
		let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);
		let col_matrix = build_col_matrix(features, rows, cols);
		let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build().unwrap();

		let params = GBDTParams {
			n_trees,
			learning_rate: 0.3,
			growth_strategy: GrowthStrategy::DepthWise { max_depth },
			gain: GainParams { reg_lambda: 1.0, ..Default::default() },
			cache_size: 256,
			..Default::default()
		};
		let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

		group.throughput(Throughput::Elements((rows * cols) as u64));
		group.bench_function(BenchmarkId::new("train", name), |b| {
			b.iter(|| black_box(trainer.train(black_box(&binned), ArrayView1::from(black_box(&targets[..])), None, &[], Parallelism::Sequential).unwrap()))
		});
	}

	group.finish();
}

fn bench_gbdt_train_binary(c: &mut Criterion) {
	let mut group = c.benchmark_group("component/train/gbdt/train_binary");
	group.sample_size(10);

	let (rows, cols, n_trees, max_depth) = (50_000usize, 100usize, 50u32, 6u32);
	let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
	let targets = synthetic_binary_targets_from_linear_score(&features, rows, cols, 7, 0.2);
	let col_matrix = build_col_matrix(features, rows, cols);
	let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build().unwrap();

	let params = GBDTParams {
		n_trees,
		learning_rate: 0.3,
		growth_strategy: GrowthStrategy::DepthWise { max_depth },
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		cache_size: 256,
		..Default::default()
	};
	let trainer = GBDTTrainer::new(LogisticLoss, LogLoss, params);

	group.throughput(Throughput::Elements((rows * cols) as u64));
	group.bench_function("train_binary", |b| {
		b.iter(|| black_box(trainer.train(black_box(&binned), ArrayView1::from(black_box(&targets[..])), None, &[], Parallelism::Sequential).unwrap()))
	});
	group.finish();
}

fn bench_gbdt_train_multiclass(c: &mut Criterion) {
	let mut group = c.benchmark_group("component/train/gbdt/train_multiclass");
	group.sample_size(10);

	let (rows, cols, n_trees, max_depth, num_classes) = (20_000usize, 50usize, 30u32, 6u32, 10usize);
	let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
	let targets = synthetic_multiclass_targets_from_linear_scores(&features, rows, cols, num_classes, 99, 0.1);
	let col_matrix = build_col_matrix(features, rows, cols);
	let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build().unwrap();

	let params = GBDTParams {
		n_trees,
		learning_rate: 0.3,
		growth_strategy: GrowthStrategy::DepthWise { max_depth },
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
		cache_size: 256,
		..Default::default()
	};
	let trainer = GBDTTrainer::new(SoftmaxLoss::new(num_classes), MulticlassLogLoss, params);

	group.throughput(Throughput::Elements((rows * cols) as u64));
	group.bench_function("train_multiclass", |b| {
		b.iter(|| black_box(trainer.train(black_box(&binned), ArrayView1::from(black_box(&targets[..])), None, &[], Parallelism::Sequential).unwrap()))
	});
	group.finish();
}

fn bench_gbdt_thread_scaling(c: &mut Criterion) {
	let mut group = c.benchmark_group("component/train/gbdt/thread_scaling");
	group.sample_size(10);

	let (rows, cols, n_trees, max_depth) = (50_000usize, 100usize, 30u32, 6u32);
	let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
	let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);
	let col_matrix = build_col_matrix(features, rows, cols);
	let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build().unwrap();

	for &n_threads in common::matrix::THREAD_COUNTS {
		let params = GBDTParams {
			n_trees,
			learning_rate: 0.3,
			growth_strategy: GrowthStrategy::DepthWise { max_depth },
			gain: GainParams { reg_lambda: 1.0, ..Default::default() },
			cache_size: 256,
			..Default::default()
		};
		let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
		group.throughput(Throughput::Elements((rows * cols) as u64));
		group.bench_function(BenchmarkId::new("train", n_threads), |b| {
			b.iter(|| {
				with_rayon_threads(n_threads, || {
					black_box(trainer.train(black_box(&binned), ArrayView1::from(black_box(&targets[..])), None, &[], Parallelism::Parallel).unwrap())
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

	let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
	let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);
	let col_matrix = build_col_matrix(features, rows, cols);
	let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build().unwrap();

	let common = GBDTParams {
		n_trees,
		learning_rate: 0.1,
		gain: GainParams { reg_lambda: 1.0, ..Default::default() },
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
	group.bench_function(BenchmarkId::new("depthwise", format!("{rows}x{cols}")), |b| {
		b.iter(|| black_box(depthwise.train(black_box(&binned), ArrayView1::from(black_box(&targets[..])), None, &[], Parallelism::Sequential).unwrap()))
	});
	group.bench_function(BenchmarkId::new("leafwise", format!("{rows}x{cols}")), |b| {
		b.iter(|| black_box(leafwise.train(black_box(&binned), ArrayView1::from(black_box(&targets[..])), None, &[], Parallelism::Sequential).unwrap()))
	});

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
