//! Component benchmarks: core prediction throughput/latency.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::models::load_boosters_model;

use boosters::data::RowMatrix;
use boosters::inference::gbdt::{Predictor, StandardTraversal, UnrolledTraversal6};
use boosters::testing::data::random_dense_f32;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_gbtree_batch_sizes(c: &mut Criterion) {
	let model = load_boosters_model("bench_medium");
	let num_features = model.num_features;
	let predictor = Predictor::<UnrolledTraversal6>::new(&model.forest);

	let mut group = c.benchmark_group("component/predict/batch_size");

	for batch_size in [1usize, 10, 100, 1_000, 10_000] {
		let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
		let matrix = RowMatrix::from_vec(input_data, batch_size, num_features);

		group.throughput(Throughput::Elements(batch_size as u64));
		group.bench_with_input(BenchmarkId::new("medium", batch_size), &matrix, |b, matrix| {
			b.iter(|| {
				let output = predictor.predict(black_box(matrix));
				black_box(output)
			});
		});
	}

	group.finish();
}

fn bench_gbtree_model_sizes(c: &mut Criterion) {
	let models = [("small", "bench_small"), ("medium", "bench_medium"), ("large", "bench_large")];
	let batch_size = 1_000usize;

	let mut group = c.benchmark_group("component/predict/model_size");

	for (label, model_name) in models {
		let model = match std::panic::catch_unwind(|| load_boosters_model(model_name)) {
			Ok(m) => m,
			Err(_) => {
				eprintln!("Skipping {label} - model not found");
				continue;
			}
		};

		let predictor = Predictor::<UnrolledTraversal6>::new(&model.forest);
		let input_data = random_dense_f32(batch_size, model.num_features, 42, -5.0, 5.0);
		let matrix = RowMatrix::from_vec(input_data, batch_size, model.num_features);

		group.throughput(Throughput::Elements(batch_size as u64));
		group.bench_with_input(BenchmarkId::new(label, batch_size), &matrix, |b, matrix| {
			b.iter(|| {
				let output = predictor.predict(black_box(matrix));
				black_box(output)
			});
		});
	}

	group.finish();
}

fn bench_gbtree_single_row(c: &mut Criterion) {
	let model = load_boosters_model("bench_medium");
	let predictor = Predictor::<UnrolledTraversal6>::new(&model.forest);

	let input_data = random_dense_f32(1, model.num_features, 42, -5.0, 5.0);
	let matrix = RowMatrix::from_vec(input_data, 1, model.num_features);

	c.bench_function("component/predict/single_row/medium", |b| {
		b.iter(|| {
			let output = predictor.predict(black_box(&matrix));
			black_box(output)
		})
	});
}

/// Compare traversal strategies: Standard vs Unrolled
fn bench_traversal_strategies(c: &mut Criterion) {
	let model = load_boosters_model("bench_medium");
	let num_features = model.num_features;
	
	let batch_size = 10_000usize;
	let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
	let matrix = RowMatrix::from_vec(input_data, batch_size, num_features);

	let mut group = c.benchmark_group("component/predict/traversal");
	group.throughput(Throughput::Elements(batch_size as u64));

	// Standard traversal (baseline)
	let standard = Predictor::<StandardTraversal>::new(&model.forest).with_block_size(64);
	group.bench_with_input(BenchmarkId::new("standard", batch_size), &matrix, |b, m| {
		b.iter(|| black_box(standard.predict(black_box(m))))
	});

	// Unrolled traversal (6 levels)
	let unrolled = Predictor::<UnrolledTraversal6>::new(&model.forest).with_block_size(64);
	group.bench_with_input(BenchmarkId::new("unrolled6", batch_size), &matrix, |b, m| {
		b.iter(|| black_box(unrolled.predict(black_box(m))))
	});

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_gbtree_batch_sizes, bench_gbtree_model_sizes, bench_gbtree_single_row, bench_traversal_strategies
}
criterion_main!(benches);
