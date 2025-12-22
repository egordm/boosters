//! Component benchmarks: traversal strategy comparisons.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::models::load_boosters_model;

use boosters::inference::gbdt::{Predictor, StandardTraversal, UnrolledTraversal6};
use boosters::testing::data::random_dense_f32;
use boosters::Parallelism;

use ndarray::Array2;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_gbtree_traversal_strategies(c: &mut Criterion) {
	let model = load_boosters_model("bench_medium");
	let forest = &model.forest;

	let std_no_block = Predictor::<StandardTraversal>::new(forest).with_block_size(100_000);
	let std_block64 = Predictor::<StandardTraversal>::new(forest).with_block_size(64);

	let unroll_no_block = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(100_000);
	let unroll_block64 = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

	let mut group = c.benchmark_group("component/predict/traversal/medium");

	for batch_size in [1_000usize, 10_000] {
		let input_data = random_dense_f32(batch_size, model.num_features, 42, -5.0, 5.0);
		let matrix = Array2::from_shape_vec((batch_size, model.num_features), input_data).unwrap();
		group.throughput(Throughput::Elements(batch_size as u64));

		group.bench_with_input(BenchmarkId::new("std_no_block", batch_size), &matrix, |b, m| {
			b.iter(|| black_box(std_no_block.predict(black_box(m.view()), Parallelism::Sequential)))
		});
		group.bench_with_input(BenchmarkId::new("std_block64", batch_size), &matrix, |b, m| {
			b.iter(|| black_box(std_block64.predict(black_box(m.view()), Parallelism::Sequential)))
		});

		group.bench_with_input(BenchmarkId::new("unroll_no_block", batch_size), &matrix, |b, m| {
			b.iter(|| black_box(unroll_no_block.predict(black_box(m.view()), Parallelism::Sequential)))
		});
		group.bench_with_input(BenchmarkId::new("unroll_block64", batch_size), &matrix, |b, m| {
			b.iter(|| black_box(unroll_block64.predict(black_box(m.view()), Parallelism::Sequential)))
		});
	}

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_gbtree_traversal_strategies
}
criterion_main!(benches);
