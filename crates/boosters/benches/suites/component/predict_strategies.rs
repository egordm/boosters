//! Component benchmarks: traversal strategy comparisons.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::models::load_boosters_model;

use boosters::data::FeaturesView;
use boosters::inference::gbdt::{Predictor, StandardTraversal, UnrolledTraversal6};
use boosters::testing::synthetic_datasets::random_features_array;
use boosters::Parallelism;

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
		let matrix = random_features_array(batch_size, model.n_features, 42, -5.0, 5.0);
		let features = FeaturesView::from_array(matrix.view());
		group.throughput(Throughput::Elements(batch_size as u64));

		group.bench_with_input(BenchmarkId::new("std_no_block", batch_size), &features, |b, f| {
			b.iter(|| black_box(std_no_block.predict(black_box(*f), Parallelism::Sequential)))
		});
		group.bench_with_input(BenchmarkId::new("std_block64", batch_size), &features, |b, f| {
			b.iter(|| black_box(std_block64.predict(black_box(*f), Parallelism::Sequential)))
		});

		group.bench_with_input(BenchmarkId::new("unroll_no_block", batch_size), &features, |b, f| {
			b.iter(|| black_box(unroll_no_block.predict(black_box(*f), Parallelism::Sequential)))
		});
		group.bench_with_input(BenchmarkId::new("unroll_block64", batch_size), &features, |b, f| {
			b.iter(|| black_box(unroll_block64.predict(black_box(*f), Parallelism::Sequential)))
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
