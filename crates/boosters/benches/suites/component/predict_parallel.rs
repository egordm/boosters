//! Component benchmarks: thread scaling for prediction.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::models::load_boosters_model;
use common::threading::with_rayon_threads;

use boosters::data::FeaturesView;
use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
use boosters::testing::synthetic_datasets::random_features_array;
use boosters::Parallelism;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_gbtree_thread_scaling(c: &mut Criterion) {
	let model = load_boosters_model("bench_medium");
	let predictor = Predictor::<UnrolledTraversal6>::new(&model.forest).with_block_size(64);

	let batch_size = 10_000usize;
	let matrix = random_features_array(batch_size, model.n_features, 42, -5.0, 5.0);

	let mut group = c.benchmark_group("component/predict/thread_scaling/medium");
	group.throughput(Throughput::Elements(batch_size as u64));

	for &n_threads in common::matrix::THREAD_COUNTS {
		group.bench_with_input(BenchmarkId::new("par_predict", n_threads), &matrix, |b, m| {
			b.iter(|| {
				let features = FeaturesView::from_array(m.view());
				with_rayon_threads(n_threads, || black_box(predictor.predict(black_box(features), Parallelism::Parallel)))
			})
		});
	}

	// Sequential baseline
	group.bench_with_input(BenchmarkId::new("predict", 1), &matrix, |b, m| {
		let features = FeaturesView::from_array(m.view());
		b.iter(|| black_box(predictor.predict(black_box(features), Parallelism::Sequential)))
	});

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_gbtree_thread_scaling
}
criterion_main!(benches);
