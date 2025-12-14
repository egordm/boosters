//! Component benchmarks: thread scaling for prediction.

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::models::load_boosters_model;
use common::threading::with_rayon_threads;

use booste_rs::data::RowMatrix;
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
use booste_rs::testing::data::random_dense_f32;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_gbtree_thread_scaling(c: &mut Criterion) {
	let model = load_boosters_model("bench_medium");
	let predictor = Predictor::<UnrolledTraversal6>::new(&model.forest).with_block_size(64);

	let batch_size = 10_000usize;
	let input_data = random_dense_f32(batch_size, model.num_features, 42, -5.0, 5.0);
	let matrix = RowMatrix::from_vec(input_data, batch_size, model.num_features);

	let mut group = c.benchmark_group("component/predict/thread_scaling/medium");
	group.throughput(Throughput::Elements(batch_size as u64));

	for &n_threads in common::matrix::THREAD_COUNTS {
		group.bench_with_input(BenchmarkId::new("par_predict", n_threads), &matrix, |b, m| {
			b.iter(|| {
				with_rayon_threads(n_threads, || black_box(predictor.par_predict(black_box(m))))
			})
		});
	}

	// Sequential baseline
	group.bench_with_input(BenchmarkId::new("predict", 1), &matrix, |b, m| {
		b.iter(|| black_box(predictor.predict(black_box(m))))
	});

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_gbtree_thread_scaling
}
criterion_main!(benches);
