//! Comparison benchmarks: booste-rs vs XGBoost prediction.
//!
//! Run with: `cargo bench --features bench-xgboost --bench prediction_xgboost`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::models::bench_models_dir;
use common::threading::with_rayon_threads;

use booste_rs::data::RowMatrix;
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
use booste_rs::testing::data::random_dense_f32;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, BatchSize};

use xgb::{Booster, DMatrix};

fn load_xgb_model_bytes(name: &str) -> Vec<u8> {
	let path = bench_models_dir().join(format!("{name}.model.json"));
	std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read XGB model: {path:?}"))
	// Note: we intentionally load from bytes (not from file) in the hot benches,
	// to avoid benchmarking filesystem overhead.
}

fn new_xgb_booster(model_bytes: &[u8]) -> Booster {
	let mut booster = Booster::load_buffer(model_bytes).expect("Failed to load XGB model from buffer");

	booster.set_param("predictor", "cpu_predictor").expect("Failed to set predictor");
	booster.set_param("nthread", "1").expect("Failed to set nthread");
	booster
}

fn reset_xgb_prediction_cache(booster: &mut Booster) {
	// Our benchmark fork adds a method to clear internal prediction caches.
	// This is intentionally best-effort (ignore error / ignore return value).
	let _ = booster.reset();
}

fn create_dmatrix(data: &[f32], num_rows: usize) -> DMatrix {
	DMatrix::from_dense(data, num_rows).expect("Failed to create DMatrix")
}

fn bench_xgboost_batch_sizes(c: &mut Criterion) {
	let boosters_model = common::models::load_boosters_model("bench_medium");
	let xgb_model_bytes = load_xgb_model_bytes("bench_medium");

	let num_features = boosters_model.num_features;
	let predictor = Predictor::<UnrolledTraversal6>::new(&boosters_model.forest).with_block_size(64);

	let mut group = c.benchmark_group("compare/predict/xgboost/batch_size/medium");

	for batch_size in [100usize, 1_000, 10_000] {
		let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
		let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

		group.throughput(Throughput::Elements(batch_size as u64));

		group.bench_with_input(BenchmarkId::new("boosters", batch_size), &matrix, |b, m| {
			b.iter(|| black_box(predictor.predict(black_box(m))))
		});

		group.bench_function(BenchmarkId::new("xgboost/cold_dmatrix", batch_size), |b| {
			let mut booster = new_xgb_booster(&xgb_model_bytes);
			b.iter_batched(
				|| create_dmatrix(&input_data, batch_size),
				|dmatrix| {
					reset_xgb_prediction_cache(&mut booster);
					let out = booster.predict(black_box(&dmatrix)).unwrap();
					black_box(out)
				},
				BatchSize::SmallInput,
			)
		});
	}

	group.finish();
}

fn bench_xgboost_single_row(c: &mut Criterion) {
	let boosters_model = common::models::load_boosters_model("bench_medium");
	let xgb_model_bytes = load_xgb_model_bytes("bench_medium");
	let mut xgb_model = new_xgb_booster(&xgb_model_bytes);

	let num_features = boosters_model.num_features;
	let predictor = Predictor::<UnrolledTraversal6>::new(&boosters_model.forest).with_block_size(64);

	let input_data = random_dense_f32(1, num_features, 42, -5.0, 5.0);
	let matrix = RowMatrix::from_vec(input_data.clone(), 1, num_features);
	let _dmatrix = create_dmatrix(&input_data, 1);

	let mut group = c.benchmark_group("compare/predict/xgboost/single_row/medium");

	group.bench_function("boosters", |b| b.iter(|| black_box(predictor.predict(black_box(&matrix)))));

	group.bench_function("xgboost/cold_dmatrix", |b| {
		b.iter_batched(
			|| create_dmatrix(&input_data, 1),
			|dmatrix| {
				reset_xgb_prediction_cache(&mut xgb_model);
				let out = xgb_model.predict(black_box(&dmatrix)).unwrap();
				black_box(out)
			},
			BatchSize::SmallInput,
		)
	});

	group.finish();
}

fn bench_xgboost_thread_scaling(c: &mut Criterion) {
	let boosters_model = common::models::load_boosters_model("bench_medium");
	let xgb_model_bytes = load_xgb_model_bytes("bench_medium");
	let mut xgb_model = new_xgb_booster(&xgb_model_bytes);

	let num_features = boosters_model.num_features;
	let predictor = Predictor::<UnrolledTraversal6>::new(&boosters_model.forest).with_block_size(64);

	let batch_size = 10_000usize;
	let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
	let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

	let mut group = c.benchmark_group("compare/predict/xgboost/thread_scaling/medium");
	group.throughput(Throughput::Elements(batch_size as u64));

	for &n_threads in common::matrix::THREAD_COUNTS {
		group.bench_with_input(BenchmarkId::new("boosters", n_threads), &matrix, |b, m| {
			b.iter(|| with_rayon_threads(n_threads, || black_box(predictor.par_predict(black_box(m)))))
		});

		let dmatrix = create_dmatrix(&input_data, batch_size);
		let threads = n_threads.to_string();
		xgb_model.set_param("nthread", &threads).expect("Failed to set nthread");
		reset_xgb_prediction_cache(&mut xgb_model);

		group.bench_function(BenchmarkId::new("xgboost/cold_cache", n_threads), |b| {
			b.iter(|| {
				reset_xgb_prediction_cache(&mut xgb_model);
				let out = xgb_model.predict(black_box(&dmatrix)).unwrap();
				black_box(out)
			})
		});
	}

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_xgboost_batch_sizes, bench_xgboost_single_row, bench_xgboost_thread_scaling
}
criterion_main!(benches);
