//! Unified comparison benchmarks: booste-rs vs XGBoost GBLinear prediction.
//!
//! Run with: `cargo bench --features bench-xgboost --bench gblinear_prediction`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::models::load_linear_model;
#[cfg(feature = "bench-xgboost")]
use common::models::bench_models_dir;

use boosters::data::RowMatrix;
use boosters::testing::data::random_dense_f32;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "bench-xgboost")]
use criterion::BatchSize;

#[cfg(feature = "bench-xgboost")]
use xgb::{Booster, DMatrix};

// =============================================================================
// Standardized Dataset Sizes
// =============================================================================

/// Small batch: latency-sensitive scenarios
const SMALL_BATCH: usize = 100;
/// Medium batch: primary comparison point
const MEDIUM_BATCH: usize = 1_000;
/// Large batch: throughput scenarios
const LARGE_BATCH: usize = 10_000;

// =============================================================================
// XGBoost Helpers
// =============================================================================

#[cfg(feature = "bench-xgboost")]
fn load_xgb_model_bytes(name: &str) -> Vec<u8> {
	let path = bench_models_dir().join(format!("{name}.model.json"));
	std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read XGB model: {path:?}"))
}

#[cfg(feature = "bench-xgboost")]
fn new_xgb_booster(model_bytes: &[u8]) -> Booster {
	Booster::load_buffer(model_bytes).expect("Failed to load XGB model from buffer")
}

#[cfg(feature = "bench-xgboost")]
fn create_dmatrix(data: &[f32], num_rows: usize) -> DMatrix {
	DMatrix::from_dense(data, num_rows).expect("Failed to create DMatrix")
}

// =============================================================================
// Batch Size Comparison
// =============================================================================

fn bench_predict_batch_sizes(c: &mut Criterion) {
	let boosters_model = load_linear_model("bench_gblinear_medium");
	let num_features = boosters_model.num_features;

	#[cfg(feature = "bench-xgboost")]
	let xgb_model_bytes = load_xgb_model_bytes("bench_gblinear_medium");

	let mut group = c.benchmark_group("compare/predict/gblinear/batch_size/medium");

	for batch_size in [SMALL_BATCH, MEDIUM_BATCH, LARGE_BATCH] {
		let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
		let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

		group.throughput(Throughput::Elements(batch_size as u64));

		// booste-rs
		group.bench_with_input(BenchmarkId::new("boosters", batch_size), &matrix, |b, m| {
			b.iter(|| black_box(boosters_model.model.predict(black_box(m), &[])))
		});

		// XGBoost
		#[cfg(feature = "bench-xgboost")]
		{
			group.bench_function(BenchmarkId::new("xgboost/cold_dmatrix", batch_size), |b| {
				let booster = new_xgb_booster(&xgb_model_bytes);
				b.iter_batched(
					|| create_dmatrix(&input_data, batch_size),
					|dmatrix| {
						let out = booster.predict(black_box(&dmatrix)).unwrap();
						black_box(out)
					},
					BatchSize::SmallInput,
				)
			});
		}
	}

	group.finish();
}

// =============================================================================
// Single Row Latency
// =============================================================================

fn bench_predict_single_row(c: &mut Criterion) {
	let boosters_model = load_linear_model("bench_gblinear_medium");
	let num_features = boosters_model.num_features;

	#[cfg(feature = "bench-xgboost")]
	let xgb_model_bytes = load_xgb_model_bytes("bench_gblinear_medium");
	#[cfg(feature = "bench-xgboost")]
	let xgb_model = new_xgb_booster(&xgb_model_bytes);

	let input_data = random_dense_f32(1, num_features, 42, -5.0, 5.0);
	let matrix = RowMatrix::from_vec(input_data.clone(), 1, num_features);

	let mut group = c.benchmark_group("compare/predict/gblinear/single_row/medium");

	// booste-rs
	group.bench_function("boosters", |b| {
		b.iter(|| black_box(boosters_model.model.predict(black_box(&matrix), &[])))
	});

	// XGBoost
	#[cfg(feature = "bench-xgboost")]
	{
		group.bench_function("xgboost/cold_dmatrix", |b| {
			b.iter_batched(
				|| create_dmatrix(&input_data, 1),
				|dmatrix| {
					let out = xgb_model.predict(black_box(&dmatrix)).unwrap();
					black_box(out)
				},
				BatchSize::SmallInput,
			)
		});
	}

	group.finish();
}

// =============================================================================
// Model Size Scaling
// =============================================================================

fn bench_model_sizes(c: &mut Criterion) {
	let models = [
		("small", load_linear_model("bench_gblinear_small")),
		("medium", load_linear_model("bench_gblinear_medium")),
		("large", load_linear_model("bench_gblinear_large")),
	];

	#[cfg(feature = "bench-xgboost")]
	let xgb_models: Vec<(&str, Booster)> = models
		.iter()
		.map(|(name, _)| {
			let bytes = load_xgb_model_bytes(&format!("bench_gblinear_{}", name));
			(*name, new_xgb_booster(&bytes))
		})
		.collect();

	let mut group = c.benchmark_group("compare/predict/gblinear/model_scaling");
	let batch_size = MEDIUM_BATCH;

	for (name, model) in &models {
		let num_features = model.num_features;
		let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
		let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

		group.throughput(Throughput::Elements(batch_size as u64));

		// booste-rs
		group.bench_with_input(BenchmarkId::new("boosters", name), &matrix, |b, m| {
			b.iter(|| black_box(model.model.predict(black_box(m), &[])))
		});

		// XGBoost
		#[cfg(feature = "bench-xgboost")]
		{
			if let Some((_, xgb_model)) = xgb_models.iter().find(|(n, _)| n == name) {
				group.bench_function(BenchmarkId::new("xgboost/cold_dmatrix", name), |b| {
					b.iter_batched(
						|| create_dmatrix(&input_data, batch_size),
						|dmatrix| {
							let out = xgb_model.predict(black_box(&dmatrix)).unwrap();
							black_box(out)
						},
						BatchSize::SmallInput,
					)
				});
			}
		}
	}

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_predict_batch_sizes, bench_predict_single_row, bench_model_sizes
}
criterion_main!(benches);
