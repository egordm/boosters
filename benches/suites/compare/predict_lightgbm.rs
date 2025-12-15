//! Comparison benchmarks: booste-rs vs LightGBM prediction.
//!
//! Run with: `cargo bench --features bench-lightgbm --bench prediction_lightgbm`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
use common::models::bench_models_dir;

use booste_rs::compat::LgbModel;
use booste_rs::data::RowMatrix;
use booste_rs::inference::gbdt::{Forest, Predictor, ScalarLeaf, UnrolledTraversal6};
use booste_rs::testing::data::random_dense_f32;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use std::fs::File;
use std::io::Read;

fn load_boosters_lgb_model(name: &str) -> (Forest<ScalarLeaf>, usize) {
	let path = bench_models_dir().join(format!("{name}.lgb.txt"));
	let mut file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open model: {path:?}"));
	let mut content = String::new();
	file.read_to_string(&mut content).expect("Failed to read model");
	let lgb_model = LgbModel::from_string(&content).unwrap_or_else(|e| panic!("Failed to parse model: {e}"));

	let num_features = lgb_model.num_features() as usize;
	let forest = lgb_model.to_forest().expect("Failed to convert model to forest");

	(forest, num_features)
}

fn bench_lightgbm_batch_sizes(c: &mut Criterion) {
	let (forest, num_features) = load_boosters_lgb_model("bench_medium");
	let lgb_booster = lightgbm3::Booster::from_file(
		bench_models_dir().join("bench_medium.lgb.txt").to_str().unwrap(),
	)
	.expect("Failed to load LightGBM model");

	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);

	let mut group = c.benchmark_group("compare/predict/lightgbm/batch_size/medium");

	for batch_size in [100usize, 1_000, 10_000] {
		let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
		let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

		group.throughput(Throughput::Elements(batch_size as u64));

		group.bench_with_input(BenchmarkId::new("boosters", batch_size), &matrix, |b, m| {
			b.iter(|| black_box(predictor.predict(black_box(m))))
		});

		let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
		let mut input_f64_b = input_f64_a.clone();
		if let Some(first) = input_f64_b.first_mut() {
			*first = f64::from_bits(first.to_bits().wrapping_add(1));
		}
		let num_feat = num_features as i32;
		group.bench_function(BenchmarkId::new("lightgbm", batch_size), |b| {
			let mut flip = false;
			b.iter(|| {
				flip = !flip;
				let input = if flip { &input_f64_a } else { &input_f64_b };
				let output = lgb_booster.predict(black_box(input), num_feat, true).unwrap();
				black_box(output)
			})
		});
	}

	group.finish();
}

fn bench_lightgbm_single_row(c: &mut Criterion) {
	let (forest, num_features) = load_boosters_lgb_model("bench_medium");
	let lgb_booster = lightgbm3::Booster::from_file(
		bench_models_dir().join("bench_medium.lgb.txt").to_str().unwrap(),
	)
	.expect("Failed to load LightGBM model");

	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
	let input_data = random_dense_f32(1, num_features, 42, -5.0, 5.0);
	let matrix = RowMatrix::from_vec(input_data.clone(), 1, num_features);

	let mut group = c.benchmark_group("compare/predict/lightgbm/single_row/medium");

	group.bench_function("boosters", |b| b.iter(|| black_box(predictor.predict(black_box(&matrix)))));

	let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
	let mut input_f64_b = input_f64_a.clone();
	if let Some(first) = input_f64_b.first_mut() {
		*first = f64::from_bits(first.to_bits().wrapping_add(1));
	}
	let num_feat = num_features as i32;
	group.bench_function("lightgbm", |b| {
		let mut flip = false;
		b.iter(|| {
			flip = !flip;
			let input = if flip { &input_f64_a } else { &input_f64_b };
			let output = lgb_booster.predict(black_box(input), num_feat, true).unwrap();
			black_box(output)
		})
	});

	group.finish();
}

fn bench_lightgbm_model_sizes(c: &mut Criterion) {
	let models = [("small", "bench_small"), ("medium", "bench_medium"), ("large", "bench_large")];
	let batch_size = 1_000usize;

	let mut group = c.benchmark_group("compare/predict/lightgbm/model_size");

	for (label, model_name) in models {
		let (forest, num_features) = match std::panic::catch_unwind(|| load_boosters_lgb_model(model_name)) {
			Ok(m) => m,
			Err(_) => {
				eprintln!("Skipping {model_name} - model not found");
				continue;
			}
		};

		let lgb_path = bench_models_dir().join(format!("{model_name}.lgb.txt"));
		let lgb_booster = match lightgbm3::Booster::from_file(lgb_path.to_str().unwrap()) {
			Ok(b) => b,
			Err(_) => {
				eprintln!("Skipping {model_name} - LightGBM model not found");
				continue;
			}
		};

		let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
		let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);
		let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);

		group.throughput(Throughput::Elements(batch_size as u64));

		group.bench_with_input(BenchmarkId::new(format!("{label}/boosters"), batch_size), &matrix, |b, m| {
			b.iter(|| black_box(predictor.predict(black_box(m))))
		});

		let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
		let mut input_f64_b = input_f64_a.clone();
		if let Some(first) = input_f64_b.first_mut() {
			*first = f64::from_bits(first.to_bits().wrapping_add(1));
		}
		let num_feat = num_features as i32;
		group.bench_function(BenchmarkId::new(format!("{label}/lightgbm"), batch_size), |b| {
			let mut flip = false;
			b.iter(|| {
				flip = !flip;
				let input = if flip { &input_f64_a } else { &input_f64_b };
				let output = lgb_booster.predict(black_box(input), num_feat, true).unwrap();
				black_box(output)
			})
		});
	}

	group.finish();
}

fn bench_lightgbm_parallel(c: &mut Criterion) {
	let (forest, num_features) = load_boosters_lgb_model("bench_medium");
	let lgb_booster = lightgbm3::Booster::from_file(
		bench_models_dir().join("bench_medium.lgb.txt").to_str().unwrap(),
	)
	.expect("Failed to load LightGBM model");

	let thread_counts = common::matrix::THREAD_COUNTS;
	let batch_size = 10_000usize;

	let input_data = random_dense_f32(batch_size, num_features, 42, -5.0, 5.0);
	let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);
	let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);

	let mut group = c.benchmark_group("compare/predict/lightgbm/parallel/medium");
	group.throughput(Throughput::Elements(batch_size as u64));

	for &n_threads in thread_counts {
		group.bench_with_input(BenchmarkId::new("boosters", n_threads), &matrix, |b, m| {
			b.iter(|| common::threading::with_rayon_threads(n_threads, || black_box(predictor.par_predict(black_box(m)))))
		});

		if n_threads == 1 {
			let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
			let mut input_f64_b = input_f64_a.clone();
			if let Some(first) = input_f64_b.first_mut() {
				*first = f64::from_bits(first.to_bits().wrapping_add(1));
			}
			let num_feat = num_features as i32;
			group.bench_function(BenchmarkId::new("lightgbm", "default"), |b| {
				let mut flip = false;
				b.iter(|| {
					flip = !flip;
					let input = if flip { &input_f64_a } else { &input_f64_b };
					let output = lgb_booster.predict(black_box(input), num_feat, true).unwrap();
					black_box(output)
				})
			});
		}
	}

	group.finish();
}

criterion_group! {
	name = benches;
	config = default_criterion();
	targets = bench_lightgbm_batch_sizes, bench_lightgbm_single_row, bench_lightgbm_model_sizes, bench_lightgbm_parallel
}
criterion_main!(benches);
