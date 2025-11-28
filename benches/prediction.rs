//! Prediction benchmarks for booste-rs.
//!
//! Benchmarks cover:
//! - Single row vs batch prediction
//! - Different batch sizes (1, 10, 100, 1K, 10K)
//! - Different model sizes (small, medium, large)
//! - Comparison against XGBoost C++ (via `xgb` crate, optional)
//!
//! # Running benchmarks
//!
//! Basic benchmarks (booste-rs only):
//! ```bash
//! cargo bench
//! ```
//!
//! With XGBoost comparison (requires libclang):
//! ```bash
//! cargo bench --features bench-xgboost
//! ```
//!
//! # Results
//!
//! HTML reports are generated in `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use booste_rs::compat::XgbModel;
use booste_rs::data::DenseMatrix;
use booste_rs::model::{FeatureInfo, Model, ModelMeta, ModelSource};
use booste_rs::objective::Objective;

// =============================================================================
// Benchmark Data Setup
// =============================================================================

/// Path to benchmark models directory.
fn bench_models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/benchmark")
}

/// Generate random dense input data.
fn generate_random_input(num_rows: usize, num_features: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..num_rows * num_features)
        .map(|_| rng.r#gen::<f32>() * 10.0 - 5.0) // Range [-5, 5]
        .collect()
}

/// Load a booste-rs model from JSON file.
fn load_boosters_model(name: &str) -> Model {
    let path = bench_models_dir().join(format!("{}.model.json", name));
    let file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open model: {:?}", path));
    let xgb_model: XgbModel =
        serde_json::from_reader(file).expect("Failed to parse model");

    let booster = xgb_model.to_booster().expect("Failed to convert model");
    let num_features = xgb_model.learner.learner_model_param.num_feature as u32;
    let base_score = xgb_model.learner.learner_model_param.base_score;

    Model::new(
        booster,
        ModelMeta {
            num_features,
            num_groups: 1,
            base_score: vec![base_score],
            source: ModelSource::XGBoostJson { version: [2, 0, 0] },
        },
        FeatureInfo::default(),
        Objective::SquaredError,
    )
}

// =============================================================================
// Benchmark Groups
// =============================================================================

/// Benchmark different batch sizes on a medium model.
fn bench_batch_sizes(c: &mut Criterion) {
    let model = load_boosters_model("bench_medium");
    let num_features = model.num_features();

    let mut group = c.benchmark_group("batch_size");

    for batch_size in [1, 10, 100, 1_000, 10_000].iter() {
        let input_data = generate_random_input(*batch_size, num_features, 42);
        let matrix = DenseMatrix::from_vec(input_data, *batch_size, num_features);

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("boosters", batch_size),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let output = model.predict(black_box(matrix));
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different model sizes with fixed batch size.
fn bench_model_sizes(c: &mut Criterion) {
    let models = [
        ("small", "bench_small"),
        ("medium", "bench_medium"),
        ("large", "bench_large"),
    ];

    let batch_size = 1000;

    let mut group = c.benchmark_group("model_size");

    for (label, model_name) in models.iter() {
        // Try to load the model, skip if not found
        let model = match std::panic::catch_unwind(|| load_boosters_model(model_name)) {
            Ok(m) => m,
            Err(_) => {
                eprintln!("Skipping {} - model not found", model_name);
                continue;
            }
        };

        let num_features = model.num_features();
        let input_data = generate_random_input(batch_size, num_features, 42);
        let matrix = DenseMatrix::from_vec(input_data, batch_size, num_features);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(BenchmarkId::new("boosters", label), &matrix, |b, matrix| {
            b.iter(|| {
                let output = model.predict(black_box(matrix));
                black_box(output)
            });
        });
    }

    group.finish();
}

/// Benchmark single-row prediction latency.
fn bench_single_row(c: &mut Criterion) {
    let model = load_boosters_model("bench_medium");
    let num_features = model.num_features();

    let input_data = generate_random_input(1, num_features, 42);
    let matrix = DenseMatrix::from_vec(input_data, 1, num_features);

    c.bench_function("single_row/boosters", |b| {
        b.iter(|| {
            let output = model.predict(black_box(&matrix));
            black_box(output)
        });
    });
}

// =============================================================================
// XGBoost Comparison Benchmarks (optional)
// =============================================================================

#[cfg(feature = "bench-xgboost")]
mod xgboost_comparison {
    use super::*;
    use xgb::{Booster, DMatrix};

    /// Load an XGBoost model using the xgb crate.
    fn load_xgb_model(name: &str) -> Booster {
        let path = bench_models_dir().join(format!("{}.model.json", name));
        Booster::load(&path).unwrap_or_else(|_| panic!("Failed to load XGB model: {:?}", path))
    }

    /// Create XGBoost DMatrix from flat f32 data.
    fn create_dmatrix(data: &[f32], num_rows: usize) -> DMatrix {
        DMatrix::from_dense(data, num_rows).expect("Failed to create DMatrix")
    }

    /// Benchmark booste-rs vs XGBoost C++ on various batch sizes.
    ///
    /// Note: XGBoost caches predictions for the same DMatrix, so we must
    /// create a fresh DMatrix inside the benchmark loop for fair comparison.
    pub fn bench_comparison(c: &mut Criterion) {
        let boosters_model = load_boosters_model("bench_medium");
        let xgb_model = load_xgb_model("bench_medium");
        let num_features = boosters_model.num_features();

        let mut group = c.benchmark_group("comparison");

        for batch_size in [100, 1_000, 10_000].iter() {
            // booste-rs: can reuse the same matrix (no caching issues)
            let input_data = generate_random_input(*batch_size, num_features, 42);
            let matrix = DenseMatrix::from_vec(input_data.clone(), *batch_size, num_features);
            group.throughput(Throughput::Elements(*batch_size as u64));
            group.bench_with_input(
                BenchmarkId::new("boosters", batch_size),
                &matrix,
                |b, matrix| {
                    b.iter(|| {
                        let output = boosters_model.predict(black_box(matrix));
                        black_box(output)
                    });
                },
            );

            // XGBoost C++: must create fresh DMatrix each iteration to avoid caching
            // Store input data for reuse, but create DMatrix inside the loop
            let input_for_xgb = input_data.clone();
            let batch = *batch_size;
            group.bench_function(BenchmarkId::new("xgboost", batch_size), |b| {
                b.iter(|| {
                    // Create fresh DMatrix to avoid XGBoost's prediction caching
                    let dmatrix = create_dmatrix(black_box(&input_for_xgb), batch);
                    let output = xgb_model.predict(&dmatrix).unwrap();
                    black_box(output)
                });
            });
        }

        group.finish();
    }

    /// Benchmark single-row latency comparison.
    ///
    /// Note: XGBoost caches predictions for the same DMatrix, so we must
    /// create a fresh DMatrix inside the benchmark loop for fair comparison.
    pub fn bench_single_row_comparison(c: &mut Criterion) {
        let boosters_model = load_boosters_model("bench_medium");
        let xgb_model = load_xgb_model("bench_medium");
        let num_features = boosters_model.num_features();

        let input_data = generate_random_input(1, num_features, 42);

        let mut group = c.benchmark_group("single_row_comparison");

        // booste-rs: can reuse the same matrix (no caching issues)
        let matrix = DenseMatrix::from_vec(input_data.clone(), 1, num_features);
        group.bench_function("boosters", |b| {
            b.iter(|| {
                let output = boosters_model.predict(black_box(&matrix));
                black_box(output)
            });
        });

        // XGBoost C++: create fresh DMatrix each iteration to avoid caching
        group.bench_function("xgboost", |b| {
            b.iter(|| {
                let dmatrix = create_dmatrix(black_box(&input_data), 1);
                let output = xgb_model.predict(&dmatrix).unwrap();
                black_box(output)
            });
        });

        group.finish();
    }
}

// =============================================================================
// Criterion Configuration
// =============================================================================

#[cfg(not(feature = "bench-xgboost"))]
criterion_group!(benches, bench_batch_sizes, bench_model_sizes, bench_single_row);

#[cfg(feature = "bench-xgboost")]
criterion_group!(
    benches,
    bench_batch_sizes,
    bench_model_sizes,
    bench_single_row,
    xgboost_comparison::bench_comparison,
    xgboost_comparison::bench_single_row_comparison
);

criterion_main!(benches);
