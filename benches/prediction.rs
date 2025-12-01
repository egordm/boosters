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
use booste_rs::data::RowMatrix;
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
        let matrix = RowMatrix::from_vec(input_data, *batch_size, num_features);

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
        let matrix = RowMatrix::from_vec(input_data, batch_size, num_features);

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
    let matrix = RowMatrix::from_vec(input_data, 1, num_features);

    c.bench_function("single_row/boosters", |b| {
        b.iter(|| {
            let output = model.predict(black_box(&matrix));
            black_box(output)
        });
    });
}

/// Comprehensive benchmark of all traversal strategy and blocking combinations.
///
/// Tests:
/// - Standard traversal: no-block, block-64
/// - Unrolled traversal: no-block, block-64
/// - SIMD traversal: no-block, block-64 (when simd feature enabled)
///
/// This helps identify whether blocking helps each strategy independently.
fn bench_all_combinations(c: &mut Criterion) {
    use booste_rs::predict::{Predictor, StandardTraversal, UnrolledTraversal6};
    #[cfg(feature = "simd")]
    use booste_rs::predict::SimdTraversal6;

    let model = load_boosters_model("bench_medium");
    let forest = model.booster.forest().expect("Benchmark model must be tree-based");
    let num_features = model.num_features();

    // Standard traversal combinations
    let std_no_block = Predictor::<StandardTraversal>::new(forest).with_block_size(100_000);
    let std_block64 = Predictor::<StandardTraversal>::new(forest).with_block_size(64);

    // Unrolled traversal combinations
    let unroll_no_block = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(100_000);
    let unroll_block64 = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

    // SIMD traversal combinations (wide crate)
    #[cfg(feature = "simd")]
    let simd_no_block = Predictor::<SimdTraversal6>::new(forest).with_block_size(100_000);
    #[cfg(feature = "simd")]
    let simd_block64 = Predictor::<SimdTraversal6>::new(forest).with_block_size(64);

    let mut group = c.benchmark_group("all_combinations");

    for batch_size in [1_000, 10_000].iter() {
        let input_data = generate_random_input(*batch_size, num_features, 42);
        let matrix = RowMatrix::from_vec(input_data, *batch_size, num_features);

        group.throughput(Throughput::Elements(*batch_size as u64));

        // Standard traversal
        group.bench_with_input(
            BenchmarkId::new("std_no_block", batch_size),
            &matrix,
            |b, matrix| b.iter(|| black_box(std_no_block.predict(black_box(matrix)))),
        );
        group.bench_with_input(
            BenchmarkId::new("std_block64", batch_size),
            &matrix,
            |b, matrix| b.iter(|| black_box(std_block64.predict(black_box(matrix)))),
        );

        // Unrolled traversal
        group.bench_with_input(
            BenchmarkId::new("unroll_no_block", batch_size),
            &matrix,
            |b, matrix| b.iter(|| black_box(unroll_no_block.predict(black_box(matrix)))),
        );
        group.bench_with_input(
            BenchmarkId::new("unroll_block64", batch_size),
            &matrix,
            |b, matrix| b.iter(|| black_box(unroll_block64.predict(black_box(matrix)))),
        );

        // SIMD traversal (wide crate)
        #[cfg(feature = "simd")]
        {
            group.bench_with_input(
                BenchmarkId::new("simd_no_block", batch_size),
                &matrix,
                |b, matrix| b.iter(|| black_box(simd_no_block.predict(black_box(matrix)))),
            );
            group.bench_with_input(
                BenchmarkId::new("simd_block64", batch_size),
                &matrix,
                |b, matrix| b.iter(|| black_box(simd_block64.predict(black_box(matrix)))),
            );
        }
    }

    group.finish();
}

// =============================================================================
// XGBoost Comparison Benchmarks (optional)
// =============================================================================

#[cfg(feature = "bench-xgboost")]
mod xgboost_comparison {
    use super::*;
    use xgb::{Booster, DMatrix};

    /// Load an XGBoost model using the xgb crate.
    ///
    /// **Important**: Configures XGBoost to use:
    /// - Single thread (nthread=1) for fair comparison with single-threaded booste-rs
    /// - CPU-only prediction (no GPU/CUDA) even if available
    fn load_xgb_model(name: &str) -> Booster {
        let path = bench_models_dir().join(format!("{}.model.json", name));
        let mut booster = Booster::load(&path)
            .unwrap_or_else(|_| panic!("Failed to load XGB model: {:?}", path));

        // Force single-thread for fair comparison
        // XGBoost uses OpenMP by default which would be unfair vs our single-threaded code
        booster.set_param("nthread", "1").expect("Failed to set nthread");

        // Force CPU predictor - disable GPU even if CUDA is available
        booster.set_param("predictor", "cpu_predictor").expect("Failed to set predictor");

        booster
    }

    /// Create XGBoost DMatrix from flat f32 data.
    fn create_dmatrix(data: &[f32], num_rows: usize) -> DMatrix {
        DMatrix::from_dense(data, num_rows).expect("Failed to create DMatrix")
    }

    /// Benchmark booste-rs vs XGBoost C++ on various batch sizes.
    ///
    /// Note: XGBoost caches predictions for the same DMatrix, so we must
    /// create a fresh DMatrix inside the benchmark loop for fair comparison.
    ///
    /// Uses UnrolledTraversal6+Block64 (our fastest) vs XGBoost single-thread.
    pub fn bench_comparison(c: &mut Criterion) {
        use booste_rs::predict::{Predictor, UnrolledTraversal6};

        let boosters_model = load_boosters_model("bench_medium");
        let xgb_model = load_xgb_model("bench_medium");
        let num_features = boosters_model.num_features();
        let forest = boosters_model.booster.forest().expect("Benchmark model must be tree-based");

        // Use our fastest configuration: UnrolledTraversal6 + Block64
        let predictor = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

        let mut group = c.benchmark_group("comparison");

        for batch_size in [100, 1_000, 10_000].iter() {
            // booste-rs: can reuse the same matrix (no caching issues)
            let input_data = generate_random_input(*batch_size, num_features, 42);
            let matrix = RowMatrix::from_vec(input_data.clone(), *batch_size, num_features);
            group.throughput(Throughput::Elements(*batch_size as u64));
            group.bench_with_input(
                BenchmarkId::new("boosters", batch_size),
                &matrix,
                |b, matrix| {
                    b.iter(|| {
                        let output = predictor.predict(black_box(matrix));
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
    ///
    /// Uses UnrolledTraversal6+Block64 (our fastest) vs XGBoost single-thread.
    pub fn bench_single_row_comparison(c: &mut Criterion) {
        use booste_rs::predict::{Predictor, UnrolledTraversal6};

        let boosters_model = load_boosters_model("bench_medium");
        let xgb_model = load_xgb_model("bench_medium");
        let num_features = boosters_model.num_features();
        let forest = boosters_model.booster.forest().expect("Benchmark model must be tree-based");

        // Use our fastest configuration
        let predictor = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

        let input_data = generate_random_input(1, num_features, 42);

        let mut group = c.benchmark_group("single_row_comparison");

        // booste-rs: can reuse the same matrix (no caching issues)
        let matrix = RowMatrix::from_vec(input_data.clone(), 1, num_features);
        group.bench_function("boosters", |b| {
            b.iter(|| {
                let output = predictor.predict(black_box(&matrix));
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

    /// Benchmark thread scaling with XGBoost comparison.
    ///
    /// Tests different thread counts for both booste-rs and XGBoost.
    pub fn bench_thread_scaling_xgboost(c: &mut Criterion) {
        use booste_rs::predict::{Predictor, UnrolledTraversal6};

        let boosters_model = load_boosters_model("bench_medium");
        let forest = boosters_model.booster.forest().expect("Benchmark model must be tree-based");
        let num_features = boosters_model.num_features();

        // Test different thread counts
        let thread_counts = [1, 2, 4, 8];
        let batch_size = 10_000;

        let input_data = generate_random_input(batch_size, num_features, 42);
        let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

        let predictor = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

        let mut group = c.benchmark_group("thread_scaling_comparison");
        group.throughput(Throughput::Elements(batch_size as u64));

        for &num_threads in &thread_counts {
            // Create a separate thread pool for booste-rs
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();

            // booste-rs parallel
            group.bench_with_input(
                BenchmarkId::new("boosters", num_threads),
                &matrix,
                |b, matrix| {
                    b.iter(|| {
                        pool.install(|| black_box(predictor.par_predict(black_box(matrix))))
                    });
                },
            );

            // XGBoost with configured thread count (must create fresh DMatrix to avoid caching)
            let input_for_xgb = input_data.clone();
            let mut xgb_booster = load_xgb_model("bench_medium");
            xgb_booster
                .set_param("nthread", &num_threads.to_string())
                .expect("Failed to set nthread");

            group.bench_function(BenchmarkId::new("xgboost", num_threads), |b| {
                b.iter(|| {
                    let dmatrix = create_dmatrix(black_box(&input_for_xgb), batch_size);
                    let output = xgb_booster.predict(&dmatrix).unwrap();
                    black_box(output)
                });
            });
        }

        group.finish();
    }
}

// =============================================================================
// Thread Scaling Benchmarks
// =============================================================================

/// Benchmark thread scaling with different core counts.
///
/// Compares sequential vs parallel prediction with controlled thread counts.
/// Both booste-rs and XGBoost are tested with the same thread counts for fairness.
fn bench_thread_scaling(c: &mut Criterion) {
    use booste_rs::predict::{Predictor, UnrolledTraversal6};

    let model = load_boosters_model("bench_medium");
    let forest = model.booster.forest().expect("Benchmark model must be tree-based");
    let num_features = model.num_features();

    // Test different thread counts
    let thread_counts = [1, 2, 4, 8];
    let batch_size = 10_000;

    let input_data = generate_random_input(batch_size, num_features, 42);
    let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

    let predictor = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

    let mut group = c.benchmark_group("thread_scaling");
    group.throughput(Throughput::Elements(batch_size as u64));

    for &num_threads in &thread_counts {
        // Create a separate thread pool for each thread count
        // Note: The global pool can only be set once, so we use install() with a custom pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("boosters_parallel", num_threads),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    // Run prediction in the custom thread pool
                    pool.install(|| black_box(predictor.par_predict(black_box(matrix))))
                });
            },
        );
    }

    // Sequential baseline (single-threaded)
    group.bench_with_input(
        BenchmarkId::new("boosters_sequential", 1),
        &matrix,
        |b, matrix| {
            b.iter(|| black_box(predictor.predict(black_box(matrix))));
        },
    );

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

#[cfg(not(feature = "bench-xgboost"))]
criterion_group!(
    benches,
    bench_batch_sizes,
    bench_model_sizes,
    bench_single_row,
    bench_all_combinations,
    bench_thread_scaling
);

#[cfg(feature = "bench-xgboost")]
criterion_group!(
    benches,
    bench_batch_sizes,
    bench_model_sizes,
    bench_single_row,
    bench_all_combinations,
    bench_thread_scaling,
    xgboost_comparison::bench_thread_scaling_xgboost,
    xgboost_comparison::bench_comparison,
    xgboost_comparison::bench_single_row_comparison
);

criterion_main!(benches);
