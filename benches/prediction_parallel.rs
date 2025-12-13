//! GBTree parallel prediction benchmarks.
//!
//! Tests how prediction performance scales with thread count on GBTree models.
//! Optionally compares against XGBoost's parallel implementation.

mod bench_utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use bench_utils::{generate_random_input, load_boosters_model};
use booste_rs::data::RowMatrix;
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};

// =============================================================================
// Thread Scaling Benchmarks
// =============================================================================

/// Benchmark thread scaling with different core counts on medium GBTree model.
///
/// Compares sequential vs parallel prediction with controlled thread counts.
fn bench_gbtree_thread_scaling(c: &mut Criterion) {
    let model = load_boosters_model("bench_medium");
    let forest = &model.forest;
    let num_features = model.num_features;

    let thread_counts = [1, 2, 4, 8];
    let batch_size = 10_000;

    let input_data = generate_random_input(batch_size, num_features, 42);
    let matrix = RowMatrix::from_vec(input_data, batch_size, num_features);

    let predictor = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

    let mut group = c.benchmark_group("gbtree/thread_scaling/medium");
    group.throughput(Throughput::Elements(batch_size as u64));

    for &num_threads in &thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("parallel", num_threads),
            &matrix,
            |b, matrix| {
                b.iter(|| pool.install(|| black_box(predictor.par_predict(black_box(matrix)))));
            },
        );
    }

    // Sequential baseline
    group.bench_with_input(
        BenchmarkId::new("sequential", 1),
        &matrix,
        |b, matrix| {
            b.iter(|| black_box(predictor.predict(black_box(matrix))));
        },
    );

    group.finish();
}

// =============================================================================
// XGBoost Comparison (optional)
// =============================================================================

#[cfg(feature = "bench-xgboost")]
mod xgboost_comparison {
    use super::*;
    use bench_utils::bench_models_dir;
    use xgb::{Booster, DMatrix};

    /// Load an XGBoost model with single-threaded configuration.
    fn load_xgb_model(name: &str) -> Booster {
        let path = bench_models_dir().join(format!("{}.model.json", name));
        let mut booster =
            Booster::load(&path).unwrap_or_else(|_| panic!("Failed to load XGB model: {:?}", path));

        booster
            .set_param("nthread", "1")
            .expect("Failed to set nthread");
        booster
            .set_param("predictor", "cpu_predictor")
            .expect("Failed to set predictor");
        booster
    }

    fn create_dmatrix(data: &[f32], num_rows: usize) -> DMatrix {
        DMatrix::from_dense(data, num_rows).expect("Failed to create DMatrix")
    }

    /// Benchmark thread scaling comparison with XGBoost (medium GBTree model).
    pub fn bench_gbtree_thread_scaling_xgboost(c: &mut Criterion) {
        let boosters_model = load_boosters_model("bench_medium");
        let forest = &boosters_model.forest;
        let num_features = boosters_model.num_features;

        let thread_counts = [1, 2, 4, 8];
        let batch_size = 10_000;

        let input_data = generate_random_input(batch_size, num_features, 42);
        let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

        let predictor = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

        let mut group = c.benchmark_group("gbtree/xgboost_comparison/medium");

        for &num_threads in &thread_counts {
            // booste-rs parallel
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();

            group.bench_with_input(
                BenchmarkId::new("boosters", num_threads),
                &matrix,
                |b, matrix| {
                    b.iter(|| pool.install(|| black_box(predictor.par_predict(black_box(matrix)))));
                },
            );

            // XGBoost with configured thread count
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

    /// Benchmark booste-rs vs XGBoost on various batch sizes (medium GBTree model).
    pub fn bench_gbtree_xgboost_comparison(c: &mut Criterion) {
        let boosters_model = load_boosters_model("bench_medium");
        let xgb_model = load_xgb_model("bench_medium");
        let num_features = boosters_model.num_features;
        let forest = &boosters_model.forest;

        let predictor = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);

        let mut group = c.benchmark_group("gbtree/xgboost/batch_size/medium");

        for batch_size in [100, 1_000, 10_000].iter() {
            let input_data = generate_random_input(*batch_size, num_features, 42);
            let matrix = RowMatrix::from_vec(input_data.clone(), *batch_size, num_features);

            group.throughput(Throughput::Elements(*batch_size as u64));

            // booste-rs
            group.bench_with_input(
                BenchmarkId::new("boosters", batch_size),
                &matrix,
                |b, matrix| {
                    b.iter(|| black_box(predictor.predict(black_box(matrix))));
                },
            );

            // XGBoost (fresh DMatrix each iteration to avoid caching)
            let input_for_xgb = input_data.clone();
            let batch = *batch_size;
            group.bench_function(BenchmarkId::new("xgboost", batch_size), |b| {
                b.iter(|| {
                    let dmatrix = create_dmatrix(black_box(&input_for_xgb), batch);
                    let output = xgb_model.predict(&dmatrix).unwrap();
                    black_box(output)
                });
            });
        }

        group.finish();
    }

    /// Benchmark single-row latency comparison (medium GBTree model).
    pub fn bench_gbtree_single_row_xgboost(c: &mut Criterion) {
        let boosters_model = load_boosters_model("bench_medium");
        let xgb_model = load_xgb_model("bench_medium");
        let num_features = boosters_model.num_features;
        let forest = &boosters_model.forest;

        let predictor = Predictor::<UnrolledTraversal6>::new(forest).with_block_size(64);
        let input_data = generate_random_input(1, num_features, 42);

        let mut group = c.benchmark_group("gbtree/xgboost/single_row/medium");

        // booste-rs
        let matrix = RowMatrix::from_vec(input_data.clone(), 1, num_features);
        group.bench_function("boosters", |b| {
            b.iter(|| black_box(predictor.predict(black_box(&matrix))));
        });

        // XGBoost
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
criterion_group!(benches, bench_gbtree_thread_scaling);

#[cfg(feature = "bench-xgboost")]
criterion_group!(
    benches,
    bench_gbtree_thread_scaling,
    xgboost_comparison::bench_gbtree_thread_scaling_xgboost,
    xgboost_comparison::bench_gbtree_xgboost_comparison,
    xgboost_comparison::bench_gbtree_single_row_xgboost
);

criterion_main!(benches);
