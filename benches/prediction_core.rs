//! Core GBTree prediction benchmarks.
//!
//! Tests basic prediction scenarios on GBTree models:
//! - Different batch sizes (1, 10, 100, 1K, 10K)
//! - Different model sizes (small, medium, large)
//! - Single row latency

mod bench_utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use bench_utils::{generate_random_input, load_boosters_model};
use booste_rs::data::RowMatrix;
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};

// =============================================================================
// Batch Size Benchmarks
// =============================================================================

/// Benchmark different batch sizes on a medium GBTree model.
fn bench_gbtree_batch_sizes(c: &mut Criterion) {
    let model = load_boosters_model("bench_medium");
    let num_features = model.num_features;
    let predictor = Predictor::<UnrolledTraversal6>::new(&model.forest);

    let mut group = c.benchmark_group("gbtree/batch_size");

    for batch_size in [1, 10, 100, 1_000, 10_000].iter() {
        let input_data = generate_random_input(*batch_size, num_features, 42);
        let matrix = RowMatrix::from_vec(input_data, *batch_size, num_features);

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("medium", batch_size),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let output = predictor.predict(black_box(matrix));
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Model Size Benchmarks
// =============================================================================

/// Benchmark different GBTree model sizes with fixed batch size.
fn bench_gbtree_model_sizes(c: &mut Criterion) {
    let models = [
        ("small", "bench_small"),
        ("medium", "bench_medium"),
        ("large", "bench_large"),
    ];

    let batch_size = 1000;

    let mut group = c.benchmark_group("gbtree/model_size");

    for (label, model_name) in models.iter() {
        // Try to load the model, skip if not found
        let model = match std::panic::catch_unwind(|| load_boosters_model(model_name)) {
            Ok(m) => m,
            Err(_) => {
                eprintln!("Skipping {} - model not found", model_name);
                continue;
            }
        };

        let num_features = model.num_features;
        let predictor = Predictor::<UnrolledTraversal6>::new(&model.forest);
        let input_data = generate_random_input(batch_size, num_features, 42);
        let matrix = RowMatrix::from_vec(input_data, batch_size, num_features);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(BenchmarkId::new(*label, batch_size), &matrix, |b, matrix| {
            b.iter(|| {
                let output = predictor.predict(black_box(matrix));
                black_box(output)
            });
        });
    }

    group.finish();
}

// =============================================================================
// Single Row Latency
// =============================================================================

/// Benchmark single-row prediction latency on medium GBTree model.
fn bench_gbtree_single_row(c: &mut Criterion) {
    let model = load_boosters_model("bench_medium");
    let num_features = model.num_features;
    let predictor = Predictor::<UnrolledTraversal6>::new(&model.forest);

    let input_data = generate_random_input(1, num_features, 42);
    let matrix = RowMatrix::from_vec(input_data, 1, num_features);

    c.bench_function("gbtree/single_row/medium", |b| {
        b.iter(|| {
            let output = predictor.predict(black_box(&matrix));
            black_box(output)
        });
    });
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_gbtree_batch_sizes,
    bench_gbtree_model_sizes,
    bench_gbtree_single_row
);

criterion_main!(benches);
