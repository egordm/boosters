//! GBTree traversal strategy comparison benchmarks.
//!
//! Compares different tree traversal strategies on GBTree models:
//! - Standard traversal (baseline)
//! - Unrolled traversal (depth-6 unrolling)
//! - SIMD traversal (wide crate, feature-gated)
//!
//! Each strategy tested with and without blocking.

mod bench_utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use bench_utils::{generate_random_input, load_boosters_model};
use booste_rs::data::RowMatrix;
use booste_rs::inference::gbdt::{Predictor, StandardTraversal, UnrolledTraversal6};

#[cfg(feature = "simd")]
use booste_rs::inference::gbdt::SimdTraversal6;

// =============================================================================
// Strategy Comparison Benchmarks
// =============================================================================

/// Comprehensive benchmark of all traversal strategy and blocking combinations.
///
/// Tests on medium GBTree model:
/// - Standard traversal: no-block, block-64
/// - Unrolled traversal: no-block, block-64
/// - SIMD traversal: no-block, block-64 (when simd feature enabled)
fn bench_gbtree_traversal_strategies(c: &mut Criterion) {
    let model = load_boosters_model("bench_medium");
    let forest = &model.forest;
    let num_features = model.num_features;

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

    let mut group = c.benchmark_group("gbtree/traversal/medium");

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
// Criterion Configuration
// =============================================================================

criterion_group!(benches, bench_gbtree_traversal_strategies);

criterion_main!(benches);
