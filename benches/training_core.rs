//! GBLinear training benchmarks.
//!
//! Benchmarks for linear model training:
//! - ColMajor vs CSC matrix formats
//! - Sequential vs Parallel (Shotgun) updaters
//! - Feature count scaling
//! - Multiclass training

mod bench_utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use bench_utils::generate_training_data;
use booste_rs::data::{CSCMatrix, ColMatrix, RowMatrix};
use booste_rs::training::{GBLinearTrainer, LossFunction, Verbosity};

// =============================================================================
// Matrix Format Benchmarks
// =============================================================================

/// Benchmark GBLinear training with different matrix formats.
fn bench_gblinear_matrix_formats(c: &mut Criterion) {
    let num_features = 100;

    // Build trainer once and reuse
    let trainer = GBLinearTrainer::builder()
        .loss(LossFunction::SquaredError)
        .num_rounds(10usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(false)
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("gblinear/matrix_format");

    for num_rows in [1_000, 10_000, 50_000] {
        let (features, labels) = generate_training_data(num_rows, num_features, 42);

        group.throughput(Throughput::Elements((num_rows * num_features) as u64));

        let row_matrix = RowMatrix::from_vec(features.clone(), num_rows, num_features);
        let col_matrix: ColMatrix = row_matrix.to_layout();
        let csc_matrix = CSCMatrix::from_dense_full(&row_matrix);

        // ColMajor
        group.bench_with_input(
            BenchmarkId::new("col_major", num_rows),
            &(&col_matrix, &labels),
            |b, (matrix, labels)| {
                b.iter(|| {
                    let model = trainer.train(black_box(*matrix), black_box(*labels), None, &[]);
                    black_box(model)
                });
            },
        );

        // CSC
        group.bench_with_input(
            BenchmarkId::new("csc", num_rows),
            &(&csc_matrix, &labels),
            |b, (matrix, labels)| {
                b.iter(|| {
                    let model = trainer.train(black_box(*matrix), black_box(*labels), None, &[]);
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark conversion overhead between formats.
fn bench_gblinear_conversion_overhead(c: &mut Criterion) {
    let num_features = 100;

    let mut group = c.benchmark_group("gblinear/conversion");

    for num_rows in [1_000, 10_000, 50_000] {
        let (features, _) = generate_training_data(num_rows, num_features, 42);
        let row_matrix = RowMatrix::from_vec(features.clone(), num_rows, num_features);

        group.throughput(Throughput::Elements((num_rows * num_features) as u64));

        // RowMajor -> ColMajor
        group.bench_with_input(
            BenchmarkId::new("row_to_col", num_rows),
            &row_matrix,
            |b, matrix| {
                b.iter(|| {
                    let col: ColMatrix = black_box(matrix).to_layout();
                    black_box(col)
                });
            },
        );

        // RowMajor -> CSC
        group.bench_with_input(
            BenchmarkId::new("row_to_csc", num_rows),
            &row_matrix,
            |b, matrix| {
                b.iter(|| {
                    let csc = CSCMatrix::from_dense_full(black_box(matrix));
                    black_box(csc)
                });
            },
        );

        // ColMajor -> RowMajor
        let col_matrix: ColMatrix = row_matrix.to_layout();
        group.bench_with_input(
            BenchmarkId::new("col_to_row", num_rows),
            &col_matrix,
            |b, matrix| {
                b.iter(|| {
                    let row: RowMatrix = black_box(matrix).to_layout();
                    black_box(row)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Updater Comparison
// =============================================================================

/// Benchmark GBLinear sequential vs parallel (shotgun) coordinate descent.
fn bench_gblinear_updater(c: &mut Criterion) {
    let num_features = 100;
    let num_rows = 10_000;
    let (features, labels) = generate_training_data(num_rows, num_features, 42);
    let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();

    let mut group = c.benchmark_group("gblinear/updater");
    group.throughput(Throughput::Elements((num_rows * num_features) as u64));

    // Sequential
    let trainer_seq = GBLinearTrainer::builder()
        .loss(LossFunction::SquaredError)
        .num_rounds(10usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(false)
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();

    group.bench_with_input(
        BenchmarkId::new("sequential", num_rows),
        &(&col_matrix, &labels),
        |b, (matrix, labels)| {
            b.iter(|| {
                let model = trainer_seq.train(black_box(*matrix), black_box(*labels), None, &[]);
                black_box(model)
            });
        },
    );

    // Parallel
    let trainer_par = GBLinearTrainer::builder()
        .loss(LossFunction::SquaredError)
        .num_rounds(10usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(true)
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();

    group.bench_with_input(
        BenchmarkId::new("parallel", num_rows),
        &(&col_matrix, &labels),
        |b, (matrix, labels)| {
            b.iter(|| {
                let model = trainer_par.train(black_box(*matrix), black_box(*labels), None, &[]);
                black_box(model)
            });
        },
    );

    group.finish();
}

// =============================================================================
// Feature Scaling
// =============================================================================

/// Benchmark how GBLinear training time scales with feature count.
fn bench_gblinear_feature_scaling(c: &mut Criterion) {
    let num_rows = 10_000;

    let trainer = GBLinearTrainer::builder()
        .loss(LossFunction::SquaredError)
        .num_rounds(10usize)
        .learning_rate(0.5f32)
        .alpha(0.0f32)
        .lambda(1.0f32)
        .parallel(true)
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("gblinear/feature_scaling");

    for num_features in [10, 50, 100, 500, 1000] {
        let (features, labels) = generate_training_data(num_rows, num_features, 42);
        let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
        let col_matrix: ColMatrix = row_matrix.to_layout();

        group.throughput(Throughput::Elements((num_rows * num_features) as u64));

        group.bench_with_input(
            BenchmarkId::new("features", num_features),
            &(&col_matrix, &labels),
            |b, (matrix, labels)| {
                b.iter(|| {
                    let model = trainer.train(black_box(*matrix), black_box(*labels), None, &[]);
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Multiclass Training
// =============================================================================

/// Benchmark GBLinear multiclass training.
fn bench_gblinear_multiclass(c: &mut Criterion) {
    let num_features = 50;
    let num_classes = 10;

    let mut group = c.benchmark_group("gblinear/multiclass");

    for num_rows in [1_000, 5_000, 10_000] {
        let (features, mut labels) = generate_training_data(num_rows, num_features, 42);
        for label in &mut labels {
            *label = (label.abs() * num_classes as f32) % num_classes as f32;
        }

        let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
        let col_matrix: ColMatrix = row_matrix.to_layout();
        let loss = LossFunction::Softmax { num_classes };

        group.throughput(Throughput::Elements((num_rows * num_features) as u64));

        // Sequential
        let trainer_seq = GBLinearTrainer::builder()
            .loss(loss.clone())
            .num_rounds(10usize)
            .learning_rate(0.5f32)
            .alpha(0.0f32)
            .lambda(1.0f32)
            .parallel(false)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("sequential", num_rows),
            &(&col_matrix, &labels),
            |b, (matrix, labels)| {
                b.iter(|| {
                    let model = trainer_seq.train(black_box(*matrix), black_box(*labels), None, &[]);
                    black_box(model)
                });
            },
        );

        // Parallel
        let trainer_par = GBLinearTrainer::builder()
            .loss(loss.clone())
            .num_rounds(10usize)
            .learning_rate(0.5f32)
            .alpha(0.0f32)
            .lambda(1.0f32)
            .parallel(true)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("parallel", num_rows),
            &(&col_matrix, &labels),
            |b, (matrix, labels)| {
                b.iter(|| {
                    let model = trainer_par.train(black_box(*matrix), black_box(*labels), None, &[]);
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_gblinear_matrix_formats,
    bench_gblinear_conversion_overhead,
    bench_gblinear_updater,
    bench_gblinear_feature_scaling,
    bench_gblinear_multiclass,
);

criterion_main!(benches);
