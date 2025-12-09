//! GBLinear memory access pattern and gradient computation benchmarks.
//!
//! Low-level benchmarks for understanding performance characteristics:
//! - Row vs column access patterns
//! - Coordinate descent core operations
//! - Softmax gradient computation

mod bench_utils;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::prelude::*;

use bench_utils::generate_training_data;
use booste_rs::data::{CSCMatrix, ColMatrix, ColumnAccess, DataMatrix, RowMatrix, RowView};
use booste_rs::training::{GradientBuffer, Loss, SoftmaxLoss};

// =============================================================================
// Row Access Pattern Benchmarks
// =============================================================================

/// Benchmark row access patterns for different matrix layouts.
fn bench_gblinear_row_access(c: &mut Criterion) {
    let num_features = 100;
    let num_rows = 10_000;
    let (features, _) = generate_training_data(num_rows, num_features, 42);

    let row_matrix = RowMatrix::from_vec(features.clone(), num_rows, num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();

    let mut group = c.benchmark_group("gblinear/row_access");
    group.throughput(Throughput::Elements(num_rows as u64));

    // RowMajor: contiguous row access via row_slice()
    group.bench_function("row_major_slice", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for row_idx in 0..num_rows {
                let row = row_matrix.row_slice(row_idx);
                sum += row.iter().sum::<f32>();
            }
            black_box(sum)
        });
    });

    // RowMajor: via DataMatrix::row() trait method
    group.bench_function("row_major_trait", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for row_idx in 0..num_rows {
                let row = row_matrix.row(row_idx);
                for (_, val) in row.iter() {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });

    // ColMajor: strided row access via row_iter()
    group.bench_function("col_major_iter", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for row_idx in 0..num_rows {
                sum += col_matrix.row_iter(row_idx).sum::<f32>();
            }
            black_box(sum)
        });
    });

    // ColMajor: via DataMatrix::row() trait method
    group.bench_function("col_major_trait", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for row_idx in 0..num_rows {
                let row = col_matrix.row(row_idx);
                for (_, val) in row.iter() {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

// =============================================================================
// Column Access Pattern Benchmarks
// =============================================================================

/// Benchmark column access patterns for different matrix layouts.
fn bench_gblinear_column_access(c: &mut Criterion) {
    let num_features = 100;
    let num_rows = 10_000;
    let (features, _) = generate_training_data(num_rows, num_features, 42);

    let row_matrix = RowMatrix::from_vec(features.clone(), num_rows, num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();
    let csc_matrix = CSCMatrix::from_dense_full(&row_matrix);

    let mut group = c.benchmark_group("gblinear/column_access");
    group.throughput(Throughput::Elements(num_features as u64));

    // RowMajor: strided column access via col_iter()
    group.bench_function("row_major_iter", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for col_idx in 0..num_features {
                sum += row_matrix.col_iter(col_idx).sum::<f32>();
            }
            black_box(sum)
        });
    });

    // ColMajor: contiguous column access via col_slice()
    group.bench_function("col_major_slice", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for col_idx in 0..num_features {
                let col = col_matrix.col_slice(col_idx);
                sum += col.iter().sum::<f32>();
            }
            black_box(sum)
        });
    });

    // CSC: sparse column access
    group.bench_function("csc_column", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for col_idx in 0..num_features {
                for (_, val) in csc_matrix.column(col_idx) {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

// =============================================================================
// Coordinate Descent Core Operation
// =============================================================================

/// Benchmark the core coordinate descent operation: weighted column sum.
fn bench_gblinear_gradient_column_sum(c: &mut Criterion) {
    let num_features = 100;
    let num_rows = 50_000;

    let (features, _) = generate_training_data(num_rows, num_features, 42);

    let row_matrix = RowMatrix::from_vec(features.clone(), num_rows, num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();
    let csc_matrix = CSCMatrix::from_dense_full(&row_matrix);

    // Simulate gradients as GradientBuffer (SoA)
    let mut buffer = GradientBuffer::new(num_rows, 1);
    for i in 0..num_rows {
        buffer.set(i, 0, (i as f32) * 0.001 - 5.0, 1.0);
    }

    let mut group = c.benchmark_group("gblinear/gradient_sum");
    group.throughput(Throughput::Elements((num_rows * num_features) as u64));

    // Direct slice access
    group.bench_function("col_slice_direct", |b| {
        let grads = buffer.grads();
        let hess = buffer.hess_slice();
        b.iter(|| {
            let mut total_grad = 0.0f32;
            let mut total_hess = 0.0f32;
            for col_idx in 0..num_features {
                let col = col_matrix.col_slice(col_idx);
                for (i, &x) in col.iter().enumerate() {
                    total_grad += x * grads[i];
                    total_hess += x * x * hess[i];
                }
            }
            black_box((total_grad, total_hess))
        });
    });

    // ColumnAccess trait on ColMatrix
    group.bench_function("col_trait_dense", |b| {
        let grads = buffer.grads();
        let hess = buffer.hess_slice();
        b.iter(|| {
            let mut total_grad = 0.0f32;
            let mut total_hess = 0.0f32;
            for col_idx in 0..num_features {
                for (row, val) in col_matrix.column(col_idx) {
                    total_grad += val * grads[row];
                    total_hess += val * val * hess[row];
                }
            }
            black_box((total_grad, total_hess))
        });
    });

    // ColumnAccess trait on CSC
    group.bench_function("csc_trait", |b| {
        let grads = buffer.grads();
        let hess = buffer.hess_slice();
        b.iter(|| {
            let mut total_grad = 0.0f32;
            let mut total_hess = 0.0f32;
            for col_idx in 0..num_features {
                for (row, val) in csc_matrix.column(col_idx) {
                    total_grad += val * grads[row];
                    total_hess += val * val * hess[row];
                }
            }
            black_box((total_grad, total_hess))
        });
    });

    group.finish();
}

// =============================================================================
// Softmax Gradient Computation
// =============================================================================

/// Benchmark softmax gradient computation for multiclass GBLinear.
fn bench_gblinear_softmax_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("gblinear/softmax_gradient");
    group.sample_size(50);

    let num_classes = 10;

    for num_samples in [1_000, 10_000, 50_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let preds: Vec<f32> = (0..num_samples * num_classes)
            .map(|_| rng.r#gen::<f32>() * 4.0 - 2.0)
            .collect();
        let labels: Vec<f32> = (0..num_samples)
            .map(|_| (rng.r#gen::<f32>() * num_classes as f32).floor())
            .collect();

        let softmax_loss = SoftmaxLoss::new(num_classes);
        let mut buffer = GradientBuffer::new(num_samples, num_classes);

        group.throughput(Throughput::Elements((num_samples * num_classes) as u64));

        group.bench_function(
            criterion::BenchmarkId::new("softmax", num_samples),
            |b| {
                b.iter(|| {
                    softmax_loss.compute_gradients(
                        black_box(&preds),
                        black_box(&labels),
                        None,
                        &mut buffer,
                    );
                    black_box(buffer.grads()[0])
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
    bench_gblinear_row_access,
    bench_gblinear_column_access,
    bench_gblinear_gradient_column_sum,
    bench_gblinear_softmax_gradient,
);

criterion_main!(benches);
