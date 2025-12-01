//! Linear model training benchmarks for booste-rs.
//!
//! Benchmarks cover:
//! - ColMajor vs CSC matrix formats for training
//! - Different dataset sizes
//! - Sequential vs Parallel (Shotgun) updaters
//! - Feature count scaling
//!
//! # Running benchmarks
//!
//! ```bash
//! cargo bench --bench linear_training
//! ```
//!
//! # Results
//!
//! HTML reports are generated in `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

use booste_rs::data::{CSCMatrix, ColMatrix, DataMatrix, RowMatrix, RowView};
use booste_rs::linear::training::{LinearTrainer, LinearTrainerConfig};
use booste_rs::training::{SquaredLoss, Verbosity};

// =============================================================================
// Benchmark Data Setup
// =============================================================================

/// Generate random dense training data.
///
/// Returns (features, labels) where labels are a simple linear function of features.
fn generate_training_data(
    num_rows: usize,
    num_features: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate random features
    let features: Vec<f32> = (0..num_rows * num_features)
        .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0) // Range [-1, 1]
        .collect();

    // Generate random true weights
    let true_weights: Vec<f32> = (0..num_features).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
    let true_bias: f32 = rng.r#gen::<f32>() * 0.5;

    // Generate labels: y = X @ w + bias + noise
    let labels: Vec<f32> = (0..num_rows)
        .map(|row| {
            let row_start = row * num_features;
            let mut y = true_bias;
            for (j, &w) in true_weights.iter().enumerate() {
                y += features[row_start + j] * w;
            }
            y += rng.r#gen::<f32>() * 0.1 - 0.05; // Small noise
            y
        })
        .collect();

    (features, labels)
}

/// Create a standard trainer config for benchmarks.
fn bench_trainer_config(parallel: bool) -> LinearTrainerConfig {
    LinearTrainerConfig {
        num_rounds: 10,
        learning_rate: 0.5,
        alpha: 0.0,
        lambda: 1.0,
        parallel,
        verbosity: Verbosity::Silent, // Suppress training logs
        ..Default::default()
    }
}

// =============================================================================
// Matrix Format Comparison Benchmarks
// =============================================================================

/// Benchmark training with different matrix formats.
///
/// Compares actual training time (no conversion) for:
/// - ColMajor dense matrix (optimal for dense data)
/// - CSC sparse matrix (optimal for sparse data, tested with dense for comparison)
fn bench_training_formats(c: &mut Criterion) {
    let num_features = 100;
    let config = bench_trainer_config(false); // Sequential for fair comparison

    let mut group = c.benchmark_group("training_format");

    for num_rows in [1_000, 10_000, 50_000] {
        let (features, labels) = generate_training_data(num_rows, num_features, 42);

        group.throughput(Throughput::Elements((num_rows * num_features) as u64));

        // Create matrices once, outside the benchmark loop
        let row_matrix = RowMatrix::from_vec(features.clone(), num_rows, num_features);
        let col_matrix: ColMatrix = row_matrix.to_layout();
        let csc_matrix = CSCMatrix::from_dense_full(&row_matrix);

        // ColMajor - optimal for dense data
        group.bench_with_input(
            BenchmarkId::new("col_major", num_rows),
            &(&col_matrix, &labels),
            |b, (matrix, labels)| {
                let trainer = LinearTrainer::new(config.clone());
                b.iter(|| {
                    let model = trainer.train(black_box(*matrix), black_box(*labels), &SquaredLoss);
                    black_box(model)
                });
            },
        );

        // CSC - optimal for sparse data (here tested with dense for comparison)
        group.bench_with_input(
            BenchmarkId::new("csc", num_rows),
            &(&csc_matrix, &labels),
            |b, (matrix, labels)| {
                let trainer = LinearTrainer::new(config.clone());
                b.iter(|| {
                    let model = trainer.train(black_box(*matrix), black_box(*labels), &SquaredLoss);
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the matrix conversion overhead itself.
///
/// Measures time to convert between formats without training.
fn bench_conversion_overhead(c: &mut Criterion) {
    let num_features = 100;

    let mut group = c.benchmark_group("conversion_overhead");

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

        // ColMajor -> RowMajor (roundtrip test)
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
// Parallel vs Sequential Benchmarks
// =============================================================================

/// Benchmark sequential vs parallel (shotgun) coordinate descent updaters.
fn bench_updater_comparison(c: &mut Criterion) {
    let num_features = 100;
    let num_rows = 10_000;
    let (features, labels) = generate_training_data(num_rows, num_features, 42);
    let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();

    let mut group = c.benchmark_group("updater");
    group.throughput(Throughput::Elements((num_rows * num_features) as u64));

    // Sequential (coordinate descent)
    let seq_config = bench_trainer_config(false);
    group.bench_with_input(
        BenchmarkId::new("sequential", num_rows),
        &(&col_matrix, &labels),
        |b, (matrix, labels)| {
            let trainer = LinearTrainer::new(seq_config.clone());
            b.iter(|| {
                let model = trainer.train(black_box(*matrix), black_box(*labels), &SquaredLoss);
                black_box(model)
            });
        },
    );

    // Parallel (shotgun)
    let par_config = bench_trainer_config(true);
    group.bench_with_input(
        BenchmarkId::new("parallel", num_rows),
        &(&col_matrix, &labels),
        |b, (matrix, labels)| {
            let trainer = LinearTrainer::new(par_config.clone());
            b.iter(|| {
                let model = trainer.train(black_box(*matrix), black_box(*labels), &SquaredLoss);
                black_box(model)
            });
        },
    );

    group.finish();
}

// =============================================================================
// Feature Count Scaling Benchmarks
// =============================================================================

/// Benchmark how training time scales with number of features.
fn bench_feature_scaling(c: &mut Criterion) {
    let num_rows = 10_000;
    let config = bench_trainer_config(true); // Use parallel for speed

    let mut group = c.benchmark_group("feature_scaling");

    for num_features in [10, 50, 100, 500, 1000] {
        let (features, labels) = generate_training_data(num_rows, num_features, 42);
        let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
        let col_matrix: ColMatrix = row_matrix.to_layout();

        group.throughput(Throughput::Elements((num_rows * num_features) as u64));

        group.bench_with_input(
            BenchmarkId::new("features", num_features),
            &(&col_matrix, &labels),
            |b, (matrix, labels)| {
                let trainer = LinearTrainer::new(config.clone());
                b.iter(|| {
                    let model = trainer.train(black_box(*matrix), black_box(*labels), &SquaredLoss);
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Row Access Pattern Benchmarks
// =============================================================================

/// Benchmark row access patterns for different layouts.
///
/// This measures the raw row iteration cost, which is relevant for:
/// - Computing predictions during training
/// - Computing gradients
fn bench_row_access(c: &mut Criterion) {
    let num_features = 100;
    let num_rows = 10_000;
    let (features, _) = generate_training_data(num_rows, num_features, 42);

    let row_matrix = RowMatrix::from_vec(features.clone(), num_rows, num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();

    let mut group = c.benchmark_group("row_access");
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

/// Benchmark column access patterns for different layouts.
///
/// This measures the raw column iteration cost, which is relevant for:
/// - Computing feature-wise gradient sums during coordinate descent
fn bench_column_access(c: &mut Criterion) {
    let num_features = 100;
    let num_rows = 10_000;
    let (features, _) = generate_training_data(num_rows, num_features, 42);

    let row_matrix = RowMatrix::from_vec(features.clone(), num_rows, num_features);
    let col_matrix: ColMatrix = row_matrix.to_layout();
    let csc_matrix = CSCMatrix::from_dense_full(&row_matrix);

    let mut group = c.benchmark_group("column_access");
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
// Coordinate Descent Core Operation Benchmark
// =============================================================================

/// Benchmark the core coordinate descent operation: weighted column sum.
///
/// This is the actual computation done in each feature update:
/// sum_i (x[i,j] * gradient[i])
///
/// Compares:
/// - Direct slice access (col_slice + zip) - optimal for dense
/// - ColumnAccess trait (column() iterator) - generic abstraction
/// - CSC column iterator - for sparse data
fn bench_cd_core_operation(c: &mut Criterion) {
    use booste_rs::data::ColumnAccess;
    use booste_rs::training::GradientBuffer;
    
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

    let mut group = c.benchmark_group("gradient_column_sum");
    group.throughput(Throughput::Elements((num_rows * num_features) as u64));

    // 1. Direct slice access with GradientBuffer
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

    // 2. ColumnAccess trait on ColMatrix with GradientBuffer
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

    // 3. ColumnAccess trait on CSC with GradientBuffer
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
// Multiclass Training Benchmarks
// =============================================================================

/// Benchmark multiclass training.
fn bench_multiclass_training(c: &mut Criterion) {
    use booste_rs::training::SoftmaxLoss;

    let num_features = 50;
    let num_classes = 10;

    let mut group = c.benchmark_group("multiclass_training");

    for num_rows in [1_000, 5_000, 10_000] {
        let (features, mut labels) = generate_training_data(num_rows, num_features, 42);
        // Convert labels to class indices
        for label in &mut labels {
            *label = (label.abs() * num_classes as f32) as f32 % num_classes as f32;
        }

        let row_matrix = RowMatrix::from_vec(features, num_rows, num_features);
        let col_matrix: ColMatrix = row_matrix.to_layout();
        let loss = SoftmaxLoss::new(num_classes);

        group.throughput(Throughput::Elements((num_rows * num_features) as u64));

        let config = bench_trainer_config(false);

        group.bench_with_input(
            BenchmarkId::new("sequential", num_rows),
            &(&col_matrix, &labels),
            |b, (matrix, labels)| {
                let trainer = LinearTrainer::new(config.clone());
                b.iter(|| {
                    let model = trainer.train_multiclass(black_box(*matrix), black_box(*labels), &loss);
                    black_box(model)
                });
            },
        );

        let config_par = bench_trainer_config(true);
        group.bench_with_input(
            BenchmarkId::new("parallel", num_rows),
            &(&col_matrix, &labels),
            |b, (matrix, labels)| {
                let trainer = LinearTrainer::new(config_par.clone());
                b.iter(|| {
                    let model = trainer.train_multiclass(black_box(*matrix), black_box(*labels), &loss);
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
    bench_training_formats,
    bench_conversion_overhead,
    bench_updater_comparison,
    bench_feature_scaling,
    bench_row_access,
    bench_column_access,
    bench_cd_core_operation,
    bench_multiclass_training,
);

criterion_main!(benches);
