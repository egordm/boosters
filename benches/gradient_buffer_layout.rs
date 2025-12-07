//! Benchmark comparing row-major vs column-major gradient buffer layouts.
//!
//! This benchmark helps decide whether to change GradientBuffer from row-major
//! (current) to column-major layout for better histogram building performance.
//!
//! Key scenarios:
//! 1. **Histogram building**: Access all samples for single output → column-major wins
//! 2. **Loss computation**: Access all outputs for single sample → row-major wins
//! 3. **Gradient copy to TreeGrower**: Column-major enables zero-copy
//!
//! Run with: cargo bench --bench gradient_buffer_layout

mod bench_utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

// =============================================================================
// Row-Major GradientBuffer (current implementation)
// =============================================================================

/// Row-major gradient buffer: [s0_o0, s0_o1, ..., s1_o0, s1_o1, ...]
/// Index: sample * n_outputs + output
#[derive(Clone)]
struct GradientBufferRowMajor {
    grads: Vec<f32>,
    hess: Vec<f32>,
    n_samples: usize,
    n_outputs: usize,
}

impl GradientBufferRowMajor {
    fn new(n_samples: usize, n_outputs: usize) -> Self {
        let size = n_samples * n_outputs;
        Self {
            grads: vec![0.0; size],
            hess: vec![0.0; size],
            n_samples,
            n_outputs,
        }
    }

    #[inline]
    fn get(&self, sample: usize, output: usize) -> (f32, f32) {
        let idx = sample * self.n_outputs + output;
        (self.grads[idx], self.hess[idx])
    }

    #[inline]
    fn set(&mut self, sample: usize, output: usize, grad: f32, hess: f32) {
        let idx = sample * self.n_outputs + output;
        self.grads[idx] = grad;
        self.hess[idx] = hess;
    }

    /// Get all outputs for a sample (contiguous slice)
    #[inline]
    fn sample_grads(&self, sample: usize) -> &[f32] {
        let start = sample * self.n_outputs;
        &self.grads[start..start + self.n_outputs]
    }

    #[inline]
    fn sample_hess(&self, sample: usize) -> &[f32] {
        let start = sample * self.n_outputs;
        &self.hess[start..start + self.n_outputs]
    }

    /// Get mutable slices for loss computation
    #[inline]
    fn as_mut_slices(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.grads, &mut self.hess)
    }

    /// Copy output to a new single-output buffer (for TreeGrower)
    fn copy_output(&self, output: usize) -> (Vec<f32>, Vec<f32>) {
        let mut grads = Vec::with_capacity(self.n_samples);
        let mut hess = Vec::with_capacity(self.n_samples);
        for sample in 0..self.n_samples {
            let (g, h) = self.get(sample, output);
            grads.push(g);
            hess.push(h);
        }
        (grads, hess)
    }
}

// =============================================================================
// Column-Major GradientBuffer (proposed)
// =============================================================================

/// Column-major gradient buffer: [s0_o0, s1_o0, ..., sN_o0, s0_o1, s1_o1, ...]
/// Index: output * n_samples + sample
/// 
/// Each output's gradients are contiguous, enabling zero-copy for TreeGrower.
#[derive(Clone)]
struct GradientBufferColMajor {
    grads: Vec<f32>,
    hess: Vec<f32>,
    n_samples: usize,
    n_outputs: usize,
}

impl GradientBufferColMajor {
    fn new(n_samples: usize, n_outputs: usize) -> Self {
        let size = n_samples * n_outputs;
        Self {
            grads: vec![0.0; size],
            hess: vec![0.0; size],
            n_samples,
            n_outputs,
        }
    }

    #[inline]
    fn get(&self, sample: usize, output: usize) -> (f32, f32) {
        let idx = output * self.n_samples + sample;
        (self.grads[idx], self.hess[idx])
    }

    #[inline]
    fn set(&mut self, sample: usize, output: usize, grad: f32, hess: f32) {
        let idx = output * self.n_samples + sample;
        self.grads[idx] = grad;
        self.hess[idx] = hess;
    }

    /// Get all samples for an output (contiguous slice) - zero-copy!
    #[inline]
    fn output_grads(&self, output: usize) -> &[f32] {
        let start = output * self.n_samples;
        &self.grads[start..start + self.n_samples]
    }

    #[inline]
    fn output_hess(&self, output: usize) -> &[f32] {
        let start = output * self.n_samples;
        &self.hess[start..start + self.n_samples]
    }

    /// Get mutable slices for an output
    #[inline]
    fn output_grads_mut(&mut self, output: usize) -> &mut [f32] {
        let start = output * self.n_samples;
        &mut self.grads[start..start + self.n_samples]
    }

    #[inline]
    fn output_hess_mut(&mut self, output: usize) -> &mut [f32] {
        let start = output * self.n_samples;
        &mut self.hess[start..start + self.n_samples]
    }
}

// =============================================================================
// Simulated Operations
// =============================================================================

/// Simulate histogram building: iterate all samples for one output
/// This is O(samples × features) and runs K times per round (K = num_outputs)
fn simulate_histogram_build_row_major(
    buffer: &GradientBufferRowMajor,
    output: usize,
    bins: &[u8],
    hist_grads: &mut [f32],
    hist_hess: &mut [f32],
) {
    for sample in 0..buffer.n_samples {
        let (grad, hess) = buffer.get(sample, output);
        let bin = bins[sample] as usize;
        hist_grads[bin] += grad;
        hist_hess[bin] += hess;
    }
}

fn simulate_histogram_build_col_major(
    grads: &[f32],  // Already a contiguous slice for this output
    hess: &[f32],
    bins: &[u8],
    hist_grads: &mut [f32],
    hist_hess: &mut [f32],
) {
    for (sample, &bin) in bins.iter().enumerate() {
        let bin = bin as usize;
        hist_grads[bin] += grads[sample];
        hist_hess[bin] += hess[sample];
    }
}

/// Simulate softmax loss computation: iterate all outputs for each sample
/// This is O(samples × outputs) and runs once per round
fn simulate_softmax_row_major(
    buffer: &mut GradientBufferRowMajor,
    preds: &[f32],
    labels: &[f32],
) {
    let num_classes = buffer.n_outputs;
    let num_samples = buffer.n_samples;
    let (grads, hess) = buffer.as_mut_slices();

    for i in 0..num_samples {
        let start = i * num_classes;
        let sample_preds = &preds[start..start + num_classes];
        let sample_grads = &mut grads[start..start + num_classes];
        let sample_hess = &mut hess[start..start + num_classes];
        let label = labels[i] as usize;

        // Compute softmax (simplified - just exp normalization)
        let max_pred = sample_preds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for k in 0..num_classes {
            let exp_val = (sample_preds[k] - max_pred).exp();
            sample_grads[k] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize and compute gradients
        for k in 0..num_classes {
            let prob = sample_grads[k] / sum_exp;
            let is_true = if k == label { 1.0 } else { 0.0 };
            sample_grads[k] = prob - is_true;
            sample_hess[k] = (prob * (1.0 - prob)).max(1e-16);
        }
    }
}

fn simulate_softmax_col_major(
    buffer: &mut GradientBufferColMajor,
    preds: &[f32],  // Still row-major: [s0_c0, s0_c1, ..., s1_c0, ...]
    labels: &[f32],
) {
    let num_classes = buffer.n_outputs;
    let num_samples = buffer.n_samples;

    // Need temporary storage for softmax computation
    let mut probs = vec![0.0f32; num_classes];

    for i in 0..num_samples {
        let start = i * num_classes;
        let sample_preds = &preds[start..start + num_classes];
        let label = labels[i] as usize;

        // Compute softmax
        let max_pred = sample_preds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for k in 0..num_classes {
            probs[k] = (sample_preds[k] - max_pred).exp();
            sum_exp += probs[k];
        }

        // Normalize and write gradients (strided writes to column-major)
        for k in 0..num_classes {
            let prob = probs[k] / sum_exp;
            let is_true = if k == label { 1.0 } else { 0.0 };
            buffer.set(i, k, prob - is_true, (prob * (1.0 - prob)).max(1e-16));
        }
    }
}

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_histogram_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_layout/histogram");

    // Configurations: (name, n_samples, n_outputs, n_features_simulated)
    let configs = [
        ("single_output_10k", 10_000, 1, 100),
        ("multi_output_10k_3class", 10_000, 3, 100),
        ("multi_output_10k_10class", 10_000, 10, 100),
        ("single_output_50k", 50_000, 1, 100),
        ("multi_output_50k_10class", 50_000, 10, 100),
    ];

    for (name, n_samples, n_outputs, n_features) in configs {
        let mut rng = StdRng::seed_from_u64(42);

        // Create buffers with random gradients
        let mut row_major = GradientBufferRowMajor::new(n_samples, n_outputs);
        let mut col_major = GradientBufferColMajor::new(n_samples, n_outputs);

        for sample in 0..n_samples {
            for output in 0..n_outputs {
                let grad = rng.r#gen::<f32>() * 2.0 - 1.0;
                let hess = rng.r#gen::<f32>().abs() + 0.1;
                row_major.set(sample, output, grad, hess);
                col_major.set(sample, output, grad, hess);
            }
        }

        // Random bin assignments (simulating quantized features)
        let bins: Vec<u8> = (0..n_samples).map(|_| rng.r#gen::<u8>()).collect();

        // For multi-output, we build histograms for each output
        // Throughput is samples × features × outputs (total histogram builds)
        group.throughput(Throughput::Elements((n_samples * n_features * n_outputs) as u64));

        // Row-major: strided access for histogram building
        group.bench_with_input(
            BenchmarkId::new("row_major", name),
            &(&row_major, &bins, n_features, n_outputs),
            |b, (buffer, bins, n_features, n_outputs)| {
                let mut hist_grads = vec![0.0f32; 256];
                let mut hist_hess = vec![0.0f32; 256];
                b.iter(|| {
                    // Simulate building histograms for all outputs and all features
                    for output in 0..*n_outputs {
                        for _feature in 0..*n_features {
                            hist_grads.fill(0.0);
                            hist_hess.fill(0.0);
                            simulate_histogram_build_row_major(
                                buffer, output, bins, &mut hist_grads, &mut hist_hess,
                            );
                        }
                    }
                    black_box(hist_grads[0])
                });
            },
        );

        // Column-major: contiguous access for histogram building
        group.bench_with_input(
            BenchmarkId::new("col_major", name),
            &(&col_major, &bins, n_features, n_outputs),
            |b, (buffer, bins, n_features, n_outputs)| {
                let mut hist_grads = vec![0.0f32; 256];
                let mut hist_hess = vec![0.0f32; 256];
                b.iter(|| {
                    for output in 0..*n_outputs {
                        let grads = buffer.output_grads(output);
                        let hess = buffer.output_hess(output);
                        for _feature in 0..*n_features {
                            hist_grads.fill(0.0);
                            hist_hess.fill(0.0);
                            simulate_histogram_build_col_major(
                                grads, hess, bins, &mut hist_grads, &mut hist_hess,
                            );
                        }
                    }
                    black_box(hist_grads[0])
                });
            },
        );
    }

    group.finish();
}

fn bench_softmax_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_layout/softmax");

    let configs = [
        ("10k_3class", 10_000, 3),
        ("10k_10class", 10_000, 10),
        ("50k_3class", 50_000, 3),
        ("50k_10class", 50_000, 10),
    ];

    for (name, n_samples, n_classes) in configs {
        let mut rng = StdRng::seed_from_u64(42);

        // Generate predictions (row-major as model output)
        let preds: Vec<f32> = (0..n_samples * n_classes)
            .map(|_| rng.r#gen::<f32>() * 4.0 - 2.0)
            .collect();

        // Generate labels
        let labels: Vec<f32> = (0..n_samples)
            .map(|_| (rng.r#gen::<f32>() * n_classes as f32).floor())
            .collect();

        group.throughput(Throughput::Elements((n_samples * n_classes) as u64));

        // Row-major: contiguous sample access
        {
            let mut buffer = GradientBufferRowMajor::new(n_samples, n_classes);
            group.bench_function(BenchmarkId::new("row_major", name), |b| {
                b.iter(|| {
                    simulate_softmax_row_major(&mut buffer, black_box(&preds), black_box(&labels));
                    black_box(buffer.grads[0])
                });
            });
        }

        // Column-major: strided sample access
        {
            let mut buffer = GradientBufferColMajor::new(n_samples, n_classes);
            group.bench_function(BenchmarkId::new("col_major", name), |b| {
                b.iter(|| {
                    simulate_softmax_col_major(&mut buffer, black_box(&preds), black_box(&labels));
                    black_box(buffer.grads[0])
                });
            });
        }
    }

    group.finish();
}

fn bench_gradient_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_layout/copy_to_grower");

    let configs = [
        ("10k_3class", 10_000, 3),
        ("10k_10class", 10_000, 10),
        ("50k_3class", 50_000, 3),
        ("50k_10class", 50_000, 10),
    ];

    for (name, n_samples, n_outputs) in configs {
        let mut rng = StdRng::seed_from_u64(42);

        let mut row_major = GradientBufferRowMajor::new(n_samples, n_outputs);
        let col_major_grads: Vec<f32> = (0..n_samples * n_outputs)
            .map(|_| rng.r#gen::<f32>())
            .collect();
        let col_major_hess: Vec<f32> = (0..n_samples * n_outputs)
            .map(|_| rng.r#gen::<f32>().abs() + 0.1)
            .collect();

        for sample in 0..n_samples {
            for output in 0..n_outputs {
                let idx = output * n_samples + sample;
                row_major.set(sample, output, col_major_grads[idx], col_major_hess[idx]);
            }
        }

        // Throughput: we're copying n_samples × n_outputs elements per output extracted
        group.throughput(Throughput::Elements((n_samples * n_outputs) as u64));

        // Row-major: must copy for each output
        group.bench_function(BenchmarkId::new("row_major_copy", name), |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for output in 0..n_outputs {
                    let (grads, hess) = row_major.copy_output(output);
                    total += grads[0] + hess[0];
                    black_box(&grads);
                    black_box(&hess);
                }
                black_box(total)
            });
        });

        // Column-major: zero-copy slice for each output
        group.bench_function(BenchmarkId::new("col_major_slice", name), |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for output in 0..n_outputs {
                    let start = output * n_samples;
                    let grads = &col_major_grads[start..start + n_samples];
                    let hess = &col_major_hess[start..start + n_samples];
                    total += grads[0] + hess[0];
                    black_box(grads);
                    black_box(hess);
                }
                black_box(total)
            });
        });
    }

    group.finish();
}

fn bench_end_to_end_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_layout/full_round");
    group.sample_size(20);

    // Simulate a full boosting round:
    // 1. Compute gradients (softmax)
    // 2. For each output: extract gradients, build histograms for all features

    let configs = [
        ("10k_3class_50feat", 10_000, 3, 50),
        ("10k_10class_100feat", 10_000, 10, 100),
        ("50k_3class_100feat", 50_000, 3, 100),
    ];

    for (name, n_samples, n_classes, n_features) in configs {
        let mut rng = StdRng::seed_from_u64(42);

        let preds: Vec<f32> = (0..n_samples * n_classes)
            .map(|_| rng.r#gen::<f32>() * 4.0 - 2.0)
            .collect();
        let labels: Vec<f32> = (0..n_samples)
            .map(|_| (rng.r#gen::<f32>() * n_classes as f32).floor())
            .collect();
        let bins: Vec<u8> = (0..n_samples).map(|_| rng.r#gen::<u8>()).collect();

        // Row-major approach (current)
        {
            let mut buffer = GradientBufferRowMajor::new(n_samples, n_classes);
            let mut hist_grads = vec![0.0f32; 256];
            let mut hist_hess = vec![0.0f32; 256];

            group.bench_function(BenchmarkId::new("row_major", name), |b| {
                b.iter(|| {
                    // 1. Compute gradients
                    simulate_softmax_row_major(&mut buffer, &preds, &labels);

                    // 2. For each output, copy and build histograms
                    for output in 0..n_classes {
                        // Copy gradients for this output (current approach)
                        let (grads, hess) = buffer.copy_output(output);

                        // Build histogram for each feature
                        for _feature in 0..n_features {
                            hist_grads.fill(0.0);
                            hist_hess.fill(0.0);
                            simulate_histogram_build_col_major(
                                &grads, &hess, &bins, &mut hist_grads, &mut hist_hess,
                            );
                        }
                    }
                    black_box(hist_grads[0])
                });
            });
        }

        // Column-major approach (proposed)
        {
            let mut buffer = GradientBufferColMajor::new(n_samples, n_classes);
            let mut hist_grads = vec![0.0f32; 256];
            let mut hist_hess = vec![0.0f32; 256];

            group.bench_function(BenchmarkId::new("col_major", name), |b| {
                b.iter(|| {
                    // 1. Compute gradients (slightly slower due to strided writes)
                    simulate_softmax_col_major(&mut buffer, &preds, &labels);

                    // 2. For each output, use zero-copy slices
                    for output in 0..n_classes {
                        // Zero-copy slice (proposed approach)
                        let grads = buffer.output_grads(output);
                        let hess = buffer.output_hess(output);

                        // Build histogram for each feature
                        for _feature in 0..n_features {
                            hist_grads.fill(0.0);
                            hist_hess.fill(0.0);
                            simulate_histogram_build_col_major(
                                grads, hess, &bins, &mut hist_grads, &mut hist_hess,
                            );
                        }
                    }
                    black_box(hist_grads[0])
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_histogram_building,
    bench_softmax_gradient,
    bench_gradient_copy,
    bench_end_to_end_round,
);

criterion_main!(benches);
