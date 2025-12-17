//! Component benchmarks: Split finding optimizations.
//!
//! This benchmark isolates the split finding hot path to measure:
//! 1. Current gain computation performance
//! 2. Parent score precomputation (proposed optimization)
//! 3. Early termination in split scans
//!
//! Run with: cargo bench --bench split_finding

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::fast_criterion;

use booste_rs::training::gbdt::split::{GainParams, GreedySplitter};
use booste_rs::training::gbdt::histograms::{FeatureMeta, HistogramBin, HistogramPool};
use booste_rs::training::gbdt::Parallelism;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// =============================================================================
// Test Data Generation
// =============================================================================

/// Generate realistic histogram data for benchmarking.
fn generate_histograms(
    n_features: usize,
    n_bins: u32,
    n_samples: u32,
    seed: u64,
) -> (HistogramPool, f64, f64, u32) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    
    // Create feature metadata
    let features: Vec<FeatureMeta> = (0..n_features)
        .map(|i| FeatureMeta {
            offset: (i as u32) * n_bins,
            n_bins,
        })
        .collect();
    
    let mut pool = HistogramPool::new(features, 4, 4);
    let slot = pool.acquire(0).slot();
    
    // Fill with realistic gradient data
    let mut total_grad = 0.0f64;
    let mut total_hess = 0.0f64;
    
    {
        let view = pool.slot_mut(slot);
        for bin in view.bins.iter_mut() {
            // Simulate gradient accumulation from samples
            let grad: f64 = rng.gen_range(-10.0..10.0);
            let hess: f64 = rng.gen_range(0.5..5.0);
            *bin = (grad, hess);
            total_grad += grad;
            total_hess += hess;
        }
    }
    
    (pool, total_grad, total_hess, n_samples)
}

// =============================================================================
// Baseline: Current Implementation
// =============================================================================

fn bench_current_gain_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/split/gain_computation");
    
    let params = GainParams::default();
    
    // Simulate typical workload: 256 bins × 100 features = 25600 candidates per node
    let n_candidates = 25_600u64;
    
    // Pre-generate test data
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let candidates: Vec<(f64, f64, f64, f64)> = (0..n_candidates)
        .map(|_| {
            let left_grad: f64 = rng.gen_range(-10.0..10.0);
            let left_hess: f64 = rng.gen_range(1.0..50.0);
            let right_grad: f64 = rng.gen_range(-10.0..10.0);
            let right_hess: f64 = rng.gen_range(1.0..50.0);
            (left_grad, left_hess, right_grad, right_hess)
        })
        .collect();
    
    let parent_grad = 0.0;
    let parent_hess = 100.0;
    
    group.throughput(Throughput::Elements(n_candidates));
    group.bench_function("current", |b| {
        b.iter(|| {
            let mut best_gain = f32::NEG_INFINITY;
            for &(gl, hl, gr, hr) in &candidates {
                let gain = params.compute_gain(
                    black_box(gl),
                    black_box(hl),
                    black_box(gr),
                    black_box(hr),
                    black_box(parent_grad),
                    black_box(parent_hess),
                );
                if gain > best_gain {
                    best_gain = gain;
                }
            }
            black_box(best_gain)
        })
    });
    
    group.finish();
}

// =============================================================================
// Proposed: Parent Score Precomputation
// =============================================================================

/// Pre-computed parent score context to avoid redundant computation.
#[derive(Clone, Debug)]
pub struct NodeGainContext {
    lambda: f64,
    /// Pre-computed: 0.5 * parent_score + min_gain
    gain_offset: f64,
}

impl NodeGainContext {
    #[inline]
    pub fn new(parent_grad: f64, parent_hess: f64, params: &GainParams) -> Self {
        let lambda = params.reg_lambda as f64;
        let parent_score = parent_grad * parent_grad / (parent_hess + lambda);
        Self {
            lambda,
            gain_offset: 0.5 * parent_score + params.min_gain as f64,
        }
    }

    /// Compute gain with pre-computed parent score.
    #[inline]
    pub fn compute_gain(&self, gl: f64, hl: f64, gr: f64, hr: f64) -> f32 {
        let sl = gl * gl / (hl + self.lambda);
        let sr = gr * gr / (hr + self.lambda);
        (0.5 * (sl + sr) - self.gain_offset) as f32
    }
}

fn bench_precomputed_gain(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/split/gain_computation");
    
    let params = GainParams::default();
    let parent_grad = 0.0;
    let parent_hess = 100.0;
    let ctx = NodeGainContext::new(parent_grad, parent_hess, &params);
    
    let n_candidates = 25_600u64;
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let candidates: Vec<(f64, f64, f64, f64)> = (0..n_candidates)
        .map(|_| {
            let left_grad: f64 = rng.gen_range(-10.0..10.0);
            let left_hess: f64 = rng.gen_range(1.0..50.0);
            let right_grad: f64 = rng.gen_range(-10.0..10.0);
            let right_hess: f64 = rng.gen_range(1.0..50.0);
            (left_grad, left_hess, right_grad, right_hess)
        })
        .collect();
    
    group.throughput(Throughput::Elements(n_candidates));
    group.bench_function("precomputed_parent", |b| {
        b.iter(|| {
            let mut best_gain = f32::NEG_INFINITY;
            for &(gl, hl, gr, hr) in &candidates {
                let gain = ctx.compute_gain(
                    black_box(gl),
                    black_box(hl),
                    black_box(gr),
                    black_box(hr),
                );
                if gain > best_gain {
                    best_gain = gain;
                }
            }
            black_box(best_gain)
        })
    });
    
    group.finish();
}

// =============================================================================
// Full Split Finding Comparison
// =============================================================================

fn bench_find_split_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/split/find_split");
    group.sample_size(50);
    
    // Test different feature/bin combinations
    let configs = [
        ("small", 20usize, 64u32),
        ("medium", 100usize, 256u32),
        ("large", 200usize, 256u32),
    ];
    
    for (name, n_features, n_bins) in configs {
        let (pool, parent_grad, parent_hess, parent_count) = 
            generate_histograms(n_features, n_bins, 10_000, 42);
        let slot = pool.slot(0);
        
        let feature_types: Vec<bool> = vec![false; n_features]; // all numerical
        let feature_has_missing: Vec<bool> = vec![false; n_features];
        let features: Vec<u32> = (0..n_features as u32).collect();
        
        let mut splitter = GreedySplitter::with_config(
            GainParams::default(),
            4,
            Parallelism::Sequential,
        );
        
        group.throughput(Throughput::Elements((n_features as u64) * (n_bins as u64)));
        group.bench_function(BenchmarkId::new("current", name), |b| {
            b.iter(|| {
                black_box(splitter.find_split(
                    black_box(&slot),
                    black_box(parent_grad),
                    black_box(parent_hess),
                    black_box(parent_count),
                    black_box(&feature_types),
                    black_box(&feature_has_missing),
                    black_box(&features),
                ))
            })
        });
    }
    
    group.finish();
}

// =============================================================================
// Numerical Split Scan with Early Termination
// =============================================================================

/// Simulates numerical split finding with early termination.
fn find_numerical_split_early_exit(
    bins: &[HistogramBin],
    parent_grad: f64,
    parent_hess: f64,
    parent_count: u32,
    min_child_weight: f64,
    min_samples_leaf: u32,
    lambda: f64,
    min_gain: f64,
) -> Option<(usize, f32)> {
    let n_bins = bins.len();
    if n_bins < 2 {
        return None;
    }
    
    // Precompute parent score
    let parent_score = parent_grad * parent_grad / (parent_hess + lambda);
    let gain_offset = 0.5 * parent_score + min_gain;
    
    let mut best: Option<(usize, f32)> = None;
    let mut left_grad = 0.0;
    let mut left_hess = 0.0;
    let mut left_count = 0u32;
    
    for bin in 0..(n_bins - 1) {
        let (bin_grad, bin_hess) = bins[bin];
        left_grad += bin_grad;
        left_hess += bin_hess;
        left_count += 1;
        
        // Early skip: left side too small
        if left_hess < min_child_weight || left_count < min_samples_leaf {
            continue;
        }
        
        let right_count = parent_count - left_count;
        
        // Early exit: once right side is too small, it stays too small
        if right_count < min_samples_leaf {
            break;
        }
        
        let right_hess = parent_hess - left_hess;
        if right_hess < min_child_weight {
            break;
        }
        
        // Compute gain (no parent score recalculation)
        let right_grad = parent_grad - left_grad;
        let sl = left_grad * left_grad / (left_hess + lambda);
        let sr = right_grad * right_grad / (right_hess + lambda);
        let gain = (0.5 * (sl + sr) - gain_offset) as f32;
        
        match best {
            Some((_, best_gain)) if gain <= best_gain => {}
            _ => best = Some((bin, gain)),
        }
    }
    
    best
}

/// Current implementation without early termination.
fn find_numerical_split_current(
    bins: &[HistogramBin],
    parent_grad: f64,
    parent_hess: f64,
    parent_count: u32,
    params: &GainParams,
) -> Option<(usize, f32)> {
    let n_bins = bins.len();
    if n_bins < 2 {
        return None;
    }
    
    let mut best: Option<(usize, f32)> = None;
    let mut left_grad = 0.0;
    let mut left_hess = 0.0;
    let mut left_count = 0u32;
    
    for bin in 0..(n_bins - 1) {
        let (bin_grad, bin_hess) = bins[bin];
        left_grad += bin_grad;
        left_hess += bin_hess;
        left_count += 1;
        
        let right_grad = parent_grad - left_grad;
        let right_hess = parent_hess - left_hess;
        let right_count = parent_count.saturating_sub(left_count);
        
        if params.is_valid_split(left_hess, right_hess, left_count, right_count) {
            let gain = params.compute_gain(
                left_grad, left_hess, right_grad, right_hess, parent_grad, parent_hess,
            );
            match best {
                Some((_, best_gain)) if gain <= best_gain => {}
                _ => best = Some((bin, gain)),
            }
        }
    }
    
    best
}

fn bench_numerical_split_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/split/numerical_scan");
    
    let params = GainParams {
        min_child_weight: 1.0,
        min_samples_leaf: 10,
        ..Default::default()
    };
    
    // Generate histogram with 256 bins
    let n_bins = 256usize;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    let bins: Vec<HistogramBin> = (0..n_bins)
        .map(|_| {
            let grad: f64 = rng.gen_range(-5.0..5.0);
            let hess: f64 = rng.gen_range(0.5..2.0);
            (grad, hess)
        })
        .collect();
    
    let parent_grad: f64 = bins.iter().map(|(g, _)| g).sum();
    let parent_hess: f64 = bins.iter().map(|(_, h)| h).sum();
    let parent_count = 1000u32;
    
    group.throughput(Throughput::Elements(n_bins as u64));
    
    group.bench_function("current", |b| {
        b.iter(|| {
            black_box(find_numerical_split_current(
                black_box(&bins),
                black_box(parent_grad),
                black_box(parent_hess),
                black_box(parent_count),
                black_box(&params),
            ))
        })
    });
    
    group.bench_function("early_exit_precomputed", |b| {
        b.iter(|| {
            black_box(find_numerical_split_early_exit(
                black_box(&bins),
                black_box(parent_grad),
                black_box(parent_hess),
                black_box(parent_count),
                params.min_child_weight as f64,
                params.min_samples_leaf,
                params.reg_lambda as f64,
                params.min_gain as f64,
            ))
        })
    });
    
    group.finish();
}

// =============================================================================
// Correctness Verification
// =============================================================================

fn bench_verify_correctness(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/split/verify");
    group.sample_size(10);
    
    // Verify that both implementations produce the same result
    let params = GainParams::default();
    let n_bins = 256usize;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    let bins: Vec<HistogramBin> = (0..n_bins)
        .map(|_| {
            let grad: f64 = rng.gen_range(-5.0..5.0);
            let hess: f64 = rng.gen_range(0.5..2.0);
            (grad, hess)
        })
        .collect();
    
    let parent_grad: f64 = bins.iter().map(|(g, _)| g).sum();
    let parent_hess: f64 = bins.iter().map(|(_, h)| h).sum();
    let parent_count = 1000u32;
    
    let current = find_numerical_split_current(&bins, parent_grad, parent_hess, parent_count, &params);
    let optimized = find_numerical_split_early_exit(
        &bins,
        parent_grad,
        parent_hess,
        parent_count,
        params.min_child_weight as f64,
        params.min_samples_leaf,
        params.reg_lambda as f64,
        params.min_gain as f64,
    );
    
    group.bench_function("verify_same_result", |b| {
        b.iter(|| {
            // Compare results
            match (current, optimized) {
                (Some((bin1, gain1)), Some((bin2, gain2))) => {
                    assert_eq!(bin1, bin2, "Split bin mismatch");
                    assert!((gain1 - gain2).abs() < 1e-5, "Gain mismatch: {} vs {}", gain1, gain2);
                }
                (None, None) => {}
                _ => panic!("One found split, other didn't"),
            }
            black_box(true)
        })
    });
    
    group.finish();
}

criterion_group! {
    name = benches;
    config = fast_criterion();
    targets = 
        bench_current_gain_computation,
        bench_precomputed_gain,
        bench_find_split_full,
        bench_numerical_split_scan,
        bench_verify_correctness
}
criterion_main!(benches);
