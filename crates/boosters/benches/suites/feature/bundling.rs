//! Feature Bundling (EFB) benchmark: compare performance with bundling enabled vs disabled.
//!
//! Tests on one-hot encoded sparse data where EFB provides maximum benefit.
//! Measures:
//! - Binning time: dataset construction with bundling on vs off
//! - Training time: short training (10 rounds) to see histogram effect
//! - Memory: reports binned column count as memory proxy
//!
//! Compares:
//! - booste-rs: BundlingConfig::auto() vs BundlingConfig::disabled()
//! - LightGBM: enable_bundle=true vs enable_bundle=false
//!
//! Run with: `cargo bench --features bench-compare --bench bundling`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::binned::{BinnedDatasetBuilder, BundlingConfig};
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};
use boosters::Parallelism;

use ndarray::ArrayView1;

fn empty_weights() -> ArrayView1<'static, f32> {
    ArrayView1::from(&[][..])
}

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "bench-lightgbm")]
use serde_json::json;

// =============================================================================
// Dataset Configuration
// =============================================================================

/// Generate one-hot encoded sparse dataset.
///
/// Creates a dataset with:
/// - `n_numerical` dense numerical features
/// - `n_categoricals` categorical features, each one-hot encoded with `cats_per_cat` categories
///
/// Total features = n_numerical + n_categoricals * cats_per_cat
fn generate_onehot_dataset(
    rows: usize,
    n_numerical: usize,
    n_categoricals: usize,
    cats_per_cat: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, usize) {
    let n_features = n_numerical + n_categoricals * cats_per_cat;
    let mut features = Vec::with_capacity(rows * n_features);
    let mut labels = Vec::with_capacity(rows);

    let mut rng_state = seed;
    let next_rand = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    for i in 0..rows {
        // Numerical features
        for _j in 0..n_numerical {
            let val = next_rand(&mut rng_state);
            features.push(val);
        }

        // Categorical features (one-hot encoded)
        let mut cat_effects = 0.0f32;
        for cat_idx in 0..n_categoricals {
            let active_cat = (i + cat_idx * 7) % cats_per_cat;
            for c in 0..cats_per_cat {
                features.push(if c == active_cat { 1.0 } else { 0.0 });
            }
            cat_effects += (active_cat as f32 - cats_per_cat as f32 / 2.0) * 0.1;
        }

        // Target: sum of numerical + categorical effects + noise
        let numerical_sum: f32 = (0..n_numerical)
            .map(|j| features[i * n_features + j] * (j as f32 + 1.0) * 0.1)
            .sum();
        let noise = next_rand(&mut rng_state) * 0.1;
        labels.push(numerical_sum + cat_effects + noise);
    }

    (features, labels, n_features)
}

// =============================================================================
// Dataset Configurations
// =============================================================================

struct DatasetConfig {
    name: &'static str,
    rows: usize,
    n_numerical: usize,
    n_categoricals: usize,
    cats_per_cat: usize,
}

const DATASETS: &[DatasetConfig] = &[
    // Small: 2 numerical + 3 categoricals × 10 cats = 32 features
    DatasetConfig {
        name: "small_sparse",
        rows: 10_000,
        n_numerical: 2,
        n_categoricals: 3,
        cats_per_cat: 10,
    },
    // Medium: 5 numerical + 5 categoricals × 20 cats = 105 features
    DatasetConfig {
        name: "medium_sparse",
        rows: 50_000,
        n_numerical: 5,
        n_categoricals: 5,
        cats_per_cat: 20,
    },
    // High sparsity: 2 numerical + 10 categoricals × 50 cats = 502 features
    DatasetConfig {
        name: "high_sparse",
        rows: 20_000,
        n_numerical: 2,
        n_categoricals: 10,
        cats_per_cat: 50,
    },
];

// =============================================================================
// booste-rs Benchmarks
// =============================================================================

/// Benchmark dataset construction (binning) - where bundling has direct impact
fn bench_boosters_binning(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature/bundling/boosters/binning");
    group.sample_size(20);

    for config in DATASETS {
        let (features, _targets, n_features) =
            generate_onehot_dataset(config.rows, config.n_numerical, config.n_categoricals, config.cats_per_cat, 42);

        group.throughput(Throughput::Elements((config.rows * n_features) as u64));

        // Pre-convert to column layout (not part of bundling)
        let row: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features.clone(), config.rows, n_features);
        let col: ColMatrix<f32> = row.to_layout();

        // WITHOUT bundling
        group.bench_function(BenchmarkId::new("no_bundling", config.name), |b| {
            b.iter(|| {
                black_box(
                    BinnedDatasetBuilder::from_matrix(&col, 256)
                        .with_bundling(BundlingConfig::disabled())
                        .build()
                        .unwrap(),
                )
            })
        });

        // WITH bundling (auto)
        group.bench_function(BenchmarkId::new("with_bundling", config.name), |b| {
            b.iter(|| {
                black_box(
                    BinnedDatasetBuilder::from_matrix(&col, 256)
                        .with_bundling(BundlingConfig::auto())
                        .build()
                        .unwrap(),
                )
            })
        });

        // Report bundling statistics
        let binned = BinnedDatasetBuilder::from_matrix(&col, 256)
            .with_bundling(BundlingConfig::auto())
            .build()
            .unwrap();
        let stats = binned.bundling_stats();
        if let Some(s) = stats {
            eprintln!(
                "[{}] Features: {} → {} binned columns ({:.1}% reduction)",
                config.name,
                n_features,
                s.bundles_created + s.standalone_features,
                (1.0 - (s.bundles_created + s.standalone_features) as f64 / n_features as f64) * 100.0
            );
        }
    }

    group.finish();
}

/// Benchmark training with pre-binned data (10 rounds for fast iteration)
fn bench_boosters_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature/bundling/boosters/training");
    group.sample_size(10);

    let (n_trees, max_depth) = (10u32, 6u32);

    for config in DATASETS {
        let (features, targets, n_features) =
            generate_onehot_dataset(config.rows, config.n_numerical, config.n_categoricals, config.cats_per_cat, 42);

        group.throughput(Throughput::Elements((config.rows * n_features) as u64));

        let params = GBDTParams {
            n_trees,
            learning_rate: 0.1,
            growth_strategy: GrowthStrategy::DepthWise { max_depth },
            gain: GainParams { reg_lambda: 1.0, ..Default::default() },
            cache_size: 32,
            ..Default::default()
        };
        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

        // Pre-convert to column layout
        let row: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features.clone(), config.rows, n_features);
        let col: ColMatrix<f32> = row.to_layout();

        // Pre-build binned datasets
        let binned_no_bundle = BinnedDatasetBuilder::from_matrix(&col, 256)
            .with_bundling(BundlingConfig::disabled())
            .build()
            .unwrap();
        let binned_with_bundle = BinnedDatasetBuilder::from_matrix(&col, 256)
            .with_bundling(BundlingConfig::auto())
            .build()
            .unwrap();

        // WITHOUT bundling (training only)
        group.bench_function(BenchmarkId::new("no_bundling", config.name), |b| {
            b.iter(|| {
                black_box(trainer.train(black_box(&binned_no_bundle), ArrayView1::from(black_box(&targets[..])), empty_weights(), &[], Parallelism::Sequential).unwrap())
            })
        });

        // WITH bundling (training only)
        group.bench_function(BenchmarkId::new("with_bundling", config.name), |b| {
            b.iter(|| {
                black_box(trainer.train(black_box(&binned_with_bundle), ArrayView1::from(black_box(&targets[..])), empty_weights(), &[], Parallelism::Sequential).unwrap())
            })
        });
    }

    group.finish();
}

// =============================================================================
// LightGBM Benchmarks
// =============================================================================

#[cfg(feature = "bench-lightgbm")]
fn bench_lightgbm_bundling(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature/bundling/lightgbm");
    group.sample_size(10);

    let (n_trees, max_depth) = (50u32, 6u32);

    for config in DATASETS {
        let (features, targets, n_features) =
            generate_onehot_dataset(config.rows, config.n_numerical, config.n_categoricals, config.cats_per_cat, 42);
        let features_f64: Vec<f64> = features.iter().map(|&x| x as f64).collect();

        group.throughput(Throughput::Elements((config.rows * n_features) as u64));

        // WITHOUT bundling
        group.bench_function(BenchmarkId::new("no_bundling", config.name), |b| {
            b.iter(|| {
                let dataset = lightgbm3::Dataset::from_slice(
                    black_box(&features_f64),
                    black_box(&targets),
                    n_features as i32,
                    true,
                )
                .unwrap();
                let params = json!({
                    "objective": "regression",
                    "metric": "l2",
                    "num_iterations": n_trees,
                    "learning_rate": 0.1,
                    "max_depth": max_depth as i32,
                    "num_leaves": 64,
                    "min_data_in_leaf": 1,
                    "lambda_l2": 1.0,
                    "enable_bundle": false,
                    "verbosity": -1,
                    "num_threads": 1
                });
                black_box(lightgbm3::Booster::train(dataset, &params).unwrap())
            })
        });

        // WITH bundling (default)
        group.bench_function(BenchmarkId::new("with_bundling", config.name), |b| {
            b.iter(|| {
                let dataset = lightgbm3::Dataset::from_slice(
                    black_box(&features_f64),
                    black_box(&targets),
                    n_features as i32,
                    true,
                )
                .unwrap();
                let params = json!({
                    "objective": "regression",
                    "metric": "l2",
                    "num_iterations": n_trees,
                    "learning_rate": 0.1,
                    "max_depth": max_depth as i32,
                    "num_leaves": 64,
                    "min_data_in_leaf": 1,
                    "lambda_l2": 1.0,
                    "enable_bundle": true,
                    "verbosity": -1,
                    "num_threads": 1
                });
                black_box(lightgbm3::Booster::train(dataset, &params).unwrap())
            })
        });
    }

    group.finish();
}

#[cfg(not(feature = "bench-lightgbm"))]
fn bench_lightgbm_bundling(_c: &mut Criterion) {
    // No-op when LightGBM not available
}

// =============================================================================
// Entry Point
// =============================================================================

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_boosters_binning, bench_boosters_training, bench_lightgbm_bundling
}
criterion_main!(benches);
