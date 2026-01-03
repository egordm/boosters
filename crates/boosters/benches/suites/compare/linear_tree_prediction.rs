//! Comparison benchmarks: booste-rs vs LightGBM linear GBDT prediction.
//!
//! Uses models trained with LightGBM's `linear_tree=True` option, loaded via
//! native booste-rs format.
//!
//! Run with: `cargo bench --features bench-lightgbm --bench linear_tree_prediction`

#[path = "../../common/mod.rs"]
mod common;

use boosters::data::Dataset;
use boosters::model::GBDTModel;
use boosters::persist::Model;
use boosters::repr::gbdt::{Forest, ScalarLeaf};
use common::criterion_config::default_criterion;
#[cfg(feature = "bench-lightgbm")]
use common::models::original_bench_models_dir;

use boosters::Parallelism;
use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
use boosters::testing::synthetic_datasets::random_features_array;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

// =============================================================================
// Helpers
// =============================================================================

/// Path to benchmark models directory (native format).
fn persist_benchmark_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/persist/benchmark")
}

fn load_linear_model(name: &str) -> (Forest<ScalarLeaf>, usize) {
    let path = persist_benchmark_dir().join(format!("{name}.model.bstr.json"));
    let model =
        Model::load_json(&path).unwrap_or_else(|e| panic!("Failed to load model {path:?}: {e}"));

    let gbdt: GBDTModel = model
        .into_gbdt()
        .expect("Expected GBDT model for linear tree benchmark");
    let n_features = gbdt.meta().n_features;
    let forest = gbdt.forest().clone();
    (forest, n_features)
}

// =============================================================================
// Linear GBDT Inference Comparison
// =============================================================================

fn bench_linear_gbdt_prediction(c: &mut Criterion) {
    // Load linear GBDT model
    let (linear_forest, n_features) = load_linear_model("bench_linear_medium");
    let linear_predictor = Predictor::<UnrolledTraversal6>::new(&linear_forest).with_block_size(64);

    // Load standard model for comparison
    let (standard_forest, _) = load_linear_model("bench_standard_medium");
    let standard_predictor =
        Predictor::<UnrolledTraversal6>::new(&standard_forest).with_block_size(64);

    // Also load LightGBM native models for comparison (using original .lgb.txt files)
    #[cfg(feature = "bench-lightgbm")]
    let lgb_linear = lightgbm3::Booster::from_file(
        original_bench_models_dir()
            .join("bench_linear_medium.lgb.txt")
            .to_str()
            .unwrap(),
    )
    .expect("Failed to load LightGBM linear model");

    #[cfg(feature = "bench-lightgbm")]
    let lgb_standard = lightgbm3::Booster::from_file(
        original_bench_models_dir()
            .join("bench_standard_medium.lgb.txt")
            .to_str()
            .unwrap(),
    )
    .expect("Failed to load LightGBM standard model");

    let batch_sizes = [100, 1_000, 10_000];

    let mut group = c.benchmark_group("compare/predict/linear_gbdt");

    for batch_size in batch_sizes {
        let input_array = random_features_array(batch_size, n_features, 42, -1.0, 1.0);

        group.throughput(Throughput::Elements(batch_size as u64));

        // booste-rs linear GBDT
        group.bench_with_input(
            BenchmarkId::new("boosters/linear", batch_size),
            &input_array,
            |b, m| {
                let dataset = Dataset::from_array(m.t(), None, None);
                b.iter(|| {
                    black_box(
                        linear_predictor.predict(black_box(&dataset), Parallelism::Sequential),
                    )
                })
            },
        );

        // booste-rs standard (baseline)
        group.bench_with_input(
            BenchmarkId::new("boosters/standard", batch_size),
            &input_array,
            |b, m| {
                let dataset = Dataset::from_array(m.t(), None, None);
                b.iter(|| {
                    black_box(
                        standard_predictor.predict(black_box(&dataset), Parallelism::Sequential),
                    )
                })
            },
        );

        // LightGBM linear GBDT
        #[cfg(feature = "bench-lightgbm")]
        {
            let input_data: &[f32] = input_array.as_slice().unwrap();
            let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
            let mut input_f64_b = input_f64_a.clone();
            if let Some(first) = input_f64_b.first_mut() {
                *first = f64::from_bits(first.to_bits().wrapping_add(1));
            }
            let n_feat = n_features as i32;

            group.bench_function(BenchmarkId::new("lightgbm/linear", batch_size), |b| {
                let mut flip = false;
                b.iter(|| {
                    flip = !flip;
                    let input = if flip { &input_f64_a } else { &input_f64_b };
                    let output = lgb_linear.predict(black_box(input), n_feat, true).unwrap();
                    black_box(output)
                })
            });

            group.bench_function(BenchmarkId::new("lightgbm/standard", batch_size), |b| {
                let mut flip = false;
                b.iter(|| {
                    flip = !flip;
                    let input = if flip { &input_f64_a } else { &input_f64_b };
                    let output = lgb_standard
                        .predict(black_box(input), n_feat, true)
                        .unwrap();
                    black_box(output)
                })
            });
        }
    }

    group.finish();
}

// =============================================================================
// Linear GBDT Overhead Measurement
// =============================================================================

fn bench_linear_gbdt_overhead(c: &mut Criterion) {
    let (linear_forest, n_features) = load_linear_model("bench_linear_medium");
    let linear_predictor = Predictor::<UnrolledTraversal6>::new(&linear_forest).with_block_size(64);

    let (standard_forest, _) = load_linear_model("bench_standard_medium");
    let standard_predictor =
        Predictor::<UnrolledTraversal6>::new(&standard_forest).with_block_size(64);

    let batch_size = 10_000;
    let input_array = random_features_array(batch_size, n_features, 42, -1.0, 1.0);
    let dataset = Dataset::from_array(input_array.t(), None, None);

    let mut group = c.benchmark_group("overhead/linear_gbdt");
    group.throughput(Throughput::Elements(batch_size as u64));

    group.bench_function("standard", |b| {
        b.iter(|| {
            black_box(standard_predictor.predict(black_box(&dataset), Parallelism::Sequential))
        })
    });

    group.bench_function("linear", |b| {
        b.iter(|| black_box(linear_predictor.predict(black_box(&dataset), Parallelism::Sequential)))
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_linear_gbdt_prediction, bench_linear_gbdt_overhead
}
criterion_main!(benches);
