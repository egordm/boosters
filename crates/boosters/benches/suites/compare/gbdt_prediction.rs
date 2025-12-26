//! Unified comparison benchmarks: booste-rs vs XGBoost vs LightGBM prediction.
//!
//! Run with: `cargo bench --features "bench-xgboost,bench-lightgbm" --bench compare_prediction`
//! Or single library: `cargo bench --features bench-xgboost --bench compare_prediction`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
#[cfg(any(feature = "bench-xgboost", feature = "bench-lightgbm"))]
use common::models::bench_models_dir;
use common::models::load_boosters_model;

use boosters::Parallelism;
use boosters::data::FeaturesView;
use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
use boosters::testing::synthetic_datasets::random_features_array;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

#[cfg(feature = "bench-xgboost")]
use criterion::BatchSize;

#[cfg(feature = "bench-xgboost")]
use xgb::{Booster, DMatrix};

// =============================================================================
// Standardized Dataset Sizes
// =============================================================================

/// Small batch: latency-sensitive scenarios
const SMALL_BATCH: usize = 100;
/// Medium batch: primary comparison point
const MEDIUM_BATCH: usize = 1_000;
/// Large batch: throughput scenarios
const LARGE_BATCH: usize = 10_000;

// =============================================================================
// XGBoost Helpers
// =============================================================================

#[cfg(feature = "bench-xgboost")]
fn load_xgb_model_bytes(name: &str) -> Vec<u8> {
    let path = bench_models_dir().join(format!("{name}.model.json"));
    std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read XGB model: {path:?}"))
}

#[cfg(feature = "bench-xgboost")]
fn new_xgb_booster(model_bytes: &[u8]) -> Booster {
    let mut booster =
        Booster::load_buffer(model_bytes).expect("Failed to load XGB model from buffer");
    booster
        .set_param("predictor", "cpu_predictor")
        .expect("Failed to set predictor");
    booster
        .set_param("nthread", "1")
        .expect("Failed to set nthread");
    booster
}

#[cfg(feature = "bench-xgboost")]
fn reset_xgb_prediction_cache(booster: &mut Booster) {
    let _ = booster.reset();
}

#[cfg(feature = "bench-xgboost")]
fn create_dmatrix(data: &[f32], n_rows: usize) -> DMatrix {
    DMatrix::from_dense(data, n_rows).expect("Failed to create DMatrix")
}

// =============================================================================
// LightGBM Helpers
// =============================================================================

#[cfg(feature = "bench-lightgbm")]
fn load_native_lgb_booster(name: &str) -> lightgbm3::Booster {
    lightgbm3::Booster::from_file(
        bench_models_dir()
            .join(format!("{name}.lgb.txt"))
            .to_str()
            .unwrap(),
    )
    .expect("Failed to load LightGBM model")
}

// =============================================================================
// Batch Size Comparison
// =============================================================================

fn bench_predict_batch_sizes(c: &mut Criterion) {
    let boosters_model = load_boosters_model("bench_medium");
    let n_features = boosters_model.n_features;
    let predictor =
        Predictor::<UnrolledTraversal6>::new(&boosters_model.forest).with_block_size(64);

    #[cfg(feature = "bench-xgboost")]
    let xgb_model_bytes = load_xgb_model_bytes("bench_medium");

    #[cfg(feature = "bench-lightgbm")]
    let lgb_booster = load_native_lgb_booster("bench_medium");

    let mut group = c.benchmark_group("compare/predict/batch_size/medium");

    for batch_size in [SMALL_BATCH, MEDIUM_BATCH, LARGE_BATCH] {
        let input_array = random_features_array(batch_size, n_features, 42, -5.0, 5.0);

        group.throughput(Throughput::Elements(batch_size as u64));

        // booste-rs
        let features = FeaturesView::from_array(input_array.view());
        group.bench_with_input(
            BenchmarkId::new("boosters", batch_size),
            &features,
            |b, f| b.iter(|| black_box(predictor.predict(black_box(*f), Parallelism::Sequential))),
        );

        // Get raw slice for XGBoost/LightGBM
        #[cfg(any(feature = "bench-xgboost", feature = "bench-lightgbm"))]
        let input_data: &[f32] = input_array.as_slice().unwrap();

        // XGBoost
        #[cfg(feature = "bench-xgboost")]
        {
            group.bench_function(BenchmarkId::new("xgboost/cold_dmatrix", batch_size), |b| {
                let mut booster = new_xgb_booster(&xgb_model_bytes);
                b.iter_batched(
                    || create_dmatrix(input_data, batch_size),
                    |dmatrix| {
                        reset_xgb_prediction_cache(&mut booster);
                        let out = booster.predict(black_box(&dmatrix)).unwrap();
                        black_box(out)
                    },
                    BatchSize::SmallInput,
                )
            });
        }

        // LightGBM
        #[cfg(feature = "bench-lightgbm")]
        {
            let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
            let mut input_f64_b = input_f64_a.clone();
            if let Some(first) = input_f64_b.first_mut() {
                *first = f64::from_bits(first.to_bits().wrapping_add(1));
            }
            let n_feat = n_features as i32;
            group.bench_function(BenchmarkId::new("lightgbm", batch_size), |b| {
                let mut flip = false;
                b.iter(|| {
                    flip = !flip;
                    let input = if flip { &input_f64_a } else { &input_f64_b };
                    let output = lgb_booster.predict(black_box(input), n_feat, true).unwrap();
                    black_box(output)
                })
            });
        }
    }

    group.finish();
}

// =============================================================================
// Single Row Latency
// =============================================================================

fn bench_predict_single_row(c: &mut Criterion) {
    let boosters_model = load_boosters_model("bench_medium");
    let n_features = boosters_model.n_features;
    let predictor =
        Predictor::<UnrolledTraversal6>::new(&boosters_model.forest).with_block_size(64);

    #[cfg(feature = "bench-xgboost")]
    let xgb_model_bytes = load_xgb_model_bytes("bench_medium");
    #[cfg(feature = "bench-xgboost")]
    let mut xgb_model = new_xgb_booster(&xgb_model_bytes);

    #[cfg(feature = "bench-lightgbm")]
    let lgb_booster = load_native_lgb_booster("bench_medium");

    let input_array = random_features_array(1, n_features, 42, -5.0, 5.0);

    #[cfg(any(feature = "bench-xgboost", feature = "bench-lightgbm"))]
    let input_data: &[f32] = input_array.as_slice().unwrap();

    let mut group = c.benchmark_group("compare/predict/single_row/medium");

    // booste-rs
    let features = FeaturesView::from_array(input_array.view());
    group.bench_function("boosters", |b| {
        b.iter(|| black_box(predictor.predict(black_box(features), Parallelism::Sequential)))
    });

    // XGBoost
    #[cfg(feature = "bench-xgboost")]
    {
        group.bench_function("xgboost/cold_dmatrix", |b| {
            b.iter_batched(
                || create_dmatrix(input_data, 1),
                |dmatrix| {
                    reset_xgb_prediction_cache(&mut xgb_model);
                    let out = xgb_model.predict(black_box(&dmatrix)).unwrap();
                    black_box(out)
                },
                BatchSize::SmallInput,
            )
        });
    }

    // LightGBM
    #[cfg(feature = "bench-lightgbm")]
    {
        let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
        let mut input_f64_b = input_f64_a.clone();
        if let Some(first) = input_f64_b.first_mut() {
            *first = f64::from_bits(first.to_bits().wrapping_add(1));
        }
        let n_feat = n_features as i32;
        group.bench_function("lightgbm", |b| {
            let mut flip = false;
            b.iter(|| {
                flip = !flip;
                let input = if flip { &input_f64_a } else { &input_f64_b };
                let output = lgb_booster.predict(black_box(input), n_feat, true).unwrap();
                black_box(output)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Thread Scaling
// =============================================================================

fn bench_predict_thread_scaling(c: &mut Criterion) {
    let boosters_model = load_boosters_model("bench_medium");
    let n_features = boosters_model.n_features;
    let predictor =
        Predictor::<UnrolledTraversal6>::new(&boosters_model.forest).with_block_size(64);

    #[cfg(feature = "bench-xgboost")]
    let xgb_model_bytes = load_xgb_model_bytes("bench_medium");
    #[cfg(feature = "bench-xgboost")]
    let mut xgb_model = new_xgb_booster(&xgb_model_bytes);

    #[cfg(feature = "bench-lightgbm")]
    let lgb_booster = load_native_lgb_booster("bench_medium");

    let batch_size = LARGE_BATCH;
    let input_array = random_features_array(batch_size, n_features, 42, -5.0, 5.0);

    #[cfg(any(feature = "bench-xgboost", feature = "bench-lightgbm"))]
    let input_data: &[f32] = input_array.as_slice().unwrap();

    let mut group = c.benchmark_group("compare/predict/thread_scaling/medium");
    group.throughput(Throughput::Elements(batch_size as u64));

    let features = FeaturesView::from_array(input_array.view());
    for &n_threads in common::matrix::THREAD_COUNTS {
        // booste-rs
        group.bench_with_input(
            BenchmarkId::new("boosters", n_threads),
            &features,
            |b, f| b.iter(|| black_box(predictor.predict(black_box(*f), Parallelism::Parallel))),
        );

        // XGBoost
        #[cfg(feature = "bench-xgboost")]
        {
            let dmatrix = create_dmatrix(input_data, batch_size);
            let threads = n_threads.to_string();
            xgb_model
                .set_param("nthread", &threads)
                .expect("Failed to set nthread");
            reset_xgb_prediction_cache(&mut xgb_model);

            group.bench_function(BenchmarkId::new("xgboost/cold_cache", n_threads), |b| {
                b.iter(|| {
                    reset_xgb_prediction_cache(&mut xgb_model);
                    let out = xgb_model.predict(black_box(&dmatrix)).unwrap();
                    black_box(out)
                })
            });
        }

        // LightGBM (single-threaded baseline only)
        #[cfg(feature = "bench-lightgbm")]
        if n_threads == 1 {
            let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
            let mut input_f64_b = input_f64_a.clone();
            if let Some(first) = input_f64_b.first_mut() {
                *first = f64::from_bits(first.to_bits().wrapping_add(1));
            }
            let n_feat = n_features as i32;
            group.bench_function(BenchmarkId::new("lightgbm", "default"), |b| {
                let mut flip = false;
                b.iter(|| {
                    flip = !flip;
                    let input = if flip { &input_f64_a } else { &input_f64_b };
                    let output = lgb_booster.predict(black_box(input), n_feat, true).unwrap();
                    black_box(output)
                })
            });
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_predict_batch_sizes, bench_predict_single_row, bench_predict_thread_scaling
}
criterion_main!(benches);
