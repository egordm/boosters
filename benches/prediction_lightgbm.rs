//! LightGBM prediction benchmark comparison.
//!
//! Compares booste-rs inference performance against LightGBM C++ library.
//! Run with: `cargo bench --features bench-lightgbm --bench prediction_lightgbm`

mod bench_utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use bench_utils::generate_random_input;
use booste_rs::compat::LgbModel;
use booste_rs::data::RowMatrix;
use booste_rs::inference::{Forest, Predictor, ScalarLeaf};
use booste_rs::inference::gbdt::UnrolledTraversal6;

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

// =============================================================================
// Model Loading Utilities
// =============================================================================

/// Path to benchmark models directory.
fn bench_models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test-cases/benchmark")
}

/// Load a booste-rs model from LightGBM text file.
fn load_boosters_lgb_model(name: &str) -> (Forest<ScalarLeaf>, usize) {
    let path = bench_models_dir().join(format!("{}.lgb.txt", name));
    let mut file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open model: {:?}", path));
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Failed to read model");
    let lgb_model =
        LgbModel::from_string(&content).unwrap_or_else(|e| panic!("Failed to parse model: {}", e));

    let num_features = lgb_model.num_features() as usize;

    let forest = lgb_model
        .to_forest()
        .expect("Failed to convert model to forest");

    (forest, num_features)
}

// =============================================================================
// LightGBM C++ Comparison Benchmarks
// =============================================================================

/// Benchmark booste-rs vs LightGBM on various batch sizes (medium model).
fn bench_lightgbm_batch_sizes(c: &mut Criterion) {
    let (forest, num_features) = load_boosters_lgb_model("bench_medium");

    let lgb_booster = lightgbm3::Booster::from_file(
        bench_models_dir()
            .join("bench_medium.lgb.txt")
            .to_str()
            .unwrap(),
    )
    .expect("Failed to load LightGBM model");

    let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);

    let mut group = c.benchmark_group("lightgbm/batch_size/medium");

    for batch_size in [100, 1_000, 10_000].iter() {
        let input_data = generate_random_input(*batch_size, num_features, 42);
        let matrix = RowMatrix::from_vec(input_data.clone(), *batch_size, num_features);

        group.throughput(Throughput::Elements(*batch_size as u64));

        // booste-rs
        group.bench_with_input(
            BenchmarkId::new("boosters", batch_size),
            &matrix,
            |b, matrix| {
                b.iter(|| black_box(predictor.predict(black_box(matrix))));
            },
        );

        // LightGBM (convert to f64 as lightgbm3 uses f64)
        let input_f64: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
        let num_feat = num_features as i32;
        group.bench_function(BenchmarkId::new("lightgbm", batch_size), |b| {
            b.iter(|| {
                let output = lgb_booster
                    .predict(black_box(&input_f64), num_feat, true)
                    .unwrap();
                black_box(output)
            });
        });
    }

    group.finish();
}

/// Benchmark single-row latency comparison (medium model).
fn bench_lightgbm_single_row(c: &mut Criterion) {
    let (forest, num_features) = load_boosters_lgb_model("bench_medium");

    let lgb_booster = lightgbm3::Booster::from_file(
        bench_models_dir()
            .join("bench_medium.lgb.txt")
            .to_str()
            .unwrap(),
    )
    .expect("Failed to load LightGBM model");

    let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);
    let input_data = generate_random_input(1, num_features, 42);

    let mut group = c.benchmark_group("lightgbm/single_row/medium");

    // booste-rs
    let matrix = RowMatrix::from_vec(input_data.clone(), 1, num_features);
    group.bench_function("boosters", |b| {
        b.iter(|| black_box(predictor.predict(black_box(&matrix))));
    });

    // LightGBM
    let input_f64: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
    let num_feat = num_features as i32;
    group.bench_function("lightgbm", |b| {
        b.iter(|| {
            let output = lgb_booster
                .predict(black_box(&input_f64), num_feat, true)
                .unwrap();
            black_box(output)
        });
    });

    group.finish();
}

/// Benchmark different model sizes (small, medium, large).
fn bench_lightgbm_model_sizes(c: &mut Criterion) {
    let models = [
        ("small", "bench_small"),
        ("medium", "bench_medium"),
        ("large", "bench_large"),
    ];

    let batch_size = 1000;

    let mut group = c.benchmark_group("lightgbm/model_size");

    for (label, model_name) in models.iter() {
        // Try to load booste-rs model
        let (forest, num_features) = match std::panic::catch_unwind(|| load_boosters_lgb_model(model_name)) {
            Ok(m) => m,
            Err(_) => {
                eprintln!("Skipping {} - model not found", model_name);
                continue;
            }
        };

        // Try to load LightGBM model
        let lgb_path = bench_models_dir().join(format!("{}.lgb.txt", model_name));
        let lgb_booster = match lightgbm3::Booster::from_file(lgb_path.to_str().unwrap()) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping {} - LightGBM model not found", model_name);
                continue;
            }
        };

        let input_data = generate_random_input(batch_size, num_features, 42);
        let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

        let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);

        group.throughput(Throughput::Elements(batch_size as u64));

        // booste-rs
        group.bench_with_input(
            BenchmarkId::new(format!("{}/boosters", label), batch_size),
            &matrix,
            |b, matrix| {
                b.iter(|| black_box(predictor.predict(black_box(matrix))));
            },
        );

        // LightGBM
        let input_f64: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
        let num_feat = num_features as i32;
        group.bench_function(
            BenchmarkId::new(format!("{}/lightgbm", label), batch_size),
            |b| {
                b.iter(|| {
                    let output = lgb_booster
                        .predict(black_box(&input_f64), num_feat, true)
                        .unwrap();
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel prediction scaling.
fn bench_lightgbm_parallel(c: &mut Criterion) {
    let (forest, num_features) = load_boosters_lgb_model("bench_medium");

    let lgb_booster = lightgbm3::Booster::from_file(
        bench_models_dir()
            .join("bench_medium.lgb.txt")
            .to_str()
            .unwrap(),
    )
    .expect("Failed to load LightGBM model");

    let thread_counts = [1, 2, 4, 8];
    let batch_size = 10_000;

    let input_data = generate_random_input(batch_size, num_features, 42);
    let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);

    let predictor = Predictor::<UnrolledTraversal6>::new(&forest).with_block_size(64);

    let mut group = c.benchmark_group("lightgbm/parallel/medium");
    group.throughput(Throughput::Elements(batch_size as u64));

    for &num_threads in &thread_counts {
        // booste-rs parallel
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("boosters", num_threads),
            &matrix,
            |b, matrix| {
                b.iter(|| pool.install(|| black_box(predictor.par_predict(black_box(matrix)))));
            },
        );

        // LightGBM with configured thread count
        // Note: lightgbm3 uses internal threading
        // For fair comparison, we just run the default configuration once
        let input_f64: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
        let num_feat = num_features as i32;
        if num_threads == 1 {
            // Only benchmark LightGBM once since we can't easily control its thread count
            group.bench_function(BenchmarkId::new("lightgbm", "default"), |b| {
                b.iter(|| {
                    let output = lgb_booster
                        .predict(black_box(&input_f64), num_feat, true)
                        .unwrap();
                    black_box(output)
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
    bench_lightgbm_batch_sizes,
    bench_lightgbm_single_row,
    bench_lightgbm_model_sizes,
    bench_lightgbm_parallel
);

criterion_main!(benches);
