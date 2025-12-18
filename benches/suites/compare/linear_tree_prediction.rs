//! Comparison benchmarks: booste-rs vs LightGBM linear tree prediction.
//!
//! Uses models trained with LightGBM's `linear_tree=True` option, loaded via
//! our LightGBM text format parser.
//!
//! Run with: `cargo bench --features bench-lightgbm --bench linear_tree_prediction`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;
#[cfg(feature = "bench-lightgbm")]
use common::models::bench_models_dir;

use booste_rs::compat::lightgbm::LgbModel;
use booste_rs::data::RowMatrix;
use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
use booste_rs::testing::data::random_dense_f32;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// =============================================================================
// Helpers
// =============================================================================

fn load_lgb_linear_model(name: &str) -> (booste_rs::inference::gbdt::Forest<booste_rs::inference::gbdt::ScalarLeaf>, usize) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/test-cases/benchmark")
        .join(format!("{name}.lgb.txt"));
    let lgb_model = LgbModel::from_file(&path)
        .unwrap_or_else(|_| panic!("Failed to parse LightGBM model: {path:?}"));
    let forest = lgb_model.to_forest()
        .expect("Failed to convert to forest");
    let num_features = lgb_model.header.max_feature_idx as usize + 1;
    (forest, num_features)
}

// =============================================================================
// Linear Tree Inference Comparison
// =============================================================================

fn bench_linear_tree_prediction(c: &mut Criterion) {
    // Load linear tree model
    let (linear_forest, num_features) = load_lgb_linear_model("bench_linear_medium");
    let linear_predictor = Predictor::<UnrolledTraversal6>::new(&linear_forest).with_block_size(64);
    
    // Load standard model for comparison
    let (standard_forest, _) = load_lgb_linear_model("bench_standard_medium");
    let standard_predictor = Predictor::<UnrolledTraversal6>::new(&standard_forest).with_block_size(64);
    
    // Also load LightGBM native models for comparison
    #[cfg(feature = "bench-lightgbm")]
    let lgb_linear = lightgbm3::Booster::from_file(
        bench_models_dir().join("bench_linear_medium.lgb.txt").to_str().unwrap()
    ).expect("Failed to load LightGBM linear model");
    
    #[cfg(feature = "bench-lightgbm")]
    let lgb_standard = lightgbm3::Booster::from_file(
        bench_models_dir().join("bench_standard_medium.lgb.txt").to_str().unwrap()
    ).expect("Failed to load LightGBM standard model");
    
    let batch_sizes = [100, 1_000, 10_000];
    
    let mut group = c.benchmark_group("compare/predict/linear_tree");
    
    for batch_size in batch_sizes {
        let input_data = random_dense_f32(batch_size, num_features, 42, -1.0, 1.0);
        let matrix = RowMatrix::from_vec(input_data.clone(), batch_size, num_features);
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        // booste-rs linear tree
        group.bench_with_input(
            BenchmarkId::new("boosters/linear", batch_size), 
            &matrix, 
            |b, m| b.iter(|| black_box(linear_predictor.predict(black_box(m))))
        );
        
        // booste-rs standard (baseline)
        group.bench_with_input(
            BenchmarkId::new("boosters/standard", batch_size), 
            &matrix, 
            |b, m| b.iter(|| black_box(standard_predictor.predict(black_box(m))))
        );
        
        // LightGBM linear tree
        #[cfg(feature = "bench-lightgbm")]
        {
            let input_f64_a: Vec<f64> = input_data.iter().map(|&x| x as f64).collect();
            let mut input_f64_b = input_f64_a.clone();
            if let Some(first) = input_f64_b.first_mut() {
                *first = f64::from_bits(first.to_bits().wrapping_add(1));
            }
            let num_feat = num_features as i32;
            
            group.bench_function(BenchmarkId::new("lightgbm/linear", batch_size), |b| {
                let mut flip = false;
                b.iter(|| {
                    flip = !flip;
                    let input = if flip { &input_f64_a } else { &input_f64_b };
                    let output = lgb_linear.predict(black_box(input), num_feat, true).unwrap();
                    black_box(output)
                })
            });
            
            group.bench_function(BenchmarkId::new("lightgbm/standard", batch_size), |b| {
                let mut flip = false;
                b.iter(|| {
                    flip = !flip;
                    let input = if flip { &input_f64_a } else { &input_f64_b };
                    let output = lgb_standard.predict(black_box(input), num_feat, true).unwrap();
                    black_box(output)
                })
            });
        }
    }
    
    group.finish();
}

// =============================================================================
// Linear Tree Overhead Measurement
// =============================================================================

fn bench_linear_tree_overhead(c: &mut Criterion) {
    let (linear_forest, num_features) = load_lgb_linear_model("bench_linear_medium");
    let linear_predictor = Predictor::<UnrolledTraversal6>::new(&linear_forest).with_block_size(64);
    
    let (standard_forest, _) = load_lgb_linear_model("bench_standard_medium");
    let standard_predictor = Predictor::<UnrolledTraversal6>::new(&standard_forest).with_block_size(64);
    
    let batch_size = 10_000;
    let input_data = random_dense_f32(batch_size, num_features, 42, -1.0, 1.0);
    let matrix = RowMatrix::from_vec(input_data, batch_size, num_features);
    
    let mut group = c.benchmark_group("overhead/linear_tree");
    group.throughput(Throughput::Elements(batch_size as u64));
    
    group.bench_function("standard", |b| {
        b.iter(|| black_box(standard_predictor.predict(black_box(&matrix))))
    });
    
    group.bench_function("linear", |b| {
        b.iter(|| black_box(linear_predictor.predict(black_box(&matrix))))
    });
    
    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_linear_tree_prediction, bench_linear_tree_overhead
}
criterion_main!(benches);
