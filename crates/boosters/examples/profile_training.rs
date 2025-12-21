//! Profiling example for training performance analysis.
//!
//! Run with samply:
//! ```bash
//! RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release --example profile_training
//! samply record target/release/examples/profile_training
//! ```

use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::training::{GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, SquaredLoss};
use boosters::Parallelism;

fn main() {
    // Large synthetic dataset for profiling
    let n_samples: usize = 50_000;
    let n_features: usize = 100;
    let n_trees: usize = 100;
    let max_depth = 6;

    println!("Generating synthetic data...");
    println!("  Samples: {}", n_samples);
    println!("  Features: {}", n_features);
    
    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    // Simple synthetic function with noise
    for i in 0..n_samples {
        let mut sum = 0.0f32;
        for f in 0..n_features {
            let val = (((i * (f + 7)) % 1000) as f32) / 100.0;
            features.push(val);
            sum += val * (1.0 / ((f + 1) as f32).sqrt());
        }
        let noise = ((i * 31) % 100) as f32 / 100.0 - 0.5;
        labels.push(sum + noise);
    }

    println!("Building binned dataset...");
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features, n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
    
    let start = std::time::Instant::now();
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build binned dataset");
    println!("  Binning took: {:?}", start.elapsed());

    // Training
    let params = GBDTParams {
        n_trees: n_trees as u32,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth },
        gain: GainParams {
            reg_lambda: 1.0,
            min_child_weight: 1.0,
            min_gain: 0.0,
            ..Default::default()
        },
        cache_size: 64,
        ..Default::default()
    };

    println!("\nTraining...");
    println!("  Trees: {}", params.n_trees);
    println!("  Depth: {}", max_depth);
    
    let start = std::time::Instant::now();
    // Use Sequential for cleaner profiling (single thread)
    let forest = GBDTTrainer::new(SquaredLoss, Rmse, params)
        .train(&dataset, &labels, &[], &[], Parallelism::SEQUENTIAL)
        .unwrap();
    let train_time = start.elapsed();

    println!("\n=== Results ===");
    println!("Training time: {:?}", train_time);
    println!("Trees: {}", forest.n_trees());
    println!("Throughput: {:.2} Melem/s", 
        (n_samples * n_features * n_trees) as f64 / train_time.as_secs_f64() / 1_000_000.0);
}
