//! Profiling example for training performance analysis.
//!
//! Run with samply:
//! ```bash
//! RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release --example profile_training
//! samply record target/release/examples/profile_training
//! ```

use boosters::data::BinningConfig;
use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{Dataset, TargetsView, WeightsView};
use boosters::training::GrowthStrategy;
use boosters::{GBDTConfig, GBDTModel, Metric, Objective, Parallelism};
use ndarray::Array2;

fn main() {
    // Large synthetic dataset for profiling
    let n_samples: usize = 50_000;
    let n_features: usize = 100;
    let n_trees: usize = 100;
    let max_depth = 6;

    println!("Generating synthetic data...");
    println!("  Samples: {}", n_samples);
    println!("  Features: {}", n_features);

    // Generate feature-major data [n_features, n_samples]
    let mut features = Array2::<f32>::zeros((n_features, n_samples));
    let mut labels = Vec::with_capacity(n_samples);

    // Simple synthetic function with noise
    for i in 0..n_samples {
        let mut sum = 0.0f32;
        for f in 0..n_features {
            let val = (((i * (f + 7)) % 1000) as f32) / 100.0;
            features[(f, i)] = val;
            sum += val * (1.0 / ((f + 1) as f32).sqrt());
        }
        let noise = ((i * 31) % 100) as f32 / 100.0 - 0.5;
        labels.push(sum + noise);
    }

    println!("Building binned dataset...");
    let features_dataset = Dataset::new(features.view(), None, None);

    let start = std::time::Instant::now();
    let dataset = BinnedDatasetBuilder::with_config(BinningConfig::builder().max_bins(256).build())
        .add_features(features_dataset.features(), Parallelism::Parallel)
        .build()
        .expect("Failed to build binned dataset");
    println!("  Binning took: {:?}", start.elapsed());

    // Training
    let config = GBDTConfig::builder()
        .objective(Objective::squared())
        .metric(Metric::rmse())
        .n_trees(n_trees as u32)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth })
        .lambda(1.0)
        .min_child_weight(1.0)
        .min_gain(0.0)
        .cache_size(64)
        .build()
        .expect("Invalid configuration");

    println!("\nTraining...");
    println!("  Trees: {}", config.n_trees);
    println!("  Depth: {}", max_depth);

    let start = std::time::Instant::now();
    // Use n_threads=1 for cleaner profiling (single thread)
    // Wrap labels in TargetsView (shape [n_outputs=1, n_samples])
    let targets_2d = ndarray::Array2::from_shape_vec((1, labels.len()), labels.clone()).unwrap();
    let targets = TargetsView::new(targets_2d.view());

    let model = GBDTModel::train_binned(&dataset, targets, WeightsView::None, &[], config, 1)
        .expect("Training failed");
    let train_time = start.elapsed();

    println!("\n=== Results ===");
    println!("Training time: {:?}", train_time);
    println!("Trees: {}", model.forest().n_trees());
    println!(
        "Throughput: {:.2} Melem/s",
        (n_samples * n_features * n_trees) as f64 / train_time.as_secs_f64() / 1_000_000.0
    );
}
