//! Feature bundling example.
//!
//! This example demonstrates how to use Exclusive Feature Bundling (EFB) to
//! reduce memory usage when training with one-hot encoded or sparse features.
//!
//! When features are mutually exclusive (never non-zero simultaneously), they
//! can be bundled into a single feature, reducing memory by up to 40x.
//!
//! Run with:
//! ```bash
//! cargo run --example train_bundling --release
//! ```

use std::time::Instant;

use boosters::data::binned::{BinnedDatasetBuilder, BundlingConfig};
use boosters::data::{transpose_to_c_order, FeaturesView};
use boosters::{GBDTConfig, GBDTModel, Metric, Objective, TreeParams};
use ndarray::{Array1, Array2};

fn main() {
    // =========================================================================
    // Generate synthetic data with one-hot encoded categorical features
    // =========================================================================
    // Simulate a dataset with:
    // - 2 numerical features (x0, x1)
    // - 1 categorical with 10 categories (one-hot: x2-x11, 90% sparse)
    // - 1 categorical with 15 categories (one-hot: x12-x26, 93% sparse)
    // Total: 27 features, but only 4 effective dimensions
    //
    // The default bundling threshold is 90% sparsity, so we use high-cardinality
    // categoricals to demonstrate bundling effectiveness.

    let n_samples = 1000;
    let n_cat1 = 10; // 10 categories → 90% sparse
    let n_cat2 = 15; // 15 categories → ~93% sparse
    let n_features = 2 + n_cat1 + n_cat2; // 27 total

    // Feature-major data [n_features, n_samples]
    let mut features = Array2::<f32>::zeros((n_features, n_samples));
    let mut labels = Array1::<f32>::zeros(n_samples);

    for i in 0..n_samples {
        // Numerical features
        let x0 = (i as f32) / (n_samples as f32) * 10.0;
        let x1 = ((i * 7) % 100) as f32 / 10.0;

        features[(0, i)] = x0;
        features[(1, i)] = x1;

        // Categorical 1: 10 categories (one-hot encoded)
        let cat1 = i % n_cat1;
        for c in 0..n_cat1 {
            features[(2 + c, i)] = if c == cat1 { 1.0 } else { 0.0 };
        }

        // Categorical 2: 15 categories (one-hot encoded)
        let cat2 = i % n_cat2;
        for c in 0..n_cat2 {
            features[(2 + n_cat1 + c, i)] = if c == cat2 { 1.0 } else { 0.0 };
        }

        // Target: combination of numerical and categorical effects
        let cat1_effect = (cat1 as f32 - 5.0) * 0.2;
        let cat2_effect = (cat2 as f32 - 7.5) * 0.1;
        let noise = ((i * 31) % 100) as f32 / 500.0 - 0.1;
        labels[i] = x0 * 0.3 + x1 * 0.2 + cat1_effect + cat2_effect + noise;
    }

    println!("Dataset: {} samples × {} features", n_samples, n_features);
    println!("  - 2 numerical features");
    println!("  - 1 categorical with {} categories (one-hot)", n_cat1);
    println!("  - 1 categorical with {} categories (one-hot)\n", n_cat2);

    // Create FeaturesView for training
    let features_view = FeaturesView::from_array(features.view());

    // For prediction, transpose to sample-major with C-order
    let samples = transpose_to_c_order(features.view());

    // =========================================================================
    // Train WITHOUT bundling (baseline)
    // =========================================================================
    println!("=== Training WITHOUT bundling ===\n");

    let start = Instant::now();
    let dataset_no_bundle = BinnedDatasetBuilder::from_matrix(&features_view, 256)
        .with_bundling(BundlingConfig::disabled())
        .build()
        .expect("Failed to build dataset");
    let binning_time_no_bundle = start.elapsed();

    // Without bundling, binned columns = original features
    let binned_cols_no_bundle = dataset_no_bundle.n_features();
    let mem_no_bundle = n_samples * binned_cols_no_bundle; // bytes (u8 per bin)

    println!("Features: {}", dataset_no_bundle.n_features());
    println!("Binned columns: {}", binned_cols_no_bundle);
    println!("Binned data size: {} bytes ({:.2} KB)", mem_no_bundle, mem_no_bundle as f64 / 1024.0);
    println!("Binning time: {:?}", binning_time_no_bundle);

    // =========================================================================
    // Train WITH bundling (optimized)
    // =========================================================================
    println!("\n=== Training WITH bundling ===\n");

    let start = Instant::now();
    let dataset_bundled = BinnedDatasetBuilder::from_matrix(&features_view, 256)
        .with_bundling(BundlingConfig::auto())
        .build()
        .expect("Failed to build dataset");
    let binning_time_bundled = start.elapsed();

    let binned_cols_bundled = dataset_bundled.bundling_stats()
        .map(|s| s.binned_columns)
        .unwrap_or(dataset_bundled.n_features());
    let mem_bundled = n_samples * binned_cols_bundled;

    println!("Features: {}", dataset_bundled.n_features());
    println!("Binned columns: {}", binned_cols_bundled);
    println!("Binned data size: {} bytes ({:.2} KB)", mem_bundled, mem_bundled as f64 / 1024.0);
    println!("Binning time: {:?}", binning_time_bundled);
    println!(
        "Bundling effective: {}",
        dataset_bundled.has_effective_bundling()
    );

    if let Some(stats) = dataset_bundled.bundling_stats() {
        println!("\n--- Bundling Statistics ---");
        println!("  Bundles created: {}", stats.bundles_created);
        println!("  Standalone features: {}", stats.standalone_features);
        println!("  Skipped features: {}", stats.skipped_features);
        println!("  Sparse features analyzed: {}", stats.original_sparse_features);
        println!("  Total conflicts: {}", stats.total_conflicts);
        println!("  Is effective: {}", stats.is_effective);
        println!(
            "  Reduction ratio: {:.1}% of original",
            stats.reduction_ratio * 100.0
        );
        println!("  Binned columns after bundling: {}", stats.binned_columns);
    }

    // =========================================================================
    // Train and compare
    // =========================================================================
    let config = GBDTConfig::builder()
        .objective(Objective::squared())
        .metric(Metric::rmse())
        .n_trees(20)
        .learning_rate(0.1)
        .tree(TreeParams::depth_wise(4))
        .build()
        .expect("Invalid configuration");

    println!("\n=== Training Models ===\n");

    // Train without bundling
    let model_no_bundle = GBDTModel::train(
        &dataset_no_bundle,
        labels.view(),
        None,
        &[],
        config.clone(),
        1,
    )
    .expect("Training failed");

    // Train with bundling
    let model_bundled = GBDTModel::train(
        &dataset_bundled,
        labels.view(),
        None,
        &[],
        config,
        1,
    )
    .expect("Training failed");

    // Evaluate both
    let predictions_no_bundle = model_no_bundle.predict(samples.view(), 1);
    let predictions_bundled = model_bundled.predict(samples.view(), 1);

    let rmse_no_bundle = compute_rmse(predictions_no_bundle.as_slice().unwrap(), labels.as_slice().unwrap());
    let rmse_bundled = compute_rmse(predictions_bundled.as_slice().unwrap(), labels.as_slice().unwrap());

    println!("=== Results ===");
    println!("Without bundling - RMSE: {:.4}", rmse_no_bundle);
    println!("With bundling    - RMSE: {:.4}", rmse_bundled);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== EFB Value Summary ===");
    let memory_reduction = if mem_no_bundle > 0 {
        (1.0 - mem_bundled as f64 / mem_no_bundle as f64) * 100.0
    } else {
        0.0
    };
    println!("Memory reduction: {:.1}% ({} → {} columns)", 
        memory_reduction, binned_cols_no_bundle, binned_cols_bundled);
    println!("Binning overhead: {:?} → {:?}", binning_time_no_bundle, binning_time_bundled);
    println!("\nNote: EFB's primary benefit is MEMORY reduction, not training speed.");
    println!("      Training time is dominated by tree building, not histogram storage.");

    // =========================================================================
    // Bundling Presets
    // =========================================================================
    println!("\n=== Available Bundling Presets ===");
    println!("  BundlingConfig::auto()       - Default, works well for most cases");
    println!("  BundlingConfig::disabled()   - Disable bundling");
    println!("  BundlingConfig::aggressive() - Bundle more aggressively (higher conflict tolerance)");
    println!("  BundlingConfig::strict()     - Only bundle truly exclusive features");
}

fn compute_rmse(predictions: &[f32], labels: &[f32]) -> f64 {
    let mse: f64 = predictions
        .iter()
        .zip(labels.iter())
        .map(|(p, l)| {
            let diff = *p as f64 - *l as f64;
            diff * diff
        })
        .sum::<f64>()
        / labels.len() as f64;
    mse.sqrt()
}
