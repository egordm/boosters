//! Feature bundling example.
//!
//! This example demonstrates Exclusive Feature Bundling (EFB) which automatically
//! groups sparse one-hot encoded features into bundles for more efficient
//! histogram building. The bundling is controlled via `BinningConfig::enable_bundling()`.
//!
//! Run with:
//! ```bash
//! cargo run --example train_bundling --release
//! ```

use std::time::Instant;

use boosters::data::BinningConfig;
use boosters::data::Dataset;
use boosters::data::binned::BinnedDataset;
use boosters::training::GrowthStrategy;
use boosters::{GBDTConfig, GBDTModel, Metric, Objective};
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

    // Create dataset with targets
    let targets_2d = labels.clone().insert_axis(ndarray::Axis(0));
    let dataset = Dataset::from_array(features.view(), Some(targets_2d), None);

    // =========================================================================
    // Train WITHOUT bundling (baseline)
    // =========================================================================
    println!("=== Training WITHOUT bundling ===\n");

    let start = Instant::now();
    let binning_config_no_bundle = BinningConfig::builder()
        .max_bins(256)
        .enable_bundling(false)
        .build();
    let dataset_no_bundle =
        BinnedDataset::from_dataset(&dataset, &binning_config_no_bundle).expect("binning failed");
    let binning_time_no_bundle = start.elapsed();

    // Without bundling, binned columns = original features
    let binned_cols_no_bundle = dataset_no_bundle.n_features();
    let mem_no_bundle = n_samples * binned_cols_no_bundle; // bytes (u8 per bin)

    println!("Features: {}", dataset_no_bundle.n_features());
    println!("Binned columns: {}", binned_cols_no_bundle);
    println!(
        "Binned data size: {} bytes ({:.2} KB)",
        mem_no_bundle,
        mem_no_bundle as f64 / 1024.0
    );
    println!("Binning time: {:?}", binning_time_no_bundle);

    // =========================================================================
    // Train WITH bundling (optimized)
    // =========================================================================
    println!("\n=== Training WITH bundling ===\n");

    let start = Instant::now();
    let binning_config_bundled = BinningConfig::builder()
        .max_bins(256)
        .enable_bundling(true)
        .build();
    let dataset_bundled =
        BinnedDataset::from_dataset(&dataset, &binning_config_bundled).expect("binning failed");
    let binning_time_bundled = start.elapsed();

    // With bundling, sparse one-hot features should be combined
    let binned_cols_bundled = dataset_bundled.n_features();
    let mem_bundled = n_samples * binned_cols_bundled;

    println!("Features: {}", dataset_bundled.n_features());
    println!("Binned columns: {}", binned_cols_bundled);
    println!(
        "Binned data size: {} bytes ({:.2} KB)",
        mem_bundled,
        mem_bundled as f64 / 1024.0
    );
    println!("Binning time: {:?}", binning_time_bundled);

    // =========================================================================
    // Train and compare
    // =========================================================================
    let config_no_bundle = GBDTConfig::builder()
        .objective(Objective::SquaredLoss)
        .metric(Metric::Rmse)
        .binning(binning_config_no_bundle.clone())
        .n_trees(20)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
        .build()
        .expect("Invalid configuration");

    let config_bundled = GBDTConfig::builder()
        .objective(Objective::SquaredLoss)
        .metric(Metric::Rmse)
        .binning(binning_config_bundled.clone())
        .n_trees(20)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
        .build()
        .expect("Invalid configuration");

    println!("\n=== Training Models ===\n");

    // Train without bundling (binning is handled internally)
    let model_no_bundle =
        GBDTModel::train(&dataset, None, config_no_bundle, 1).expect("Training failed");

    // Train with bundling (binning is handled internally)
    let model_bundled =
        GBDTModel::train(&dataset, None, config_bundled, 1).expect("Training failed");

    // Evaluate both - dataset is already feature-major
    let predictions_no_bundle = model_no_bundle.predict(&dataset, 1);
    let predictions_bundled = model_bundled.predict(&dataset, 1);

    let rmse_no_bundle = compute_rmse(
        predictions_no_bundle.as_slice().unwrap(),
        labels.as_slice().unwrap(),
    );
    let rmse_bundled = compute_rmse(
        predictions_bundled.as_slice().unwrap(),
        labels.as_slice().unwrap(),
    );

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
    println!(
        "Memory reduction: {:.1}% ({} → {} columns)",
        memory_reduction, binned_cols_no_bundle, binned_cols_bundled
    );
    println!(
        "Binning overhead: {:?} → {:?}",
        binning_time_no_bundle, binning_time_bundled
    );
    println!("\nNote: EFB's primary benefit is MEMORY reduction, not training speed.");
    println!("      Training time is dominated by tree building, not histogram storage.");

    // =========================================================================
    // Bundling Configuration
    // =========================================================================
    println!("\n=== Bundling Configuration ===");
    println!("  BinningConfig::builder().enable_bundling(true)  - Enable bundling (default)");
    println!("  BinningConfig::builder().enable_bundling(false) - Disable bundling");
    println!("\nBundling algorithm parameters can be customized via BundlingConfig.");
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
