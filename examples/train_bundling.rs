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
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::training::{GBDTParams, GBDTTrainer, GrowthStrategy, Metric, Rmse, SquaredLoss};

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

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Numerical features
        let x0 = (i as f32) / (n_samples as f32) * 10.0;
        let x1 = ((i * 7) % 100) as f32 / 10.0;

        features.push(x0);
        features.push(x1);

        // Categorical 1: 10 categories (one-hot encoded)
        let cat1 = i % n_cat1;
        for c in 0..n_cat1 {
            features.push(if c == cat1 { 1.0 } else { 0.0 });
        }

        // Categorical 2: 15 categories (one-hot encoded)
        let cat2 = i % n_cat2;
        for c in 0..n_cat2 {
            features.push(if c == cat2 { 1.0 } else { 0.0 });
        }

        // Target: combination of numerical and categorical effects
        let cat1_effect = (cat1 as f32 - 5.0) * 0.2;
        let cat2_effect = (cat2 as f32 - 7.5) * 0.1;
        let noise = ((i * 31) % 100) as f32 / 500.0 - 0.1;
        labels.push(x0 * 0.3 + x1 * 0.2 + cat1_effect + cat2_effect + noise);
    }

    println!("Dataset: {} samples × {} features", n_samples, n_features);
    println!("  - 2 numerical features");
    println!("  - 1 categorical with {} categories (one-hot)", n_cat1);
    println!("  - 1 categorical with {} categories (one-hot)\n", n_cat2);

    // =========================================================================
    // Train WITHOUT bundling (baseline)
    // =========================================================================
    println!("=== Training WITHOUT bundling ===\n");

    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features.clone(), n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

    let start = Instant::now();
    let dataset_no_bundle = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
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
    let dataset_bundled = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
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
    let params = GBDTParams {
        n_trees: 20,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
        ..Default::default()
    };

    println!("\n=== Training Models ===\n");

    // Train without bundling
    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params.clone());
    let forest_no_bundle = trainer
        .train(&dataset_no_bundle, &labels, &[], &[])
        .unwrap();

    // Train with bundling
    let forest_bundled = trainer
        .train(&dataset_bundled, &labels, &[], &[])
        .unwrap();

    // Evaluate both
    let predictions_no_bundle: Vec<f32> = features
        .chunks(n_features)
        .map(|row| forest_no_bundle.predict_row(row)[0])
        .collect();

    let predictions_bundled: Vec<f32> = features
        .chunks(n_features)
        .map(|row| forest_bundled.predict_row(row)[0])
        .collect();

    let rmse_no_bundle = Rmse.compute(n_samples, 1, &predictions_no_bundle, &labels, &[]);
    let rmse_bundled = Rmse.compute(n_samples, 1, &predictions_bundled, &labels, &[]);

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
