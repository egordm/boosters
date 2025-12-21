//! GOSS (Gradient-based One-Side Sampling) comparison example.
//!
//! Compares training speed and model quality between:
//! - No sampling (baseline)
//! - GOSS sampling (top_rate=0.2, other_rate=0.1)
//!
//! Run with:
//! ```bash
//! cargo run --example train_goss --release
//! ```

use std::time::Instant;

use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::testing::data::{random_dense_f32, synthetic_regression_targets_linear};
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, MetricFn, Rmse, RowSamplingParams,
    SquaredLoss,
};
use boosters::Parallelism;

fn main() {
    // =========================================================================
    // Generate larger synthetic dataset to see sampling benefits
    // =========================================================================
    let n_samples = 100_000;
    let n_features = 100;
    let n_trees = 100;
    let max_depth = 6;

    println!("Generating dataset: {} rows × {} features", n_samples, n_features);
    let features = random_dense_f32(n_samples, n_features, 42, -1.0, 1.0);
    let (labels, _weights, _bias) =
        synthetic_regression_targets_linear(&features, n_samples, n_features, 1337, 0.1);

    // Split train/test (80/20)
    let split_idx = (n_samples as f32 * 0.8) as usize;
    let train_features: Vec<f32> = features[..split_idx * n_features].to_vec();
    let train_labels: Vec<f32> = labels[..split_idx].to_vec();
    let test_features: Vec<f32> = features[split_idx * n_features..].to_vec();
    let test_labels: Vec<f32> = labels[split_idx..].to_vec();
    let n_train = split_idx;
    let n_test = n_samples - split_idx;

    println!("Train: {} samples, Test: {} samples\n", n_train, n_test);

    // Build binned dataset for training
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(train_features.clone(), n_train, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build binned dataset");

    // =========================================================================
    // Configuration 1: No sampling (baseline)
    // =========================================================================
    println!("=== No Sampling (Baseline) ===");
    let params_baseline = GBDTParams {
        n_trees,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        row_sampling: RowSamplingParams::None,
        cache_size: 32,
        ..Default::default()
    };

    let trainer_baseline = GBDTTrainer::new(SquaredLoss, Rmse, params_baseline);

    let start = Instant::now();
    let forest_baseline = trainer_baseline
        .train(&dataset, &train_labels, &[], &[], Parallelism::SEQUENTIAL)
        .unwrap();
    let time_baseline = start.elapsed();

    let preds_baseline: Vec<f32> = test_features
        .chunks(n_features)
        .map(|row| forest_baseline.predict_row(row)[0])
        .collect();
    let rmse_baseline = Rmse.compute(n_test, 1, &preds_baseline, &test_labels, &[]);

    println!("  Training time: {:?}", time_baseline);
    println!("  Test RMSE: {:.6}", rmse_baseline);
    println!();

    // =========================================================================
    // Configuration 2: GOSS (LightGBM defaults: top_rate=0.2, other_rate=0.1)
    // =========================================================================
    println!("=== GOSS Sampling (top=0.2, other=0.1) ===");
    let params_goss = GBDTParams {
        n_trees,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        row_sampling: RowSamplingParams::goss(0.2, 0.1),
        cache_size: 32,
        ..Default::default()
    };

    let trainer_goss = GBDTTrainer::new(SquaredLoss, Rmse, params_goss);

    let start = Instant::now();
    let forest_goss = trainer_goss
        .train(&dataset, &train_labels, &[], &[], Parallelism::SEQUENTIAL)
        .unwrap();
    let time_goss = start.elapsed();

    let preds_goss: Vec<f32> = test_features
        .chunks(n_features)
        .map(|row| forest_goss.predict_row(row)[0])
        .collect();
    let rmse_goss = Rmse.compute(n_test, 1, &preds_goss, &test_labels, &[]);

    println!("  Training time: {:?}", time_goss);
    println!("  Test RMSE: {:.6}", rmse_goss);
    println!("  Effective sample rate: 30% (20% top + 10% other)");
    println!();

    // =========================================================================
    // Configuration 3: Uniform sampling (30% for fair comparison)
    // =========================================================================
    println!("=== Uniform Sampling (30%) ===");
    let params_uniform = GBDTParams {
        n_trees,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        row_sampling: RowSamplingParams::uniform(0.3),
        cache_size: 32,
        ..Default::default()
    };

    let trainer_uniform = GBDTTrainer::new(SquaredLoss, Rmse, params_uniform);

    let start = Instant::now();
    let forest_uniform = trainer_uniform
        .train(&dataset, &train_labels, &[], &[], Parallelism::SEQUENTIAL)
        .unwrap();
    let time_uniform = start.elapsed();

    let preds_uniform: Vec<f32> = test_features
        .chunks(n_features)
        .map(|row| forest_uniform.predict_row(row)[0])
        .collect();
    let rmse_uniform = Rmse.compute(n_test, 1, &preds_uniform, &test_labels, &[]);

    println!("  Training time: {:?}", time_uniform);
    println!("  Test RMSE: {:.6}", rmse_uniform);
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("=== Summary ===");
    println!(
        "{:<20} {:>12} {:>12} {:>12}",
        "Strategy", "Time", "RMSE", "Speedup"
    );
    println!("{:-<58}", "");

    let baseline_ms = time_baseline.as_secs_f64() * 1000.0;
    let goss_ms = time_goss.as_secs_f64() * 1000.0;
    let uniform_ms = time_uniform.as_secs_f64() * 1000.0;

    println!(
        "{:<20} {:>10.1}ms {:>12.6} {:>11.2}x",
        "No sampling", baseline_ms, rmse_baseline, 1.0
    );
    println!(
        "{:<20} {:>10.1}ms {:>12.6} {:>11.2}x",
        "GOSS (0.2, 0.1)",
        goss_ms,
        rmse_goss,
        baseline_ms / goss_ms
    );
    println!(
        "{:<20} {:>10.1}ms {:>12.6} {:>11.2}x",
        "Uniform (0.3)",
        uniform_ms,
        rmse_uniform,
        baseline_ms / uniform_ms
    );

    println!();
    println!("GOSS vs Uniform (same sample rate):");
    println!("  GOSS RMSE delta: {:+.6}", rmse_goss - rmse_baseline);
    println!("  Uniform RMSE delta: {:+.6}", rmse_uniform - rmse_baseline);

    if rmse_goss < rmse_uniform {
        println!(
            "  → GOSS achieves {:.1}% better RMSE than uniform sampling!",
            (1.0 - rmse_goss / rmse_uniform) * 100.0
        );
    }
}
