//! GOSS (Gradient-based One-Side Sampling) comparison example.
//!
//! Compares training speed and model quality between:
//! - No sampling (baseline)
//! - GOSS sampling (top_rate=0.2, other_rate=0.1)
//!
//! **Note:** GOSS is an advanced feature that requires the lower-level
//! `GBDTTrainer` API. For most use cases, use `GBDTModel` with uniform
//! subsampling via `SamplingParams`.
//!
//! Run with:
//! ```bash
//! cargo run --example train_goss --release
//! ```

use std::time::Instant;

use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::BinningConfig;
use boosters::dataset::{Dataset, TargetsView, WeightsView};
use boosters::inference::gbdt::SimplePredictor;
use boosters::repr::gbdt::Forest;
use boosters::testing::data::synthetic_regression;
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, RowSamplingParams,
    SquaredLoss,
};
use boosters::Parallelism;
use ndarray::Array2;

/// Predict a single row using the predictor.
fn predict_row(forest: &Forest, features: &[f32]) -> Vec<f32> {
    let predictor = SimplePredictor::new(forest);
    let mut output = vec![0.0; predictor.n_groups()];
    predictor.predict_row_into(features, None, &mut output);
    output
}

fn main() {
    // =========================================================================
    // Generate larger synthetic dataset to see sampling benefits
    // =========================================================================
    let n_samples = 100_000;
    let n_features = 100;
    let n_trees = 100;
    let max_depth = 6;

    println!("Generating dataset: {} rows × {} features", n_samples, n_features);
    let dataset = synthetic_regression(n_samples, n_features, 42, 0.1);

    // Split train/test (80/20)
    let split_idx = (n_samples as f32 * 0.8) as usize;
    let n_train = split_idx;
    let n_test = n_samples - split_idx;

    println!("Train: {} samples, Test: {} samples\n", n_train, n_test);

    // Get train features from the dataset's feature-major layout
    // The dataset has feature-major: [n_features, n_samples]
    let features_fm = dataset.features.view();
    let labels = dataset.targets_slice();
    
    // Extract train/test splits
    let mut train_features = Array2::<f32>::zeros((n_features, n_train));
    for f in 0..n_features {
        for i in 0..n_train {
            train_features[(f, i)] = features_fm[(f, i)];
        }
    }
    let train_labels: Vec<f32> = labels[..n_train].to_vec();
    
    // For test data: create sample-major array for predict_row
    let mut test_features: Vec<f32> = Vec::with_capacity(n_test * n_features);
    for i in split_idx..n_samples {
        for f in 0..n_features {
            test_features.push(features_fm[(f, i)]);
        }
    }
    let test_labels: Vec<f32> = labels[split_idx..].to_vec();

    // Build binned dataset for training
    let features_dataset = Dataset::new(train_features.view(), None, None);
    let dataset = BinnedDatasetBuilder::new(BinningConfig::builder().max_bins(256).build())
        .add_features(features_dataset.features(), Parallelism::Parallel)
        .build()
        .expect("Failed to build binned dataset");

    // =========================================================================
    // Configuration 1: No sampling (baseline)
    // =========================================================================
    println!("=== No Sampling (Baseline) ===");
    let params_baseline = GBDTParams {
        n_trees: n_trees as u32,
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

    // Wrap labels in TargetsView (shape [n_outputs=1, n_samples])
    let targets_2d = ndarray::Array2::from_shape_vec((1, train_labels.len()), train_labels.clone()).unwrap();
    let targets = TargetsView::new(targets_2d.view());

    let start = Instant::now();
    let forest_baseline = trainer_baseline
        .train(&dataset, targets.clone(), WeightsView::None, &[], Parallelism::Sequential)
        .unwrap();
    let time_baseline = start.elapsed();

    let preds_baseline: Vec<f32> = test_features
        .chunks(n_features)
        .map(|row| predict_row(&forest_baseline, row)[0])
        .collect();
    let rmse_baseline = compute_rmse(&preds_baseline, &test_labels);

    println!("  Training time: {:?}", time_baseline);
    println!("  Test RMSE: {:.6}", rmse_baseline);
    println!();

    // =========================================================================
    // Configuration 2: GOSS (LightGBM defaults: top_rate=0.2, other_rate=0.1)
    // =========================================================================
    println!("=== GOSS Sampling (top=0.2, other=0.1) ===");
    let params_goss = GBDTParams {
        n_trees: n_trees as u32,
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
        .train(&dataset, targets.clone(), WeightsView::None, &[], Parallelism::Sequential)
        .unwrap();
    let time_goss = start.elapsed();

    let preds_goss: Vec<f32> = test_features
        .chunks(n_features)
        .map(|row| predict_row(&forest_goss, row)[0])
        .collect();
    let rmse_goss = compute_rmse(&preds_goss, &test_labels);

    println!("  Training time: {:?}", time_goss);
    println!("  Test RMSE: {:.6}", rmse_goss);
    println!("  Effective sample rate: 30% (20% top + 10% other)");
    println!();

    // =========================================================================
    // Configuration 3: Uniform sampling (30% for fair comparison)
    // =========================================================================
    println!("=== Uniform Sampling (30%) ===");
    let params_uniform = GBDTParams {
        n_trees: n_trees as u32,
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
        .train(&dataset, targets, WeightsView::None, &[], Parallelism::Sequential)
        .unwrap();
    let time_uniform = start.elapsed();

    let preds_uniform: Vec<f32> = test_features
        .chunks(n_features)
        .map(|row| predict_row(&forest_uniform, row)[0])
        .collect();
    let rmse_uniform = compute_rmse(&preds_uniform, &test_labels);

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
