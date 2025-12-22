//! Class imbalance handling with sample weights.
//!
//! This example demonstrates how to use sample weights to improve
//! model performance on imbalanced classification datasets.
//!
//! ## The Problem
//!
//! When one class significantly outnumbers another (e.g., fraud detection,
//! rare disease diagnosis), standard training tends to predict the majority
//! class. Sample weights allow us to emphasize minority class samples.
//!
//! ## Solution
//!
//! Give higher weights to minority class samples so they contribute more
//! to the loss function during training.
//!
//! Run with:
//! ```bash
//! cargo run --example train_imbalanced
//! ```

use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{ColMatrix, DenseMatrix, RowMajor};
use boosters::training::{Accuracy, GBDTParams, GBDTTrainer, GrowthStrategy, LogLoss, LogisticLoss, MetricFn};
use boosters::Parallelism;
use ndarray::{Array2, ArrayView1};

fn main() {
    // =========================================================================
    // Generate Imbalanced Dataset (10:1 ratio)
    // =========================================================================
    // Class 0 (majority): 900 samples centered at (3, 3)
    // Class 1 (minority): 100 samples centered at (5, 5)
    let n_majority = 900;
    let n_minority = 100;
    let n_features = 2;
    let n_samples = n_majority + n_minority;

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_majority {
        let noise1 = ((i * 17) % 200) as f32 / 40.0 - 2.5;
        let noise2 = ((i * 31) % 200) as f32 / 40.0 - 2.5;
        features.push(3.0 + noise1);
        features.push(3.0 + noise2);
        labels.push(0.0);
    }

    for i in 0..n_minority {
        let noise1 = ((i * 23) % 200) as f32 / 40.0 - 2.5;
        let noise2 = ((i * 37) % 200) as f32 / 40.0 - 2.5;
        features.push(5.0 + noise1);
        features.push(5.0 + noise2);
        labels.push(1.0);
    }

    // Create binned dataset
    let row_matrix: DenseMatrix<f32, RowMajor> =
        DenseMatrix::from_vec(features.clone(), n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
    let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .expect("Failed to build binned dataset");

    // =========================================================================
    // Compute Class Weights (inverse frequency)
    // =========================================================================
    let weight_class_0 = n_samples as f32 / (2.0 * n_majority as f32);
    let weight_class_1 = n_samples as f32 / (2.0 * n_minority as f32);

    let class_weights: Vec<f32> = labels
        .iter()
        .map(|&l| if l < 0.5 { weight_class_0 } else { weight_class_1 })
        .collect();

    println!("=== Imbalanced Classification Example ===\n");
    println!("Dataset: {} majority, {} minority ({:.0}:1 ratio)\n",
        n_majority, n_minority, n_majority as f32 / n_minority as f32);
    println!("Class weights: majority={:.2}, minority={:.2}\n", weight_class_0, weight_class_1);

    let params = GBDTParams {
        n_trees: 30,
        learning_rate: 0.1,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
        cache_size: 32,
        ..Default::default()
    };

    let trainer = GBDTTrainer::new(LogisticLoss, LogLoss, params);

    // =========================================================================
    // Train WITHOUT weights (baseline)
    // =========================================================================
    println!("--- Training WITHOUT weights ---");
    let forest_unweighted = trainer
        .train(&dataset, ArrayView1::from(&labels[..]), None, &[], Parallelism::Sequential)
        .unwrap();

    // Convert logits to probabilities and compute accuracy + recall
    let probs_uw: Vec<f32> = features
        .chunks(n_features)
        .map(|row| {
            let logit = forest_unweighted.predict_row(row)[0];
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();

    let pred_arr_uw = Array2::from_shape_vec((1, probs_uw.len()), probs_uw.to_vec()).unwrap();
    let acc_uw = Accuracy::default().compute(pred_arr_uw.view(), ArrayView1::from(&labels[..]), None);
    let recall_1_uw = compute_recall(&probs_uw, &labels, 1.0);
    println!("  Accuracy: {:.1}%", acc_uw * 100.0);
    println!("  Minority recall: {:.1}%\n", recall_1_uw * 100.0);

    // =========================================================================
    // Train WITH class weights
    // =========================================================================
    println!("--- Training WITH class weights ---");
    let forest_weighted = trainer
        .train(&dataset, ArrayView1::from(&labels[..]), Some(ArrayView1::from(&class_weights[..])), &[], Parallelism::Sequential)
        .unwrap();

    let probs_w: Vec<f32> = features
        .chunks(n_features)
        .map(|row| {
            let logit = forest_weighted.predict_row(row)[0];
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();

    let pred_arr_w = Array2::from_shape_vec((1, probs_w.len()), probs_w.to_vec()).unwrap();
    let acc_w = Accuracy::default().compute(pred_arr_w.view(), ArrayView1::from(&labels[..]), None);
    let recall_1_w = compute_recall(&probs_w, &labels, 1.0);
    println!("  Accuracy: {:.1}%", acc_w * 100.0);
    println!("  Minority recall: {:.1}%\n", recall_1_w * 100.0);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("=== Summary ===");
    println!("Minority recall improvement: {:.1}%", (recall_1_w - recall_1_uw) * 100.0);

    if recall_1_w > recall_1_uw {
        println!("✓ Weighted training improved minority class recall!");
    }
}

/// Compute recall for a specific class.
fn compute_recall(probs: &[f32], labels: &[f32], target_class: f32) -> f32 {
    let mut tp = 0;
    let mut total = 0;
    for (&prob, &label) in probs.iter().zip(labels) {
        if (label - target_class).abs() < 0.5 {
            total += 1;
            let pred = if prob >= 0.5 { 1.0 } else { 0.0 };
            if (pred - target_class).abs() < 0.5 {
                tp += 1;
            }
        }
    }
    if total > 0 { tp as f32 / total as f32 } else { 0.0 }
}
