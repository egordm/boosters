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

use boosters::data::Dataset;
use boosters::training::GrowthStrategy;
use boosters::{GBDTConfig, GBDTModel, Metric, Objective};
use ndarray::{Array1, Array2};

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

    // Generate feature-major data [n_features, n_samples]
    let mut features = Array2::<f32>::zeros((n_features, n_samples));
    let mut labels = Array1::<f32>::zeros(n_samples);

    for i in 0..n_majority {
        let noise1 = ((i * 17) % 200) as f32 / 40.0 - 2.5;
        let noise2 = ((i * 31) % 200) as f32 / 40.0 - 2.5;
        features[(0, i)] = 3.0 + noise1;
        features[(1, i)] = 3.0 + noise2;
        labels[i] = 0.0;
    }

    for i in 0..n_minority {
        let sample_idx = n_majority + i;
        let noise1 = ((i * 23) % 200) as f32 / 40.0 - 2.5;
        let noise2 = ((i * 37) % 200) as f32 / 40.0 - 2.5;
        features[(0, sample_idx)] = 5.0 + noise1;
        features[(1, sample_idx)] = 5.0 + noise2;
        labels[sample_idx] = 1.0;
    }

    // Wrap labels in targets array (shape [n_outputs=1, n_samples])
    let targets_2d = labels.clone().insert_axis(ndarray::Axis(0));
    let dataset = Dataset::from_array(features.view(), Some(targets_2d.clone()), None);

    // =========================================================================
    // Compute Class Weights (inverse frequency)
    // =========================================================================
    let weight_class_0 = n_samples as f32 / (2.0 * n_majority as f32);
    let weight_class_1 = n_samples as f32 / (2.0 * n_minority as f32);

    let class_weights: Vec<f32> = labels
        .iter()
        .map(|&l| {
            if l < 0.5 {
                weight_class_0
            } else {
                weight_class_1
            }
        })
        .collect();

    println!("=== Imbalanced Classification Example ===\n");
    println!(
        "Dataset: {} majority, {} minority ({:.0}:1 ratio)\n",
        n_majority,
        n_minority,
        n_majority as f32 / n_minority as f32
    );
    println!(
        "Class weights: majority={:.2}, minority={:.2}\n",
        weight_class_0, weight_class_1
    );

    let config_unweighted = GBDTConfig::builder()
        .objective(Objective::LogisticLoss)
        .metric(Metric::LogLoss)
        .n_trees(30)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
        .cache_size(32)
        .build()
        .expect("Invalid configuration");

    let config_weighted = GBDTConfig::builder()
        .objective(Objective::LogisticLoss)
        .metric(Metric::LogLoss)
        .n_trees(30)
        .learning_rate(0.1)
        .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
        .cache_size(32)
        .build()
        .expect("Invalid configuration");

    // =========================================================================
    // Train WITHOUT weights (baseline)
    // =========================================================================
    println!("--- Training WITHOUT weights ---");
    let model_unweighted =
        GBDTModel::train(&dataset, None, config_unweighted, 1).expect("Training failed");

    let probs_uw = model_unweighted.predict(&dataset, 1);

    let acc_uw = compute_accuracy(probs_uw.as_slice().unwrap(), labels.as_slice().unwrap());
    let recall_1_uw = compute_recall(
        probs_uw.as_slice().unwrap(),
        labels.as_slice().unwrap(),
        1.0,
    );
    println!("  Accuracy: {:.1}%", acc_uw * 100.0);
    println!("  Minority recall: {:.1}%\n", recall_1_uw * 100.0);

    // =========================================================================
    // Train WITH class weights
    // =========================================================================
    println!("--- Training WITH class weights ---");
    let dataset_weighted = Dataset::from_array(
        features.view(),
        Some(targets_2d),
        Some(Array1::from_vec(class_weights)),
    );
    let model_weighted =
        GBDTModel::train(&dataset_weighted, None, config_weighted, 1).expect("Training failed");

    let probs_w = model_weighted.predict(&dataset_weighted, 1);

    let acc_w = compute_accuracy(probs_w.as_slice().unwrap(), labels.as_slice().unwrap());
    let recall_1_w = compute_recall(probs_w.as_slice().unwrap(), labels.as_slice().unwrap(), 1.0);
    println!("  Accuracy: {:.1}%", acc_w * 100.0);
    println!("  Minority recall: {:.1}%\n", recall_1_w * 100.0);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("=== Summary ===");
    println!(
        "Minority recall improvement: {:.1}%",
        (recall_1_w - recall_1_uw) * 100.0
    );

    if recall_1_w > recall_1_uw {
        println!("✓ Weighted training improved minority class recall!");
    }
}

/// Compute classification accuracy.
fn compute_accuracy(probs: &[f32], labels: &[f32]) -> f32 {
    let correct: usize = probs
        .iter()
        .zip(labels.iter())
        .filter(|(prob, label)| {
            let pred = if **prob >= 0.5 { 1.0 } else { 0.0 };
            (pred - **label).abs() < 0.5
        })
        .count();
    correct as f32 / labels.len() as f32
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
    if total > 0 {
        tp as f32 / total as f32
    } else {
        0.0
    }
}
