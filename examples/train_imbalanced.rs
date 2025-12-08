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

use booste_rs::data::{ColMatrix, RowMatrix};
use booste_rs::predict::{Predictor, StandardTraversal};
use booste_rs::training::{GBTreeTrainer, LossFunction, Verbosity};

fn main() {
    // =========================================================================
    // Generate Imbalanced Dataset
    // =========================================================================

    // Create a dataset with 10:1 class imbalance
    // The classes overlap significantly to make classification challenging
    // Class 0 (majority): 900 samples centered at (3, 3)
    // Class 1 (minority): 100 samples centered at (5, 5)
    let n_majority = 900;
    let n_minority = 100;
    let n_features = 2;
    let n_samples = n_majority + n_minority;

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    // Majority class (0) - centered around (3, 3) with high variance
    for i in 0..n_majority {
        let noise1 = ((i * 17) % 200) as f32 / 40.0 - 2.5; // [-2.5, 2.5]
        let noise2 = ((i * 31) % 200) as f32 / 40.0 - 2.5;
        features.push(3.0 + noise1);
        features.push(3.0 + noise2);
        labels.push(0.0);
    }

    // Minority class (1) - centered around (5, 5) with overlap
    for i in 0..n_minority {
        let noise1 = ((i * 23) % 200) as f32 / 40.0 - 2.5;
        let noise2 = ((i * 37) % 200) as f32 / 40.0 - 2.5;
        features.push(5.0 + noise1);
        features.push(5.0 + noise2);
        labels.push(1.0);
    }

    // Create matrices
    let row_matrix = RowMatrix::from_vec(features, n_samples, n_features);
    let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

    // =========================================================================
    // Compute Class Weights
    // =========================================================================

    // Option 1: Inverse frequency weighting
    // Weight = total_samples / (num_classes * class_count)
    let weight_class_0 = n_samples as f32 / (2.0 * n_majority as f32);
    let weight_class_1 = n_samples as f32 / (2.0 * n_minority as f32);

    let weights: Vec<f32> = labels
        .iter()
        .map(|&label| {
            if label < 0.5 {
                weight_class_0
            } else {
                weight_class_1
            }
        })
        .collect();

    println!("=== Imbalanced Classification Example ===\n");
    println!("Dataset:");
    println!("  Majority class (0): {} samples", n_majority);
    println!("  Minority class (1): {} samples", n_minority);
    println!("  Imbalance ratio: {:.0}:1\n", n_majority as f32 / n_minority as f32);
    println!("Weights:");
    println!("  Majority class weight: {:.2}", weight_class_0);
    println!("  Minority class weight: {:.2}\n", weight_class_1);

    // =========================================================================
    // Train Without Weights (Baseline)
    // =========================================================================

    let trainer = GBTreeTrainer::builder()
        .loss(LossFunction::Logistic)
        .num_rounds(30u32)
        .max_depth(4u8)
        .learning_rate(0.1f32)
        .verbosity(Verbosity::Silent)
        .build()
        .unwrap();

    println!("--- Training WITHOUT weights ---");
    let forest_unweighted = trainer.train(&col_matrix, &labels, None, &[]);

    let predictor_unweighted = Predictor::<StandardTraversal>::new(&forest_unweighted);
    let preds_unweighted = predictor_unweighted.predict(&row_matrix).into_vec();

    let (acc_uw, recall_0_uw, recall_1_uw) = compute_metrics(&preds_unweighted, &labels);
    println!("  Accuracy:       {:.1}%", acc_uw * 100.0);
    println!("  Recall class 0: {:.1}%", recall_0_uw * 100.0);
    println!("  Recall class 1: {:.1}% (minority)\n", recall_1_uw * 100.0);

    // =========================================================================
    // Train With Weights (Class-Balanced)
    // =========================================================================

    println!("--- Training WITH class weights ---");
    let forest_weighted = trainer.train(&col_matrix, &labels, Some(&weights), &[]);

    let predictor_weighted = Predictor::<StandardTraversal>::new(&forest_weighted);
    let preds_weighted = predictor_weighted.predict(&row_matrix).into_vec();

    let (acc_w, recall_0_w, recall_1_w) = compute_metrics(&preds_weighted, &labels);
    println!("  Accuracy:       {:.1}%", acc_w * 100.0);
    println!("  Recall class 0: {:.1}%", recall_0_w * 100.0);
    println!("  Recall class 1: {:.1}% (minority)\n", recall_1_w * 100.0);

    // =========================================================================
    // Compare Results
    // =========================================================================

    println!("=== Summary ===\n");
    println!("Minority class recall improvement: {:.1}%",
        (recall_1_w - recall_1_uw) * 100.0);

    if recall_1_w > recall_1_uw {
        println!("\n✓ Weighted training improved minority class recall!");
    } else if (recall_1_w - recall_1_uw).abs() < 0.01 {
        println!("\n≈ Both models performed similarly on minority class.");
    } else {
        println!("\n✗ Unweighted model had better minority recall (unusual).");
    }
}

/// Compute accuracy and per-class recall.
fn compute_metrics(predictions: &[f32], labels: &[f32]) -> (f32, f32, f32) {
    let mut tp0 = 0;
    let mut fn0 = 0; // Predicted 1, actual 0
    let mut tp1 = 0;
    let mut fn1 = 0; // Predicted 0, actual 1
    let mut correct = 0;

    for (pred, &label) in predictions.iter().zip(labels.iter()) {
        let prob = 1.0 / (1.0 + (-pred).exp()); // sigmoid
        let predicted_class = if prob >= 0.5 { 1.0 } else { 0.0 };

        if (predicted_class - label).abs() < 0.5 {
            correct += 1;
            if label < 0.5 {
                tp0 += 1;
            } else {
                tp1 += 1;
            }
        } else if label < 0.5 {
            fn0 += 1;
        } else {
            fn1 += 1;
        }
    }

    let accuracy = correct as f32 / predictions.len() as f32;
    let recall_0 = if tp0 + fn0 > 0 {
        tp0 as f32 / (tp0 + fn0) as f32
    } else {
        0.0
    };
    let recall_1 = if tp1 + fn1 > 0 {
        tp1 as f32 / (tp1 + fn1) as f32
    } else {
        0.0
    };

    (accuracy, recall_0, recall_1)
}
