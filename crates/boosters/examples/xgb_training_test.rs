//! Quick test to verify XGBoost training times
//!
//! Run with: cargo run --example xgb_training_test --features bench-xgboost --release

#[cfg(feature = "bench-xgboost")]
fn main() {
    use std::time::Instant;
    use xgb::parameters::BoosterType;
    use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
    use xgb::parameters::{BoosterParametersBuilder, TrainingParametersBuilder};
    use xgb::{Booster, DMatrix};

    let n_samples = 1000;
    let n_features = 20;
    let n_trees = 100u32;
    let max_depth = 6u32;

    // Generate data
    let features: Vec<f32> = (0..n_samples * n_features)
        .map(|i| (i as f32 * 0.1).sin() * 2.0)
        .collect();
    let labels: Vec<f32> = (0..n_samples)
        .map(|i| i as f32 / n_samples as f32)
        .collect();

    println!("Data: {} samples, {} features", n_samples, n_features);
    println!("Training {} trees with max_depth={}", n_trees, max_depth);
    println!("Using 1 thread (nthread=1)");
    println!();

    let tree_params = TreeBoosterParametersBuilder::default()
        .eta(0.3)
        .max_depth(max_depth)
        .lambda(1.0)
        .alpha(0.0)
        .gamma(0.0)
        .min_child_weight(1.0)
        .tree_method(TreeMethod::Hist)
        .max_bin(256u32)
        .build()
        .unwrap();

    let booster_params = BoosterParametersBuilder::default()
        .booster_type(BoosterType::Tree(tree_params))
        .verbose(false)
        .threads(Some(1))
        .build()
        .unwrap();

    // Test 1: With caching (normal behavior)
    println!("=== Test 1: Normal training (with caching) ===");
    let iterations = 5;
    for i in 0..iterations {
        let iter_start = Instant::now();

        let mut dtrain = DMatrix::from_dense(&features, n_samples).unwrap();
        dtrain.set_labels(&labels).unwrap();

        let training_params = TrainingParametersBuilder::default()
            .dtrain(&dtrain)
            .boost_rounds(n_trees)
            .booster_params(booster_params.clone())
            .evaluation_sets(None)
            .build()
            .unwrap();

        let model = Booster::train(&training_params).unwrap();
        let _preds = model.predict(&dtrain).unwrap();

        println!("  Iteration {}: {:?}", i, iter_start.elapsed());
    }

    // Test 2: With reset() - should disable caching
    println!();
    println!("=== Test 2: Training with reset() between iterations ===");
    for i in 0..iterations {
        let iter_start = Instant::now();

        let mut dtrain = DMatrix::from_dense(&features, n_samples).unwrap();
        dtrain.set_labels(&labels).unwrap();

        let training_params = TrainingParametersBuilder::default()
            .dtrain(&dtrain)
            .boost_rounds(n_trees)
            .booster_params(booster_params.clone())
            .evaluation_sets(None)
            .build()
            .unwrap();

        let mut model = Booster::train(&training_params).unwrap();
        let _preds = model.predict(&dtrain).unwrap();

        // Reset the booster to clear caches
        model.reset().unwrap();

        println!("  Iteration {}: {:?}", i, iter_start.elapsed());
    }

    // Test 3: Manual training loop with fresh booster each time
    println!();
    println!("=== Test 3: Fresh Booster with cached dmats ===");
    for i in 0..iterations {
        let iter_start = Instant::now();

        let mut dtrain = DMatrix::from_dense(&features, n_samples).unwrap();
        dtrain.set_labels(&labels).unwrap();

        // Create booster WITH cached dmat so it knows num_features
        let mut bst = Booster::new_with_cached_dmats(&booster_params, &[&dtrain]).unwrap();

        for round in 0..n_trees as i32 {
            bst.update(&dtrain, round).unwrap();
        }

        let _preds = bst.predict(&dtrain).unwrap();

        println!("  Iteration {}: {:?}", i, iter_start.elapsed());
    }
}

#[cfg(not(feature = "bench-xgboost"))]
fn main() {
    println!("Run with --features bench-xgboost");
}
