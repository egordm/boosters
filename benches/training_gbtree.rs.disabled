//! GBTree training benchmark.
//!
//! Times GBTree training at various scales to compare with XGBoost.
//!
//! This benchmark compares booste-rs histogram-based GBTree training against
//! XGBoost's C++ implementation using rust bindings. Both implementations:
//! - Use histogram-based tree building (max_bin=256)
//! - Single-threaded execution (nthread=1)
//! - Same hyperparameters (eta=0.3, max_depth=6, lambda=1.0)
//!
//! Note: We use a local fork of rust-xgboost with reset() method to ensure
//! fair benchmarking by clearing internal caches between iterations.
//!
//! Run with: cargo bench --bench training_gbtree --features bench-xgboost

mod bench_utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use bench_utils::generate_training_data;
use booste_rs::data::{ColMatrix, RowMatrix};
use booste_rs::training::{GBTreeTrainer, GrowthStrategy, LossFunction, Verbosity};

// =============================================================================
// Regression Benchmarks - booste-rs vs XGBoost
// =============================================================================

fn bench_gbtree_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbtree/regression");
    group.sample_size(10); // Fewer samples for slower benchmarks

    let configs: [(_, usize, usize, u32, u8); 3] = [
        ("small", 1000, 20, 100, 6),
        ("medium", 5000, 50, 100, 6),
        ("large", 20000, 100, 100, 6),
    ];

    for (name, n_samples, n_features, n_trees, max_depth) in configs {
        let (features, labels) = generate_training_data(n_samples, n_features, 42);
        let row_matrix = RowMatrix::from_vec(features.clone(), n_samples, n_features);
        let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

        // Build trainer with benchmark params
        let trainer = GBTreeTrainer::builder()
            .loss(LossFunction::SquaredError)
            .num_rounds(n_trees)
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: max_depth as u32 })
            .learning_rate(0.3f32)
            .reg_lambda(1.0f32)
            .reg_alpha(0.0f32)
            .min_split_gain(0.0f32)
            .min_child_weight(1.0f32)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        // booste-rs benchmark
        group.bench_with_input(
            BenchmarkId::new("boosters", name),
            &(&col_matrix, &labels),
            |b, (col_matrix, labels)| {
                b.iter(|| {
                    let forest = trainer.train(black_box(*col_matrix), black_box(*labels), None, &[]);
                    black_box(forest)
                });
            },
        );

        // XGBoost benchmark (when feature enabled)
        #[cfg(feature = "bench-xgboost")]
        {
            use xgb::parameters::{BoosterParametersBuilder, TrainingParametersBuilder};
            use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
            use xgb::parameters::BoosterType;
            use xgb::{Booster, DMatrix};

            // Clone data for use inside benchmark closure
            let features_for_xgb = features.clone();
            let labels_for_xgb = labels.clone();

            let tree_params = TreeBoosterParametersBuilder::default()
                .eta(0.3)
                .max_depth(max_depth as u32)
                .lambda(1.0)
                .alpha(0.0)
                .gamma(0.0)
                .min_child_weight(1.0)
                .tree_method(TreeMethod::Hist)
                .max_bin(256)
                .build()
                .unwrap();

            let booster_params = BoosterParametersBuilder::default()
                .booster_type(BoosterType::Tree(tree_params))
                .verbose(false)
                .threads(Some(1)) // Single-threaded for fair comparison
                .build()
                .unwrap();

            group.bench_function(BenchmarkId::new("xgboost", name), |b| {
                b.iter(|| {
                    // Create fresh DMatrix each iteration to avoid caching
                    let mut dtrain = DMatrix::from_dense(
                        black_box(&features_for_xgb),
                        n_samples,
                    )
                    .expect("Failed to create DMatrix");
                    dtrain
                        .set_labels(black_box(&labels_for_xgb))
                        .expect("Failed to set labels");

                    let training_params = TrainingParametersBuilder::default()
                        .dtrain(&dtrain)
                        .boost_rounds(n_trees)
                        .booster_params(booster_params.clone())
                        .evaluation_sets(None) // No early stopping
                        .build()
                        .unwrap();

                    let model = Booster::train(&training_params).expect("XGBoost training failed");
                    black_box(model)
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Classification Benchmarks - booste-rs vs XGBoost
// =============================================================================

fn bench_gbtree_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbtree/classification");
    group.sample_size(10);

    let configs: [(_, usize, usize, u32, u8); 2] = [
        ("small", 1000, 20, 100, 6),
        ("medium", 5000, 50, 100, 6),
    ];

    for (name, n_samples, n_features, n_trees, max_depth) in configs {
        let (features, raw_labels) = generate_training_data(n_samples, n_features, 42);
        // Convert to binary labels
        let labels: Vec<f32> = raw_labels
            .iter()
            .map(|&y| if y > 0.0 { 1.0 } else { 0.0 })
            .collect();

        let row_matrix = RowMatrix::from_vec(features.clone(), n_samples, n_features);
        let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

        // Build trainer with benchmark params
        let trainer = GBTreeTrainer::builder()
            .loss(LossFunction::Logistic)
            .num_rounds(n_trees)
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: max_depth as u32 })
            .learning_rate(0.3f32)
            .reg_lambda(1.0f32)
            .reg_alpha(0.0f32)
            .min_split_gain(0.0f32)
            .min_child_weight(1.0f32)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        // booste-rs benchmark
        group.bench_with_input(
            BenchmarkId::new("boosters", name),
            &(&col_matrix, &labels),
            |b, (col_matrix, labels)| {
                b.iter(|| {
                    let forest = trainer.train(black_box(*col_matrix), black_box(*labels), None, &[]);
                    black_box(forest)
                });
            },
        );

        // XGBoost benchmark (when feature enabled)
        #[cfg(feature = "bench-xgboost")]
        {
            use xgb::parameters::{
                learning::{LearningTaskParametersBuilder, Objective},
                BoosterParametersBuilder, TrainingParametersBuilder,
            };
            use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
            use xgb::parameters::BoosterType;
            use xgb::{Booster, DMatrix};

            // Clone data for use inside benchmark closure
            let features_for_xgb = features.clone();
            let labels_for_xgb = labels.clone();

            let tree_params = TreeBoosterParametersBuilder::default()
                .eta(0.3)
                .max_depth(max_depth as u32)
                .lambda(1.0)
                .alpha(0.0)
                .gamma(0.0)
                .min_child_weight(1.0)
                .tree_method(TreeMethod::Hist)
                .max_bin(256)
                .build()
                .unwrap();

            let learning_params = LearningTaskParametersBuilder::default()
                .objective(Objective::BinaryLogistic)
                .build()
                .unwrap();

            let booster_params = BoosterParametersBuilder::default()
                .booster_type(BoosterType::Tree(tree_params))
                .learning_params(learning_params)
                .verbose(false)
                .threads(Some(1)) // Single-threaded for fair comparison
                .build()
                .unwrap();

            group.bench_function(BenchmarkId::new("xgboost", name), |b| {
                b.iter(|| {
                    // Create fresh DMatrix each iteration to avoid caching
                    let mut dtrain = DMatrix::from_dense(
                        black_box(&features_for_xgb),
                        n_samples,
                    )
                    .expect("Failed to create DMatrix");
                    dtrain
                        .set_labels(black_box(&labels_for_xgb))
                        .expect("Failed to set labels");

                    let training_params = TrainingParametersBuilder::default()
                        .dtrain(&dtrain)
                        .boost_rounds(n_trees)
                        .booster_params(booster_params.clone())
                        .evaluation_sets(None) // No early stopping
                        .build()
                        .unwrap();

                    let model = Booster::train(&training_params).expect("XGBoost training failed");
                    black_box(model)
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(benches, bench_gbtree_regression, bench_gbtree_classification,);

criterion_main!(benches);

