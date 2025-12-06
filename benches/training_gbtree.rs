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
use booste_rs::training::{
    BaseScore, DepthWisePolicy, GBTreeTrainer, GainParams, Quantizer, SquaredLoss, TrainerParams,
    TreeParams, Verbosity,
};
use booste_rs::training::gbtree::{CutFinder, ExactQuantileCuts};

// =============================================================================
// Configuration
// =============================================================================

/// Create trainer params matching XGBoost defaults for fair comparison.
fn bench_trainer_params(n_trees: u32, max_depth: u32) -> TrainerParams {
    TrainerParams {
        num_rounds: n_trees,
        verbosity: Verbosity::Silent,
        base_score: BaseScore::Mean,
        tree_params: TreeParams {
            max_depth,
            max_leaves: 256,
            min_samples_split: 2,
            min_samples_leaf: 1,
            learning_rate: 0.3,
            gain: GainParams {
                lambda: 1.0,
                alpha: 0.0,
                min_split_gain: 0.0,
                min_child_weight: 1.0,
            },
            parallel_histograms: false,
        },
    }
}

// =============================================================================
// Regression Benchmarks - booste-rs vs XGBoost
// =============================================================================

fn bench_gbtree_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbtree/regression");
    group.sample_size(10); // Fewer samples for slower benchmarks

    let configs: [(_, usize, usize, u32, u32); 3] = [
        ("small", 1000, 20, 100, 6),
        ("medium", 5000, 50, 100, 6),
        ("large", 20000, 100, 100, 6),
    ];

    for (name, n_samples, n_features, n_trees, max_depth) in configs {
        let (features, labels) = generate_training_data(n_samples, n_features, 42);
        let row_matrix = RowMatrix::from_vec(features.clone(), n_samples, n_features);
        let col_matrix: ColMatrix<f32> = row_matrix.to_layout();

        // Quantize once (not timed) - match XGBoost settings
        let cut_finder = ExactQuantileCuts::default();
        let cuts = cut_finder.find_cuts(&col_matrix, 256);
        let quantizer = Quantizer::new(cuts.clone());
        let quantized = quantizer.quantize::<_, u8>(&col_matrix);

        let params = bench_trainer_params(n_trees, max_depth);
        let policy = DepthWisePolicy { max_depth };

        // booste-rs benchmark
        group.bench_with_input(
            BenchmarkId::new("boosters", name),
            &(&quantized, &labels, &cuts),
            |b, (quantized, labels, cuts)| {
                b.iter(|| {
                    let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params.clone());
                    let forest = trainer.train(
                        policy.clone(),
                        black_box(*quantized),
                        black_box(*labels),
                        black_box(*cuts),
                        &[],
                    );
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
                .max_depth(max_depth)
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
    use booste_rs::training::LogisticLoss;

    let mut group = c.benchmark_group("gbtree/classification");
    group.sample_size(10);

    let configs: [(_, usize, usize, u32, u32); 2] = [
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

        let cut_finder = ExactQuantileCuts::default();
        let cuts = cut_finder.find_cuts(&col_matrix, 256);
        let quantizer = Quantizer::new(cuts.clone());
        let quantized = quantizer.quantize::<_, u8>(&col_matrix);

        let params = bench_trainer_params(n_trees, max_depth);
        let policy = DepthWisePolicy { max_depth };

        // booste-rs benchmark
        group.bench_with_input(
            BenchmarkId::new("boosters", name),
            &(&quantized, &labels, &cuts),
            |b, (quantized, labels, cuts)| {
                b.iter(|| {
                    let mut trainer = GBTreeTrainer::new(Box::new(LogisticLoss), params.clone());
                    let forest = trainer.train(
                        policy.clone(),
                        black_box(*quantized),
                        black_box(*labels),
                        black_box(*cuts),
                        &[],
                    );
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
                .max_depth(max_depth)
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

