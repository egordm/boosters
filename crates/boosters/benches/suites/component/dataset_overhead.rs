//! Dataset Consolidation Overhead Benchmarks
//!
//! Measures overhead of using BinnedDataset instead of types/Dataset for:
//! 1. GBLinear training (critical: linear models don't need bins)
//! 2. GBDT training (baseline: trees need bins anyway)
//! 3. Prediction (important: inference uses raw features, not bins)
//! 4. Dataset construction (measures binning cost)
//!
//! See docs/backlogs/dataset-consolidation.md for context.
//!
//! Run with: `cargo bench --features bench-training --bench dataset_overhead`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::Parallelism;
use boosters::data::binned::BinnedDatasetBuilder;
use boosters::data::{BinningConfig, Dataset, TargetsView, WeightsView};
use boosters::inference::UnrolledPredictor6;
use boosters::testing::synthetic_datasets::synthetic_regression;
use boosters::training::{
    GBDTParams, GBDTTrainer, GBLinearParams, GBLinearTrainer, GainParams, GrowthStrategy, Rmse,
    SquaredLoss, Verbosity,
};

use ndarray::Array2;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

// =============================================================================
// Dataset Construction Overhead
// =============================================================================

/// Benchmark: Compare construction time of Dataset vs BinnedDataset
///
/// This measures the binning overhead - how much extra time does BinnedDataset
/// construction take compared to raw Dataset construction?
fn bench_dataset_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/overhead/construction");

    for n_rows in [1_000usize, 10_000, 50_000] {
        let n_features = 100usize;
        let seed = 42u64;

        // Generate synthetic data (feature-major layout)
        let synth = synthetic_regression(n_rows, n_features, seed, 0.05);

        group.throughput(Throughput::Elements((n_rows * n_features) as u64));

        // Benchmark: types/Dataset construction (raw features only)
        let features_view = synth.features.view();
        group.bench_with_input(
            BenchmarkId::new("Dataset", n_rows),
            &features_view,
            |b, features| {
                b.iter(|| {
                    black_box(Dataset::new(black_box(*features), None, None))
                })
            },
        );

        // Benchmark: BinnedDataset construction (includes binning)
        let dataset = Dataset::new(synth.features.view(), None, None);
        group.bench_with_input(
            BenchmarkId::new("BinnedDataset", n_rows),
            &dataset,
            |b, ds| {
                b.iter(|| {
                    black_box(
                        BinnedDatasetBuilder::with_config(
                            BinningConfig::builder().max_bins(256).build(),
                        )
                        .add_features(black_box(ds).features(), Parallelism::Parallel)
                        .build()
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// GBLinear Training Overhead (CRITICAL)
// =============================================================================

/// Benchmark: GBLinear training with BinnedDataset
///
/// After migration, GBLinear uses BinnedDataset directly via for_each_feature_value().
/// This benchmark measures actual training performance with the new unified path.
/// Target: <2x on small datasets compared to raw feature access.
fn bench_gblinear_training_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/overhead/gblinear_train");
    group.sample_size(10);

    // Test small and medium datasets - overhead matters more on small ones
    for n_rows in [1_000usize, 10_000, 50_000] {
        let n_features = 100usize;
        let seed = 42u64;

        let synth = synthetic_regression(n_rows, n_features, seed, 0.05);
        let targets: Vec<f32> = synth.targets.to_vec();
        let targets_2d =
            Array2::from_shape_vec((1, targets.len()), targets.clone()).unwrap();

        // Create Dataset (raw features)
        let dataset = Dataset::new(
            synth.features.view(),
            Some(targets_2d.view()),
            None,
        );

        // Create BinnedDataset (bundling disabled for GBLinear - simpler features)
        let binned = BinnedDatasetBuilder::with_config(
            BinningConfig::builder()
                .max_bins(256)
                .enable_bundling(false) // GBLinear doesn't use feature bundling
                .build(),
        )
        .add_features(dataset.features(), Parallelism::Parallel)
        .build()
        .unwrap();

        let params = GBLinearParams {
            n_rounds: 10,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: true,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);

        group.throughput(Throughput::Elements((n_rows * n_features) as u64));

        // Benchmark GBLinear training with BinnedDataset (current unified path)
        let targets_view = TargetsView::new(targets_2d.view());
        group.bench_with_input(
            BenchmarkId::new("BinnedDataset", n_rows),
            &(&binned, targets_view.clone()),
            |b, (binned_ds, targets)| {
                b.iter(|| {
                    black_box(
                        trainer
                            .train(black_box(*binned_ds), targets.clone(), WeightsView::None, &[])
                            .unwrap(),
                    )
                })
            },
        );

        // Benchmark raw access pattern overhead (BinnedDataset)
        group.bench_with_input(
            BenchmarkId::new("BinnedDataset_raw_access", n_rows),
            &binned,
            |b, binned_ds| {
                // Simulate GBLinear's access pattern: iterate features via for_each_feature_value
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for feature_idx in 0..binned_ds.n_features() {
                        binned_ds.for_each_feature_value(feature_idx, |_row, value| {
                            sum += value;
                        });
                    }
                    black_box(sum)
                })
            },
        );

        // Benchmark Dataset raw access for comparison (baseline)
        let features_view = dataset.features();
        group.bench_with_input(
            BenchmarkId::new("Dataset_raw_access", n_rows),
            &features_view,
            |b, fv| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for feature_idx in 0..fv.n_features() {
                        let slice = fv.feature(feature_idx);
                        sum += slice.iter().sum::<f32>();
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// GBDT Training Overhead (BASELINE)
// =============================================================================

/// Benchmark: GBDT training overhead
///
/// GBDT always uses BinnedDataset (needs bins for histograms), so this is our
/// baseline. We measure to ensure there's no regression from consolidation.
fn bench_gbdt_training_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/overhead/gbdt_train");
    group.sample_size(10);

    let n_rows = 10_000usize;
    let n_features = 50usize;
    let seed = 42u64;

    let synth = synthetic_regression(n_rows, n_features, seed, 0.05);
    let targets: Vec<f32> = synth.targets.to_vec();
    let targets_2d = Array2::from_shape_vec((1, targets.len()), targets).unwrap();

    let dataset = Dataset::new(synth.features.view(), None, None);
    let binned = BinnedDatasetBuilder::with_config(
        BinningConfig::builder().max_bins(256).build(),
    )
    .add_features(dataset.features(), Parallelism::Parallel)
    .build()
    .unwrap();

    let params = GBDTParams {
        n_trees: 10,
        learning_rate: 0.3,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 6 },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        cache_size: 256,
        ..Default::default()
    };
    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

    group.throughput(Throughput::Elements((n_rows * n_features) as u64));

    // GBDT with BinnedDataset (only valid path)
    let targets_view = TargetsView::new(targets_2d.view());
    group.bench_function("BinnedDataset", |b| {
        b.iter(|| {
            black_box(
                trainer
                    .train(
                        black_box(&binned),
                        targets_view,
                        WeightsView::None,
                        &[],
                        Parallelism::Sequential,
                    )
                    .unwrap(),
            )
        })
    });

    group.finish();
}

// =============================================================================
// Prediction Overhead
// =============================================================================

/// Benchmark: Prediction overhead with Dataset vs BinnedDataset
///
/// Prediction uses raw features, not bins. This measures the overhead of
/// accessing raw features from BinnedDataset vs Dataset.
fn bench_prediction_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("component/overhead/predict");
    group.sample_size(10);

    let n_rows = 10_000usize;
    let n_features = 100usize;
    let n_trees = 50u32;
    let seed = 42u64;

    // Generate data and train a model
    let synth = synthetic_regression(n_rows, n_features, seed, 0.05);
    let targets: Vec<f32> = synth.targets.to_vec();
    let targets_2d = Array2::from_shape_vec((1, targets.len()), targets).unwrap();

    let dataset = Dataset::new(synth.features.view(), None, None);
    let binned = BinnedDatasetBuilder::with_config(
        BinningConfig::builder().max_bins(256).build(),
    )
    .add_features(dataset.features(), Parallelism::Parallel)
    .build()
    .unwrap();

    // Train a GBDT model for prediction
    let params = GBDTParams {
        n_trees,
        learning_rate: 0.3,
        growth_strategy: GrowthStrategy::DepthWise { max_depth: 6 },
        gain: GainParams {
            reg_lambda: 1.0,
            ..Default::default()
        },
        cache_size: 256,
        ..Default::default()
    };
    let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
    let targets_view = TargetsView::new(targets_2d.view());
    let forest = trainer
        .train(&binned, targets_view, WeightsView::None, &[], Parallelism::Sequential)
        .unwrap();

    let predictor = UnrolledPredictor6::new(&forest);

    group.throughput(Throughput::Elements((n_rows * n_trees as usize) as u64));

    // Prediction with Dataset/FeaturesView (current path)
    let features_view = dataset.features();
    group.bench_with_input(
        BenchmarkId::new("Dataset", n_rows),
        &features_view,
        |b, fv| {
            b.iter(|| {
                let output = predictor.predict(black_box(*fv), Parallelism::Sequential);
                black_box(output)
            })
        },
    );

    // Raw feature access overhead: Dataset vs BinnedDataset
    // This measures the overhead of accessing raw features from BinnedDataset
    // compared to Dataset. The actual prediction uses the same predictor.
    group.bench_with_input(
        BenchmarkId::new("BinnedDataset_raw_access", n_rows),
        &binned,
        |b, binned_ds| {
            // Simulate prediction access pattern by iterating raw features
            b.iter(|| {
                let mut sum = 0.0f32;
                for sample in 0..binned_ds.n_samples() {
                    for feature in 0..binned_ds.n_features().min(10) {
                        if let Some(slice) = binned_ds.raw_feature_slice(feature) {
                            sum += slice[sample];
                        }
                    }
                }
                black_box(sum)
            })
        },
    );

    // Dataset raw feature access for comparison
    group.bench_with_input(
        BenchmarkId::new("Dataset_raw_access", n_rows),
        &features_view,
        |b, fv| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for sample in 0..fv.n_samples() {
                    for feature in 0..fv.n_features().min(10) {
                        sum += fv.get(sample, feature);
                    }
                }
                black_box(sum)
            })
        },
    );

    // End-to-end prediction using BinnedDataset + SampleBlocks
    // This is the recommended path for BinnedDataset prediction per Story 2.2a
    let block_size = predictor.block_size();
    group.bench_with_input(
        BenchmarkId::new("BinnedDataset_SampleBlocks", n_rows),
        &(&binned, &predictor, block_size),
        |b, (binned_ds, pred, blk_size)| {
            b.iter(|| {
                let n_groups = pred.n_groups();
                let n_samples = binned_ds.n_samples();
                let mut output = Array2::<f32>::zeros((n_groups, n_samples));

                // Use SampleBlocks iterator to get contiguous sample-major blocks
                // Then transpose to feature-major for predict_into
                for (block_idx, block) in binned_ds.sample_blocks(*blk_size).iter().enumerate() {
                    let start_idx = block_idx * *blk_size;
                    let block_len = block.nrows();

                    // Transpose to feature-major: [samples, features] -> [features, samples]
                    let block_transposed = block.t().to_owned();
                    let block_features =
                        boosters::data::FeaturesView::from_array(block_transposed.view());

                    // Predict this block
                    let mut block_output = Array2::<f32>::zeros((n_groups, block_len));
                    pred.predict_into(
                        block_features,
                        None,
                        Parallelism::Sequential,
                        block_output.view_mut(),
                    );

                    // Copy to output
                    output
                        .slice_mut(ndarray::s![.., start_idx..start_idx + block_len])
                        .assign(&block_output);
                }

                black_box(output)
            })
        },
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_dataset_construction,
        bench_gblinear_training_overhead,
        bench_gbdt_training_overhead,
        bench_prediction_overhead
}
criterion_main!(benches);
