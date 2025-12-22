//! Sampling strategies benchmark: GOSS vs Uniform vs No sampling.
//!
//! Compares training speed and quality impact of different sampling strategies.
//! All configurations use depth-wise expansion.
//!
//! Run with: `cargo bench --bench sampling`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::data::{binned::BinnedDatasetBuilder, ColMatrix, DenseMatrix, RowMajor};
use boosters::testing::data::{random_dense_f32, synthetic_regression_targets_linear};
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, RowSamplingParams, SquaredLoss,
};
use boosters::Parallelism;

use ndarray::ArrayView1;

fn empty_weights() -> ArrayView1<'static, f32> {
    ArrayView1::from(&[][..])
}

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// =============================================================================
// Dataset Configuration
// =============================================================================

const DATASET: (usize, usize) = (100_000, 100);
const N_TREES: u32 = 100;
const MAX_DEPTH: u32 = 6;

// =============================================================================
// Helpers
// =============================================================================

fn build_col_matrix(features_row_major: Vec<f32>, rows: usize, cols: usize) -> ColMatrix<f32> {
    let row: DenseMatrix<f32, RowMajor> = DenseMatrix::from_vec(features_row_major, rows, cols);
    row.to_layout()
}

// =============================================================================
// Sampling Strategies Benchmark
// =============================================================================

fn bench_sampling_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature/sampling");
    group.sample_size(10);

    let (rows, cols) = DATASET;
    let features = random_dense_f32(rows, cols, 42, -1.0, 1.0);
    let (targets, _w, _b) = synthetic_regression_targets_linear(&features, rows, cols, 1337, 0.05);

    group.throughput(Throughput::Elements((rows * cols) as u64));

    // Pre-build binned dataset
    let col_matrix = build_col_matrix(features.clone(), rows, cols);
    let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256)
        .build()
        .unwrap();

    // Sampling configurations to test
    let configs: Vec<(&str, RowSamplingParams)> = vec![
        ("no_sampling", RowSamplingParams::None),
        ("uniform_0.5", RowSamplingParams::uniform(0.5)),
        ("uniform_0.3", RowSamplingParams::uniform(0.3)),
        // GOSS with LightGBM defaults: top_rate=0.2, other_rate=0.1
        ("goss_default", RowSamplingParams::goss(0.2, 0.1)),
        // More aggressive GOSS
        ("goss_aggressive", RowSamplingParams::goss(0.1, 0.05)),
    ];

    for (name, row_sampling) in configs {
        let params = GBDTParams {
            n_trees: N_TREES,
            learning_rate: 0.1,
            growth_strategy: GrowthStrategy::DepthWise {
                max_depth: MAX_DEPTH,
            },
            gain: GainParams {
                reg_lambda: 1.0,
                ..Default::default()
            },
            row_sampling,
            cache_size: 32,
            ..Default::default()
        };
        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);

        group.bench_function(BenchmarkId::new("boosters", name), |b| {
            b.iter(|| {
                black_box(
                    trainer
                        .train(black_box(&binned), ArrayView1::from(black_box(&targets[..])), empty_weights(), &[], Parallelism::Sequential)
                        .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_sampling_strategies
}
criterion_main!(benches);
