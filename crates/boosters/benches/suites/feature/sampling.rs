//! Sampling strategies benchmark: GOSS vs Uniform vs No sampling.
//!
//! Compares training speed and quality impact of different sampling strategies.
//! All configurations use depth-wise expansion.
//!
//! Run with: `cargo bench --bench sampling`

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::dataset::{TargetsView, WeightsView};
use boosters::testing::data::synthetic_regression;
use boosters::training::{
    GBDTParams, GBDTTrainer, GainParams, GrowthStrategy, Rmse, RowSamplingParams, SquaredLoss,
};
use boosters::Parallelism;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// =============================================================================
// Dataset Configuration
// =============================================================================

const DATASET: (usize, usize) = (100_000, 100);
const N_TREES: u32 = 100;
const MAX_DEPTH: u32 = 6;

// =============================================================================
// Sampling Strategies Benchmark
// =============================================================================

fn bench_sampling_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature/sampling");
    group.sample_size(10);

    let (rows, cols) = DATASET;
    let dataset = synthetic_regression(rows, cols, 42, 0.05);
    let binned = dataset.to_binned(256);
    let targets = TargetsView::new(dataset.targets.view().insert_axis(ndarray::Axis(0)));

    group.throughput(Throughput::Elements((rows * cols) as u64));

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
                        .train(black_box(&binned), black_box(targets), WeightsView::None, &[], Parallelism::Sequential)
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
