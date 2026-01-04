//! Training throughput benchmarks using the high-level model API.

#[path = "common/criterion_config.rs"]
mod criterion_config;

use boosters::model::GBDTModel;
use boosters::model::gbdt::GBDTConfig;
use boosters::testing::synthetic_datasets::synthetic_regression;
use boosters::training::{Metric, Objective};

use criterion_config::default_criterion;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

fn bench_train_gbdt(c: &mut Criterion) {
    let mut group = c.benchmark_group("train/gbdt");

    // Keep sizes modest: training includes binning.
    let cases = [
        ("small", 5_000usize, 50usize, 50usize),
        ("medium", 50_000usize, 100usize, 100usize),
    ];

    for (label, n_samples, n_features, n_trees) in cases {
        let dataset = synthetic_regression(n_samples, n_features, 42, 0.05);

        let config = GBDTConfig::builder()
            .objective(Objective::SquaredLoss)
            .metric(Metric::Rmse)
            .n_trees(n_trees as u32)
            .learning_rate(0.1)
            .build()
            .expect("valid config");

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new(label, format!("{n_samples}x{n_features}x{n_trees}")),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let model = GBDTModel::train(black_box(dataset), None, config.clone(), 0)
                        .expect("training should succeed");
                    black_box(model)
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_train_gbdt
}
criterion_main!(benches);
