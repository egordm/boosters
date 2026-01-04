//! Prediction throughput benchmarks using the high-level model API.

#[path = "common/mod.rs"]
mod common;

use boosters::data::Dataset;
use boosters::model::{GBDTModel, ModelMeta, OutputTransform};
use boosters::testing::synthetic_datasets::random_features_array;

use common::criterion_config::default_criterion;
use common::models::load_boosters_model;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

fn bench_predict_gbdt(c: &mut Criterion) {
    let models = [
        ("small", "bench_small"),
        ("medium", "bench_medium"),
        ("large", "bench_large"),
    ];

    let mut group = c.benchmark_group("predict/gbdt");

    for (label, model_name) in models {
        let loaded = load_boosters_model(model_name);
        let n_features = loaded.n_features;
        let meta = ModelMeta::for_regression(n_features);
        let model = GBDTModel::from_parts(loaded.forest, meta, OutputTransform::Identity);

        for &n_threads in &[1usize, 0] {
            for &batch_size in &[1usize, 128, 1_024, 16_384] {
                let matrix = random_features_array(batch_size, n_features, 42, -5.0, 5.0);
                let dataset = Dataset::from_array(matrix.t(), None, None);

                group.throughput(Throughput::Elements(batch_size as u64));
                group.bench_with_input(
                    BenchmarkId::new(format!("{label}/threads_{n_threads}"), batch_size),
                    &dataset,
                    |b, dataset| {
                        b.iter(|| {
                            let output = model.predict(black_box(dataset), n_threads);
                            black_box(output)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_predict_gbdt
}
criterion_main!(benches);
