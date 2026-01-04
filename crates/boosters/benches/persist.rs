//! Persistence benchmarks (binary + JSON) using the streaming `SerializableModel` API.

#[path = "common/mod.rs"]
mod common;

use boosters::model::{GBDTModel, ModelMeta, OutputTransform};
use boosters::persist::{
    BinaryReadOptions, BinaryWriteOptions, JsonWriteOptions, SerializableModel,
};

use common::criterion_config::default_criterion;
use common::models::load_boosters_model;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_persist_gbdt(c: &mut Criterion) {
    let loaded = load_boosters_model("bench_medium");
    let meta = ModelMeta::for_regression(loaded.n_features);
    let model = GBDTModel::from_parts(loaded.forest, meta, OutputTransform::Identity);

    c.bench_function("persist/gbdt/binary_write", |b| {
        let options = BinaryWriteOptions::default();
        b.iter(|| {
            let mut buf = Vec::new();
            model
                .write_into(&mut buf, &options)
                .expect("binary write should succeed");
            black_box(buf)
        });
    });

    let binary = {
        let mut buf = Vec::new();
        model
            .write_into(&mut buf, &BinaryWriteOptions::default())
            .expect("binary write should succeed");
        buf
    };

    c.bench_function("persist/gbdt/binary_read", |b| {
        let options = BinaryReadOptions::default();
        b.iter(|| {
            let loaded = GBDTModel::read_from(black_box(binary.as_slice()), &options)
                .expect("binary read should succeed");
            black_box(loaded)
        });
    });

    c.bench_function("persist/gbdt/json_write", |b| {
        let options = JsonWriteOptions::compact();
        b.iter(|| {
            let mut buf = Vec::new();
            model
                .write_json_into(&mut buf, &options)
                .expect("json write should succeed");
            black_box(buf)
        });
    });

    let json = {
        let mut buf = Vec::new();
        model
            .write_json_into(&mut buf, &JsonWriteOptions::compact())
            .expect("json write should succeed");
        buf
    };

    c.bench_function("persist/gbdt/json_read", |b| {
        b.iter(|| {
            let loaded = GBDTModel::read_json_from(black_box(json.as_slice()))
                .expect("json read should succeed");
            black_box(loaded)
        });
    });
}

criterion_group! {
    name = benches;
    config = default_criterion();
    targets = bench_persist_gbdt
}
criterion_main!(benches);
