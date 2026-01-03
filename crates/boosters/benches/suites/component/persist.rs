//! Component benchmarks: persistence read/write performance.
//!
//! Performance targets from RFC-0016:
//! - Write (100 trees, 1K nodes): <20ms
//! - Write (1000 trees, 1K nodes): <200ms
//! - Read (100 trees): <10ms
//! - Inspect (header only): <1ms

#[path = "../../common/mod.rs"]
mod common;

use common::criterion_config::default_criterion;

use boosters::model::gbdt::GBDTConfig;
use boosters::model::{GBDTModel, ModelMeta, TaskKind};
use boosters::persist::{
    BinaryReadOptions, BinaryWriteOptions, JsonWriteOptions, ModelInfo, SerializableModel,
};
use boosters::repr::gbdt::{Forest, MutableTree, ScalarLeaf};
use boosters::training::Objective;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::io::Cursor;

/// Build a model with a given number of trees and nodes per tree.
fn build_test_model(n_trees: usize, nodes_per_tree: usize) -> GBDTModel {
    let mut forest = Forest::for_regression().with_base_score(vec![0.5]);

    for _ in 0..n_trees {
        // Build a tree with the specified number of nodes
        let tree = build_tree_with_nodes(nodes_per_tree);
        forest.push_tree(tree, 0);
    }

    let meta = ModelMeta {
        n_features: 10,
        n_groups: 1,
        task: TaskKind::Regression,
        base_scores: vec![0.5],
        feature_names: None,
        feature_types: None,
        best_iteration: None,
    };

    let config = GBDTConfig::builder()
        .objective(Objective::squared())
        .n_trees(n_trees as u32)
        .build()
        .unwrap();

    GBDTModel::from_parts(forest, meta, config)
}

/// Build a tree with approximately the specified number of nodes.
///
/// Creates a complete binary tree with the given depth, or as close as possible.
fn build_tree_with_nodes(target_nodes: usize) -> boosters::repr::gbdt::Tree<ScalarLeaf> {
    // A complete binary tree of depth d has 2^(d+1) - 1 nodes
    // We'll create a balanced tree with approximately target_nodes
    let depth = (target_nodes as f64).log2().floor() as u32;
    let depth = depth.clamp(1, 10);

    // Pre-calculate number of nodes for a complete binary tree of this depth
    let n_nodes = (1u32 << (depth + 1)) - 1;

    let mut builder = MutableTree::<ScalarLeaf>::with_capacity(n_nodes as usize);
    builder.init_root_with_n_nodes(n_nodes as usize);

    // Build complete binary tree recursively by setting each node
    build_subtree(&mut builder, 0, 0, depth);

    builder.freeze()
}

fn build_subtree(
    builder: &mut MutableTree<ScalarLeaf>,
    node_id: u32,
    feature: u32,
    remaining_depth: u32,
) {
    if remaining_depth == 0 {
        builder.make_leaf(node_id, ScalarLeaf(0.1 * node_id as f32));
    } else {
        let threshold = (remaining_depth as f32) * 0.1;
        // In a complete binary tree, left child is 2*i+1, right is 2*i+2
        let left_id = 2 * node_id + 1;
        let right_id = 2 * node_id + 2;

        builder.set_numeric_split(node_id, feature % 10, threshold, true, left_id, right_id);

        build_subtree(builder, left_id, (feature + 1) % 10, remaining_depth - 1);
        build_subtree(builder, right_id, (feature + 2) % 10, remaining_depth - 1);
    }
}

fn bench_write_small_model(c: &mut Criterion) {
    let model = build_test_model(100, 1000);
    let expected_size = model.forest().n_trees() * 1000; // approximate

    let mut group = c.benchmark_group("component/persist/write");
    group.throughput(Throughput::Elements(expected_size as u64));

    group.bench_function(BenchmarkId::new("binary", "100_trees"), |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(1024 * 1024);
            model
                .write_into(&mut buf, &BinaryWriteOptions::default())
                .unwrap();
            black_box(buf)
        });
    });

    group.finish();
}

fn bench_write_large_model(c: &mut Criterion) {
    let model = build_test_model(1000, 1000);
    let expected_size = model.forest().n_trees() * 1000;

    let mut group = c.benchmark_group("component/persist/write");
    group.throughput(Throughput::Elements(expected_size as u64));

    group.bench_function(BenchmarkId::new("binary", "1000_trees"), |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(10 * 1024 * 1024);
            model
                .write_into(&mut buf, &BinaryWriteOptions::default())
                .unwrap();
            black_box(buf)
        });
    });

    group.finish();
}

fn bench_read_model(c: &mut Criterion) {
    let model = build_test_model(100, 1000);
    let mut buf = Vec::new();
    model
        .write_into(&mut buf, &BinaryWriteOptions::default())
        .unwrap();

    let mut group = c.benchmark_group("component/persist/read");
    group.throughput(Throughput::Bytes(buf.len() as u64));

    group.bench_function(BenchmarkId::new("binary", "100_trees"), |b| {
        b.iter(|| {
            let cursor = Cursor::new(&buf);
            let loaded = GBDTModel::read_from(cursor, &BinaryReadOptions::default()).unwrap();
            black_box(loaded)
        });
    });

    group.finish();
}

fn bench_read_large_model(c: &mut Criterion) {
    let model = build_test_model(1000, 1000);
    let mut buf = Vec::new();
    model
        .write_into(&mut buf, &BinaryWriteOptions::default())
        .unwrap();

    let mut group = c.benchmark_group("component/persist/read");
    group.throughput(Throughput::Bytes(buf.len() as u64));

    group.bench_function(BenchmarkId::new("binary", "1000_trees"), |b| {
        b.iter(|| {
            let cursor = Cursor::new(&buf);
            let loaded = GBDTModel::read_from(cursor, &BinaryReadOptions::default()).unwrap();
            black_box(loaded)
        });
    });

    group.finish();
}

fn bench_write_json_small_model(c: &mut Criterion) {
    let model = build_test_model(100, 1000);
    let mut group = c.benchmark_group("component/persist/write_json");

    group.bench_function(BenchmarkId::new("json", "100_trees"), |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(10 * 1024 * 1024);
            model
                .write_json_into(&mut buf, &JsonWriteOptions::compact())
                .unwrap();
            black_box(buf)
        });
    });

    group.finish();
}

fn bench_inspect_header(c: &mut Criterion) {
    let model = build_test_model(100, 1000);
    let mut buf = Vec::new();
    model
        .write_into(&mut buf, &BinaryWriteOptions::default())
        .unwrap();

    let mut group = c.benchmark_group("component/persist/inspect");

    group.bench_function("header_only", |b| {
        b.iter(|| {
            let cursor = Cursor::new(&buf);
            let info = ModelInfo::inspect_binary(cursor).unwrap();
            black_box((info.schema_version, info.model_type, info.format))
        });
    });

    group.finish();
}

criterion_group! {
    name = persist_benches;
    config = default_criterion();
    targets =
        bench_write_small_model,
        bench_write_large_model,
        bench_write_json_small_model,
        bench_read_model,
        bench_read_large_model,
        bench_inspect_header,
}

criterion_main!(persist_benches);
