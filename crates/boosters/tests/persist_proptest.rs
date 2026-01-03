//! Property-based tests for the persist module.
//!
//! These tests use proptest to generate arbitrary models and verify round-trip
//! serialization preserves data correctly.

use std::io::Cursor;

use proptest::collection::vec as prop_vec;
use proptest::prelude::*;

use boosters::model::{GBDTModel, GBLinearModel, ModelMeta, TaskKind};
use boosters::persist::{
    BinaryReadOptions, BinaryWriteOptions, JsonWriteOptions, Model, SerializableModel,
};
use boosters::repr::gbdt::{Forest, MutableTree, ScalarLeaf};
use boosters::repr::gblinear::LinearModel;

// =============================================================================
// Arbitrary Model Generators
// =============================================================================

/// Strategy for generating valid f32 values (no NaN/Inf).
fn arb_finite_f32() -> impl Strategy<Value = f32> {
    prop::num::f32::ANY
        .prop_filter("must be finite", |x| x.is_finite())
        .prop_map(|x| x.clamp(-1e6, 1e6))
}

/// Build a simple tree with the given depth using MutableTree.
fn build_tree_from_depth(
    depth: usize,
    leaf_values: &[f32],
    thresholds: &[f32],
) -> boosters::repr::gbdt::Tree<ScalarLeaf> {
    match depth {
        0 => {
            // Single leaf
            let leaf_val = leaf_values.first().copied().unwrap_or(0.0);
            let mut tree = MutableTree::<ScalarLeaf>::with_capacity(1);
            tree.init_root_with_n_nodes(1);
            tree.make_leaf(0, ScalarLeaf(leaf_val));
            tree.freeze()
        }
        1 => {
            // One split, two leaves
            let thresh = thresholds.first().copied().unwrap_or(0.5);
            let lv0 = leaf_values.first().copied().unwrap_or(1.0);
            let lv1 = leaf_values.get(1).copied().unwrap_or(2.0);

            let mut tree = MutableTree::<ScalarLeaf>::with_capacity(3);
            tree.init_root_with_n_nodes(3);
            tree.set_numeric_split(0, 0, thresh, true, 1, 2);
            tree.make_leaf(1, ScalarLeaf(lv0));
            tree.make_leaf(2, ScalarLeaf(lv1));
            tree.freeze()
        }
        _ => {
            // Two+ splits: three internal nodes, four leaves
            let t0 = thresholds.first().copied().unwrap_or(0.5);
            let t1 = thresholds.get(1).copied().unwrap_or(0.25);
            let t2 = thresholds.get(2).copied().unwrap_or(0.75);
            let lv0 = leaf_values.first().copied().unwrap_or(1.0);
            let lv1 = leaf_values.get(1).copied().unwrap_or(2.0);
            let lv2 = leaf_values.get(2).copied().unwrap_or(3.0);
            let lv3 = leaf_values.get(3).copied().unwrap_or(4.0);

            let mut tree = MutableTree::<ScalarLeaf>::with_capacity(7);
            tree.init_root_with_n_nodes(7);
            tree.set_numeric_split(0, 0, t0, true, 1, 2);
            tree.set_numeric_split(1, 1, t1, true, 3, 4);
            tree.set_numeric_split(2, 1, t2, true, 5, 6);
            tree.make_leaf(3, ScalarLeaf(lv0));
            tree.make_leaf(4, ScalarLeaf(lv1));
            tree.make_leaf(5, ScalarLeaf(lv2));
            tree.make_leaf(6, ScalarLeaf(lv3));
            tree.freeze()
        }
    }
}

/// Strategy for generating a simple tree.
fn arb_simple_tree() -> impl Strategy<Value = boosters::repr::gbdt::Tree<ScalarLeaf>> {
    (
        0usize..=2,
        prop_vec(arb_finite_f32(), 4),
        prop_vec(arb_finite_f32(), 3),
    )
        .prop_map(|(depth, leaves, thresholds)| build_tree_from_depth(depth, &leaves, &thresholds))
}

/// Strategy for generating a task with consistent n_groups.
///
/// TaskKind determines n_groups:
/// - Regression: 1
/// - BinaryClassification: 1
/// - MulticlassClassification(n): n (3+)
fn arb_task_kind() -> impl Strategy<Value = (TaskKind, usize)> {
    prop_oneof![
        Just((TaskKind::Regression, 1)),
        Just((TaskKind::BinaryClassification, 1)),
        (3usize..=5).prop_map(|n| (TaskKind::MulticlassClassification { n_classes: n }, n)),
    ]
}

/// Strategy for generating a forest with a consistent task.
fn arb_forest_with_task() -> impl Strategy<Value = (Forest<ScalarLeaf>, TaskKind, usize)> {
    arb_task_kind().prop_flat_map(|(task, n_groups)| {
        let n_trees = 1usize..=10;
        let base_scores = prop_vec(arb_finite_f32(), n_groups);
        let trees = prop_vec(arb_simple_tree(), n_trees);

        (base_scores, trees).prop_map(move |(base, trees)| {
            let mut forest = Forest::new(n_groups as u32).with_base_score(base);
            for (i, tree) in trees.into_iter().enumerate() {
                forest.push_tree(tree, (i % n_groups) as u32);
            }
            (forest, task, n_groups)
        })
    })
}

/// Strategy for generating a GBDT model.
fn arb_gbdt_model() -> impl Strategy<Value = GBDTModel> {
    (1usize..=100, arb_forest_with_task()).prop_map(|(n_features, (forest, task, n_groups))| {
        let meta = ModelMeta {
            n_features,
            n_groups,
            task,
            base_scores: forest.base_score().to_vec(),
            feature_names: None,
            feature_types: None,
            best_iteration: None,
        };
        GBDTModel::from_forest(forest, meta)
    })
}

/// Strategy for generating a GBLinear model.
fn arb_gblinear_model() -> impl Strategy<Value = GBLinearModel> {
    (1usize..=50, arb_task_kind()).prop_flat_map(|(n_features, (task, n_groups))| {
        let total = (n_features + 1) * n_groups;
        prop_vec(arb_finite_f32(), total).prop_map(move |weights| {
            let arr =
                ndarray::Array2::from_shape_vec((n_features + 1, n_groups), weights).unwrap();
            let linear = LinearModel::new(arr);
            let meta = ModelMeta {
                n_features,
                n_groups,
                task,
                base_scores: vec![0.0; n_groups],
                feature_names: None,
                feature_types: None,
                best_iteration: None,
            };
            GBLinearModel::from_linear_model(linear, meta)
        })
    })
}

// =============================================================================
// Round-Trip Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// GBDT binary round-trip preserves all data.
    #[test]
    fn gbdt_binary_roundtrip(model in arb_gbdt_model()) {
        let mut buf = Vec::new();
        model.write_into(&mut buf, &BinaryWriteOptions::default()).unwrap();
        let loaded = GBDTModel::read_from(Cursor::new(&buf), &BinaryReadOptions::default()).unwrap();

        prop_assert_eq!(loaded.meta().n_features, model.meta().n_features);
        prop_assert_eq!(loaded.meta().n_groups, model.meta().n_groups);
        prop_assert_eq!(loaded.meta().task, model.meta().task);
        prop_assert_eq!(loaded.forest().n_trees(), model.forest().n_trees());
        prop_assert_eq!(loaded.forest().n_groups(), model.forest().n_groups());

        for (a, b) in model.forest().base_score().iter().zip(loaded.forest().base_score().iter()) {
            prop_assert!((a - b).abs() < 1e-5, "base score mismatch: {} vs {}", a, b);
        }
    }

    /// GBDT JSON round-trip preserves all data.
    #[test]
    fn gbdt_json_roundtrip(model in arb_gbdt_model()) {
        let mut buf = Vec::new();
        model.write_json_into(&mut buf, &JsonWriteOptions::compact()).unwrap();
        let loaded = GBDTModel::read_json_from(Cursor::new(&buf)).unwrap();

        prop_assert_eq!(loaded.meta().n_features, model.meta().n_features);
        prop_assert_eq!(loaded.meta().n_groups, model.meta().n_groups);
        prop_assert_eq!(loaded.forest().n_trees(), model.forest().n_trees());
    }

    /// GBLinear binary round-trip preserves all data.
    #[test]
    fn gblinear_binary_roundtrip(model in arb_gblinear_model()) {
        let mut buf = Vec::new();
        model.write_into(&mut buf, &BinaryWriteOptions::default()).unwrap();
        let loaded = GBLinearModel::read_from(Cursor::new(&buf), &BinaryReadOptions::default()).unwrap();

        prop_assert_eq!(loaded.meta().n_features, model.meta().n_features);
        prop_assert_eq!(loaded.meta().n_groups, model.meta().n_groups);

        let orig = model.linear().as_slice();
        let load = loaded.linear().as_slice();
        prop_assert_eq!(orig.len(), load.len());
        for (a, b) in orig.iter().zip(load.iter()) {
            prop_assert!((a - b).abs() < 1e-5, "weight mismatch: {} vs {}", a, b);
        }
    }

    /// GBLinear JSON round-trip preserves all data.
    #[test]
    fn gblinear_json_roundtrip(model in arb_gblinear_model()) {
        let mut buf = Vec::new();
        model.write_json_into(&mut buf, &JsonWriteOptions::compact()).unwrap();
        let loaded = GBLinearModel::read_json_from(Cursor::new(&buf)).unwrap();

        prop_assert_eq!(loaded.meta().n_features, model.meta().n_features);
        prop_assert_eq!(loaded.meta().n_groups, model.meta().n_groups);
    }

    /// Polymorphic loading returns correct type for GBDT.
    #[test]
    fn polymorphic_gbdt_binary(model in arb_gbdt_model()) {
        let mut buf = Vec::new();
        model.write_into(&mut buf, &BinaryWriteOptions::default()).unwrap();
        let loaded = Model::read_binary(Cursor::new(&buf), &BinaryReadOptions::default()).unwrap();
        prop_assert!(loaded.as_gbdt().is_some());
        prop_assert!(loaded.as_gblinear().is_none());
    }

    /// Polymorphic loading returns correct type for GBLinear.
    #[test]
    fn polymorphic_gblinear_binary(model in arb_gblinear_model()) {
        let mut buf = Vec::new();
        model.write_into(&mut buf, &BinaryWriteOptions::default()).unwrap();
        let loaded = Model::read_binary(Cursor::new(&buf), &BinaryReadOptions::default()).unwrap();
        prop_assert!(loaded.as_gblinear().is_some());
        prop_assert!(loaded.as_gbdt().is_none());
    }
}

// =============================================================================
// Negative Tests (Validation)
// =============================================================================

#[test]
fn invalid_magic_rejected() {
    let mut buf = vec![0x00, 0x01, 0x02, 0x03];
    buf.extend_from_slice(&[0u8; 100]);
    let result = GBDTModel::read_from(Cursor::new(&buf), &BinaryReadOptions::default());
    assert!(result.is_err());
}

#[test]
fn corrupted_payload_rejected() {
    let tree = boosters::scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(1.0),
        2 => leaf(2.0),
    };
    let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
    forest.push_tree(tree, 0);
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_forest(forest, meta);

    let mut buf = Vec::new();
    model
        .write_into(&mut buf, &BinaryWriteOptions::default())
        .unwrap();

    if buf.len() > 50 {
        buf[40] ^= 0xFF;
        buf[41] ^= 0xFF;
        buf[42] ^= 0xFF;
    }

    let result = GBDTModel::read_from(Cursor::new(&buf), &BinaryReadOptions::default());
    assert!(result.is_err());
}

#[test]
fn truncated_file_rejected() {
    let tree = boosters::scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(1.0),
        2 => leaf(2.0),
    };
    let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
    forest.push_tree(tree, 0);
    let meta = ModelMeta::for_regression(2);
    let model = GBDTModel::from_forest(forest, meta);

    let mut buf = Vec::new();
    model
        .write_into(&mut buf, &BinaryWriteOptions::default())
        .unwrap();

    buf.truncate(buf.len() / 2);
    let result = GBDTModel::read_from(Cursor::new(&buf), &BinaryReadOptions::default());
    assert!(result.is_err());
}

#[test]
fn invalid_json_rejected() {
    let bad_json = r#"{"bstr_version": 1, "model_type": "gbdt", "model": "not_an_object"}"#;
    let result = GBDTModel::read_json_from(Cursor::new(bad_json.as_bytes()));
    assert!(result.is_err());
}

#[test]
fn wrong_model_type_rejected() {
    let weights = ndarray::array![[0.1, 0.2], [0.3, 0.4], [0.01, 0.02]];
    let linear = LinearModel::new(weights);
    let task = TaskKind::BinaryClassification;
    let meta = ModelMeta {
        n_features: 2,
        n_groups: 2,
        task,
        base_scores: vec![0.0, 0.0],
        feature_names: None,
        feature_types: None,
        best_iteration: None,
    };
    let model = GBLinearModel::from_linear_model(linear, meta);

    let mut buf = Vec::new();
    model
        .write_json_into(&mut buf, &JsonWriteOptions::compact())
        .unwrap();

    let result = GBDTModel::read_json_from(Cursor::new(&buf));
    assert!(result.is_err());
}
