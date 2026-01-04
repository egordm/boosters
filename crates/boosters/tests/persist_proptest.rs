//! Property-based tests for the persist module.
//!
//! These tests use proptest to generate arbitrary models and verify round-trip
//! serialization preserves data correctly.

use std::io::Cursor;

use proptest::collection::vec as prop_vec;
use proptest::prelude::*;

use boosters::model::{GBDTModel, GBLinearModel, ModelMeta, OutputTransform};
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
    // IMPORTANT: avoid `prop_filter(is_finite)` here.
    // Many generated models contain *lots* of floats; filtering any single float
    // causes the whole model strategy to reject, which quickly leads to
    // `Too many local rejects`.
    prop::num::f32::NORMAL.prop_map(|x| x.clamp(-1e6, 1e6))
}

/// Build a simple tree with the given depth using MutableTree.
#[derive(Clone, Debug)]
enum SplitSpec {
    Numeric {
        feature: u32,
        threshold: f32,
        default_left: bool,
    },
    Categorical {
        feature: u32,
        category_bitset_words: Vec<u32>,
        default_left: bool,
    },
}

#[derive(Clone, Debug)]
struct LinearLeafSpec {
    feature_indices: Vec<u32>,
    coefficients: Vec<f32>,
    intercept: f32,
}

#[derive(Clone, Debug)]
enum NodeSpec {
    Leaf {
        value: f32,
        linear: Option<LinearLeafSpec>,
    },
    Split {
        split: SplitSpec,
        left: Box<NodeSpec>,
        right: Box<NodeSpec>,
    },
}

fn count_nodes(spec: &NodeSpec) -> usize {
    match spec {
        NodeSpec::Leaf { .. } => 1,
        NodeSpec::Split { left, right, .. } => 1 + count_nodes(left) + count_nodes(right),
    }
}

fn build_tree_from_spec(spec: &NodeSpec) -> boosters::repr::gbdt::Tree<ScalarLeaf> {
    let n_nodes = count_nodes(spec);
    let mut builder = MutableTree::<ScalarLeaf>::with_capacity(n_nodes);
    builder.init_root_with_n_nodes(n_nodes);

    fn fill(builder: &mut MutableTree<ScalarLeaf>, spec: &NodeSpec, next_id: &mut u32) -> u32 {
        let my_id = *next_id;
        *next_id += 1;

        match spec {
            NodeSpec::Leaf { value, linear } => {
                builder.make_leaf(my_id, ScalarLeaf(*value));
                if let Some(l) = linear {
                    builder.set_linear_leaf(
                        my_id,
                        l.feature_indices.clone(),
                        l.intercept,
                        l.coefficients.clone(),
                    );
                }
            }
            NodeSpec::Split { split, left, right } => {
                let left_id = fill(builder, left, next_id);
                let right_id = fill(builder, right, next_id);

                match split {
                    SplitSpec::Numeric {
                        feature,
                        threshold,
                        default_left,
                    } => builder.set_numeric_split(
                        my_id,
                        *feature,
                        *threshold,
                        *default_left,
                        left_id,
                        right_id,
                    ),
                    SplitSpec::Categorical {
                        feature,
                        category_bitset_words,
                        default_left,
                    } => builder.set_categorical_split(
                        my_id,
                        *feature,
                        category_bitset_words.clone(),
                        *default_left,
                        left_id,
                        right_id,
                    ),
                }
            }
        }

        my_id
    }

    let mut next_id = 0u32;
    let root_id = fill(&mut builder, spec, &mut next_id);
    debug_assert_eq!(root_id, 0);
    debug_assert_eq!(next_id as usize, n_nodes);

    builder.freeze()
}

fn arb_split_spec(n_features: u32) -> impl Strategy<Value = SplitSpec> {
    let max_feat = n_features.max(1);
    prop_oneof![
        (0u32..max_feat, arb_finite_f32(), any::<bool>(),).prop_map(
            |(feature, threshold, default_left)| SplitSpec::Numeric {
                feature,
                threshold,
                default_left,
            }
        ),
        (0u32..max_feat, prop_vec(any::<u32>(), 1..=4), any::<bool>(),).prop_map(
            |(feature, bitset, default_left)| SplitSpec::Categorical {
                feature,
                category_bitset_words: bitset,
                default_left,
            }
        ),
    ]
}

fn arb_linear_leaf_spec(n_features: u32) -> impl Strategy<Value = LinearLeafSpec> {
    let max_feat = n_features.max(1);

    // Keep these tiny so trees stay cheap.
    (1usize..=3).prop_flat_map(move |len| {
        (
            prop_vec(0u32..max_feat, len),
            prop_vec(arb_finite_f32(), len),
            arb_finite_f32(),
        )
            .prop_map(|(mut idxs, coeffs, intercept)| {
                idxs.sort_unstable();
                idxs.dedup();
                // If dedup shrank, truncate coeffs to match deterministically.
                let coeffs = coeffs.into_iter().take(idxs.len()).collect::<Vec<_>>();
                LinearLeafSpec {
                    feature_indices: idxs,
                    coefficients: coeffs,
                    intercept,
                }
            })
    })
}

fn arb_node_spec(max_depth: u32, n_features: u32) -> impl Strategy<Value = NodeSpec> {
    let leaf = (
        arb_finite_f32(),
        prop::option::of(arb_linear_leaf_spec(n_features)),
    )
        .prop_map(|(value, linear)| NodeSpec::Leaf { value, linear });

    leaf.prop_recursive(max_depth, 2048, 2, move |inner| {
        (arb_split_spec(n_features), inner.clone(), inner).prop_map(|(split, left, right)| {
            NodeSpec::Split {
                split,
                left: Box::new(left),
                right: Box::new(right),
            }
        })
    })
}

fn arb_tree(
    max_depth: u32,
    n_features: u32,
) -> impl Strategy<Value = boosters::repr::gbdt::Tree<ScalarLeaf>> {
    arb_node_spec(max_depth, n_features).prop_map(|spec| build_tree_from_spec(&spec))
}

/// Strategy for generating (output transform, n_groups) pairs.
fn arb_output_semantics() -> impl Strategy<Value = (OutputTransform, usize)> {
    prop_oneof![
        Just((OutputTransform::Identity, 1)),
        Just((OutputTransform::Sigmoid, 1)),
        (3usize..=5).prop_map(|n| (OutputTransform::Softmax, n)),
    ]
}

/// Strategy for generating a forest with consistent output semantics.
fn arb_forest_with_semantics()
-> impl Strategy<Value = (Forest<ScalarLeaf>, OutputTransform, usize, usize)> {
    arb_output_semantics().prop_flat_map(|(output_transform, n_groups)| {
        let n_trees = 1usize..=100;
        let base_scores = prop_vec(arb_finite_f32(), n_groups);
        (1usize..=100, 1u32..=20, base_scores, n_trees)
            .prop_flat_map(move |(n_features, max_depth, base, n_trees)| {
                let trees = prop_vec(arb_tree(max_depth, n_features as u32), n_trees);
                (Just(n_features), Just(max_depth), Just(base), trees)
            })
            .prop_map(move |(n_features, _max_depth, base, trees)| {
                let mut forest = Forest::new(n_groups as u32).with_base_score(base);
                for (i, tree) in trees.into_iter().enumerate() {
                    forest.push_tree(tree, (i % n_groups) as u32);
                }
                (forest, output_transform, n_groups, n_features)
            })
    })
}

/// Strategy for generating a GBDT model.
fn arb_gbdt_model() -> impl Strategy<Value = GBDTModel> {
    arb_forest_with_semantics().prop_map(|(forest, output_transform, n_groups, n_features)| {
        let meta = ModelMeta {
            n_features,
            n_groups,
            base_scores: forest.base_score().to_vec(),
            feature_names: None,
            feature_types: None,
            best_iteration: None,
        };
        GBDTModel::from_parts(forest, meta, output_transform)
    })
}

/// Strategy for generating a GBLinear model.
fn arb_gblinear_model() -> impl Strategy<Value = GBLinearModel> {
    (1usize..=50, arb_output_semantics()).prop_flat_map(
        |(n_features, (output_transform, n_groups))| {
            let total = (n_features + 1) * n_groups;
            prop_vec(arb_finite_f32(), total).prop_map(move |weights| {
                let arr =
                    ndarray::Array2::from_shape_vec((n_features + 1, n_groups), weights).unwrap();
                let linear = LinearModel::new(arr);
                let meta = ModelMeta {
                    n_features,
                    n_groups,
                    base_scores: vec![0.0; n_groups],
                    feature_names: None,
                    feature_types: None,
                    best_iteration: None,
                };
                GBLinearModel::from_parts(linear, meta, output_transform)
            })
        },
    )
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
        prop_assert_eq!(loaded.output_transform(), model.output_transform());
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
        prop_assert_eq!(loaded.output_transform(), model.output_transform());
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
        prop_assert_eq!(loaded.output_transform(), model.output_transform());

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
        prop_assert_eq!(loaded.output_transform(), model.output_transform());
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
    let model = GBDTModel::from_parts(forest, meta, OutputTransform::Identity);

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
    let model = GBDTModel::from_parts(forest, meta, OutputTransform::Identity);

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
    let meta = ModelMeta {
        n_features: 2,
        n_groups: 2,
        base_scores: vec![0.0, 0.0],
        feature_names: None,
        feature_types: None,
        best_iteration: None,
    };
    let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Sigmoid);

    let mut buf = Vec::new();
    model
        .write_json_into(&mut buf, &JsonWriteOptions::compact())
        .unwrap();

    let result = GBDTModel::read_json_from(Cursor::new(&buf));
    assert!(result.is_err());
}
