//! Targeted validation-failure tests for RFC-0016 invariants.

use std::io::Cursor;

use boosters::model::{GBDTModel, ModelMeta, OutputTransform};
use boosters::persist::{JsonWriteOptions, Model, ReadError, SerializableModel};
use boosters::repr::gbdt::Forest;
use serde_json::Value;

fn load_fixture_value() -> Value {
    // Create a tiny, deterministic model and serialize it to the persisted JSON envelope.
    // This keeps validation tests self-contained (no filesystem fixtures).
    let tree = boosters::scalar_tree! {
        0 => num(0, 0.5, L) -> 1, 2,
        1 => leaf(1.0),
        2 => leaf(2.0),
    };

    let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
    forest.push_tree(tree, 0);

    let model = GBDTModel::from_parts(
        forest,
        ModelMeta::for_regression(1),
        OutputTransform::Identity,
    );

    let mut buf = Vec::new();
    model
        .write_json_into(&mut buf, &JsonWriteOptions::compact())
        .expect("serialize model");

    serde_json::from_slice(&buf).expect("parse fixture json")
}

fn roundtrip_model_err(v: Value) -> ReadError {
    let bytes = serde_json::to_vec(&v).expect("serialize mutated json");
    Model::read_json(Cursor::new(&bytes)).expect_err("expected error")
}

#[test]
fn validation_fails_on_mismatched_array_lengths() {
    let mut v = load_fixture_value();

    let tree = v.pointer_mut("/model/forest/trees/0").expect("tree exists");

    let split_indices = tree
        .get_mut("split_indices")
        .and_then(|x| x.as_array_mut())
        .expect("split_indices array");

    // Remove one element so len != num_nodes.
    split_indices.pop();

    let err = roundtrip_model_err(v);
    assert!(matches!(err, ReadError::Validation(_)), "got: {err:?}");
}

#[test]
fn validation_fails_on_out_of_bounds_child_index() {
    let mut v = load_fixture_value();

    let tree = v.pointer_mut("/model/forest/trees/0").expect("tree exists");

    // Force an invalid child index at the root.
    let children_left = tree
        .get_mut("children_left")
        .and_then(|x| x.as_array_mut())
        .expect("children_left array");

    children_left[0] = Value::from(9_999_999u64);

    let err = roundtrip_model_err(v);
    assert!(matches!(err, ReadError::Validation(_)), "got: {err:?}");
}

#[test]
fn validation_fails_on_tree_groups_length_mismatch() {
    let mut v = load_fixture_value();

    let forest = v.pointer_mut("/model/forest").expect("forest exists");

    let trees_len = forest
        .get("trees")
        .and_then(|x| x.as_array())
        .expect("trees array")
        .len();

    // Make tree_groups one shorter than trees.
    forest["tree_groups"] = Value::from(vec![0usize; trees_len.saturating_sub(1)]);

    let err = roundtrip_model_err(v);
    assert!(matches!(err, ReadError::Validation(_)), "got: {err:?}");
}

#[test]
fn validation_fails_on_base_score_length_mismatch() {
    let mut v = load_fixture_value();

    let forest = v.pointer_mut("/model/forest").expect("forest exists");

    forest["n_groups"] = Value::from(2u64);

    // Keep base_score at length 1, making it invalid for n_groups=2.
    let err = roundtrip_model_err(v);
    assert!(matches!(err, ReadError::Validation(_)), "got: {err:?}");
}
