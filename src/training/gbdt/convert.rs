//! Conversion utilities between training-time and inference-time representations.
//!
//! Phase 1 of representation unification: training still uses the node-based
//! `training::gbdt::tree::*` structures for growth, but exposes the trained model
//! publicly as `inference::gbdt::{Forest, TreeStorage}`.

use crate::data::BinnedDataset;
use crate::inference::gbdt::{
    categories_to_bitset, CategoriesStorage, Forest as InferenceForest, ScalarLeaf, SplitType,
    TreeStorage,
};

use super::tree::{Forest as TrainingForest, Tree};

#[inline]
fn next_up_f32(x: f32) -> f32 {
    // Mirrors `f32::next_up` without relying on a specific MSRV.
    if x.is_nan() {
        return x;
    }
    if x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }

    let bits = x.to_bits();
    if x.is_sign_positive() {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

fn tree_to_inference(tree: &Tree, dataset: &BinnedDataset) -> Option<TreeStorage<ScalarLeaf>> {
    let num_nodes = tree.n_nodes();

    let mut split_indices = vec![0u32; num_nodes];
    let mut split_thresholds = vec![0.0f32; num_nodes];
    let mut left_children = vec![0u32; num_nodes];
    let mut right_children = vec![0u32; num_nodes];
    let mut default_left = vec![false; num_nodes];
    let mut is_leaf = vec![false; num_nodes];
    let mut leaf_values = vec![ScalarLeaf::default(); num_nodes];
    let mut split_types = vec![SplitType::Numeric; num_nodes];

    // Categorical storage: per-node segments into a flat bitset array.
    let mut cat_segments = vec![(0u32, 0u32); num_nodes];
    let mut cat_bitsets = Vec::<u32>::new();

    for (node_idx_usize, node) in tree.nodes().iter().enumerate() {
        let node_idx = node_idx_usize as u32;
        if node.is_leaf {
            is_leaf[node_idx_usize] = true;
            leaf_values[node_idx_usize] = ScalarLeaf(node.value);
            continue;
        }

        let feature = node.feature;
        split_indices[node_idx_usize] = feature;
        left_children[node_idx_usize] = node.left;
        right_children[node_idx_usize] = node.right;
        default_left[node_idx_usize] = node.default_left;

        if let Some(cat_split) = tree.categorical_split(node_idx) {
            // Training: categories in left_cats go LEFT.
            // Inference: categories in bitset go RIGHT.
            // Preserve semantics by storing training-left categories in the inference bitset
            // and swapping children (and default direction) for this node.
            split_types[node_idx_usize] = SplitType::Categorical;

            // Swap children.
            left_children[node_idx_usize] = node.right;
            right_children[node_idx_usize] = node.left;
            default_left[node_idx_usize] = !node.default_left;

            let mapper = dataset.bin_mapper(feature as usize);
            let mut categories = Vec::<u32>::new();
            for bin in cat_split.left_cats.iter() {
                let cat_f64 = mapper.bin_to_value(bin);
                let cat_i32 = cat_f64 as i32;
                if (cat_i32 as f64) != cat_f64 {
                    return None;
                }
                if cat_i32 < 0 {
                    return None;
                }
                categories.push(cat_i32 as u32);
            }

            categories.sort_unstable();
            categories.dedup();

            let bitset = categories_to_bitset(&categories);
            if !bitset.is_empty() {
                let start = cat_bitsets.len() as u32;
                let size = bitset.len() as u32;
                cat_segments[node_idx_usize] = (start, size);
                cat_bitsets.extend(bitset);
            }
        } else {
            // Training: go left if bin <= threshold.
            // Inference: go left if value < threshold.
            // Use the bin upper bound + next_up to make `<=` match `<`.
            let mapper = dataset.bin_mapper(feature as usize);
            let bound_f32 = mapper.bin_to_value(node.threshold as u32) as f32;
            split_thresholds[node_idx_usize] = next_up_f32(bound_f32);
            split_types[node_idx_usize] = SplitType::Numeric;
        }
    }

    let categories = CategoriesStorage::new(cat_bitsets, cat_segments);

    Some(TreeStorage::with_categories(
        split_indices,
        split_thresholds,
        left_children,
        right_children,
        default_left,
        is_leaf,
        leaf_values,
        split_types,
        categories,
    ))
}

/// Convert a training-time forest into an inference-time forest.
pub fn forest_to_inference(
    forest: &TrainingForest,
    dataset: &BinnedDataset,
) -> Option<InferenceForest<ScalarLeaf>> {
    let num_groups_usize = forest.n_outputs();
    let num_groups = u32::try_from(num_groups_usize).ok()?;

    let mut out = InferenceForest::new(num_groups).with_base_score(forest.base_scores().to_vec());

    for (idx, tree) in forest.trees().iter().enumerate() {
        let group = (idx % num_groups_usize) as u32;
        let soa = tree_to_inference(tree, dataset)?;
        out.push_tree(soa, group);
    }

    Some(out)
}
