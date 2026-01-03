use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use boosters::repr::gbdt::MutableTree;
use boosters::repr::gbdt::{Forest, ScalarLeaf};

/// Minimal model wrapper for benchmarks.
///
/// Benchmarks typically need only the parsed tree ensemble plus metadata like feature count.
pub struct LoadedForestModel {
    pub forest: Forest<ScalarLeaf>,
    pub n_features: usize,
}
fn build_balanced_tree(
    depth: u32,
    n_features: u32,
    rng: &mut Xoshiro256PlusPlus,
) -> boosters::repr::gbdt::Tree<ScalarLeaf> {
    let depth = depth.clamp(1, 12);
    let n_nodes = (1u32 << (depth + 1)) - 1;

    let mut builder = MutableTree::<ScalarLeaf>::with_capacity(n_nodes as usize);
    builder.init_root_with_n_nodes(n_nodes as usize);

    // Fill the complete binary tree in index order.
    for node_id in 0..n_nodes {
        let is_leaf = node_id >= (1u32 << depth) - 1;
        if is_leaf {
            let v = rng.gen_range(-1.0f32..=1.0f32);
            builder.make_leaf(node_id, ScalarLeaf(v));
        } else {
            let feature = rng.gen_range(0u32..n_features.max(1));
            let threshold = rng.gen_range(-5.0f32..=5.0f32);
            let default_left = rng.gen_bool(0.5);
            let left_id = 2 * node_id + 1;
            let right_id = 2 * node_id + 2;
            builder.set_numeric_split(node_id, feature, threshold, default_left, left_id, right_id);
        }
    }

    builder.freeze()
}

fn build_forest(n_features: usize, n_trees: usize, depth: u32, seed: u64) -> Forest<ScalarLeaf> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
    for _ in 0..n_trees {
        let tree = build_balanced_tree(depth, n_features as u32, &mut rng);
        forest.push_tree(tree, 0);
    }
    forest
}

/// Load a deterministic in-memory GBDT model for benchmarks.
pub fn load_boosters_model(name: &str) -> LoadedForestModel {
    let (n_features, n_trees, depth, seed) = match name {
        // Sizes chosen for stable iteration speed while still being non-trivial.
        "bench_small" => (50usize, 100usize, 6u32, 0xB001),
        "bench_medium" => (100usize, 500usize, 8u32, 0xB002),
        "bench_large" => (200usize, 1_000usize, 10u32, 0xB003),
        _ => panic!("Unknown benchmark model name: {name}"),
    };

    let forest = build_forest(n_features, n_trees, depth, seed);
    LoadedForestModel { forest, n_features }
}
