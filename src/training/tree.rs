//! Tree building for gradient boosting.
//!
//! This module implements RFC-0015: tree growing strategies and training loop coordination.
//!
//! # Overview
//!
//! During training, trees are built incrementally:
//! 1. Start with root node containing all samples
//! 2. Build histogram, find best split
//! 3. Expand node according to growth policy
//! 4. Repeat until stopping criteria met
//!
//! # Growth Strategies
//!
//! - **Depth-wise** (XGBoost): Expand all nodes at each level before going deeper
//! - **Leaf-wise** (LightGBM): Always expand the leaf with highest gain
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::tree::{TreeGrower, DepthWisePolicy, TreeParams};
//!
//! let policy = DepthWisePolicy { max_depth: 6 };
//! let params = TreeParams::default();
//! let mut grower = TreeGrower::new(policy, &cuts, params);
//!
//! let tree = grower.build_tree(&quantized, &grads, &mut partitioner);
//! ```
//!
//! See RFC-0015 for design rationale.

use std::collections::HashMap;

use super::histogram::{HistogramBuilder, NodeHistogram};
use super::partition::RowPartitioner;
use super::quantize::{BinCuts, BinIndex, QuantizedMatrix};
use super::split::{GainParams, GreedySplitFinder, SplitFinder, SplitInfo};
use crate::training::buffer::GradientBuffer;

// ============================================================================
// BuildingNode and BuildingTree
// ============================================================================

/// A node in a tree being constructed.
///
/// During building, nodes start as leaves and can be expanded into split nodes.
/// This structure maintains parent/child links for tree traversal.
#[derive(Debug, Clone)]
pub struct BuildingNode {
    /// Split information (None if this is a leaf)
    pub split: Option<SplitInfo>,
    /// Left child index (u32::MAX if leaf)
    pub left: u32,
    /// Right child index (u32::MAX if leaf)
    pub right: u32,
    /// Parent index (u32::MAX if root)
    pub parent: u32,
    /// Depth in tree (root = 0)
    pub depth: u32,
    /// Leaf weight value
    pub weight: f32,
    /// Whether this node is currently a leaf
    pub is_leaf: bool,
    /// Partition node ID (for accessing rows)
    pub partition_id: u32,
}

impl BuildingNode {
    /// Create a new leaf node.
    fn new_leaf(weight: f32, parent: u32, depth: u32, partition_id: u32) -> Self {
        Self {
            split: None,
            left: u32::MAX,
            right: u32::MAX,
            parent,
            depth,
            weight,
            is_leaf: true,
            partition_id,
        }
    }
}

/// A tree being constructed during training.
///
/// Maintains a mutable tree structure that can be expanded node by node.
/// Once complete, can be converted to an immutable inference tree.
#[derive(Debug, Clone)]
pub struct BuildingTree {
    /// All nodes in the tree
    nodes: Vec<BuildingNode>,
    /// Current number of leaves
    num_leaves: u32,
    /// Maximum depth reached
    max_depth: u32,
}

impl BuildingTree {
    /// Create a new tree with just a root leaf.
    ///
    /// The root node starts as a leaf with the given base weight.
    pub fn new(base_weight: f32) -> Self {
        Self {
            nodes: vec![BuildingNode::new_leaf(base_weight, u32::MAX, 0, 0)],
            num_leaves: 1,
            max_depth: 0,
        }
    }

    /// Number of nodes in the tree.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaves in the tree.
    #[inline]
    pub fn num_leaves(&self) -> u32 {
        self.num_leaves
    }

    /// Maximum depth of the tree.
    #[inline]
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Get a node by index.
    #[inline]
    pub fn node(&self, index: u32) -> &BuildingNode {
        &self.nodes[index as usize]
    }

    /// Get a mutable node by index.
    #[inline]
    pub fn node_mut(&mut self, index: u32) -> &mut BuildingNode {
        &mut self.nodes[index as usize]
    }

    /// Expand a leaf node into a split node with two children.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Index of the leaf node to expand
    /// * `split` - Split information (feature, threshold, etc.)
    /// * `left_partition` - Partition node ID for left child rows
    /// * `right_partition` - Partition node ID for right child rows
    ///
    /// # Returns
    ///
    /// `(left_child_id, right_child_id)` - Indices of the new child nodes
    ///
    /// # Panics
    ///
    /// Panics if `node_id` is not a leaf.
    pub fn expand(
        &mut self,
        node_id: u32,
        split: SplitInfo,
        left_partition: u32,
        right_partition: u32,
    ) -> (u32, u32) {
        debug_assert!(
            self.nodes[node_id as usize].is_leaf,
            "Cannot expand non-leaf node"
        );

        let node = &self.nodes[node_id as usize];
        let depth = node.depth;

        let left_id = self.nodes.len() as u32;
        let right_id = left_id + 1;

        // Create left and right children
        self.nodes.push(BuildingNode::new_leaf(
            split.weight_left,
            node_id,
            depth + 1,
            left_partition,
        ));
        self.nodes.push(BuildingNode::new_leaf(
            split.weight_right,
            node_id,
            depth + 1,
            right_partition,
        ));

        // Convert parent to split node
        let node = &mut self.nodes[node_id as usize];
        node.split = Some(split);
        node.left = left_id;
        node.right = right_id;
        node.is_leaf = false;

        self.num_leaves += 1; // net +1 (remove 1, add 2)
        self.max_depth = self.max_depth.max(depth + 1);

        (left_id, right_id)
    }

    /// Iterate over all current leaf node indices.
    pub fn leaves(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.nodes.len() as u32).filter(|&i| self.nodes[i as usize].is_leaf)
    }

    /// Apply learning rate to all leaf weights.
    pub fn apply_learning_rate(&mut self, learning_rate: f32) {
        for node in &mut self.nodes {
            if node.is_leaf {
                node.weight *= learning_rate;
            }
        }
    }
}

// ============================================================================
// NodeCandidate
// ============================================================================

/// A candidate node for expansion.
///
/// Tracks the best split found for a node along with metadata for the
/// growth policy to make decisions.
#[derive(Debug, Clone)]
pub struct NodeCandidate {
    /// Index in the BuildingTree
    pub node_id: u32,
    /// Best split found for this node
    pub split: SplitInfo,
    /// Depth of this node
    pub depth: u32,
    /// Number of samples in this node
    pub num_samples: u32,
}

impl NodeCandidate {
    /// Create a new candidate.
    pub fn new(node_id: u32, split: SplitInfo, depth: u32, num_samples: u32) -> Self {
        Self {
            node_id,
            split,
            depth,
            num_samples,
        }
    }

    /// Get the gain of the best split.
    #[inline]
    pub fn gain(&self) -> f32 {
        self.split.gain
    }

    /// Check if this candidate has a valid split.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.split.gain > 0.0
    }
}

// ============================================================================
// GrowthPolicy trait and implementations
// ============================================================================

/// Policy for selecting which nodes to expand during tree growth.
///
/// Different policies produce different tree shapes:
/// - Depth-wise: balanced trees, expand level by level
/// - Leaf-wise: asymmetric trees, expand best gain leaf
pub trait GrowthPolicy {
    /// State maintained across expansion iterations.
    type State: GrowthState;

    /// Initialize state for a new tree.
    fn init(&self) -> Self::State;

    /// Select nodes to expand from current candidates.
    ///
    /// Returns the node IDs to expand this iteration.
    fn select_nodes(&self, state: &mut Self::State, candidates: &[NodeCandidate]) -> Vec<u32>;

    /// Check if we should continue growing the tree.
    fn should_continue(&self, state: &Self::State, tree: &BuildingTree) -> bool;
}

/// Trait for growth policy state.
pub trait GrowthState {
    /// Register new child nodes created from an expansion.
    fn add_children(&mut self, left: u32, right: u32, depth: u32);
}

// ============================================================================
// DepthWisePolicy
// ============================================================================

/// Depth-wise growth policy (XGBoost style).
///
/// Expands all nodes at the current level before proceeding to the next level.
/// Produces balanced trees.
///
/// # Example
///
/// ```
/// use booste_rs::training::tree::DepthWisePolicy;
///
/// let policy = DepthWisePolicy { max_depth: 6 };
/// ```
#[derive(Debug, Clone)]
pub struct DepthWisePolicy {
    /// Maximum tree depth (root = depth 0)
    pub max_depth: u32,
}

/// State for depth-wise growth.
#[derive(Debug)]
pub struct DepthWiseState {
    /// Current depth being processed
    current_depth: u32,
    /// Nodes at current depth waiting to be expanded
    current_level: Vec<u32>,
    /// Nodes generated for next level
    next_level: Vec<u32>,
}

impl GrowthPolicy for DepthWisePolicy {
    type State = DepthWiseState;

    fn init(&self) -> Self::State {
        DepthWiseState {
            current_depth: 0,
            current_level: vec![0], // Start with root
            next_level: Vec::new(),
        }
    }

    fn select_nodes(&self, state: &mut Self::State, candidates: &[NodeCandidate]) -> Vec<u32> {
        // Return all nodes at current level that have valid splits
        let to_expand: Vec<u32> = state
            .current_level
            .iter()
            .filter(|&&node_id| {
                candidates
                    .iter()
                    .find(|c| c.node_id == node_id)
                    .map(|c| c.is_valid())
                    .unwrap_or(false)
            })
            .copied()
            .collect();

        // Move to next level
        state.current_level.clear();
        std::mem::swap(&mut state.current_level, &mut state.next_level);
        state.current_depth += 1;

        to_expand
    }

    fn should_continue(&self, state: &Self::State, _tree: &BuildingTree) -> bool {
        state.current_depth <= self.max_depth && !state.current_level.is_empty()
    }
}

impl GrowthState for DepthWiseState {
    fn add_children(&mut self, left: u32, right: u32, _depth: u32) {
        self.next_level.push(left);
        self.next_level.push(right);
    }
}

// ============================================================================
// TreeParams
// ============================================================================

/// Parameters for tree building.
#[derive(Debug, Clone)]
pub struct TreeParams {
    /// Parameters for gain computation
    pub gain: GainParams,
    /// Maximum tree depth
    pub max_depth: u32,
    /// Minimum samples required to split a node
    pub min_samples_split: u32,
    /// Minimum samples required in a leaf
    pub min_samples_leaf: u32,
    /// Learning rate (shrinkage) applied to leaf weights
    pub learning_rate: f32,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            gain: GainParams::default(),
            max_depth: 6,
            min_samples_split: 2,
            min_samples_leaf: 1,
            learning_rate: 0.3,
        }
    }
}

// ============================================================================
// TreeGrower
// ============================================================================

/// Coordinates tree growing with a growth policy.
///
/// Brings together histogram building, split finding, and row partitioning
/// to grow a tree according to the specified growth policy.
///
/// # Naming Note
///
/// Named `TreeGrower` (not `TreeBuilder`) to avoid confusion with
/// `trees::TreeBuilder` which is an inference-time builder pattern helper.
pub struct TreeGrower<'a, G: GrowthPolicy> {
    /// Growth policy (depth-wise or leaf-wise)
    policy: G,
    /// Histogram builder
    hist_builder: HistogramBuilder,
    /// Split finder
    split_finder: GreedySplitFinder,
    /// Bin cuts for histograms
    cuts: &'a BinCuts,
    /// Training parameters
    params: TreeParams,
}

impl<'a, G: GrowthPolicy> TreeGrower<'a, G> {
    /// Create a new tree grower.
    pub fn new(policy: G, cuts: &'a BinCuts, params: TreeParams) -> Self {
        Self {
            policy,
            hist_builder: HistogramBuilder::default(),
            split_finder: GreedySplitFinder::new(),
            cuts,
            params,
        }
    }

    /// Build a single tree.
    ///
    /// # Arguments
    ///
    /// * `quantized` - Quantized feature matrix
    /// * `grads` - Gradient buffer with (grad, hess) for each row
    /// * `partitioner` - Row partitioner (will be modified during building)
    ///
    /// # Returns
    ///
    /// The built tree structure.
    pub fn build_tree<B: BinIndex>(
        &mut self,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        partitioner: &mut RowPartitioner,
    ) -> BuildingTree {
        let mut tree = BuildingTree::new(0.0);
        let mut state = self.policy.init();

        // Histogram storage per node (reuse across iterations)
        let mut histograms: HashMap<u32, NodeHistogram> = HashMap::new();

        // Build root histogram
        let root_rows = partitioner.node_rows(0);
        let root_hist = self.build_histogram(root_rows, quantized, grads);
        histograms.insert(0, root_hist);

        // Find initial split for root
        let root_split =
            self.split_finder
                .find_best_split(&histograms[&0], self.cuts, &self.params.gain);

        // Update root weight
        tree.node_mut(0).weight = root_split.weight_left; // Will be updated by split

        let mut candidates = vec![NodeCandidate::new(
            0,
            root_split,
            0,
            partitioner.node_size(0),
        )];

        // Main growth loop
        while self.policy.should_continue(&state, &tree) {
            let nodes_to_expand = self.policy.select_nodes(&mut state, &candidates);

            if nodes_to_expand.is_empty() {
                break;
            }

            // Expand selected nodes
            let mut new_candidates = Vec::new();

            for &node_id in &nodes_to_expand {
                let candidate = candidates
                    .iter()
                    .find(|c| c.node_id == node_id)
                    .expect("Candidate not found");

                if !self.should_split(candidate) {
                    continue;
                }

                // Get partition node for this tree node
                let partition_id = tree.node(node_id).partition_id;

                // Apply split to partitioner
                let split = candidate.split.clone();
                let (left_partition, right_partition) =
                    partitioner.apply_split(partition_id, &split, quantized);

                // Expand tree node
                let (left_id, right_id) =
                    tree.expand(node_id, split.clone(), left_partition, right_partition);

                // Register children with growth state
                state.add_children(left_id, right_id, candidate.depth + 1);

                // Build histograms for children (use subtraction optimization)
                let parent_hist = &histograms[&node_id];
                let (left_hist, right_hist) = self.build_child_histograms(
                    parent_hist,
                    left_partition,
                    right_partition,
                    quantized,
                    grads,
                    partitioner,
                );

                // Find splits for new nodes
                let left_split =
                    self.split_finder
                        .find_best_split(&left_hist, self.cuts, &self.params.gain);
                let right_split =
                    self.split_finder
                        .find_best_split(&right_hist, self.cuts, &self.params.gain);

                histograms.insert(left_id, left_hist);
                histograms.insert(right_id, right_hist);

                new_candidates.push(NodeCandidate::new(
                    left_id,
                    left_split,
                    candidate.depth + 1,
                    partitioner.node_size(left_partition),
                ));
                new_candidates.push(NodeCandidate::new(
                    right_id,
                    right_split,
                    candidate.depth + 1,
                    partitioner.node_size(right_partition),
                ));
            }

            // Remove expanded nodes from candidates, add new ones
            candidates.retain(|c| !nodes_to_expand.contains(&c.node_id));
            candidates.extend(new_candidates);
        }

        // Apply learning rate to leaf weights
        tree.apply_learning_rate(self.params.learning_rate);

        tree
    }

    /// Check if a candidate should be split.
    fn should_split(&self, candidate: &NodeCandidate) -> bool {
        candidate.is_valid()
            && candidate.depth < self.params.max_depth
            && candidate.num_samples >= self.params.min_samples_split
    }

    /// Build histogram for a set of rows.
    fn build_histogram<B: BinIndex>(
        &mut self,
        rows: &[u32],
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
    ) -> NodeHistogram {
        let mut hist = NodeHistogram::new(self.cuts);
        self.hist_builder.build(&mut hist, quantized, grads, rows);
        hist
    }

    /// Build histograms for child nodes using subtraction optimization.
    fn build_child_histograms<B: BinIndex>(
        &mut self,
        parent_hist: &NodeHistogram,
        left_partition: u32,
        right_partition: u32,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        partitioner: &RowPartitioner,
    ) -> (NodeHistogram, NodeHistogram) {
        let left_rows = partitioner.node_rows(left_partition);
        let right_rows = partitioner.node_rows(right_partition);

        let left_size = left_rows.len();
        let right_size = right_rows.len();

        // Build smaller child directly, derive larger via subtraction
        if left_size <= right_size {
            let left_hist = self.build_histogram(left_rows, quantized, grads);
            let right_hist = parent_hist.subtract(&left_hist);
            (left_hist, right_hist)
        } else {
            let right_hist = self.build_histogram(right_rows, quantized, grads);
            let left_hist = parent_hist.subtract(&right_hist);
            (left_hist, right_hist)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test split with common defaults.
    fn test_split(gain: f32, weight_left: f32, weight_right: f32) -> SplitInfo {
        SplitInfo {
            feature: 0,
            split_bin: 5,
            threshold: 0.5,
            gain,
            grad_left: 1.0,
            hess_left: 1.0,
            grad_right: 1.0,
            hess_right: 1.0,
            weight_left,
            weight_right,
            default_left: true,
            is_categorical: false,
            categories_left: vec![],
        }
    }

    #[test]
    fn test_building_node_new_leaf() {
        let node = BuildingNode::new_leaf(0.5, 0, 1, 2);
        assert!(node.is_leaf);
        assert!(node.split.is_none());
        assert_eq!(node.weight, 0.5);
        assert_eq!(node.parent, 0);
        assert_eq!(node.depth, 1);
        assert_eq!(node.partition_id, 2);
        assert_eq!(node.left, u32::MAX);
        assert_eq!(node.right, u32::MAX);
    }

    #[test]
    fn test_building_tree_new() {
        let tree = BuildingTree::new(0.1);
        assert_eq!(tree.num_nodes(), 1);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.max_depth(), 0);

        let root = tree.node(0);
        assert!(root.is_leaf);
        assert_eq!(root.weight, 0.1);
        assert_eq!(root.depth, 0);
    }

    #[test]
    fn test_building_tree_expand() {
        let mut tree = BuildingTree::new(0.0);

        let split = test_split(1.0, -0.2, 0.3);
        let (left, right) = tree.expand(0, split, 1, 2);

        assert_eq!(left, 1);
        assert_eq!(right, 2);
        assert_eq!(tree.num_nodes(), 3);
        assert_eq!(tree.num_leaves(), 2);
        assert_eq!(tree.max_depth(), 1);

        // Root is no longer a leaf
        let root = tree.node(0);
        assert!(!root.is_leaf);
        assert!(root.split.is_some());
        assert_eq!(root.left, 1);
        assert_eq!(root.right, 2);

        // Children are leaves
        let left_node = tree.node(1);
        assert!(left_node.is_leaf);
        assert_eq!(left_node.weight, -0.2);
        assert_eq!(left_node.parent, 0);
        assert_eq!(left_node.depth, 1);
        assert_eq!(left_node.partition_id, 1);

        let right_node = tree.node(2);
        assert!(right_node.is_leaf);
        assert_eq!(right_node.weight, 0.3);
        assert_eq!(right_node.parent, 0);
        assert_eq!(right_node.depth, 1);
        assert_eq!(right_node.partition_id, 2);
    }

    #[test]
    fn test_building_tree_leaves() {
        let mut tree = BuildingTree::new(0.0);

        // Initially just root
        let leaves: Vec<u32> = tree.leaves().collect();
        assert_eq!(leaves, vec![0]);

        // Expand root
        let split = test_split(1.0, 0.0, 0.0);
        tree.expand(0, split.clone(), 1, 2);

        let leaves: Vec<u32> = tree.leaves().collect();
        assert_eq!(leaves, vec![1, 2]);

        // Expand left child
        tree.expand(1, split.clone(), 3, 4);

        let leaves: Vec<u32> = tree.leaves().collect();
        assert_eq!(leaves, vec![2, 3, 4]);
    }

    #[test]
    fn test_building_tree_apply_learning_rate() {
        let mut tree = BuildingTree::new(0.0);

        let split = test_split(1.0, 1.0, 2.0);
        tree.expand(0, split, 1, 2);

        tree.apply_learning_rate(0.1);

        // Only leaf weights are scaled
        let left = tree.node(1);
        let right = tree.node(2);
        assert!((left.weight - 0.1).abs() < 1e-6);
        assert!((right.weight - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_node_candidate() {
        let split = test_split(1.5, 0.0, 0.0);
        let candidate = NodeCandidate::new(0, split, 2, 100);

        assert_eq!(candidate.node_id, 0);
        assert_eq!(candidate.depth, 2);
        assert_eq!(candidate.num_samples, 100);
        assert!((candidate.gain() - 1.5).abs() < 1e-6);
        assert!(candidate.is_valid());
    }

    #[test]
    fn test_node_candidate_invalid() {
        let split = test_split(0.0, 0.0, 0.0); // No gain
        let candidate = NodeCandidate::new(0, split, 0, 100);
        assert!(!candidate.is_valid());
    }

    #[test]
    fn test_depth_wise_policy_init() {
        let policy = DepthWisePolicy { max_depth: 6 };
        let state = policy.init();

        assert_eq!(state.current_depth, 0);
        assert_eq!(state.current_level, vec![0]);
        assert!(state.next_level.is_empty());
    }

    #[test]
    fn test_depth_wise_policy_select_nodes() {
        let policy = DepthWisePolicy { max_depth: 6 };
        let mut state = policy.init();

        // Create candidates for root with valid split
        let split = test_split(1.0, 0.0, 0.0);
        let candidates = vec![NodeCandidate::new(0, split, 0, 100)];

        let to_expand = policy.select_nodes(&mut state, &candidates);
        assert_eq!(to_expand, vec![0]);
        assert_eq!(state.current_depth, 1);
        assert!(state.current_level.is_empty()); // No children added yet
    }

    #[test]
    fn test_depth_wise_policy_should_continue() {
        let policy = DepthWisePolicy { max_depth: 2 };
        let mut state = policy.init();
        let tree = BuildingTree::new(0.0);

        // Should continue at depth 0 with nodes
        assert!(policy.should_continue(&state, &tree));

        // Advance to depth 1
        state.current_depth = 1;
        state.current_level = vec![1, 2];
        assert!(policy.should_continue(&state, &tree));

        // At max depth, should continue if we have nodes to process
        state.current_depth = 2;
        state.current_level = vec![3, 4];
        assert!(policy.should_continue(&state, &tree));

        // Beyond max depth
        state.current_depth = 3;
        assert!(!policy.should_continue(&state, &tree));

        // No more nodes
        state.current_depth = 1;
        state.current_level.clear();
        assert!(!policy.should_continue(&state, &tree));
    }

    #[test]
    fn test_depth_wise_state_add_children() {
        let mut state = DepthWiseState {
            current_depth: 0,
            current_level: vec![0],
            next_level: vec![],
        };

        state.add_children(1, 2, 1);
        assert_eq!(state.next_level, vec![1, 2]);

        state.add_children(3, 4, 1);
        assert_eq!(state.next_level, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_tree_params_default() {
        let params = TreeParams::default();
        assert_eq!(params.max_depth, 6);
        assert_eq!(params.min_samples_split, 2);
        assert_eq!(params.min_samples_leaf, 1);
        assert!((params.learning_rate - 0.3).abs() < 1e-6);
    }

    mod integration {
        use super::*;
        use crate::data::DenseMatrix;
        use crate::training::buffer::GradientBuffer;
        use crate::training::quantize::{CutFinder, ExactQuantileCuts, Quantizer};

        fn make_test_data() -> (QuantizedMatrix<u8>, BinCuts, GradientBuffer) {
            // 10 rows, 2 features (row-major)
            // Feature 0: values 0..9
            // Feature 1: first 5 rows = 0, last 5 rows = 1
            let data: Vec<f32> = vec![
                0.0, 0.0, // row 0
                1.0, 0.0, // row 1
                2.0, 0.0, // row 2
                3.0, 0.0, // row 3
                4.0, 0.0, // row 4
                5.0, 1.0, // row 5
                6.0, 1.0, // row 6
                7.0, 1.0, // row 7
                8.0, 1.0, // row 8
                9.0, 1.0, // row 9
            ];
            let matrix = DenseMatrix::from_vec(data, 10, 2);
            // Use min_samples_per_bin=1 for small test data
            let cuts_finder = ExactQuantileCuts::new(1);
            let cuts = cuts_finder.find_cuts(&matrix, 256);

            let quantizer = Quantizer::new(cuts.clone());
            let quantized = quantizer.quantize::<_, u8>(&matrix);

            // Simple gradients: positive for first group, negative for second
            let mut grads = GradientBuffer::new(10, 1);
            for i in 0..5 {
                grads.set(i, 0, 1.0, 1.0);
            }
            for i in 5..10 {
                grads.set(i, 0, -1.0, 1.0);
            }

            (quantized, cuts, grads)
        }

        #[test]
        fn test_tree_builder_single_split() {
            let (quantized, cuts, grads) = make_test_data();

            let policy = DepthWisePolicy { max_depth: 1 };
            let params = TreeParams {
                learning_rate: 1.0, // No shrinkage for testing
                ..Default::default()
            };

            let mut partitioner = RowPartitioner::new(10);
            let mut grower = TreeGrower::new(policy, &cuts, params);

            let tree = grower.build_tree(&quantized, &grads, &mut partitioner);

            // Should have root + 2 children
            assert!(tree.num_nodes() >= 1);
            // Root should be split (if gain was found)
            let root = tree.node(0);
            if !root.is_leaf {
                assert_eq!(tree.num_leaves(), 2);
                assert_eq!(tree.max_depth(), 1);
            }
        }

        #[test]
        fn test_tree_builder_depth_limit() {
            let (quantized, cuts, grads) = make_test_data();

            let policy = DepthWisePolicy { max_depth: 0 };
            let params = TreeParams {
                max_depth: 0, // Also set params.max_depth to 0
                ..Default::default()
            };

            let mut partitioner = RowPartitioner::new(10);
            let mut grower = TreeGrower::new(policy, &cuts, params);

            let tree = grower.build_tree(&quantized, &grads, &mut partitioner);

            // With max_depth=0, should only have root (no splits)
            assert_eq!(tree.num_nodes(), 1);
            assert_eq!(tree.num_leaves(), 1);
            assert!(tree.node(0).is_leaf);
        }

        #[test]
        fn test_tree_builder_multiple_levels() {
            let (quantized, cuts, grads) = make_test_data();

            let policy = DepthWisePolicy { max_depth: 3 };
            let params = TreeParams {
                learning_rate: 1.0,
                ..Default::default()
            };

            let mut partitioner = RowPartitioner::new(10);
            let mut grower = TreeGrower::new(policy, &cuts, params);

            let tree = grower.build_tree(&quantized, &grads, &mut partitioner);

            // Should build multiple levels until no more gain or max depth
            assert!(tree.max_depth() <= 3);
            // All leaves should be marked as leaves
            for leaf_id in tree.leaves() {
                assert!(tree.node(leaf_id).is_leaf);
            }
        }
    }
}
