//! Growth policies for tree building.
//!
//! This module defines how trees are grown during training:
//!
//! - **Depth-wise** (XGBoost style): Expand all nodes at each level before going deeper
//! - **Leaf-wise** (LightGBM style): Always expand the leaf with highest gain
//!
//! # Usage
//!
//! ```ignore
//! use boosters::training::{GrowthStrategy, GrowthState};
//!
//! // Depth-wise (XGBoost style)
//! let mut state = GrowthStrategy::DepthWise { max_depth: 6 }.init();
//!
//! // Leaf-wise (LightGBM style)  
//! let mut state = GrowthStrategy::LeafWise { max_leaves: 31 }.init();
//!
//! // Use state methods for tree building
//! state.push_root(candidate);
//! while state.should_continue() {
//!     let nodes = state.pop_next();
//!     // ... expand nodes ...
//!     for child in children {
//!         state.push(child);
//!     }
//!     state.advance();
//! }
//! ```

use std::collections::BinaryHeap;

use crate::repr::gbdt::NodeId;

use super::split::SplitInfo;

// ============================================================================
// GrowthStrategy enum (config only)
// ============================================================================

/// Growth strategy configuration.
///
/// This is a config-only enum that creates the appropriate [`GrowthState`].
/// All tree-building logic is on `GrowthState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStrategy {
    /// Depth-wise growth (XGBoost style) - expand all nodes at each level.
    DepthWise {
        /// Maximum tree depth (root = depth 0).
        max_depth: u32,
    },
    /// Leaf-wise growth (LightGBM style) - expand best gain leaf.
    LeafWise {
        /// Maximum number of leaves.
        max_leaves: u32,
    },
}

impl Default for GrowthStrategy {
    fn default() -> Self {
        Self::DepthWise { max_depth: 6 }
    }
}

impl GrowthStrategy {
    /// Create a depth-wise strategy with default max_depth=6.
    pub fn depth_wise() -> Self {
        Self::DepthWise { max_depth: 6 }
    }

    /// Create a leaf-wise strategy with default max_leaves=31.
    pub fn leaf_wise() -> Self {
        Self::LeafWise { max_leaves: 31 }
    }

    /// Create a depth-wise strategy with custom max_depth.
    pub fn with_max_depth(max_depth: u32) -> Self {
        Self::DepthWise { max_depth }
    }

    /// Create a leaf-wise strategy with custom max_leaves.
    pub fn with_max_leaves(max_leaves: u32) -> Self {
        Self::LeafWise { max_leaves }
    }

    /// Initialize growth state for this strategy.
    pub fn init(self) -> GrowthState {
        match self {
            GrowthStrategy::DepthWise { max_depth } => GrowthState::DepthWise {
                max_depth,
                current_depth: 0,
                current_level: Vec::new(),
                next_level: Vec::new(),
            },
            GrowthStrategy::LeafWise { max_leaves } => GrowthState::LeafWise {
                max_leaves,
                candidates: BinaryHeap::new(),
                n_leaves: 1, // Start with root as single leaf
            },
        }
    }
}

// ============================================================================
// GrowthState enum (all logic here)
// ============================================================================

/// Growth state with all tree-building methods.
///
/// Created via `GrowthStrategy::init()`. Contains both config params
/// and runtime state in enum variants.
pub enum GrowthState {
    /// Depth-wise state.
    DepthWise {
        /// Maximum tree depth.
        max_depth: u32,
        /// Current depth being processed.
        current_depth: u32,
        /// Nodes at current depth waiting to be expanded.
        current_level: Vec<NodeCandidate>,
        /// Nodes generated for next level.
        next_level: Vec<NodeCandidate>,
    },
    /// Leaf-wise state.
    LeafWise {
        /// Maximum number of leaves.
        max_leaves: u32,
        /// Priority queue of candidate leaves (max-heap by gain).
        candidates: BinaryHeap<NodeCandidate>,
        /// Current number of leaves in the tree.
        n_leaves: u32,
    },
}

impl GrowthState {
    /// Check if we should continue growing the tree.
    pub fn should_continue(&self) -> bool {
        match self {
            GrowthState::DepthWise {
                max_depth,
                current_depth,
                current_level,
                ..
            } => *current_depth < *max_depth && !current_level.is_empty(),
            GrowthState::LeafWise {
                max_leaves,
                candidates,
                n_leaves,
            } => *n_leaves < *max_leaves && !candidates.is_empty(),
        }
    }

    /// Pop the next node(s) to expand.
    ///
    /// For depth-wise: returns all valid nodes at the current level.
    /// For leaf-wise: returns the single best node by gain.
    pub fn pop_next(&mut self) -> Vec<NodeCandidate> {
        match self {
            GrowthState::DepthWise { current_level, .. } => std::mem::take(current_level),
            GrowthState::LeafWise {
                candidates,
                n_leaves,
                ..
            } => {
                if let Some(best) = candidates.pop() {
                    // Splitting a leaf: removes 1 leaf, adds 2 = net +1
                    *n_leaves += 1;
                    vec![best]
                } else {
                    vec![]
                }
            }
        }
    }

    /// Advance to the next iteration (depth-wise only).
    ///
    /// Called after all children have been added for the current iteration.
    pub fn advance(&mut self) {
        match self {
            GrowthState::DepthWise {
                current_depth,
                current_level,
                next_level,
                ..
            } => {
                std::mem::swap(current_level, next_level);
                next_level.clear();
                *current_depth += 1;
            }
            GrowthState::LeafWise { .. } => {
                // No-op for leaf-wise (handled per-node)
            }
        }
    }

    /// Add a candidate node for expansion.
    ///
    /// For depth-wise, adds to next_level.
    /// For leaf-wise, adds to the priority queue.
    pub fn push(&mut self, candidate: NodeCandidate) {
        match self {
            GrowthState::DepthWise { next_level, .. } => {
                next_level.push(candidate);
            }
            GrowthState::LeafWise { candidates, .. } => {
                candidates.push(candidate);
            }
        }
    }

    /// Initialize with root candidate.
    pub fn push_root(&mut self, candidate: NodeCandidate) {
        match self {
            GrowthState::DepthWise { current_level, .. } => {
                current_level.push(candidate);
            }
            GrowthState::LeafWise { candidates, .. } => {
                candidates.push(candidate);
            }
        }
    }

    /// Get current number of leaves.
    pub fn n_leaves(&self) -> u32 {
        match self {
            GrowthState::DepthWise { .. } => 0, // Not tracked for depth-wise
            GrowthState::LeafWise { n_leaves, .. } => *n_leaves,
        }
    }
}

// ============================================================================
// NodeCandidate
// ============================================================================

/// A node candidate for expansion.
///
/// Contains all information needed to decide whether to expand a node
/// and to perform the expansion.
#[derive(Clone, Debug)]
pub struct NodeCandidate {
    /// Node ID in the partitioner.
    pub node: NodeId,
    /// Tree builder node ID.
    pub tree_node: NodeId,
    /// Current depth.
    pub depth: u16,
    /// Best split found for this node.
    pub split: SplitInfo,
    /// Gradient sum.
    pub grad_sum: f64,
    /// Hessian sum.
    pub hess_sum: f64,
    /// Row count.
    pub row_count: u32,
}

impl NodeCandidate {
    /// Create a new candidate.
    #[inline]
    pub fn new(
        node: NodeId,
        tree_node: NodeId,
        depth: u16,
        split: SplitInfo,
        grad_sum: f64,
        hess_sum: f64,
        row_count: u32,
    ) -> Self {
        Self {
            node,
            tree_node,
            depth,
            split,
            grad_sum,
            hess_sum,
            row_count,
        }
    }

    /// Check if this candidate has a valid split.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.split.is_valid()
    }

    /// Get the split gain.
    #[inline]
    pub fn gain(&self) -> f32 {
        self.split.gain
    }
}

// Implement ordering by gain for max-heap behavior.
// Higher gain = higher priority.
impl PartialEq for NodeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.split.gain == other.split.gain && self.node == other.node
    }
}

impl Eq for NodeCandidate {}

impl PartialOrd for NodeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order by gain (higher = better), break ties by node_id
        self.split
            .gain
            .partial_cmp(&other.split.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(node: NodeId, gain: f32) -> NodeCandidate {
        NodeCandidate {
            node,
            tree_node: node,
            depth: 0,
            split: SplitInfo::numerical(0, 1, gain, false),
            grad_sum: 0.0,
            hess_sum: 1.0,
            row_count: 10,
        }
    }

    #[test]
    fn test_growth_strategy_default() {
        let strategy = GrowthStrategy::default();
        assert!(matches!(
            strategy,
            GrowthStrategy::DepthWise { max_depth: 6 }
        ));
    }

    #[test]
    fn test_depth_wise_init() {
        let state = GrowthStrategy::DepthWise { max_depth: 3 }.init();

        match state {
            GrowthState::DepthWise {
                max_depth,
                current_depth,
                current_level,
                next_level,
            } => {
                assert_eq!(max_depth, 3);
                assert_eq!(current_depth, 0);
                assert!(current_level.is_empty());
                assert!(next_level.is_empty());
            }
            _ => panic!("Expected DepthWise state"),
        }
    }

    #[test]
    fn test_leaf_wise_init() {
        let state = GrowthStrategy::LeafWise { max_leaves: 31 }.init();

        match state {
            GrowthState::LeafWise {
                max_leaves,
                n_leaves,
                candidates,
            } => {
                assert_eq!(max_leaves, 31);
                assert_eq!(n_leaves, 1);
                assert!(candidates.is_empty());
            }
            _ => panic!("Expected LeafWise state"),
        }
    }

    #[test]
    fn test_depth_wise_flow() {
        let mut state = GrowthStrategy::DepthWise { max_depth: 2 }.init();

        // Push root
        state.push_root(make_candidate(0, 1.0));
        assert!(state.should_continue());

        // Pop root
        let nodes = state.pop_next();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node, 0);

        // Push children to next level
        state.push(make_candidate(1, 0.5));
        state.push(make_candidate(2, 0.8));

        // Advance to next depth
        state.advance();
        assert!(state.should_continue());

        // Pop children
        let nodes = state.pop_next();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_leaf_wise_max_heap() {
        let mut state = GrowthStrategy::LeafWise { max_leaves: 10 }.init();

        // Push candidates
        state.push_root(make_candidate(0, 0.5));
        state.push(make_candidate(1, 0.8));
        state.push(make_candidate(2, 0.3));

        // Should pop highest gain first
        let nodes = state.pop_next();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node, 1);
        assert!((nodes[0].gain() - 0.8).abs() < 1e-6);

        let nodes = state.pop_next();
        assert_eq!(nodes[0].node, 0);

        let nodes = state.pop_next();
        assert_eq!(nodes[0].node, 2);
    }

    #[test]
    fn test_depth_wise_should_continue() {
        let mut state = GrowthStrategy::DepthWise { max_depth: 2 }.init();

        // Empty - should not continue
        assert!(!state.should_continue());

        // Add node
        state.push_root(make_candidate(0, 1.0));
        assert!(state.should_continue());

        // Simulate advancing through depths
        state.pop_next();
        state.push(make_candidate(1, 0.5));
        state.advance();
        assert!(state.should_continue()); // depth 1

        state.pop_next();
        state.push(make_candidate(2, 0.3));
        state.advance();
        // At depth == max_depth, we stop expanding (leaves can be at depth=max_depth).
        assert!(!state.should_continue());
    }

    #[test]
    fn test_leaf_wise_should_continue() {
        let mut state = GrowthStrategy::LeafWise { max_leaves: 3 }.init();

        // Empty - should not continue
        assert!(!state.should_continue());

        // Add candidate
        state.push_root(make_candidate(0, 1.0));
        assert!(state.should_continue()); // 1 < 3 leaves

        // Pop (n_leaves becomes 2)
        state.pop_next();
        state.push(make_candidate(1, 0.5));
        assert!(state.should_continue()); // 2 < 3 leaves

        // Pop (n_leaves becomes 3)
        state.pop_next();
        state.push(make_candidate(2, 0.3));
        assert!(!state.should_continue()); // 3 >= 3 leaves
    }

    #[test]
    fn test_candidate_ordering() {
        let c1 = make_candidate(0, 0.5);
        let c2 = make_candidate(1, 0.8);
        let c3 = make_candidate(2, 0.3);

        // Higher gain should be greater
        assert!(c2 > c1);
        assert!(c1 > c3);
        assert!(c2 > c3);
    }
}
