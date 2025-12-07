//! Growth policies for tree building.
//!
//! This module defines how trees are grown during training:
//!
//! - **Depth-wise** (XGBoost style): Expand all nodes at each level before going deeper
//! - **Leaf-wise** (LightGBM style): Always expand the leaf with highest gain
//!
//! # Usage
//!
//! Use [`GrowthStrategy`] to select the growth mode:
//!
//! ```
//! use booste_rs::training::GrowthStrategy;
//!
//! // Depth-wise (XGBoost style)
//! let strategy = GrowthStrategy::DepthWise { max_depth: 6 };
//!
//! // Leaf-wise (LightGBM style)
//! let strategy = GrowthStrategy::LeafWise { max_leaves: 31 };
//! ```

use std::collections::BinaryHeap;

use super::building::NodeCandidate;

// ============================================================================
// GrowthStrategy enum
// ============================================================================

/// Growth strategy for tree building.
///
/// Controls how the tree is grown during training:
/// - **DepthWise**: Expand all nodes at each level before going deeper (XGBoost style)
/// - **LeafWise**: Always expand the leaf with highest gain (LightGBM style)
///
/// # Example
///
/// ```
/// use booste_rs::training::GrowthStrategy;
///
/// let strategy = GrowthStrategy::LeafWise { max_leaves: 31 };
/// ```
#[derive(Debug, Clone, Copy)]
pub enum GrowthStrategy {
    /// Depth-wise growth (XGBoost style) - expand all nodes at each level
    DepthWise {
        /// Maximum tree depth (root = depth 0)
        max_depth: u32,
    },
    /// Leaf-wise growth (LightGBM style) - expand best gain leaf
    LeafWise {
        /// Maximum number of leaves
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

    /// Initialize growth state for this strategy.
    pub(crate) fn init(&self) -> GrowthState {
        match self {
            GrowthStrategy::DepthWise { .. } => GrowthState::DepthWise(DepthWiseState {
                current_depth: 0,
                current_level: vec![0], // Start with root
                next_level: Vec::new(),
            }),
            GrowthStrategy::LeafWise { .. } => GrowthState::LeafWise(LeafWiseState {
                candidates: BinaryHeap::new(),
                num_leaves: 1, // Start with root as single leaf
                started: false,
                pending_children: 0,
            }),
        }
    }

    /// Select nodes to expand from current candidates.
    ///
    /// Returns the node IDs to expand this iteration.
    pub(crate) fn select_nodes(
        &self,
        state: &mut GrowthState,
        candidates: &[NodeCandidate],
    ) -> Vec<u32> {
        match (self, state) {
            (GrowthStrategy::DepthWise { .. }, GrowthState::DepthWise(s)) => {
                // Return all nodes at current level that have valid splits
                s.current_level
                    .iter()
                    .filter(|&&node_id| {
                        candidates
                            .iter()
                            .find(|c| c.node_id == node_id)
                            .map(|c| c.is_valid())
                            .unwrap_or(false)
                    })
                    .copied()
                    .collect()
            }
            (GrowthStrategy::LeafWise { .. }, GrowthState::LeafWise(s)) => {
                // Mark that we've started processing
                s.started = true;
                s.pending_children = 0;

                // Update priority queue with new candidates
                for cand in candidates {
                    if cand.is_valid() {
                        s.candidates
                            .push(LeafCandidate::new(cand.node_id, cand.gain()));
                    }
                }

                // Pop the single best candidate (leaf-wise expands one at a time)
                if let Some(best) = s.candidates.pop() {
                    // Splitting a leaf: removes 1 leaf, adds 2 = net +1
                    s.num_leaves += 1;
                    vec![best.node_id]
                } else {
                    vec![]
                }
            }
            _ => panic!("Mismatched GrowthStrategy and GrowthState"),
        }
    }

    /// Check if we should continue growing the tree.
    pub(crate) fn should_continue(&self, state: &GrowthState) -> bool {
        match (self, state) {
            (GrowthStrategy::DepthWise { max_depth }, GrowthState::DepthWise(s)) => {
                s.current_depth <= *max_depth && !s.current_level.is_empty()
            }
            (GrowthStrategy::LeafWise { max_leaves }, GrowthState::LeafWise(s)) => {
                let has_work =
                    !s.started || !s.candidates.is_empty() || s.pending_children > 0;
                s.num_leaves < *max_leaves && has_work
            }
            _ => panic!("Mismatched GrowthStrategy and GrowthState"),
        }
    }

    /// Advance to the next iteration.
    ///
    /// Called after all children have been added for the current iteration.
    pub(crate) fn advance(&self, state: &mut GrowthState) {
        match (self, state) {
            (GrowthStrategy::DepthWise { .. }, GrowthState::DepthWise(s)) => {
                // Move to next level
                s.current_level.clear();
                std::mem::swap(&mut s.current_level, &mut s.next_level);
                s.current_depth += 1;
            }
            (GrowthStrategy::LeafWise { .. }, GrowthState::LeafWise(_)) => {
                // No-op for leaf-wise
            }
            _ => panic!("Mismatched GrowthStrategy and GrowthState"),
        }
    }
}

// ============================================================================
// GrowthState enum
// ============================================================================

/// State maintained during tree growth.
///
/// This is an internal enum that tracks the growth progress.
/// Created via `GrowthStrategy::init()`.
pub(crate) enum GrowthState {
    DepthWise(DepthWiseState),
    LeafWise(LeafWiseState),
}

impl GrowthState {
    /// Register new child nodes created from an expansion.
    pub(crate) fn add_children(&mut self, left: u32, right: u32, depth: u32) {
        match self {
            GrowthState::DepthWise(s) => {
                let _ = depth; // unused for depth-wise
                s.next_level.push(left);
                s.next_level.push(right);
            }
            GrowthState::LeafWise(s) => {
                let _ = (left, right, depth); // unused - we track pending count
                s.pending_children += 2;
            }
        }
    }
}

// ============================================================================
// DepthWiseState
// ============================================================================

/// State for depth-wise growth.
pub(crate) struct DepthWiseState {
    /// Current depth being processed
    pub current_depth: u32,
    /// Nodes at current depth waiting to be expanded
    pub current_level: Vec<u32>,
    /// Nodes generated for next level
    pub next_level: Vec<u32>,
}

// ============================================================================
// LeafWiseState
// ============================================================================

/// State for leaf-wise growth.
pub(crate) struct LeafWiseState {
    /// Priority queue of candidate leaves (max-heap by gain)
    pub candidates: BinaryHeap<LeafCandidate>,
    /// Current number of leaves in the tree
    pub num_leaves: u32,
    /// Whether we've processed at least one batch of candidates
    pub started: bool,
    /// Number of pending children not yet added to the queue
    pub pending_children: u32,
}

// ============================================================================
// LeafCandidate (for leaf-wise priority queue)
// ============================================================================

/// A leaf candidate for expansion in leaf-wise growth.
///
/// Wraps node ID and gain to enable priority queue ordering.
#[derive(Debug, Clone)]
pub(crate) struct LeafCandidate {
    /// Node ID in the building tree
    pub node_id: u32,
    /// Split gain for this node (used for ordering)
    pub gain: f32,
}

impl LeafCandidate {
    /// Create a new leaf candidate.
    pub fn new(node_id: u32, gain: f32) -> Self {
        Self { node_id, gain }
    }
}

// Implement ordering by gain for max-heap behavior.
// Higher gain = higher priority.
impl PartialEq for LeafCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain && self.node_id == other.node_id
    }
}

impl Eq for LeafCandidate {}

impl PartialOrd for LeafCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LeafCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order by gain (higher = better), break ties by node_id
        self.gain
            .partial_cmp(&other.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_growth_strategy_default() {
        let strategy = GrowthStrategy::default();
        assert!(matches!(strategy, GrowthStrategy::DepthWise { max_depth: 6 }));
    }

    #[test]
    fn test_depth_wise_init() {
        let strategy = GrowthStrategy::DepthWise { max_depth: 3 };
        let state = strategy.init();
        
        match state {
            GrowthState::DepthWise(s) => {
                assert_eq!(s.current_depth, 0);
                assert_eq!(s.current_level, vec![0]);
                assert!(s.next_level.is_empty());
            }
            _ => panic!("Expected DepthWiseState"),
        }
    }

    #[test]
    fn test_leaf_wise_init() {
        let strategy = GrowthStrategy::LeafWise { max_leaves: 31 };
        let state = strategy.init();
        
        match state {
            GrowthState::LeafWise(s) => {
                assert_eq!(s.num_leaves, 1);
                assert!(!s.started);
                assert!(s.candidates.is_empty());
            }
            _ => panic!("Expected LeafWiseState"),
        }
    }

    #[test]
    fn test_depth_wise_should_continue() {
        let strategy = GrowthStrategy::DepthWise { max_depth: 2 };
        let mut state = strategy.init();
        
        // Should continue at depth 0
        assert!(strategy.should_continue(&state));
        
        // Simulate advancing through depths
        if let GrowthState::DepthWise(s) = &mut state {
            s.next_level.push(1);
            s.next_level.push(2);
        }
        strategy.advance(&mut state);
        assert!(strategy.should_continue(&state)); // depth 1
        
        if let GrowthState::DepthWise(s) = &mut state {
            s.next_level.push(3);
            s.next_level.push(4);
        }
        strategy.advance(&mut state);
        assert!(strategy.should_continue(&state)); // depth 2
        
        if let GrowthState::DepthWise(s) = &mut state {
            s.next_level.push(5);
        }
        strategy.advance(&mut state);
        assert!(!strategy.should_continue(&state)); // depth 3 > max_depth
    }

    #[test]
    fn test_leaf_wise_should_continue() {
        let strategy = GrowthStrategy::LeafWise { max_leaves: 3 };
        let mut state = strategy.init();
        
        // Should continue (not started yet)
        assert!(strategy.should_continue(&state));
        
        // Simulate starting and adding leaves
        if let GrowthState::LeafWise(s) = &mut state {
            s.started = true;
            s.num_leaves = 2;
            s.candidates.push(LeafCandidate::new(1, 0.5));
        }
        assert!(strategy.should_continue(&state)); // 2 < 3 leaves
        
        if let GrowthState::LeafWise(s) = &mut state {
            s.num_leaves = 3;
            s.candidates.clear();
        }
        assert!(!strategy.should_continue(&state)); // 3 >= 3 leaves
    }

    #[test]
    fn test_add_children_depth_wise() {
        let strategy = GrowthStrategy::DepthWise { max_depth: 3 };
        let mut state = strategy.init();
        
        state.add_children(1, 2, 1);
        
        if let GrowthState::DepthWise(s) = &state {
            assert_eq!(s.next_level, vec![1, 2]);
        } else {
            panic!("Expected DepthWiseState");
        }
    }

    #[test]
    fn test_add_children_leaf_wise() {
        let strategy = GrowthStrategy::LeafWise { max_leaves: 31 };
        let mut state = strategy.init();
        
        state.add_children(1, 2, 1);
        
        if let GrowthState::LeafWise(s) = &state {
            assert_eq!(s.pending_children, 2);
        } else {
            panic!("Expected LeafWiseState");
        }
    }
}
