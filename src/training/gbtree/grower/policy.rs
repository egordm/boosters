//! Growth policies for tree building.
//!
//! This module defines how trees are grown during training:
//!
//! - **Depth-wise** (XGBoost style): Expand all nodes at each level before going deeper
//! - **Leaf-wise** (LightGBM style): Always expand the leaf with highest gain
//!
//! # Usage
//!
//! For compile-time known strategies, use the policy types directly:
//!
//! ```ignore
//! use booste_rs::training::DepthWisePolicy;
//!
//! let policy = DepthWisePolicy { max_depth: 6 };
//! ```
//!
//! For runtime selection, use [`GrowthStrategy`]:
//!
//! ```ignore
//! use booste_rs::training::GrowthStrategy;
//!
//! let strategy = GrowthStrategy::LeafWise { max_leaves: 31 };
//! ```

use super::building::{BuildingTree, NodeCandidate};

// ============================================================================
// GrowthPolicy trait
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

    /// Advance to the next iteration.
    ///
    /// Called after all children have been added for the current iteration.
    /// For depth-wise, this swaps current/next level buffers.
    /// For leaf-wise, this is typically a no-op.
    fn advance(&self, _state: &mut Self::State) {
        // Default: no-op (leaf-wise doesn't need level management)
    }
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
/// use booste_rs::training::DepthWisePolicy;
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
    pub current_depth: u32,
    /// Nodes at current depth waiting to be expanded
    pub current_level: Vec<u32>,
    /// Nodes generated for next level
    pub next_level: Vec<u32>,
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

        to_expand
    }

    fn should_continue(&self, state: &Self::State, _tree: &BuildingTree) -> bool {
        state.current_depth <= self.max_depth && !state.current_level.is_empty()
    }

    fn advance(&self, state: &mut Self::State) {
        // Move to next level - called AFTER all children have been added
        state.current_level.clear();
        std::mem::swap(&mut state.current_level, &mut state.next_level);
        state.current_depth += 1;
    }
}

impl GrowthState for DepthWiseState {
    fn add_children(&mut self, left: u32, right: u32, _depth: u32) {
        self.next_level.push(left);
        self.next_level.push(right);
    }
}

// ============================================================================
// LeafWisePolicy (LightGBM style)
// ============================================================================

/// Leaf-wise growth policy (LightGBM style).
///
/// Always expands the leaf with highest gain, regardless of depth.
/// This is a **best-first search** strategy that produces asymmetric trees
/// and typically achieves lower loss with fewer leaves than depth-wise.
///
/// # Algorithm
///
/// Uses a **priority queue** (max-heap) to track candidate leaves by gain:
/// 1. Start with root in priority queue
/// 2. Pop highest-gain candidate
/// 3. If valid and under max_leaves limit, expand it
/// 4. Push children to priority queue
/// 5. Repeat until max_leaves reached or no valid candidates
///
/// # Example
///
/// ```
/// use booste_rs::training::LeafWisePolicy;
///
/// let policy = LeafWisePolicy { max_leaves: 31 };
/// ```
#[derive(Debug, Clone)]
pub struct LeafWisePolicy {
    /// Maximum number of leaves in the tree
    pub max_leaves: u32,
}

/// State for leaf-wise growth.
///
/// Maintains a priority queue of candidate leaves ordered by gain.
#[derive(Debug)]
pub struct LeafWiseState {
    /// Priority queue of candidate leaves (max-heap by gain).
    /// Uses BinaryHeap which is a max-heap - highest gain at top.
    pub candidates: std::collections::BinaryHeap<LeafCandidate>,
    /// Current number of leaves in the tree
    pub num_leaves: u32,
    /// Whether we've processed at least one batch of candidates.
    /// Needed because priority queue is empty until first select_nodes call.
    pub started: bool,
    /// Number of pending children that haven't been added to the queue yet.
    /// This tracks children created via add_children that will be processed
    /// in the next select_nodes call.
    pub pending_children: u32,
}

/// A leaf candidate for expansion in leaf-wise growth.
///
/// Wraps node ID and gain to enable priority queue ordering.
/// Implements `Ord` to order by gain (highest first).
#[derive(Debug, Clone)]
pub struct LeafCandidate {
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

impl GrowthPolicy for LeafWisePolicy {
    type State = LeafWiseState;

    fn init(&self) -> Self::State {
        LeafWiseState {
            candidates: std::collections::BinaryHeap::new(),
            num_leaves: 1, // Start with root as single leaf
            started: false,
            pending_children: 0,
        }
    }

    fn select_nodes(&self, state: &mut Self::State, candidates: &[NodeCandidate]) -> Vec<u32> {
        // Mark that we've started processing
        state.started = true;

        // Clear pending count - the candidates passed in represent the pending children
        state.pending_children = 0;

        // Update priority queue with new candidates
        for cand in candidates {
            if cand.is_valid() {
                state
                    .candidates
                    .push(LeafCandidate::new(cand.node_id, cand.gain()));
            }
        }

        // Pop the single best candidate (leaf-wise expands one at a time)
        if let Some(best) = state.candidates.pop() {
            // Splitting a leaf: removes 1 leaf, adds 2 = net +1
            state.num_leaves += 1;
            vec![best.node_id]
        } else {
            vec![]
        }
    }

    fn should_continue(&self, state: &Self::State, _tree: &BuildingTree) -> bool {
        // Continue if we haven't reached max leaves AND either:
        // 1. We haven't started yet (need to process root)
        // 2. There are candidates in the priority queue
        // 3. There are pending children that will be processed in the next iteration
        let has_work =
            !state.started || !state.candidates.is_empty() || state.pending_children > 0;
        state.num_leaves < self.max_leaves && has_work
    }
}

impl GrowthState for LeafWiseState {
    fn add_children(&mut self, _left: u32, _right: u32, _depth: u32) {
        // Track that children were created and will be available in next iteration.
        // This ensures should_continue returns true when there's pending work.
        self.pending_children += 2;
    }
}

// ============================================================================
// GrowthStrategy enum (runtime selection)
// ============================================================================

/// Runtime-selectable growth strategy.
///
/// Use this when the growth strategy is determined at runtime (e.g., from config).
/// For compile-time known strategies, use the policy types directly for better performance.
///
/// # Example
///
/// ```
/// use booste_rs::training::GrowthStrategy;
///
/// // From configuration
/// let strategy = GrowthStrategy::LeafWise { max_leaves: 31 };
/// ```
#[derive(Debug, Clone, Copy)]
pub enum GrowthStrategy {
    /// Depth-wise growth (XGBoost style) - expand all nodes at each level
    DepthWise {
        /// Maximum tree depth
        max_depth: u32,
    },
    /// Leaf-wise growth (LightGBM style) - expand best gain leaf
    LeafWise {
        /// Maximum number of leaves
        max_leaves: u32,
    },
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
}
