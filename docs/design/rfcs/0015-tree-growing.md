# RFC-0015: Tree Growing Strategies

- **Status**: Draft
- **Created**: 2024-11-30
- **Updated**: 2024-11-30
- **Depends on**: RFC-0012 (Histograms), RFC-0013 (Split Finding), RFC-0014 (Row Partitioning)
- **Scope**: Tree building strategies (depth-wise, leaf-wise) and training loop coordination

## Summary

This RFC defines how trees are grown during gradient boosting:

1. **Growth strategies**: Depth-wise (XGBoost) vs leaf-wise (LightGBM)
2. **Training loop**: Coordinates histogram building, split finding, and partitioning
3. **Stopping criteria**: Max depth, max leaves, min gain, early stopping

## Motivation

The tree growth strategy determines:

- **Tree shape**: Balanced (depth-wise) vs asymmetric (leaf-wise)
- **Overfitting risk**: Leaf-wise can overfit on small data
- **Training speed**: Leaf-wise often faster (fewer nodes for same loss reduction)
- **Implementation complexity**: Leaf-wise needs priority queue

Both strategies are valuable and should be supported with a common interface.

## Design

### Overview

```
Tree growth strategies:
━━━━━━━━━━━━━━━━━━━━━━

Depth-wise (XGBoost style):       Leaf-wise (LightGBM style):
Expand all nodes at each level    Expand best leaf globally

Level 0:    [root]                [root]
                ↓                     ↓
Level 1:  [L1]    [R1]            [L1]    [R1]   ← Expand L1 (best gain)
              ↓      ↓                 ↓
Level 2: [LL] [LR] [RL] [RR]     [LL] [LR] [R1]  ← Expand LR (best gain)
                                       ↓
                                 [LL] [LRL] [LRR] [R1]  ← Expand LRL...

Depth-wise: O(2^depth) nodes      Leaf-wise: Exactly max_leaves nodes
Balanced trees                    Unbalanced, deeper paths
```

### Growth Policy Trait

```rust
/// Policy for selecting which nodes to expand
pub trait GrowthPolicy {
    /// State maintained across expansions
    type State;
    
    /// Initialize state for a new tree
    fn init(&self) -> Self::State;
    
    /// Select nodes to expand given current candidates
    /// Returns nodes to expand this iteration
    fn select_nodes(&self, state: &mut Self::State, candidates: &[NodeCandidate]) -> Vec<u32>;
    
    /// Check if we should continue growing
    fn should_continue(&self, state: &Self::State, tree: &BuildingTree) -> bool;
}

/// Candidate node for expansion
pub struct NodeCandidate {
    /// Node index in building tree
    pub node_id: u32,
    /// Best split found for this node
    pub split: SplitInfo,
    /// Depth of this node
    pub depth: u32,
    /// Number of samples in this node
    pub num_samples: u32,
}

impl NodeCandidate {
    pub fn gain(&self) -> f32 {
        self.split.gain
    }
}
```

### Depth-wise Policy

```rust
/// Depth-wise growth: expand all nodes at current level before going deeper
pub struct DepthWisePolicy {
    /// Maximum tree depth
    pub max_depth: u32,
}

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
            current_level: vec![0],  // Start with root
            next_level: Vec::new(),
        }
    }
    
    fn select_nodes(&self, state: &mut Self::State, candidates: &[NodeCandidate]) -> Vec<u32> {
        // Return all nodes at current level that have valid splits
        let to_expand: Vec<u32> = state.current_level.iter()
            .filter(|&&node_id| {
                candidates.iter()
                    .find(|c| c.node_id == node_id)
                    .map(|c| c.split.is_valid())
                    .unwrap_or(false)
            })
            .copied()
            .collect();
        
        // Prepare for next level
        state.current_level.clear();
        std::mem::swap(&mut state.current_level, &mut state.next_level);
        state.current_depth += 1;
        
        to_expand
    }
    
    fn should_continue(&self, state: &Self::State, _tree: &BuildingTree) -> bool {
        state.current_depth < self.max_depth && !state.current_level.is_empty()
    }
}
```

### Leaf-wise Policy

```rust
/// Leaf-wise growth: always expand the leaf with highest gain
pub struct LeafWisePolicy {
    /// Maximum number of leaves
    pub max_leaves: u32,
}

pub struct LeafWiseState {
    /// Priority queue of candidate leaves (max-heap by gain)
    candidates: BinaryHeap<LeafCandidate>,
    /// Current number of leaves
    num_leaves: u32,
}

#[derive(Clone)]
struct LeafCandidate {
    node_id: u32,
    gain: f32,
}

impl Ord for LeafCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain.partial_cmp(&other.gain).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for LeafCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl GrowthPolicy for LeafWisePolicy {
    type State = LeafWiseState;
    
    fn init(&self) -> Self::State {
        LeafWiseState {
            candidates: BinaryHeap::new(),
            num_leaves: 1,  // Start with root as single leaf
        }
    }
    
    fn select_nodes(&self, state: &mut Self::State, candidates: &[NodeCandidate]) -> Vec<u32> {
        // Update priority queue with current candidates
        for cand in candidates {
            if cand.split.is_valid() {
                state.candidates.push(LeafCandidate {
                    node_id: cand.node_id,
                    gain: cand.gain(),
                });
            }
        }
        
        // Pop the best candidate
        if let Some(best) = state.candidates.pop() {
            state.num_leaves += 1;  // Split creates net +1 leaf
            vec![best.node_id]
        } else {
            vec![]
        }
    }
    
    fn should_continue(&self, state: &Self::State, _tree: &BuildingTree) -> bool {
        state.num_leaves < self.max_leaves && !state.candidates.is_empty()
    }
}
```

### Building Tree Structure

```rust
/// Tree being constructed (mutable)
pub struct BuildingTree {
    /// Nodes in the tree
    nodes: Vec<BuildingNode>,
    /// Leaf values (for completed leaves)
    leaf_values: Vec<f32>,
    /// Current number of leaves
    num_leaves: u32,
    /// Maximum depth reached
    max_depth: u32,
}

pub struct BuildingNode {
    /// Split information (None if leaf)
    pub split: Option<SplitInfo>,
    /// Left child index (u32::MAX if leaf)
    pub left: u32,
    /// Right child index (u32::MAX if leaf)
    pub right: u32,
    /// Parent index (u32::MAX if root)
    pub parent: u32,
    /// Depth in tree
    pub depth: u32,
    /// Leaf value (if this is a leaf)
    pub leaf_value: f32,
    /// Whether this node is a leaf
    pub is_leaf: bool,
}

impl BuildingTree {
    pub fn new() -> Self {
        Self {
            nodes: vec![BuildingNode::new_leaf(0, u32::MAX, 0)],
            leaf_values: Vec::new(),
            num_leaves: 1,
            max_depth: 0,
        }
    }
    
    /// Expand a leaf into a split node with two children
    pub fn expand(&mut self, node_id: u32, split: SplitInfo) -> (u32, u32) {
        let node = &self.nodes[node_id as usize];
        let depth = node.depth;
        
        let left_id = self.nodes.len() as u32;
        let right_id = left_id + 1;
        
        // Create left and right children
        self.nodes.push(BuildingNode::new_leaf(
            split.weight_left,
            node_id,
            depth + 1,
        ));
        self.nodes.push(BuildingNode::new_leaf(
            split.weight_right,
            node_id,
            depth + 1,
        ));
        
        // Convert parent to split node
        let node = &mut self.nodes[node_id as usize];
        node.split = Some(split);
        node.left = left_id;
        node.right = right_id;
        node.is_leaf = false;
        
        self.num_leaves += 1;  // net +1 (remove 1, add 2)
        self.max_depth = self.max_depth.max(depth + 1);
        
        (left_id, right_id)
    }
    
    /// Get all current leaf nodes
    pub fn leaves(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.nodes.len() as u32).filter(|&i| self.nodes[i as usize].is_leaf)
    }
    
    /// Convert to immutable SoATree for inference
    pub fn freeze(&self) -> SoATreeStorage<ScalarLeaf> {
        // Convert BuildingTree to inference-optimized format
        todo!("Convert to SoATreeStorage")
    }
}

impl BuildingNode {
    fn new_leaf(value: f32, parent: u32, depth: u32) -> Self {
        Self {
            split: None,
            left: u32::MAX,
            right: u32::MAX,
            parent,
            depth,
            leaf_value: value,
            is_leaf: true,
        }
    }
}
```

### Tree Builder (Training Loop)

```rust
/// Coordinates tree building with a growth policy
pub struct TreeBuilder<G: GrowthPolicy> {
    /// Growth policy (depth-wise or leaf-wise)
    policy: G,
    /// Histogram builder
    hist_builder: HistogramBuilder,
    /// Split finder
    split_finder: GreedySplitFinder,
    /// Training parameters
    params: TreeParams,
}

/// Training parameters
pub struct TreeParams {
    /// Gain computation parameters
    pub gain: GainParams,
    /// Maximum depth (for depth-wise, or absolute limit for leaf-wise)
    pub max_depth: u32,
    /// Maximum leaves (for leaf-wise)
    pub max_leaves: u32,
    /// Minimum samples to split
    pub min_samples_split: u32,
    /// Minimum samples per leaf
    pub min_samples_leaf: u32,
    /// Learning rate for leaf weights
    pub learning_rate: f32,
}

impl<G: GrowthPolicy> TreeBuilder<G> {
    /// Build a single tree
    pub fn build_tree(
        &mut self,
        index: &GHistIndexMatrix,
        grads: &GradientBuffer,
        partitioner: &mut RowPartitioner,
    ) -> BuildingTree {
        let mut tree = BuildingTree::new();
        let mut state = self.policy.init();
        
        // Histogram storage per node (reuse across iterations)
        let mut histograms: HashMap<u32, NodeHistogram> = HashMap::new();
        
        // Build root histogram
        let root_hist = self.build_histogram(0, index, grads, partitioner);
        histograms.insert(0, root_hist);
        
        // Find initial split for root
        let root_split = self.split_finder.find_best_split(
            &histograms[&0],
            &index.cuts,
            &self.params.gain,
        );
        
        let mut candidates = vec![NodeCandidate {
            node_id: 0,
            split: root_split,
            depth: 0,
            num_samples: partitioner.node_size(0),
        }];
        
        // Main growth loop
        while self.policy.should_continue(&state, &tree) {
            let nodes_to_expand = self.policy.select_nodes(&mut state, &candidates);
            
            if nodes_to_expand.is_empty() {
                break;
            }
            
            // Expand selected nodes
            let mut new_candidates = Vec::new();
            
            for node_id in nodes_to_expand {
                let candidate = candidates.iter().find(|c| c.node_id == node_id).unwrap();
                
                if !self.should_split(candidate) {
                    continue;
                }
                
                // Apply split
                let split = candidate.split.clone();
                let (left_id, right_id) = tree.expand(node_id, split.clone());
                let (left_part, right_part) = partitioner.apply_split(node_id, &split, index);
                
                // Build histograms for children (use subtraction optimization)
                let (left_hist, right_hist) = self.build_child_histograms(
                    node_id, left_part, right_part,
                    index, grads, partitioner,
                    &histograms,
                );
                
                // Find splits for new nodes
                let left_split = self.split_finder.find_best_split(&left_hist, &index.cuts, &self.params.gain);
                let right_split = self.split_finder.find_best_split(&right_hist, &index.cuts, &self.params.gain);
                
                histograms.insert(left_id, left_hist);
                histograms.insert(right_id, right_hist);
                
                new_candidates.push(NodeCandidate {
                    node_id: left_id,
                    split: left_split,
                    depth: candidate.depth + 1,
                    num_samples: partitioner.node_size(left_part),
                });
                new_candidates.push(NodeCandidate {
                    node_id: right_id,
                    split: right_split,
                    depth: candidate.depth + 1,
                    num_samples: partitioner.node_size(right_part),
                });
            }
            
            // Remove expanded nodes from candidates, add new ones
            candidates.retain(|c| !nodes_to_expand.contains(&c.node_id));
            candidates.extend(new_candidates);
        }
        
        // Apply learning rate to leaf weights
        for node in &mut tree.nodes {
            if node.is_leaf {
                node.leaf_value *= self.params.learning_rate;
            }
        }
        
        tree
    }
    
    fn should_split(&self, candidate: &NodeCandidate) -> bool {
        candidate.split.is_valid()
            && candidate.depth < self.params.max_depth
            && candidate.num_samples >= self.params.min_samples_split
    }
    
    fn build_histogram(
        &mut self,
        node: u32,
        index: &GHistIndexMatrix,
        grads: &GradientBuffer,
        partitioner: &RowPartitioner,
    ) -> NodeHistogram {
        let mut hist = NodeHistogram::new(&index.cuts);
        let rows = partitioner.node_rows(node);
        self.hist_builder.build(&mut hist, index, grads, rows);
        hist
    }
    
    fn build_child_histograms(
        &mut self,
        parent: u32,
        left_node: u32,
        right_node: u32,
        index: &GHistIndexMatrix,
        grads: &GradientBuffer,
        partitioner: &RowPartitioner,
        histograms: &HashMap<u32, NodeHistogram>,
    ) -> (NodeHistogram, NodeHistogram) {
        let parent_hist = &histograms[&parent];
        
        let left_size = partitioner.node_size(left_node);
        let right_size = partitioner.node_size(right_node);
        
        // Build smaller child, derive larger via subtraction
        if left_size <= right_size {
            let left_hist = self.build_histogram(left_node, index, grads, partitioner);
            let mut right_hist = NodeHistogram::new(&index.cuts);
            HistogramSubtractor::compute_sibling(parent_hist, &left_hist, &mut right_hist);
            (left_hist, right_hist)
        } else {
            let right_hist = self.build_histogram(right_node, index, grads, partitioner);
            let mut left_hist = NodeHistogram::new(&index.cuts);
            HistogramSubtractor::compute_sibling(parent_hist, &right_hist, &mut left_hist);
            (left_hist, right_hist)
        }
    }
}
```

### Gradient Boosting Trainer

```rust
/// Full gradient boosting trainer
pub struct GBTreeTrainer<G: GrowthPolicy> {
    /// Tree builder
    tree_builder: TreeBuilder<G>,
    /// Number of boosting rounds
    num_rounds: u32,
    /// Loss function
    loss: Box<dyn Loss>,
    /// Early stopping configuration
    early_stopping: Option<EarlyStoppingConfig>,
}

pub struct EarlyStoppingConfig {
    /// Number of rounds without improvement to stop
    pub patience: u32,
    /// Evaluation metric
    pub metric: Box<dyn Metric>,
    /// Minimum improvement to count as progress
    pub min_delta: f32,
}

impl<G: GrowthPolicy> GBTreeTrainer<G> {
    /// Train a gradient boosted ensemble
    pub fn train(
        &mut self,
        data: &GHistIndexMatrix,
        labels: &[f32],
        eval_set: Option<(&GHistIndexMatrix, &[f32])>,
    ) -> SoAForest<ScalarLeaf> {
        let num_rows = data.num_rows();
        
        // Initialize predictions (base score)
        let base_score = self.loss.base_score(labels);
        let mut predictions = vec![base_score; num_rows as usize];
        
        // Gradient buffer
        let mut grads = GradientBuffer::new(num_rows);
        
        // Row partitioner (reused per tree)
        let mut partitioner = RowPartitioner::new(num_rows);
        
        // Trees
        let mut trees = Vec::with_capacity(self.num_rounds as usize);
        
        // Early stopping state
        let mut best_score = f32::INFINITY;
        let mut rounds_without_improvement = 0;
        
        for round in 0..self.num_rounds {
            // Compute gradients
            self.loss.gradient(labels, &predictions, &mut grads);
            
            // Reset partitioner for new tree
            partitioner.reset(num_rows);
            
            // Build tree
            let tree = self.tree_builder.build_tree(data, &grads, &mut partitioner);
            
            // Update predictions
            self.update_predictions(&tree, data, &mut predictions);
            
            // Evaluation and early stopping
            if let Some(ref config) = self.early_stopping {
                if let Some((eval_data, eval_labels)) = eval_set {
                    let score = self.evaluate(&trees, &tree, eval_data, eval_labels, &*config.metric);
                    
                    if score < best_score - config.min_delta {
                        best_score = score;
                        rounds_without_improvement = 0;
                    } else {
                        rounds_without_improvement += 1;
                        if rounds_without_improvement >= config.patience {
                            break;
                        }
                    }
                }
            }
            
            trees.push(tree);
        }
        
        // Convert to inference format
        self.freeze_forest(trees, base_score)
    }
    
    fn update_predictions(
        &self,
        tree: &BuildingTree,
        data: &GHistIndexMatrix,
        predictions: &mut [f32],
    ) {
        // Traverse tree for each row, add leaf value to prediction
        for row in 0..data.num_rows() {
            let leaf_value = self.predict_row(tree, data, row);
            predictions[row as usize] += leaf_value;
        }
    }
    
    fn predict_row(&self, tree: &BuildingTree, data: &GHistIndexMatrix, row: u32) -> f32 {
        let mut node_id = 0u32;
        
        while !tree.nodes[node_id as usize].is_leaf {
            let node = &tree.nodes[node_id as usize];
            let split = node.split.as_ref().unwrap();
            let bin = data.get(row, split.feature);
            
            let goes_left = if bin == 0 {
                split.default_left
            } else if split.is_categorical {
                split.categories_left.contains(&(bin as u32))
            } else {
                let threshold_bin = data.cuts.bin_value(split.feature, split.threshold);
                bin <= threshold_bin
            };
            
            node_id = if goes_left { node.left } else { node.right };
        }
        
        tree.nodes[node_id as usize].leaf_value
    }
    
    fn freeze_forest(&self, trees: Vec<BuildingTree>, base_score: f32) -> SoAForest<ScalarLeaf> {
        // Convert BuildingTree list to SoAForest
        todo!("Convert to SoAForest")
    }
}
```

### Enum-based Strategy Selection

```rust
/// Runtime-selectable growth strategy
pub enum GrowthStrategy {
    DepthWise { max_depth: u32 },
    LeafWise { max_leaves: u32 },
}

impl GrowthStrategy {
    /// Create builder with this strategy
    pub fn create_builder(&self, params: TreeParams) -> Box<dyn TreeBuilderTrait> {
        match self {
            GrowthStrategy::DepthWise { max_depth } => {
                Box::new(TreeBuilder::new(
                    DepthWisePolicy { max_depth: *max_depth },
                    params,
                ))
            }
            GrowthStrategy::LeafWise { max_leaves } => {
                Box::new(TreeBuilder::new(
                    LeafWisePolicy { max_leaves: *max_leaves },
                    params,
                ))
            }
        }
    }
}
```

## Design Decisions

### DD-1: Trait-based Growth Policy

**Context**: Need to support multiple growth strategies with different behavior.

**Options considered**:

1. **Enum with match**: Single TreeBuilder with enum variant
2. **Trait object**: `Box<dyn GrowthPolicy>` for runtime polymorphism
3. **Generic parameter**: `TreeBuilder<G: GrowthPolicy>` for static dispatch

**Decision**: Generic parameter with trait for zero-cost abstraction.

**Rationale**:

- Growth policy methods called in hot loop
- Static dispatch enables inlining
- Enum wrapper available for runtime selection when needed

### DD-2: Histogram Caching Strategy

**Context**: Histograms can be reused or recomputed.

**Options considered**:

1. **Cache all**: Store all node histograms
2. **Cache none**: Recompute as needed
3. **Cache active**: Store only active level/candidates

**Decision**: Cache active nodes only, remove on expansion.

**Rationale**:

- Full caching uses O(nodes × features × bins) memory
- Active-only caching uses O(active_nodes × features × bins)
- For depth-wise: active = one level
- For leaf-wise: active = frontier

### DD-3: Depth-wise Parallelism

**Context**: How to parallelize depth-wise tree building.

**Options considered**:

1. **Per-node parallel**: Build histograms for all nodes at level in parallel
2. **Per-feature parallel**: Within a node, parallelize over features
3. **Nested**: Both levels of parallelism

**Decision**: Per-node parallel for depth-wise, per-feature for leaf-wise.

**Rationale**:

- Depth-wise naturally has many independent nodes at each level
- Leaf-wise expands one node at a time, use per-feature parallelism
- Avoid nested parallelism complexity

## Integration

| Component | Integration Point | Notes |
|-----------|-------------------|-------|
| RFC-0011 (Quantization) | `QuantizedMatrix` | Input data |
| RFC-0012 (Histograms) | `HistogramBuilder` | Build histograms |
| RFC-0013 (Split Finding) | `SplitFinder` | Find best splits |
| RFC-0014 (Row Partition) | `RowPartitioner` | Track row assignments |
| RFC-0002 (Tree Storage) | `SoATreeStorage` | Output format |

### Integration with Existing Code

- **`src/trees/storage.rs`**: `SoATreeStorage` is the target inference format
- **`src/forest/storage.rs`**: `SoAForest` aggregates trees for inference
- **`src/training/mod.rs`**: Export trainer alongside existing `Loss`, `GradientBuffer`, `EarlyStopping`
- **`src/training/callback.rs`**: Reuse `EarlyStopping` callback for early stopping logic
- **`src/training/logger.rs`**: Reuse `TrainingLogger` for training output
- **New module**: `src/training/gbtree.rs` for `TreeBuilder`, `GBTreeTrainer`, growth policies

## Open Questions

1. **Oblivious trees**: CatBoost-style symmetric trees. **Low priority** — nice-to-have but not essential for accuracy/performance. Can add as `ObliviousPolicy` later.

2. **Histogram memory reuse**: **Yes** — use a pool allocator if histogram allocation shows up in profiling.

3. **Incremental training**: **Low priority** — if easy to implement during initial design, do it. Otherwise defer.

## Future Work

- [ ] Oblivious tree growth strategy
- [ ] Histogram memory pool
- [ ] Incremental/continued training
- [ ] Multi-threaded tree building

## References

- [XGBoost updater_quantile_hist.cc](https://github.com/dmlc/xgboost/blob/master/src/tree/updater_quantile_hist.cc)
- [LightGBM serial_tree_learner.cpp](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/serial_tree_learner.cpp)
- [Feature Overview](../FEATURE_OVERVIEW.md) - Priority and design context

## Changelog

- 2024-11-30: Initial draft
