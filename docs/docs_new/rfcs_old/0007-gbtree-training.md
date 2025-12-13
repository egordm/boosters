```markdown
# RFC-0007: GBTree Training

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: RFC-0001 (Data Matrix), RFC-0002 (Objectives), RFC-0008 (Quantization), RFC-0009 (Histogram Building)
- **Scope**: Gradient boosted tree training loop and tree growing

## Summary

GBTree trains an ensemble of decision trees via gradient boosting. Each round computes gradients from the current predictions, grows a tree to fit those gradients, and updates predictions. This RFC covers the training loop, tree growing, expansion strategies, and sampling. Histogram building details are in RFC-0009.

## Overview

### Component Hierarchy

```text
GBTreeTrainer<O: Objective>
│
├── params: GBTreeParams
│   ├── n_rounds
│   ├── tree: TreeParams
│   └── sampling: SamplingParams
│
├── row_sampler: RowSampler          ← Samples rows per round (Random, GOSS)
│
└── grower: TreeGrower
    ├── params: TreeParams
    ├── expansion: ExpansionStrategy  ← DepthWise or LeafWise
    ├── histogram_builder             ← RFC-0009
    ├── partitioner: RowPartitioner   ← Tracks row→node assignment
    └── col_sampler: ColSampler       ← Samples features per tree/level/node
```

### Data Flow

```text
Features (col-major) ──► Quantize ──► QuantizedMatrix (col-major)
                                              │
Labels ──► Objective ──► Gradients (col-major)│
                              │               │
                              ▼               ▼
                         TreeGrower ◄────────────
                              │
                              ▼
                            Tree ──► Update Predictions
```

### Multi-Output Support

Multi-output (multiclass, multi-target, multi-quantile) is supported by design:

- Gradients: `ColMatrix<f32>` with shape `[n_samples × n_outputs]`
- Each round trains `n_outputs` trees, one per gradient column
- All outputs share row sampling and quantization

## Components

### GBTreeTrainer

```rust
pub struct GBTreeTrainer<O: Objective> {
    objective: O,
    params: GBTreeParams,
    row_sampler: RowSampler,
    grower: TreeGrower,
}

pub struct GBTreeParams {
    pub n_rounds: usize,
    pub tree: TreeParams,
    pub row_sampling: RowSamplingConfig,
    pub col_sampling: ColSamplingConfig,
}
```

### Training Loop

```text
train(features, labels, weights?):
  // Setup (once)
  quantized = quantize(features)            // RFC-0008
  predictions = init_base_predictions()     // Shape: [n_samples × n_outputs]
  grower = TreeGrower::new(params, quantized.cuts())
  best_model: Option<TreeEnsemble> = None
  
  // Boosting rounds
  for round in 0..n_rounds:
    // Compute gradients for all outputs
    (grads, hess) = objective.gradients(predictions, labels, weights)
    
    rows = row_sampler.sample(grads, hess)    // ◄── Row sampling called here
    
    // Train one tree per output
    for k in 0..n_outputs:
      tree = grower.grow(quantized, grads.col(k), hess.col(k), rows)
      update_predictions(predictions.col_mut(k), tree)
      model.add_tree(tree, output=k)
    
    // Evaluation with action-based flow (RFC-0006)
    match evaluator.evaluate(round, &predictions, eval_sets):
      EvalAction::NewBest { .. } => best_model = Some(model.clone())
      EvalAction::Stop { .. } => break
      _ => {}
  
  return best_model.unwrap_or(model)
```

### RowSampler

Samples rows once per boosting round:

```rust
pub enum RowSamplingConfig {
    None,
    Random { subsample: f32 },
    Goss { top_rate: f32, other_rate: f32 },
}

pub struct RowSampler {
    config: RowSamplingConfig,
    rng: Rng,
    indices: Vec<u32>,       // Reused buffer
    grad_weights: Vec<f32>,  // Amplification factors for GOSS
}

impl RowSampler {
    /// Sample rows for this round. GOSS also modifies gradient weights.
    pub fn sample(&mut self, grads: &mut ColMatrix<f32>, hess: &mut ColMatrix<f32>) -> &[u32];
}
```

### GOSS Algorithm (Gradient-based One-Side Sampling)

GOSS keeps all high-gradient samples (hard examples) and randomly samples low-gradient samples, achieving ~70% data reduction with minimal accuracy loss.

```text
goss_sample(grads, hess, top_rate, other_rate):
  n = grads.n_rows()
  n_outputs = grads.n_cols()
  
  // Step 1: Compute gradient magnitude per row (L2 norm across outputs)
  magnitudes = [0.0; n]
  for i in 0..n:
    for k in 0..n_outputs:
      magnitudes[i] += grads[k][i] * grads[k][i]
    magnitudes[i] = sqrt(magnitudes[i])
  
  // Step 2: Find top-k by magnitude
  top_k = (n * top_rate) as usize
  other_k = (n * other_rate) as usize
  
  // Partial sort to find threshold (O(n) via quickselect)
  threshold = quickselect(magnitudes, top_k)
  
  // Step 3: Select indices
  selected = []
  small_selected = []
  
  for i in 0..n:
    if magnitudes[i] >= threshold:
      selected.push(i)           // Keep all large gradients
    else:
      small_selected.push(i)     // Candidates for sampling
  
  // Reservoir sample from small gradients
  shuffle(small_selected)
  for i in small_selected[..other_k]:
    selected.push(i)
  
  // Step 4: Amplify sampled small gradients to correct bias
  amplify = (n - top_k) as f32 / other_k as f32
  for i in small_selected[..other_k]:
    for k in 0..n_outputs:
      grads[k][i] *= amplify
      hess[k][i] *= amplify
  
  return selected
```

**Parameters**:

- `top_rate`: Fraction of large-gradient samples to keep (default: 0.2)
- `other_rate`: Fraction of small-gradient samples to sample (default: 0.1)
- Effective sample rate: `top_rate + other_rate` ≈ 0.3 (70% reduction)

**When to use GOSS**:

```text
should_use_goss(n_rows, round, learning_rate):
  // Don't use on small data
  if n_rows < 100_000: return false
  
  // Warm-up period: use full data initially
  warmup_rounds = (1.0 / learning_rate) as usize
  if round < warmup_rounds: return false
  
  return true
```

### TreeGrower

Grows a single tree using histogram-based split finding:

```rust
pub struct TreeGrower {
    params: TreeParams,
    expansion: ExpansionStrategy,
    histogram_builder: HistogramBuilder,  // RFC-0009
    partitioner: RowPartitioner,
    col_sampler: ColSampler,
}

pub struct TreeParams {
    pub max_depth: u16,        // 0 = unlimited
    pub max_leaves: u16,       // 0 = unlimited  
    pub min_child_weight: f32, // Min hessian sum per leaf
    pub lambda: f32,           // L2 regularization
    pub gamma: f32,            // Min split gain
    pub learning_rate: f32,
}
```

### ColSampler

Samples features at tree, level, and node granularity:

```rust
pub struct ColSamplingConfig {
    pub colsample_bytree: f32,   // Fraction per tree (default: 1.0)
    pub colsample_bylevel: f32,  // Fraction per depth level (default: 1.0)
    pub colsample_bynode: f32,   // Fraction per node (default: 1.0)
}

pub struct ColSampler {
    config: ColSamplingConfig,
    rng: Rng,
    n_features: usize,
    tree_features: Vec<u32>,   // Features for current tree
    level_features: Vec<u32>,  // Further filtered by level
}

impl ColSampler {
    /// Called at start of each tree. Samples colsample_bytree.
    pub fn sample_tree(&mut self);
    
    /// Called at each depth level. Further samples colsample_bylevel.
    pub fn sample_level(&mut self, depth: u16);
    
    /// Called for each node. Returns features for split finding.
    /// Further samples colsample_bynode from level_features.
    pub fn sample_node(&mut self) -> &[u32];
}
```

### Tree Growing Algorithm

```text
grow(quantized, grads, hess, rows):
  tree = Tree::new()
  partitioner.init(rows)
  col_sampler.sample_tree()                    // ◄── Tree-level sampling
  
  // Root node
  root_hist = histogram_builder.build(ROOT, quantized, grads, hess, rows)
  col_sampler.sample_level(depth=0)            // ◄── Level sampling
  allowed = col_sampler.sample_node()          // ◄── Node sampling
  root_split = find_best_split(root_hist, allowed)
  queue.push(Candidate { node: ROOT, depth: 0, split: root_split })
  
  // Expansion loop
  while queue.has_candidates():
    candidates = queue.pop_batch()
    next_depth = candidates[0].depth + 1
    col_sampler.sample_level(next_depth)       // ◄── Level sampling for children
    
    for candidate in candidates:
      if !should_expand(candidate): continue
      
      (left, right) = tree.apply_split(candidate)
      partitioner.partition(candidate.node, candidate.split, quantized)
      
      (left_hist, right_hist) = histogram_builder.build_children(...)
      
      for (child, hist) in [(left, left_hist), (right, right_hist)]:
        allowed = col_sampler.sample_node()    // ◄── Node sampling
        split = find_best_split(hist, allowed)
        queue.push(Candidate { node: child, depth: next_depth, split })
  
  // Finalize leaf weights
  for leaf in tree.leaves():
    leaf.weight *= learning_rate
  
  return tree
```

### Expansion Strategy

Controls the order of node expansion:

```rust
pub enum ExpansionStrategy {
    /// Expand all nodes at depth d before any at d+1. (XGBoost default)
    /// Better parallelism: entire level processed together.
    DepthWise,
    /// Always expand highest-gain node. (LightGBM default)
    /// More efficient for same leaf count, but less parallel.
    LeafWise,
}
```

| Strategy | Queue Type | Batch Size | Parallelism | Default In |
|----------|------------|------------|-------------|------------|
| DepthWise | FIFO | All at depth d | High (level-wise) | XGBoost |
| LeafWise | Max-heap | 1 | Low | LightGBM |

### Row Partitioner

Tracks row-to-node assignment, partitions rows after splits:

```rust
pub struct RowPartitioner {
    /// All row indices, grouped by node.
    rows: Vec<u32>,
    ranges: Vec<(usize, usize)>,  // (start, end) per node
}

impl RowPartitioner {
    pub fn init(&mut self, rows: &[u32]);
    pub fn rows(&self, node: u32) -> &[u32];
    pub fn partition(&mut self, node: u32, split: &SplitInfo, quantized: &QuantizedMatrix) -> (u32, u32);
}
```

### Partition Algorithms

Two partition algorithms: serial (simple) and parallel (scalable).

**Serial (Lomuto-style, in-place)**:

```text
partition_serial(node, split):
  rows = self.rows_mut(node)
  write_idx = 0
  
  for i in 0..rows.len():
    bin = quantized.get_bin(split.feature, rows[i])
    go_left = if is_missing(bin): split.default_left else: bin <= split.bin
    
    if go_left:
      swap(rows[write_idx], rows[i])
      write_idx += 1
  
  // rows = [left_rows | right_rows]
  return (left_id, right_id)
```

**Parallel (Three-phase, from XGBoost)**:

For large nodes, parallel partitioning avoids becoming a bottleneck:

```text
partition_parallel(node, split, n_threads):
  rows = self.rows_mut(node)
  n = rows.len()
  block_size = (n + n_threads - 1) / n_threads
  
  // Phase 1: Count left/right per block (parallel)
  left_counts = [0; n_threads]
  parallel_for block_id in 0..n_threads:
    start = block_id * block_size
    end = min(start + block_size, n)
    for i in start..end:
      if goes_left(rows[i], split):
        left_counts[block_id] += 1
  
  // Phase 2: Compute prefix sums (serial, cheap)
  left_offsets = [0; n_threads + 1]
  right_offsets = [0; n_threads + 1]
  total_left = 0
  total_right = 0
  for i in 0..n_threads:
    left_offsets[i] = total_left
    right_offsets[i] = total_right
    total_left += left_counts[i]
    total_right += (block_size - left_counts[i])
  
  // Phase 3: Write to final positions (parallel)
  output = [0; n]
  parallel_for block_id in 0..n_threads:
    left_write = left_offsets[block_id]
    right_write = total_left + right_offsets[block_id]
    
    start = block_id * block_size
    end = min(start + block_size, n)
    for i in start..end:
      if goes_left(rows[i], split):
        output[left_write] = rows[i]
        left_write += 1
      else:
        output[right_write] = rows[i]
        right_write += 1
  
  rows.copy_from_slice(&output)
  return (left_id with total_left rows, right_id with n - total_left rows)
```

**When to use parallel partition**:

```text
select_partition_algorithm(n_rows, n_threads):
  if n_rows < 10_000 or n_threads == 1:
    return Serial
  else:
    return Parallel
```

### Split Info

Result of split finding (done in RFC-0009):

```rust
pub struct SplitInfo {
    pub feature: u32,
    pub bin: u16,
    pub gain: f32,
    pub left_grad: f32,
    pub left_hess: f32,
    pub right_grad: f32,
    pub right_hess: f32,
    pub default_left: bool,  // Direction for missing values
}
```

## Design Decisions

### DD-1: Multi-Output via Explicit Loop

**Context**: How to handle multiclass, multi-target, multi-quantile?

**Decision**: Train `n_outputs` trees per round in an explicit loop.

**Rationale**:

- Matches XGBoost/LightGBM behavior
- Each tree trains on its own gradient column
- Shared: quantization, row sampling, tree structure
- Not shared: gradients, predictions per output
- Makes multi-output explicit rather than hidden

### DD-2: Three-Level Column Sampling

**Context**: XGBoost has `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`. How to organize?

**Decision**: ColSampler with `sample_tree()`, `sample_level()`, `sample_node()` methods.

**Rationale**:

- `bytree`: Called once per tree, sets `tree_features`
- `bylevel`: Called once per depth, filters `tree_features` → `level_features`
- `bynode`: Called per node, filters `level_features` for split finding
- Multiplicative: effective rate = bytree × bylevel × bynode

### DD-3: Row Sampling Modifies Gradients In-Place

**Context**: GOSS needs to amplify gradients for sampled rows.

**Decision**: `RowSampler::sample()` takes mutable gradients, modifies weights in place.

**Rationale**: (LightGBM approach)

- Avoids allocating weighted gradient copies
- Gradients are recomputed each round anyway
- Clear lifecycle: compute → sample (modify) → train

### DD-4: TreeGrower Owns Partitioner and ColSampler

**Context**: Where should these live?

**Decision**: TreeGrower owns both.

**Rationale**:

- Both have tree-specific state (reset between trees)
- Clear lifecycle: init at tree start, use during growth, reset
- Avoids passing them through every function

### DD-5: ExpansionStrategy as Enum

**Context**: DepthWise and LeafWise have different queue types.

**Decision**: Enum with unified `ExpansionQueue` abstraction.

**Rationale**:

- Both strategies share the same tree-growing structure
- Only difference is `pop_batch()` semantics
- Enum allows exhaustive matching, no dynamic dispatch

## Integration

| Component | Source |
|-----------|--------|
| QuantizedMatrix, BinCuts | RFC-0008 |
| HistogramBuilder, SplitInfo | RFC-0009 |
| Objective, gradients | RFC-0002 |
| Evaluator, early stopping | RFC-0006 |

## Future Work

- Monotonic constraints
- Interaction constraints
- GPU training
- Distributed training

## Categorical Features

Native categorical feature support avoids one-hot encoding overhead and finds optimal category partitions.

### Categorical Split Types

```rust
/// Split on a categorical feature.
pub enum CategoricalSplit {
    /// One category vs rest (for low-cardinality features).
    OneHot { category: u32 },
    /// Set of categories go left (for high-cardinality features).
    Partition { left_categories: BitSet },
}
```

### One-Hot vs Partition Selection

```text
select_categorical_method(n_categories, max_cat_to_onehot):
  if n_categories <= max_cat_to_onehot:  // Default: 4
    return OneHot   // O(k) algorithm: try each category alone
  else:
    return Partition  // O(k log k) algorithm: sorted greedy
```

### One-Hot Algorithm

For features with few categories (≤4), try each category as a split point:

```text
find_best_onehot_split(histogram, n_categories):
  best = SplitInfo::invalid()
  
  for cat in 0..n_categories:
    // Split: category cat goes left, all others go right
    left_g = histogram.grads[cat]
    left_h = histogram.hess[cat]
    right_g = total_g - left_g
    right_h = total_h - left_h
    
    gain = compute_gain(left_g, left_h, right_g, right_h)
    if gain > best.gain:
      best = CategoricalSplit::OneHot { category: cat }
  
  return best
```

### Partition Algorithm (Many-vs-Many)

For features with many categories, sort by gradient statistic and find optimal partition:

```text
find_best_partition_split(histogram, n_categories, max_cat_per_split):
  // Step 1: Compute sorting criterion (gradient / hessian + smoothing)
  cat_smooth = 10.0  // Regularization for noisy categories
  scores = []
  for cat in 0..n_categories:
    if histogram.counts[cat] > 0:
      score = histogram.grads[cat] / (histogram.hess[cat] + cat_smooth)
      scores.push((cat, score))
  
  // Step 2: Sort categories by score
  scores.sort_by(|a, b| a.1.cmp(&b.1))
  
  // Step 3: Try cumulative partitions from sorted order
  best = SplitInfo::invalid()
  left_categories = BitSet::new()
  left_g, left_h = 0, 0
  
  for (cat, _) in scores[..min(max_cat_per_split, scores.len())]:
    left_categories.insert(cat)
    left_g += histogram.grads[cat]
    left_h += histogram.hess[cat]
    right_g = total_g - left_g
    right_h = total_h - left_h
    
    gain = compute_gain(left_g, left_h, right_g, right_h)
    if gain > best.gain:
      best = CategoricalSplit::Partition { left_categories: left_categories.clone() }
  
  return best
```

### Categorical Feature Metadata

```rust
pub struct FeatureMetadata {
    pub feature_type: FeatureType,
    pub n_categories: Option<u32>,  // For categorical features
}

pub enum FeatureType {
    Numerical,
    Categorical,
}

impl BinCuts {
    /// Get feature metadata for split finding.
    pub fn feature_metadata(&self, feature: usize) -> &FeatureMetadata;
    
    /// Check if feature is categorical.
    pub fn is_categorical(&self, feature: usize) -> bool;
}
```

### Split Finding Integration

```text
find_best_split(histogram, feature, metadata, params):
  match metadata.feature_type:
    Numerical:
      return find_best_numerical_split(histogram, params)
    
    Categorical:
      n_cat = metadata.n_categories
      if n_cat <= params.max_cat_to_onehot:
        return find_best_onehot_split(histogram, n_cat)
      else:
        return find_best_partition_split(histogram, n_cat, params.max_cat_per_split)
```

## References

- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)