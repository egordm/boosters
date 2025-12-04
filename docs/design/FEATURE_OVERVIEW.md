# Feature Overview: GBTree Training Design

**Status**: Planning Phase  
**Goal**: Design booste-rs GBTree training to match or exceed XGBoost/LightGBM in accuracy and performance.

---

## Current State Inventory

### What booste-rs Has Today

| Component | Status | Notes |
|-----------|--------|-------|
| **Inference** | ✅ Complete | 3-9x faster than XGBoost C++ |
| **Tree Storage** | ✅ Complete | SoATreeStorage, UnrolledTreeLayout |
| **Forest Storage** | ✅ Complete | SoAForest with multi-group support |
| **Data Input** | ✅ Complete | RowMatrix, ColMatrix, CSCMatrix |
| **XGBoost Loading** | ✅ Complete | JSON format with categoricals |
| **GBLinear Training** | ✅ Complete | Coordinate descent with elastic net |
| **Loss Functions** | ✅ Complete | Squared, Logistic, Softmax, Quantile, Huber, Hinge |
| **Metrics** | ✅ Complete | RMSE, MAE, LogLoss, Accuracy, AUC |
| **Categorical Splits** | ✅ Inference only | Bitset storage, partition-based |

### What XGBoost Implements

| Component | Description | Priority for booste-rs |
|-----------|-------------|------------------------|
| **Histogram Building** | Aggregate gradients into bins | P0 - Core |
| **Quantile Sketch** | GK/T-Digest for bin boundaries | P0 - Core |
| **Split Finding** | Greedy best split from histogram | P0 - Core |
| **Row Partitioning** | Assign rows to nodes | P0 - Core |
| **Depth-wise Growth** | Level-by-level tree building | P1 - Important |
| **Histogram Subtraction** | Derive sibling from parent | P0 - Core |
| **Monotonic Constraints** | Feature monotonicity | P2 - Nice to have |
| **Interaction Constraints** | Feature interaction limits | P3 - Future |
| **GPU Training (ELLPACK)** | CUDA acceleration | P3 - Future |
| **Distributed Training** | Rabit/Federated | P4 - Out of scope |

### What LightGBM Implements

| Component | Description | Priority for booste-rs |
|-----------|-------------|------------------------|
| **Leaf-wise Growth** | Best-first tree building | P0 - Core |
| **GOSS Sampling** | Gradient-based row sampling | P1 - Important |
| **Native Categoricals** | Gradient-sorted categorical splits | P1 - Important |
| **Gradient Quantization** | 16/32-bit packed gradients on CPU | P2 - Nice to have |
| **Feature Bundling (EFB)** | Bundle mutually exclusive features | P2 - Nice to have |
| **Linear Trees** | Linear models in leaves | P2 - Nice to have |
| **Multi-output Trees** | Vector-valued leaf predictions | P2 - Nice to have |
| **Histogram Caching** | Cache histograms across iterations | P2 - Nice to have |

---

## Design Decisions

### 1. Tree Growth Strategy: Support Both, Default Leaf-wise

**Decision**: Implement both strategies behind a common interface.

**Rationale**:
- Leaf-wise (LightGBM) achieves lower loss with same number of leaves
- Depth-wise (XGBoost) provides more balanced trees, better for small datasets
- Users should choose based on their data characteristics

```rust
pub enum GrowthStrategy {
    LeafWise { num_leaves: u32 },      // Default: LightGBM style
    DepthWise { max_depth: u32 },      // XGBoost style
}
```

**RFC Required**: Yes - Growth strategy abstraction

---

### 2. Histogram Building: Row-major with Column Iteration

**Decision**: Build histograms by iterating over quantized feature columns.

**Rationale**:
- Column iteration enables histogram subtraction optimization
- Matches LightGBM's proven approach
- Compatible with both dense and sparse storage

**Key structures**:
- `GHistIndexMatrix`: Quantized features (u8 bin indices)
- `HistogramBuilder`: Builds per-node histograms
- `GradientHistogram`: Aggregated (sum_grad, sum_hess, count) per bin

**RFC Required**: Yes - Histogram building and storage

---

### 3. Categorical Features: Native Support (LightGBM-style)

**Decision**: Implement gradient-sorted categorical splits.

**Rationale**:
- One-hot encoding explodes memory for high-cardinality features
- Gradient-sorted approach finds optimal binary partition in O(k log k)
- Already have categorical inference support

**Algorithm**:
1. Sort categories by gradient sum / hessian sum
2. Binary search for optimal split point
3. Store category partition in split node

**RFC Required**: Yes - Categorical feature handling in training

---

### 4. Row Partitioning: Bitset-based Position Tracking

**Decision**: Use bitsets or position lists to track row-to-node mapping.

**Rationale**:
- XGBoost uses `RowSetCollection` (sparse positions per node)
- LightGBM uses `DataPartition` (row index mapping)
- Bitsets are cache-friendly and support parallel updates

**RFC Required**: Yes - Part of split application

---

### 5. Quantization: Two-phase with Optional Streaming

**Decision**: Support both exact quantile and streaming (QuantileDMatrix style).

**Rationale**:
- Small data: exact quantile via sorting
- Large data: GK sketch or T-Digest streaming
- Training should work with pre-quantized data

**RFC Required**: Yes - Quantization and binning

---

### 6. Loss Functions: Reuse Existing Infrastructure

**Decision**: Extend current `Loss` / `MulticlassLoss` traits.

**Rationale**:
- Already have squared, logistic, softmax, quantile, huber, hinge
- GBTree uses identical gradient computation
- Add second-order gradient support where needed

**No RFC Required**: Existing implementation sufficient

---

### 7. Regularization: Full XGBoost Parity

**Decision**: Implement all XGBoost regularization parameters.

**Parameters**:
- `lambda` (L2 regularization)
- `alpha` (L1 regularization)
- `gamma` (minimum split gain)
- `min_child_weight` (minimum sum of hessians in child)

**RFC Required**: Yes - Part of split finding

---

### 8. Sampling: GOSS as Optional Optimization

**Decision**: Implement random sampling first, GOSS as optimization.

**Rationale**:
- Random sampling is simpler and well-understood
- GOSS requires gradient sorting (O(n log n) overhead)
- Benefit is proportional to data size

```rust
pub enum SamplingStrategy {
    None,
    Random { rate: f32 },
    GOSS { top_rate: f32, other_rate: f32 },
}
```

**RFC Required**: Yes - Part of training config

---

### 9. Linear Trees: Phase 2 Feature

**Decision**: Defer linear trees to after basic GBTree is complete.

**Rationale**:
- Requires fitting linear model at each leaf during training
- Adds complexity to inference path
- Nice-to-have, not core functionality

**RFC Required**: Yes - Separate RFC after core GBTree

---

### 10. Multi-output Trees: Design Now, Implement Later

**Decision**: Design histogram and leaf structures to support vector-valued outputs.

**Rationale**:
- Multi-class classification benefits from shared tree structure
- LightGBM supports this with `num_class` × gradient dimensions
- Retrofitting after scalar-only design is painful

**Design considerations**:
- `GradientBuffer` should support `[grad, hess]` per class
- Histogram bins store sum vectors, not scalars
- Split gain computation aggregates across classes
- Leaf values are vectors `[f32; num_class]`

**RFC Required**: Yes - Phase 2, but design hooks in Phase 1

---

### 11. Feature Bundling (EFB): Design Now, Implement Later

**Decision**: Design quantization to be bundle-aware.

**Rationale**:
- Mutually exclusive features can share histogram bins
- Reduces memory and computation proportional to sparsity
- LightGBM achieves significant speedups on sparse data

**Design considerations**:
- `GHistIndexMatrix` should map feature → bundle
- Histogram building iterates bundles, not raw features
- Split finding maps bundle bin → original feature + threshold

**RFC Required**: Yes - Phase 2

---

### 12. GPU Training: Phase 3 Feature

**Decision**: Defer GPU to after CPU is optimized.

**Rationale**:
- CPU implementation validates design
- GPU requires different data layouts (ELLPACK)
- Significant engineering effort

**RFC Required**: Yes - Separate epic

---

## Deferred Features (Design Now, Implement Later)

These features are **not in Phase 1** but should be considered during design to avoid
paintful refactors later:

| Feature | Priority | Design Consideration |
|---------|----------|----------------------|
| **Multi-output Trees** | P2 | Leaf values should support vector outputs; histogram/split structures must handle multi-dimensional gradients |
| **Feature Bundling (EFB)** | P2 | Data structures should allow bundled feature indices; histogram building should be bundle-aware |
| **Linear Trees** | P2 | Leaf fitting should be pluggable (constant vs linear model) |

## What We Will NOT Implement

| Feature | Reason |
|---------|--------|
| **Distributed Training** | Complexity, different problem space |
| **DART Boosting** | Inference works, training is niche |
| **Custom Objectives in Training** | Start with standard objectives |

---

## RFC Roadmap

Based on the analysis above, here's the recommended RFC order:

### Phase 1: Core Training Infrastructure

| RFC | Topic | Dependencies | Priority |
|-----|-------|--------------|----------|
| **RFC-0011** | Quantization & Binning | None | P0 |
| **RFC-0012** | Histogram Building | RFC-0011 | P0 |
| **RFC-0013** | Split Finding & Gain | RFC-0012 | P0 |
| **RFC-0014** | Row Partitioning | RFC-0013 | P0 |
| **RFC-0015** | Tree Growing Strategies | RFC-0014 | P0 |

### Phase 2: Optimizations & Features

| RFC | Topic | Dependencies | Priority |
|-----|-------|--------------|----------|
| **RFC-0016** | Categorical Training | RFC-0015 | P1 |
| **RFC-0017** | Sampling Strategies | RFC-0015 | P1 |
| **RFC-0018** | Multi-output Trees | RFC-0012, RFC-0013 | P2 |
| **RFC-0019** | Feature Bundling (EFB) | RFC-0011 | P2 |
| **RFC-0020** | Gradient Quantization | RFC-0012 | P2 |
| **RFC-0021** | Linear Trees | RFC-0015 | P2 |

### Phase 3: Advanced

| RFC | Topic | Dependencies | Priority |
|-----|-------|--------------|----------|
| **RFC-0022** | GPU Training | RFC-0015 | P3 |
| **RFC-0023** | Constraints (Monotonic, Interaction) | RFC-0015 | P3 |

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GBTree Trainer                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │   Quantizer     │    │  TreeBuilder    │    │   Objective     │      │
│  │                 │    │                 │    │                 │      │
│  │ - QuantileSketch│    │ - GrowthStrategy│    │ - Loss trait    │      │
│  │ - HistogramCuts │    │ - SplitFinder   │    │ - Gradients     │      │
│  │ - GHistIndex    │    │ - NodeExpander  │    │ - Hessians      │      │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘      │
│           │                      │                      │                │
│           ▼                      ▼                      ▼                │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                     Training Loop                               │     │
│  │                                                                 │     │
│  │  for round in 0..num_rounds:                                    │     │
│  │    1. Compute gradients from predictions                        │     │
│  │    2. Sample rows (optional GOSS)                               │     │
│  │    3. Build tree:                                               │     │
│  │       - Initialize root with all rows                           │     │
│  │       - While can_grow:                                         │     │
│  │         a. Select node(s) to expand (leaf-wise or depth-wise)   │     │
│  │         b. Build histogram for selected node(s)                 │     │
│  │         c. Find best split for each node                        │     │
│  │         d. Apply splits, partition rows                         │     │
│  │    4. Add tree to ensemble                                      │     │
│  │    5. Update predictions                                        │     │
│  │    6. Evaluate metrics, check early stopping                    │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │ RowPartitioner  │    │ HistogramCache  │    │   Callbacks     │      │
│  │                 │    │                 │    │                 │      │
│  │ - PositionList  │    │ - NodeHistogram │    │ - EarlyStopping │      │
│  │ - ApplySplit    │    │ - Subtraction   │    │ - Logger        │      │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Structure Hierarchy

```
Training Data Flow:
━━━━━━━━━━━━━━━━━━

Raw Features (RowMatrix<f32>)
        │
        ▼
┌───────────────────┐
│  HistogramCuts    │  ← Bin boundaries per feature
│  - cut_values[]   │     (from quantile sketch)
│  - cut_ptrs[]     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  GHistIndexMatrix │  ← Quantized features
│  - index[row,col] │     (u8 bin indices, column-major)
│  - num_bins       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  GradientBuffer   │  ← Per-row gradients
│  - grads[row]     │     (SoA layout)
│  - hess[row]      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  NodeHistogram    │  ← Per-bin aggregates
│  - sum_grad[bin]  │
│  - sum_hess[bin]  │
│  - count[bin]     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  SplitInfo        │  ← Best split for node
│  - feature        │
│  - threshold      │
│  - gain           │
│  - left_sum, etc  │
└───────────────────┘
```

---

## Zero-Cost Abstraction Goals

### Growth Strategy

```rust
// Compile-time strategy selection
trait GrowthPolicy {
    fn select_nodes(&self, candidates: &[NodeId]) -> Vec<NodeId>;
    fn can_continue(&self, tree: &BuildingTree) -> bool;
}

struct LeafWisePolicy { num_leaves: u32 }
struct DepthWisePolicy { max_depth: u32 }

// Usage with monomorphization
fn build_tree<G: GrowthPolicy>(policy: &G, data: &GHistIndexMatrix) -> Tree {
    // Compiler specializes for each policy
}
```

### Histogram Building

```rust
// Compile-time layout selection
trait HistogramLayout {
    type Storage;
    fn aggregate(&mut self, bin: u8, grad: f32, hess: f32);
    fn get(&self, bin: u8) -> (f32, f32, u32);
}

// Dense layout (most common)
struct DenseHistogram { data: Vec<(f32, f32, u32)> }

// Sparse layout (for high-cardinality categoricals)
struct SparseHistogram { data: HashMap<u8, (f32, f32, u32)> }
```

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Single tree build** | ≤ XGBoost | Histogram approach is proven |
| **Memory per feature** | ≤ 1 byte | u8 bin indices |
| **Histogram build** | O(n_rows × n_features / 256) | Vectorizable |
| **Split finding** | O(n_bins) | Linear scan |
| **Parallel efficiency** | >80% on 8 cores | Rayon work-stealing |

---

## Next Steps

1. **Write RFC-0011**: Quantization & Binning
   - HistogramCuts structure
   - Quantile sketch algorithm
   - GHistIndexMatrix layout

2. **Write RFC-0012**: Histogram Building
   - NodeHistogram structure
   - Aggregation algorithms
   - Histogram subtraction

3. **Prototype**: Simple depth-wise trainer
   - Validate design with real data
   - Benchmark against XGBoost

---

## Open Questions

1. **Sparse data handling**: Should sparse use same histogram approach or special-case?
2. **Multi-threading granularity**: Per-node, per-feature, or per-row parallelism?
3. **Memory allocation**: Pre-allocate histograms or grow dynamically?
4. **Cache optimization**: Should we cache parent histograms for subtraction?

These will be resolved in individual RFCs.

---

## References

- [XGBoost GBTree Research](research/xgboost-gbtree/)
- [LightGBM Research](research/lightgbm/)
- [XGBoost vs LightGBM Comparison](research/comparison.md)
- [Existing RFCs](rfcs/)
