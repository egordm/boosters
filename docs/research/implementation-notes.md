# Implementation Notes

Key lessons, design decisions, and trade-offs learned from studying XGBoost and LightGBM.
This document synthesizes insights from both libraries into actionable guidance.

---

## Algorithmic Foundations

### 1. Histogram-Based Training is Essential

Both XGBoost (`tree_method=hist`) and LightGBM use histogram-based training:

- **O(bins)** split finding instead of **O(n)** per feature
- Better cache efficiency (small working set)
- Natural sparsity handling

**Key insight**: The histogram approach is the foundation of modern gradient boosting.
Without it, training large datasets is impractical.

### 2. Histogram Subtraction Halves Work

The subtraction trick `child = parent - sibling` reduces histogram builds:

```text
Instead of building histograms for both children:
  1. Build histogram for SMALLER child only
  2. Derive larger child: larger = parent - smaller
```

This nearly halves histogram building work. Both XGBoost and LightGBM implement this.

### 3. Quantile Sketching Enables Scalability

For large datasets, computing exact quantiles requires sorting. Streaming sketches provide:

- O(1/ε²) memory regardless of dataset size
- Single-pass quantile computation  
- Mergeable across distributed workers

**Trade-off**: Sketch approximation vs exact quantiles. For most applications, sketch
error is negligible compared to other sources of variance.

---

## Data Structure Decisions

### 4. Separate Training and Inference Representations

XGBoost and LightGBM both use different tree formats:

| Aspect | Training | Inference |
|--------|----------|-----------|
| Mutability | Mutable (growing tree) | Immutable |
| Layout | AoS (node-centric) | SoA (cache-friendly) |
| Priority | Fast updates | Fast traversal |

**Recommendation**: Build mutable tree during training, convert to SoA for inference.

### 5. Bin Index Size is Configurable

Both libraries adapt bin index type to max_bins:

| max_bins | Type |
|----------|------|
| ≤256 | u8 |
| ≤65536 | u16 |
| >65536 | u32 |

**Default recommendation**: Use u8 for 256 bins (covers most cases).

### 6. Gradient Precision Trade-offs

| Precision | Memory | Overflow Risk | Use Case |
|-----------|--------|---------------|----------|
| f32 | 4 bytes | Rare | Default |
| f64 | 8 bytes | Never | Very large datasets |
| int8 (quantized) | 1 byte | Managed | LightGBM optimization |

**Recommendation**: f32 default, support f64 via feature flag.

---

## Parallelization Strategy

### 7. Multiple Levels of Parallelism

Both libraries parallelize at several levels:

1. **Features**: Split finding (independent per feature)
2. **Rows**: Histogram building (partition + reduce)
3. **Nodes**: Depth-wise processes all nodes at same level
4. **Trees**: Can build trees in parallel (independent for bagging)

**Key insight**: Feature-level parallelism for split finding, block-level for histogram
building. Rayon handles this well with work-stealing.

### 8. Block-Based Processing

XGBoost uses fixed block size of 64 rows for prediction. LightGBM uses similar
block-based approaches for histogram building.

**Why 64?**
- Matches typical cache line sizes
- Good SIMD vectorization potential
- Balances memory footprint vs amortization

**Recommendation**: Configurable block size, default 64.

---

## Memory Management

### 9. Memory Reuse Matters

Hot paths should avoid allocations:

- **Histogram ring buffer**: Only keep 2 levels (parent + current)
- **Row index buffers**: Swap between iterations
- **Thread-local histograms**: Reuse across trees

**Pattern**: Pre-allocate buffers sized to maximum expected usage, reuse aggressively.

### 10. Ordered Gradients for Cache Efficiency

LightGBM reorders gradients to match data indices:

```text
BAD: Random access pattern
  for idx in data_indices:
      gradient = gradients[idx]  ← Cache miss likely

GOOD: Sequential access (pre-reordered)
  for (i, idx) in enumerate(data_indices):
      gradient = ordered_gradients[i]  ← Sequential, cache-friendly
```

**Cost**: One reordering pass per iteration. **Benefit**: Much faster histogram building.

---

## Missing Value Handling

### 11. Learn Default Direction

Both XGBoost and LightGBM learn `default_left` for each split:

```text
During split evaluation:
  1. Try all missing → left: compute gain
  2. Try all missing → right: compute gain
  3. Choose direction with higher gain
  4. Store direction for inference
```

This requires bidirectional histogram scanning but provides optimal handling.

---

## Categorical Features

### 12. Native Handling is Worth It

LightGBM's native categorical support:

1. **Low cardinality (≤4)**: One-hot strategy, O(k)
2. **High cardinality**: Gradient-sorted partition, O(k log k)

**Key insight**: Native handling finds optimal binary partitions, not just one-vs-rest.
This is particularly valuable for high-cardinality categoricals.

**Storage**: Bitset for category membership (CSR-like for variable sizes).

---

## Performance Optimizations

### 13. Algorithmic > Micro-optimizations

Priority order for performance:

1. **Algorithmic**: Histogram, subtraction, quantization
2. **Memory layout**: SoA, cache alignment, prefetching
3. **Parallelization**: Rayon, block-based
4. **SIMD**: Nice-to-have but not critical

**Key insight**: Most gains come from algorithmic optimizations. SIMD helps but is not
the primary differentiator.

### 14. Software Prefetching Helps

LightGBM uses explicit prefetch hints in histogram building:

```cpp
PREFETCH_T0(data_.data() + pf_idx);  // Load ahead of use
```

**When useful**: Random access patterns where next index is predictable (e.g., during
histogram building with known indices).

### 15. Template Specialization Eliminates Branches

Both libraries use templates/generics to eliminate runtime branches:

```cpp
template <bool HAS_MISSING, bool HAS_CATEGORICAL>
void Traverse(...) {
    if constexpr (HAS_MISSING) { /* ... */ }
}
```

This produces 4 specialized code paths with no runtime branching.

**Rust equivalent**: Const generics or trait-based dispatch.

---

## API Design

### 16. Builder Pattern for Configuration

Both libraries have many parameters. Good practices:

- **Sensible defaults**: `max_depth=6`, `learning_rate=0.3`
- **Named parameters**: Clear what each does
- **Validation**: Check parameter combinations

### 17. Progress Reporting is Useful

Support optional progress callbacks:

```rust
pub trait TrainingCallback {
    fn on_iteration(&mut self, iteration: usize, metrics: &Metrics);
    fn should_stop(&self) -> bool;
}
```

---

## What We Defer

Some features are out of initial scope:

| Feature | Reason |
|---------|--------|
| Distributed training | Start single-machine |
| GPU support | CPU-first |
| External memory | In-memory only initially |
| Approximate tree method | Exact histogram only |
| Linear trees | After core GBDT |

---

## Summary Table

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training algorithm | Histogram-based | Standard, O(bins) splits |
| Tree growth | Leaf-wise default | More efficient (LightGBM style) |
| Histogram subtraction | Yes | Halves work |
| Gradient precision | f32 default | 2x memory savings |
| Bin index type | u8 default | Sufficient for 256 bins |
| Parallelism | Rayon | Work-stealing, ergonomic |
| Missing values | Learned direction | Optimal handling |
| Categoricals | Native support | LightGBM approach |
| Tree storage | AoS train, SoA infer | Optimized for each use case |
| Block size | 64 rows | Cache-aligned, proven |

---

## References

### XGBoost Source Files

| Component | File |
|-----------|------|
| Training loop | `src/tree/updater_quantile_hist.cc` |
| Histogram | `src/common/hist_util.cc` |
| Split finding | `src/tree/split_evaluator.cc` |
| Row partitioning | `src/tree/row_set.h` |

### LightGBM Source Files

| Component | File |
|-----------|------|
| Training loop | `src/treelearner/serial_tree_learner.cpp` |
| Histogram | `src/treelearner/feature_histogram.hpp` |
| GOSS | `src/boosting/goss.hpp` |
| Categorical | `src/treelearner/feature_histogram.cpp` |
| Data partition | `src/treelearner/data_partition.hpp` |
