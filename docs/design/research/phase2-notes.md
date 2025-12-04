# Phase 2 Research Notes

**Purpose**: Capture knowledge from Phase 1 RFC research for future Phase 2 RFC writing.
**Created**: 2024-11-30
**Context**: These notes preserve insights from XGBoost/LightGBM source code analysis that will inform Phase 2 RFCs (Categorical Training, Sampling, Multi-output, Feature Bundling, etc.)

---

## RFC-0016: Categorical Training

### Key Insights from LightGBM

**Gradient-sorted categorical splits** (from `feature_histogram.hpp`):
1. Sort categories by `grad_sum / hess_sum` ratio
2. Linear scan to find optimal binary partition
3. Complexity: O(k log k) for k categories

**Implementation pattern**:
```cpp
// LightGBM approach
struct CatSortEntry {
  uint32_t category;
  double grad_sum;
  double hess_sum;
};
// Sort by grad/hess ratio, scan for best split
```

**Missing value handling for categoricals**:
- LightGBM treats missing as separate category (bin 0)
- Missing can go left or right like numerical features

**High-cardinality optimization**:
- LightGBM uses max_cat_threshold (default 32) to limit categories considered
- Categories below min_data_in_leaf_category are grouped

### Extension Points in Phase 1

- `SplitInfo.is_categorical` and `categories_left: Vec<u32>` already defined
- `CategoricalSplitFinder` skeleton in RFC-0013
- `QuantizedMatrix` bin indices work for categorical (category = bin)

### Open Research Questions

1. How does LightGBM handle one-vs-all categorical splits?
2. What's the optimal threshold for switching from gradient-sorted to one-hot?
3. How to handle new categories at inference time?

---

## RFC-0017: Sampling Strategies

### GOSS (Gradient-based One-Side Sampling)

**LightGBM implementation** (from research notes):
```
1. Sort samples by |gradient|
2. Keep top_rate% samples with largest gradients
3. Randomly sample other_rate% from remaining
4. Weight the random samples by (1 - top_rate) / other_rate
```

**Key parameters**:
- `top_rate`: typically 0.1-0.2 (keep 10-20% high gradient)
- `other_rate`: typically 0.1-0.2 (sample 10-20% from rest)

**Integration point**: `RowPartitioner::with_goss()` skeleton in RFC-0014

### Random Sampling

XGBoost's `subsample` parameter:
- Applied per-tree, not per-node
- Uniform random without replacement
- Integration: `RowPartitioner::with_sampling()`

### Column Sampling

XGBoost supports:
- `colsample_bytree`: sample features per tree
- `colsample_bylevel`: sample features per depth level
- `colsample_bynode`: sample features per node

**Integration point**: `GreedySplitFinder.feature_subset` in RFC-0013

---

## RFC-0018: Multi-Output Trees

### XGBoost Multi-output

XGBoost builds separate trees per output (class), combining in forest:
- `num_class` trees per round for multi-class
- Gradients computed independently per class

### LightGBM Multi-output

LightGBM can build shared trees with vector leaf values:
- Single tree, K outputs per leaf
- Split gain sums across all classes
- More efficient but different accuracy characteristics

### Phase 1 Extension Points

- `MultiOutputFeatureHistogram<const NUM_CLASSES>` in RFC-0012
- `DynMultiOutputFeatureHistogram` for dynamic class count
- `MultiOutputSplitFinder<NUM_CLASSES>` skeleton in RFC-0013
- `GradientBuffer` already supports `n_outputs` dimension

### Design Decision Needed

Should booste-rs support:
1. **XGBoost-style**: Separate tree per output (simpler, proven)
2. **LightGBM-style**: Shared tree with vector outputs (more efficient)
3. **Both**: Configuration option

Recommendation: Start with XGBoost-style (separate trees), add shared trees later.

---

## RFC-0019: Feature Bundling (EFB)

### LightGBM EFB Algorithm

**Exclusive Feature Bundling** groups mutually exclusive features:
1. Build conflict graph (features that have non-zero together)
2. Greedy coloring to find bundles
3. Merge bundled features into single histogram

**Bin offset encoding**:
```
bundle_bin = feature_a_bin + feature_b_offset + feature_b_bin
```

Where `feature_b_offset = num_bins_a`

### Phase 1 Extension Points

- `BinCutsWithBundles` structure defined in RFC-0011
- `bundle_map`, `bundles`, `bin_offsets` fields ready

### Key Implementation Details

From LightGBM `bin.cpp`:
- Build conflict matrix in O(features²) 
- Use graph coloring heuristic
- Handle sparse features specially (zero = no value)

---

## RFC-0020: Gradient Quantization

### LightGBM 16-bit Gradients

**From bin.h research**:
```cpp
// LightGBM histogram construction with quantized gradients
void ConstructHistogramInt16(...);  // 16-bit quantized
void ConstructHistogramInt32(...);  // 32-bit quantized  
void ConstructHistogramInt8(...);   // 8-bit quantized
```

**Quantization approach**:
- Scale gradients to fit in int16/int32
- Use fixed-point arithmetic
- Dequantize when computing split gain

**Benefits**:
- Reduced memory bandwidth
- Faster SIMD accumulation (8 int16 vs 4 float32 per vector)

### Phase 1 Extension Points

- `FeatureHistogram` SoA layout supports different storage types
- Could add `QuantizedFeatureHistogram<T: QuantizedGrad>`

---

## RFC-0021: Linear Trees

### LightGBM Linear Trees

**From training/linear_trees.md research**:

At each leaf, fit linear model instead of constant:
```
leaf_value = w₀ + Σ wᵢ × xᵢ
```

**Training approach**:
1. Normal tree growing to determine structure
2. At each leaf, solve least squares for linear weights
3. Regularize with L2 penalty

**Key complexity**:
- Need raw feature values at leaves (not just bins)
- Fitting linear model at each leaf adds O(features²) per leaf

### Phase 1 Extension Points

- `BuildingNode.leaf_value` could become `LeafModel` enum
- Split finding unchanged (still uses histograms)
- Extra step at leaf finalization

---

## RFC-0022: GPU Training

### XGBoost ELLPACK Format

**From gpu_training.md research**:
- ELLPACK: Compressed row storage for GPU
- Fixed-width per row (max non-zeros)
- Coalesced memory access patterns

**Histogram building on GPU**:
- Use atomic adds (cooperative)
- Or parallel reduction (more complex)

### LightGBM CUDA

- Uses chunked histogram building
- Radix sort for row partitioning

### Phase 1 Considerations

- CPU implementations should not prevent GPU path
- Keep data structures GPU-friendly where possible

---

## RFC-0023: Constraints

### Interaction Constraints

**XGBoost FeatureInteractionConstraintHost**:
```cpp
// Maintains per-node allowed feature sets
// When split on feature F, children inherit only features
// that can interact with F according to constraint groups
```

**Data structure**:
- `constraint_sets: Vec<Vec<u32>>` - groups that can interact
- `node_constraints: Vec<BitSet>` - allowed features per node

### Monotonic Constraints

**Implementation** (RFC-0013 already has skeleton):
- Check `weight_left <= weight_right` for increasing
- Filter splits that violate constraint

---

## Integration Summary

### Extension Points Built Into Phase 1

| RFC | Extension Point | For Phase 2 RFC |
|-----|-----------------|-----------------|
| 0011 | `BinCutsWithBundles` | 0019 (EFB) |
| 0011 | `BinIndex` generic | 0020 (Grad Quant) |
| 0012 | `MultiOutputFeatureHistogram` | 0018 (Multi-output) |
| 0013 | `CategoricalSplitFinder` | 0016 (Categorical) |
| 0013 | `ConstrainedSplitFinder` | 0023 (Constraints) |
| 0013 | `feature_subset` | 0017 (Sampling) |
| 0014 | `with_goss()` | 0017 (Sampling) |
| 0015 | `GrowthPolicy` trait | 0022 (GPU, Oblivious) |

### Research Files for Reference

| Topic | XGBoost | LightGBM |
|-------|---------|----------|
| Histogram building | `training/histogram_building.md` | `training/histogram_building.md` |
| Split finding | `training/split_finding.md` | - |
| Row partitioning | `training/row_partitioning.md` | - |
| Tree growing | `training/tree_growing.md` | `training/leaf_wise_growth.md` |
| Quantization | `training/quantization.md` | - |
| Categorical | - | `training/categorical_features.md` |
| Linear trees | - | `training/linear_trees.md` |
| GPU | `training/gpu_training.md` | - |

---

## Source Code References

These source files were studied and contain valuable implementation details:

### XGBoost
- `src/common/hist_util.h` - Quantile sketch, cuts
- `src/tree/updater_quantile_hist.cc` - Main training loop
- `src/tree/split_evaluator.h` - Gain computation
- `src/common/row_set.h` - Row partitioning

### LightGBM  
- `include/LightGBM/bin.h` - BinMapper, SparseBin templates
- `src/io/bin.cpp` - Binning implementation
- `src/treelearner/feature_histogram.hpp` - Histogram building
- `src/treelearner/serial_tree_learner.cpp` - Tree growing
- `src/treelearner/data_partition.hpp` - Row partitioning
