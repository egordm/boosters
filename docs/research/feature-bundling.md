# Feature Bundling and Sparse Feature Optimization

This document explores techniques for optimizing gradient boosting when datasets contain
many sparse or binary features, particularly from one-hot encoding. The key insight is
that mutually exclusive features (like one-hot encoded columns) can be **bundled** together,
dramatically reducing memory usage and computation.

---

## Problem Statement

### The One-Hot Encoding Problem

When categorical features are one-hot encoded, they create a sparse matrix:

```text
Original: color = {red, blue, green}  →  3 binary features

Row 1: color=red   →  [1, 0, 0]
Row 2: color=blue  →  [0, 1, 0]  
Row 3: color=green →  [0, 0, 1]
Row 4: color=red   →  [1, 0, 0]
```

**Observation**: In any row, **at most one** of these features is non-zero.
These features are **mutually exclusive**.

### Waste in Current Approach

With separate features, we:
1. Store 3 separate binned columns (3× memory vs 1 categorical column)
2. Build 3 separate histograms (3× histogram work)
3. Find splits for 3 features (3× split-finding work)

For a dataset with 50 categorical features of average cardinality 20:
- **1000 one-hot features** instead of 50 categorical features
- **20× memory** and compute overhead

### Real-World Impact

Our quality benchmark shows boosters is 4% worse on the Adult dataset (105 one-hot
features from 14 original) compared to LightGBM. This is partly due to:
1. Equal-width binning on binary features (addressed separately via quantile binning)
2. No feature bundling optimization (this RFC)
3. No native categorical support (future work)

---

## Exclusive Feature Bundling (EFB)

### Core Idea

**Key insight**: Mutually exclusive features can be merged into a single feature without
losing information. If features $f_1, f_2, f_3$ never have non-zero values in the same
row, we can encode them as one feature with distinct value ranges:

```text
Original features:
  f1 ∈ {0, 1}  (0 = absent, 1 = present)
  f2 ∈ {0, 1}
  f3 ∈ {0, 1}

Bundled feature f_bundle:
  0 = all absent (f1=f2=f3=0)
  1 = f1 is active
  2 = f2 is active
  3 = f3 is active

Bundle value: offset_f1 + bin(f1) + offset_f2 + bin(f2) + ...
            = 0 + 1 + 0 + 0 + 0 + 0 = 1  (when f1=1)
```

### Formal Definition

**Exclusive Feature Bundling**: Given features $F = \{f_1, ..., f_k\}$, we say they
can be bundled if for every sample $i$:
$$|\{f_j : f_j(x_i) \neq 0\}| \leq 1$$

i.e., at most one feature has a non-zero value per sample.

**Relaxation**: LightGBM allows a small **conflict rate** (default 0.01%) for near-exclusive
features. This trades a tiny accuracy loss for significant efficiency gains.

### Algorithm: Bundle Construction

```text
ALGORITHM: ConstructExclusiveBundles(features, max_conflict_rate)
-----------------------------------------------------------------
INPUT:
  features: List of sparse features
  max_conflict_rate: Maximum allowed conflict (default 0.0001)

OUTPUT:
  bundles: List of feature sets that can be bundled

1. // Build conflict graph: edge (i,j) if features i,j conflict
2. conflict_graph = ComputeConflicts(features)

3. // Greedy graph coloring (NP-hard to optimize, greedy is good enough)
4. bundles = []
5. sorted_features = SortByNonZeroCount(features, descending=True)

6. FOR feature IN sorted_features:
7.     best_bundle = None
8.     best_conflict = infinity
9.     
10.    FOR bundle IN bundles:
11.        conflict = CountConflicts(feature, bundle, features)
12.        IF conflict <= max_conflict_rate * n_samples:
13.            IF conflict < best_conflict:
14.                best_bundle = bundle
15.                best_conflict = conflict
16.    
17.    IF best_bundle is not None:
18.        best_bundle.append(feature)
19.    ELSE:
20.        bundles.append([feature])

21. RETURN bundles
```

**Complexity**: O(n_features² × n_samples) worst case, but conflict detection uses
sparse indices making it O(n_features² × avg_nnz).

### Algorithm: Bundle Value Encoding

Once features are bundled, encode them as a single feature:

```text
ALGORITHM: EncodeBundleValue(sample, bundle, bin_offsets)
---------------------------------------------------------
INPUT:
  sample: Feature values for one row
  bundle: List of (feature_idx, bin_mapper) in the bundle
  bin_offsets: Cumulative bin offsets for each feature in bundle

OUTPUT:
  bin_value: Single bin index representing all features

1. bundle_value = 0  // Default: all features are zero/missing

2. FOR i, (feat_idx, mapper) IN enumerate(bundle):
3.     val = sample[feat_idx]
4.     IF val != 0:  // Only one should be non-zero (by exclusivity)
5.         bin = mapper.GetBin(val)
6.         bundle_value = bin_offsets[i] + bin
7.         BREAK  // At most one active feature

8. RETURN bundle_value
```

### Decoding at Split Time

When finding splits, we need to map bundle bins back to original features:

```text
Bundle: [f1 (2 bins), f2 (3 bins), f3 (4 bins)]
Offsets: [0, 2, 5, 9]

Bundle bin 0: Default (all zero)
Bundle bin 1: f1 = bin 0 (offset 0 + 1 = 1)
Bundle bin 2: f1 = bin 1 (offset 0 + 2 = 2... wait, that's f2's range)

Corrected:
  Bins 1-2: f1 (bins 0-1)
  Bins 3-5: f2 (bins 0-2)  
  Bins 6-9: f3 (bins 0-3)
```

Split on bundle bin 4 → "f2 <= threshold_1"

---

## Conflict Detection

### Conflict Definition

Two features **conflict** on sample $i$ if both have non-zero values:
$$\text{conflict}(f_a, f_b) = |\{i : f_a(x_i) \neq 0 \land f_b(x_i) \neq 0\}|$$

### Efficient Conflict Detection

For sparse features (stored as non-zero indices), use set intersection:

```text
ALGORITHM: CountConflicts(feature_a, bundle, features)
------------------------------------------------------
1. conflicts = 0
2. indices_a = NonZeroIndices(feature_a)

3. FOR feature_b IN bundle:
4.     indices_b = NonZeroIndices(feature_b)
5.     conflicts += |indices_a ∩ indices_b|

6. RETURN conflicts
```

**Optimization**: Use bitsets for conflict tracking:

```text
conflict_mask[bundle] = bitset of rows where bundle has any non-zero
new_conflicts = popcount(conflict_mask[bundle] & indices_a)
```

### LightGBM's Threshold

From LightGBM source (`dataset.cpp:118`):

```cpp
const data_size_t single_val_max_conflict_cnt =
    static_cast<data_size_t>(total_sample_cnt / 10000);
```

Default: Allow conflicts in 0.01% of samples.

---

## Multi-Value Bins (Dense Bundling)

### When Bundles Become Dense

If bundled features collectively cover many rows (e.g., >40% non-zero), the bundle
behaves like a dense feature. LightGBM uses two modes:

1. **Single-Value Bin**: Each row has one bin value (works when exclusive)
2. **Multi-Value Bin**: Each row can have multiple active features

### Multi-Value Storage

For dense bundles (>40% coverage), LightGBM stores multiple values per row:

```text
Traditional: bins[row] = single_bin_value

Multi-value: bins[row] = [bin1, bin2, ...]  // CSR-like format
```

**Use case**: When exclusivity assumption breaks down for subset of features.

---

## Binary Feature Detection

### Identifying Binary Features

A feature is binary if it has only 2 unique values (typically 0 and 1):

```text
ALGORITHM: IsBinaryFeature(feature)
-----------------------------------
1. unique_values = CountUnique(feature)
2. RETURN unique_values == 2
```

### Binary Feature Optimization

Binary features need only **1 bit** per sample instead of a full bin index:

| Storage | Memory per sample | For 1M samples |
|---------|-------------------|----------------|
| u8 bin  | 8 bits | 1 MB |
| 1-bit   | 1 bit | 125 KB |

### Bitpacking Multiple Binary Features

Pack 8 binary features into a single byte:

```text
Features: [is_male, is_employed, has_car, ..., is_urban]
Bitpacked: byte = (is_male << 0) | (is_employed << 1) | (has_car << 2) | ...

Reading: is_employed = (byte >> 1) & 1
```

**Histogram optimization**: For 8 packed features, need 2^8 = 256 combined bins.
But can use bit manipulation to extract individual feature histograms efficiently.

---

## Integration with Quantile Binning

### Binary Features Don't Need Quantile Binning

Quantile binning is designed for continuous features with skewed distributions.
For binary features:
- Only 2 possible values
- Sorting/percentile computation is wasted effort
- Equal-width and quantile binning give identical results

**Optimization**: Detect binary features and skip expensive quantile computation.

```text
ALGORITHM: BinFeature(feature, config)
--------------------------------------
1. unique_vals = CountUnique(feature)

2. IF unique_vals == 1:
3.     RETURN TrivialBinMapper()  // Constant feature, skip

4. IF unique_vals == 2:
5.     RETURN BinaryBinMapper(min_val, max_val)  // Special fast path

6. IF config.strategy == Quantile:
7.     RETURN QuantileBinMapper(feature, config.max_bins)
8. ELSE:
9.     RETURN EqualWidthBinMapper(feature, config.max_bins)
```

---

## Performance Analysis

### Memory Savings

For Adult dataset (105 one-hot features from 14 original):

| Approach | Features | Memory (256 bins, 48K samples) |
|----------|----------|--------------------------------|
| No bundling | 105 | 105 × 48K = 5.04 MB |
| Perfect bundling (14 bundles) | 14 | 14 × 48K = 0.67 MB |
| With bitpacking | ~3 bundles | ~0.2 MB |

**Reduction**: Up to **25× less memory**.

### Histogram Building Speedup

Fewer features = fewer histograms to build:

| Approach | Histograms per leaf |
|----------|---------------------|
| 105 features | 105 × 256 bins |
| 14 bundles | 14 × 256 bins |

**Speedup**: ~7× for Adult dataset.

### Split Finding

Must still consider original features for interpretability, but histogram work is reduced:
- Build histogram for bundle
- Map splits back to original features

---

## Library Comparison

### LightGBM (Full EFB)

- Automatic exclusive feature bundling
- Multi-value bins for dense bundles
- Conflict-tolerant bundling (0.01% threshold)
- Enabled by default (`enable_bundle=true`)

### XGBoost (Limited)

- No explicit EFB
- Some sparse optimizations in `tree_method=hist`
- Relies on native categorical support instead

### CatBoost

- Native categorical encoding (ordered target statistics)
- No one-hot explosion
- Different approach: avoid the problem rather than optimize it

---

## References

- Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
  - Section 3.1: Exclusive Feature Bundling
- LightGBM source:
  - `src/io/dataset.cpp`: FastFeatureBundling, FindGroups
  - `include/LightGBM/feature_group.h`: FeatureGroup class
  - `src/treelearner/feature_histogram.hpp`: Histogram with bundles
- XGBoost source:
  - `include/xgboost/tree_model.h`: Sparse handling

---

## Summary

| Technique | Benefit | Complexity | Priority |
|-----------|---------|------------|----------|
| Binary detection | Skip quantile, potential bitpacking | Low | High |
| Exclusive bundling | 5-25× memory reduction | Medium | Medium |
| Multi-value bins | Handles non-exclusive | High | Low |
