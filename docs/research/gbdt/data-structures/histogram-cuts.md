# Histogram Cuts (Bin Boundaries)

## Overview

Histogram cuts (also called bin boundaries or split candidates) partition continuous feature values into discrete intervals. Given $n$ cut points, we create $n+1$ bins. The quantization function maps:

$$f: \mathbb{R} \rightarrow \{0, 1, \ldots, n\}$$

The cuts are typically chosen via **quantile sketch** algorithms to ensure each bin contains approximately the same number of samples (equal-frequency binning), which maximizes information gain potential per bin.

## Why Quantile-Based?

Consider two binning strategies for a feature with values clustered around 0 and 100:

| Strategy | Bins | Problem |
|----------|------|---------|
| **Equal-width** | [0-25], [25-50], [50-75], [75-100] | Most data in first/last bin only |
| **Equal-frequency** | Adaptive based on data density | Each bin equally useful for splits |

Equal-frequency (quantile) binning ensures:

- Each bin contains roughly the same number of samples
- Dense regions get more resolution (more bins where data concentrates)
- Sparse regions don't waste bins
- Approximately balanced gradient/hessian sums per bin

This directly relates to split quality: a bin with 10% of data can only improve the split by at most 10% of total information, so giving more bins to dense regions maximizes the chance of finding good splits.

## Theory: How Cuts Are Computed

### Method 1: Exact Quantiles (Small Data)

For data that fits in memory, compute exact quantiles:

```
ALGORITHM: ExactQuantileCuts(feature_values, max_bins)
-------------------------------------------------------
1. sorted_values <- SORT(feature_values)
2. n <- LENGTH(sorted_values)
3. cuts <- []
4. FOR i FROM 1 TO max_bins - 1:
       q <- i / max_bins              // Target quantile (e.g., 0.25, 0.5, 0.75)
       idx <- FLOOR(q * n)            // Index in sorted array
       APPEND(cuts, sorted_values[idx])
5. RETURN cuts
```

**Time complexity**: O(n log n) for sorting, where n is the number of samples.

### Method 2: Weighted Quantile Sketch (Large/Distributed Data)

For distributed or streaming data where exact sorting is impractical, XGBoost uses a **weighted quantile sketch** algorithm (described in the XGBoost paper, Section 3.3).

The key insight is that we want bins to contain equal **weighted sums**, not equal counts. In gradient boosting, the natural weight is the Hessian (second derivative):

$$\text{Find cuts } \{c_1, \ldots, c_k\} \text{ such that } \sum_{x_i < c_j} h_i \approx \sum_{c_j \le x_i < c_{j+1}} h_i$$

This is called **adaptive binning** or **Hessian-weighted quantiles**.

**Why weight by Hessian?** The optimal leaf value is $-G/H$ (gradient sum over Hessian sum). Regions with high Hessian have more "curvature" in the loss function and benefit from finer binning resolution.

The sketch algorithm:

1. Process data in chunks (summaries)
2. Each summary maintains approximate quantile information
3. Merge summaries from different workers (associative operation)
4. Guarantees bounded approximation error: epsilon-approximate quantiles

> **Reference**: XGBoost paper Section 3.3, "Weighted Quantile Sketch", and `xgboost/src/common/quantile.h`

### LightGBM's Approach

LightGBM's `BinMapper` class uses similar quantile-based binning with additional optimizations:

- Stores bin upper bounds (`bin_upper_bound_` array)
- Tracks the most-frequent bin for zero-value optimization
- Supports explicit missing value bins
- Handles categorical features with a separate mapping

> **Reference**: `LightGBM/include/LightGBM/bin.h`, class `BinMapper`

## Conceptual Storage Structure

### XGBoost: HistogramCuts (CSR-like Layout)

XGBoost stores all feature cuts in a single concatenated array with pointers to delimit each feature's range:

```
+-----------------------------------------------------------+
| cut_values_: [0.5, 1.5 | 10, 20, 30 | 0.0, 0.5]           |
|               |-feat 0-| |--feat 1--| |-feat 2-|          |
+-----------------------------------------------------------+
| cut_ptrs_:   [0, 2, 5, 7]                                 |
|               |  |  |  +-- end (total bins)               |
|               |  |  +-- feat 2 starts at index 5          |
|               |  +-- feat 1 starts at index 2             |
|               +-- feat 0 starts at index 0                |
+-----------------------------------------------------------+
| min_vals_:   [0.0, 5.0, -1.0]  (minimum per feature)      |
+-----------------------------------------------------------+
```

This CSR-like (Compressed Sparse Row) layout:

- Concatenates all cuts into one contiguous array
- Uses pointer array to delimit each feature's range  
- Enables O(log b) bin lookup via binary search (b = bins per feature)
- Memory efficient: no per-feature allocation overhead

> **Reference**: `xgboost/src/common/hist_util.h`, class `HistogramCuts`

### LightGBM: BinMapper (Per-Feature)

LightGBM uses a separate `BinMapper` object for each feature:

```
+-----------------------------------------------------------+
| bin_upper_bound_: [0.5, 1.5, 2.5, ...]                    |
+-----------------------------------------------------------+
| default_bin_:     3      (bin for value 0.0)              |
| most_freq_bin_:   2      (most common bin, for GOSS)      |
| num_bin_:         256                                     |
| missing_type_:    NaN    (how missing values encoded)     |
+-----------------------------------------------------------+
```

Each feature has its own `BinMapper` instance storing:

- Upper bounds of each bin
- Metadata for optimization (sparse handling, most frequent bin)
- Missing value strategy

> **Reference**: `LightGBM/include/LightGBM/bin.h`, class `BinMapper`

## Bin Lookup: Theory and Algorithm

### The Binary Search Approach

Finding which bin a value belongs to is essentially a **binary search** for the first cut point greater than the value:

```
ALGORITHM: SearchBin(value, cuts[0..n-1])
-----------------------------------------
1. IF value < cuts[0]:
       RETURN 0                    // Below first cut -> bin 0
2. IF value >= cuts[n-1]:
       RETURN n                    // Above last cut -> last bin
3. RETURN UPPER_BOUND(cuts, value) // Binary search
```

The XGBoost implementation uses `std::upper_bound` to find the first cut strictly greater than the value.

> **Reference**: `xgboost/src/common/hist_util.h`, method `HistogramCuts::SearchBin()`

### Worked Example

```
cuts = [0.5, 1.5, 2.5]  -> Creates 4 bins: (-inf, 0.5), [0.5, 1.5), [1.5, 2.5), [2.5, +inf)

value = 0.3  -> bin 0  (less than first cut 0.5)
value = 0.5  -> bin 1  (equals first cut -> goes to right bin)  
value = 1.0  -> bin 1  (between cuts[0]=0.5 and cuts[1]=1.5)
value = 3.0  -> bin 3  (greater than last cut 2.5)
```

**Complexity**: O(log b) per lookup, where b is the number of bins for that feature.

## Global Bin Indexing

During histogram building, we need a **global bin index** that spans all features, enabling a single flat histogram array.

### The Offset Scheme

```
Feature 0: 3 bins -> local indices {0, 1, 2}
Feature 1: 4 bins -> local indices {0, 1, 2, 3}  
Feature 2: 3 bins -> local indices {0, 1, 2}

feature_offsets = [0, 3, 7, 10]   // Cumulative sum of bin counts

Global bin index for (feature=1, local_bin=2):
  = feature_offsets[1] + local_bin
  = 3 + 2 = 5
```

This indexing scheme enables:

- Single contiguous histogram array of size `total_bins`
- O(1) mapping from (feature, local_bin) to global index
- Cache-efficient histogram accumulation

### Why This Matters

With global bin indexing, histogram building becomes a simple scatter-add operation:

```
histogram[global_bin_index] += gradient_pair
```

No per-feature histogram allocation, no complex indexing—just one array, one index computation.

## Missing Value Handling

Missing values (NaN, null) require special treatment in histogram-based methods:

| Strategy | Description | Used By |
|----------|-------------|---------|
| **Separate bin** | NaN gets its own bin (typically the last bin) | LightGBM |
| **Learn direction** | During training, try NaN going left and right; pick better | XGBoost |
| **Default direction** | Use a configurable default (left or right) | Both |

### XGBoost's Learned Direction

XGBoost's key innovation is **learning** the optimal direction for missing values:

1. When evaluating a split, compute gain assuming NaN goes left
2. Also compute gain assuming NaN goes right
3. Pick the direction with higher gain
4. Store this direction in the tree node for inference

This happens automatically during training—no user configuration needed.

> **Reference**: XGBoost paper Section 3.4, "Sparsity-aware Split Finding"

### LightGBM's Explicit Bin

LightGBM assigns missing values to a dedicated bin and tracks the `missing_type_` (None, Zero, or NaN) per feature. The `default_bin_` field stores which bin represents missing/zero values.

## Configuration and Trade-offs

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `max_bins` | 256 | Maximum bins per feature |
| `min_child_weight` | Varies | Minimum Hessian sum per bin |

### Why 256 Bins?

The magic number 256 is popular because:

- Fits in a single byte (`uint8`): 4x memory reduction vs float32
- Provides sufficient precision for most datasets
- More bins have diminishing returns (the 257th split point rarely helps)
- Powers of 2 enable certain SIMD optimizations

### Trade-off Analysis

| Bins | Memory | Precision | Training Speed |
|------|--------|-----------|----------------|
| 64 | Very low | May underfit | Faster |
| 256 | Low | Good balance | Good |
| 512 | Medium | Slightly better | Slower |
| 4096 | High | Marginal gain | Much slower |

Beyond ~256 bins, you're mostly spending memory and time for negligible accuracy gains.

## Memory Impact: A Worked Example

Consider a dataset with 1 million samples and 100 features:

```
Without quantization:
  Storage = 1M x 100 x 4 bytes (float32) = 400 MB

With quantization (256 bins = uint8):
  Quantized matrix = 1M x 100 x 1 byte = 100 MB
  Cut storage      = 100 features x 256 bins x 4 bytes = 100 KB
  Total = ~100 MB

Memory reduction: 4x (from 400 MB to 100 MB)
```

For larger datasets or more features, the savings multiply. This is why histogram-based methods scale to datasets that don't fit in memory with traditional exact methods.

## Summary

| Aspect | Description |
|--------|-------------|
| **Purpose** | Map continuous feature values -> discrete bin indices |
| **Core algorithm** | Quantile sketch for equal-frequency bins |
| **Storage** | CSR-like concatenated cuts (XGBoost) or per-feature arrays (LightGBM) |
| **Lookup** | Binary search, O(log b) |
| **Key benefit** | 4x memory reduction, cache efficiency, integer comparisons |
| **Typical config** | 256 bins per feature |

### Key References

- XGBoost paper: "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016), Section 3.3
- `xgboost/src/common/hist_util.h` — `HistogramCuts` class
- `xgboost/src/common/quantile.h` — Weighted quantile sketch implementation
- `LightGBM/include/LightGBM/bin.h` — `BinMapper` class
