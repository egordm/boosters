# Quantized Feature Matrix

A quantized feature matrix stores pre-computed bin indices for all (sample, feature) pairs,
enabling O(1) histogram updates instead of O(log b) per-sample bin lookups during training.

---

## Core Concept

Given histogram cuts computed during preprocessing, the quantization function:

$$Q(x_{i,j}) = \text{SearchBin}(x_{i,j}, \text{cuts}_j)$$

maps each raw feature value to its bin index. This transformation is performed once
during data loading, amortizing the O(log b) binary search cost.

### The Space-Time Trade-off

```text
Traditional approach:
  - Store raw floats (4 bytes each)
  - Every histogram build: O(n × log(bins)) for bin lookups

Quantized approach:
  - Pre-compute bins once: O(n × log(bins))
  - Store bin indices (1 byte each for ≤256 bins)
  - Every histogram build: O(n) direct indexing
```

Since histogram building happens many times (once per node, per tree, per iteration),
the one-time quantization cost is quickly amortized.

---

## Benefits

| Benefit | Explanation |
|---------|-------------|
| **Memory reduction** | 1 byte (uint8) vs 4 bytes (float32) per value — 4× reduction |
| **Cache efficiency** | 4× more data fits in cache, reducing memory bandwidth pressure |
| **Integer operations** | Bin comparisons are integer ops, faster than float comparisons |
| **Determinism** | Integer comparisons are exact across platforms (no float issues) |

### Type Selection

The storage type adapts to the maximum number of bins:

| max_bins | Storage Type | Bytes per Value |
|----------|--------------|-----------------|
| ≤ 256    | uint8        | 1 |
| ≤ 65536  | uint16       | 2 |
| > 65536  | uint32       | 4 |

**256 bins is the sweet spot**: sufficient precision for most datasets while fitting
in a single byte.

---

## Memory Layouts

### Dense (Row-Major)

For data without missing values, store bin indices in a contiguous 2D array:

```text
Layout: bins[row × n_features + feature]

Row 0: [b₀₀, b₀₁, b₀₂, ..., b₀ₙ]
Row 1: [b₁₀, b₁₁, b₁₂, ..., b₁ₙ]
Row 2: [b₂₀, b₂₁, b₂₂, ..., b₂ₙ]
...
```

**Access pattern**: O(1) per cell, excellent cache locality for row-wise iteration.

**Best for**: Dense data, histogram building (iterates by row).

### Sparse (CSR)

For sparse data or data with many missing values, Compressed Sparse Row format:

```text
row_ptr: [0, 2, 3, 6]           ← Row boundaries  
col_idx: [0, 2, 1, 0, 1, 2]     ← Feature indices  
values:  [b₀, b₁, b₂, b₃, b₄, b₅]  ← Bin values

Row 0: 2 values at features 0, 2
Row 1: 1 value at feature 1
Row 2: 3 values at features 0, 1, 2
```

**Invariant**: Missing values are implicit (not stored), saving memory.

**Best for**: >50% sparsity, memory-constrained environments.

### GPU (ELLPACK)

GPUs require uniform memory access patterns for coalesced reads:

```text
ELLPACK (padded dense format):
  row_stride = max_features_per_row

  Row 0: [b₀, b₁, PAD]
  Row 1: [b₀, PAD, PAD]  
  Row 2: [b₀, b₁, b₂]
```

All rows padded to the same width, enabling predictable strided access.

**Why GPUs need this**: Threads in a warp (32 threads) must access adjacent memory
addresses for efficient coalesced reads. Variable-length rows break this pattern.

**Trade-off**: Padding wastes memory for sparse data, but enables massive parallelism.

---

## Global Bin Indexing

During histogram building, we need a **global bin index** across all features to use
a single flat histogram array:

```text
Feature 0: 3 bins → local indices {0, 1, 2}
Feature 1: 4 bins → local indices {0, 1, 2, 3}  
Feature 2: 3 bins → local indices {0, 1, 2}

feature_offsets = [0, 3, 7, 10]   // Cumulative sum

Global bin = feature_offsets[feature] + local_bin

Example: feature=1, local_bin=2 → global_bin = 3 + 2 = 5
```

This enables O(1) mapping and a single contiguous histogram array.

---

## Quantization Algorithm

```text
ALGORITHM: BuildQuantizedMatrix(data, max_bins)
───────────────────────────────────────────────
Input:  Raw feature matrix data[n_rows × n_features]
        Maximum bins per feature
Output: Quantized matrix with bin indices

1. cuts ← BuildHistogramCuts(data, max_bins)  // See histogram-cuts.md

2. SELECT storage type:
   IF max_bins ≤ 256:   use uint8
   ELIF max_bins ≤ 65536: use uint16
   ELSE: use uint32

3. PARALLEL FOR row ∈ [0, n_rows):
       FOR feature ∈ [0, n_features):
           value ← data[row, feature]
           IF is_missing(value):
               bin ← MISSING_BIN  // Special sentinel
           ELSE:
               bin ← BinarySearch(value, cuts[feature])
           index[row, feature] ← bin

4. RETURN QuantizedMatrix(index, cuts, feature_offsets)
```

**Complexity**: O(n × d × log(bins)) for n samples, d features.

**Parallelization**: Embarrassingly parallel — each cell independent.

---

## Access During Histogram Building

The histogram building loop uses the quantized matrix for O(1) bin lookup:

```text
ALGORITHM: BuildHistogram(node_samples, gradients, qmatrix)
───────────────────────────────────────────────────────────
histogram ← zeros(total_bins)

FOR row ∈ node_samples:
    grad ← gradients[row]
    FOR feature ∈ [0, n_features):
        local_bin ← qmatrix[row, feature]
        global_bin ← feature_offset[feature] + local_bin
        histogram[global_bin] += grad

RETURN histogram
```

**Key insight**: The inner loop is just an array index + accumulate — no binary search.

---

## Missing Value Handling

Different strategies for representing missing values:

| Strategy | Description | Memory | Lookup |
|----------|-------------|--------|--------|
| **Sentinel bin** | Assign bin_id = max_bins to missing | Same | O(1) |
| **Separate mask** | Boolean array tracking missing | +12.5% | O(1) |
| **Implicit (CSR)** | Missing values not stored | Sparse wins | O(log nnz) |

XGBoost uses sentinel bins. LightGBM uses a mix depending on sparsity.

---

## Memory Estimates

For 1 million samples, 100 features:

| Format | Size | Notes |
|--------|------|-------|
| Raw float32 | 400 MB | 4 bytes × 100M cells |
| Quantized uint8 | 100 MB | 1 byte × 100M cells |
| CSR (50% sparse) | ~75 MB | Saves on missing, adds indices |
| ELLPACK (GPU) | Varies | Depends on row_stride, bit-packing |

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Purpose** | Pre-computed bin indices for O(1) histogram updates |
| **Key trade-off** | One-time O(log b) quantization vs per-histogram lookup |
| **Memory** | 4× reduction with uint8 bins |
| **CPU format** | Dense or CSR depending on sparsity |
| **GPU format** | ELLPACK (padded, bit-packed) |
| **Missing values** | Sentinel bin or implicit (CSR) |
