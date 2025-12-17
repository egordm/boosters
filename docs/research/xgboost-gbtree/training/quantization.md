# Quantization and Sketching

## Overview

Quantization converts continuous feature values into discrete bin indices. This is the
foundation of histogram-based tree training — instead of sorting N samples per feature,
we aggregate into ~256 bins and scan those bins.

## The Quantization Problem

**Goal**: Map continuous values to discrete bins such that split quality is preserved.

**Naive approach**: Uniform binning (equal-width bins)
- Problem: If values are skewed, most samples land in few bins
- Example: Income data with 90% under $100k gets one bin, missing important splits

**Better approach**: Quantile binning (equal-population bins)
- Each bin contains approximately equal number of samples
- More bins where data is dense, fewer where sparse
- Preserves split quality for any distribution

```text
Feature Distribution (skewed):
[1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200, 500]

Uniform bins (width=100):
  Bin 0 [0-100]:   1,2,3,4,5,6,7,8,9,100  (10 samples)
  Bin 1 [100-200]: 200                      (1 sample)
  Bin 2 [200-300]: 500                      (1 sample)
  → Most variation lost in Bin 0

Quantile bins (3 bins):
  Bin 0: 1,2,3,4     (cuts at 4)
  Bin 1: 5,6,7,8,9   (cuts at 9)
  Bin 2: 100,200,500 (cuts at ∞)
  → Better split granularity where data is dense
```

## Quantile Sketching

Computing exact quantiles requires storing all values — infeasible for large datasets.
XGBoost uses **weighted quantile sketching** (based on Greenwald-Khanna algorithm) to
compute approximate quantiles in streaming fashion.

### Key Properties

1. **Streaming**: Process data in batches, don't need all values in memory
2. **Mergeable**: Sketches from different workers can be merged (distributed training)
3. **Weighted**: Support sample weights for gradient-weighted quantiles
4. **Bounded error**: Guarantees on quantile approximation quality

### Weighted Quantiles

For gradient boosting, we use **hessian-weighted quantiles**. This ensures more bins
where the loss function is more sensitive (higher hessian = steeper curvature).

```text
Standard quantile: each sample has equal weight
Weighted quantile: sample i has weight h_i (hessian)

Why hessians?
- High hessian = loss is sensitive here = want fine-grained bins
- Low hessian = loss is flat here = coarse bins are fine
```

### XGBoost Implementation

From `src/common/quantile.h`:

```cpp
// Simplified sketch structure
struct SketchEntry {
  float value;     // Feature value
  float rmin;      // Minimum rank
  float rmax;      // Maximum rank  
  float wmin;      // Minimum weighted rank
};

// Key operations:
// 1. Push values into sketch
// 2. Prune/compress sketch when too large
// 3. Query quantile at rank r
```

## HistogramCuts Structure

The result of quantization is `HistogramCuts` — a CSC-like structure storing bin
boundaries for all features.

```cpp
class HistogramCuts {
  HostDeviceVector<float> cut_values_;   // Bin boundaries (upper bounds)
  HostDeviceVector<uint32_t> cut_ptrs_;  // Feature offsets into cut_values_
  HostDeviceVector<float> min_vals_;     // Minimum value per feature
};
```

**Example**:
```text
Feature 0: bins [0, 5), [5, 10), [10, ∞)
Feature 1: bins [0, 2), [2, 4), [4, 6), [6, ∞)

cut_ptrs_  = [0, 2, 5]        // Feature 0 has 2 cuts, Feature 1 has 3 cuts
cut_values_ = [5, 10, 2, 4, 6] // All cut points concatenated
min_vals_  = [0, 0]           // Minimum value per feature
```

### Bin Index Lookup

To find the bin for a value:
```cpp
bst_bin_t SearchBin(float value, bst_feature_t fidx) {
  auto begin = cut_ptrs_[fidx];
  auto end = cut_ptrs_[fidx + 1];
  // Binary search for first cut > value
  auto it = upper_bound(cut_values_ + begin, cut_values_ + end, value);
  return it - cut_values_.begin();
}
```

## GHistIndexMatrix

Once cuts are computed, we convert the entire dataset to bin indices:

```cpp
class GHistIndexMatrix {
  common::Index index;         // Bin indices (u8/u16/u32)
  std::vector<size_t> row_ptr; // Row pointers (CSR format for sparse)
  HistogramCuts cut;           // The cuts used for quantization
  bool isDense_;               // Dense or sparse storage
};
```

### Dense vs Sparse

**Dense storage** (no missing values):
- Simple 2D array: `index[row * n_features + col]`
- Bin index can be compressed by subtracting feature offset

**Sparse storage** (missing values present):
- CSR format: `row_ptr` + flat bin indices
- Bin indices are global (not compressed)

### Compressed Bin Indices

When `max_bin ≤ 256`:
- Store bin indices as `u8` (1 byte per value)
- Need to add back feature offset when reading

When `256 < max_bin ≤ 65536`:
- Store as `u16` (2 bytes per value)

This compression significantly reduces memory bandwidth.

## XGBoost Quantization Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│  DMatrix (raw features)                                         │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  SketchOnDMatrix / DeviceSketch                                 │
│  - Create quantile sketch per feature                           │
│  - Process in batches for large data                            │
│  - Support sample weights (hessians) if provided                │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  HistogramCuts                                                  │
│  - Finalized bin boundaries per feature                         │
│  - Stored with model for inference                              │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  GHistIndexMatrix                                               │
│  - All features converted to bin indices                        │
│  - Ready for histogram building                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Considerations for booste-rs

### What We Need

1. **HistogramCuts structure**: Store bin boundaries per feature
2. **SearchBin function**: Convert float → bin index efficiently
3. **GHistIndexMatrix**: Store quantized dataset

### What We Can Simplify

1. **Skip streaming sketches initially**: Load all data, compute exact quantiles
2. **Skip hessian-weighted quantiles initially**: Use equal-population bins
3. **Focus on dense data first**: Simpler storage, easier to optimize

### Potential Improvements

1. **SIMD bin lookup**: Vectorize the binary search for multiple values
2. **Batch quantization**: Quantize many values at once for cache efficiency
3. **Lazy quantization**: Only quantize features used in current tree (column sampling)

## Source Code References

| Component | XGBoost Source |
|-----------|----------------|
| Quantile Sketch | `src/common/quantile.h` |
| HistogramCuts | `src/common/hist_util.h` |
| GHistIndexMatrix | `src/data/gradient_index.{h,cc}` |
| SketchOnDMatrix | `src/common/hist_util.cc` |
| QuantileDMatrix | `src/data/quantile_dmatrix.{h,cc}` |
