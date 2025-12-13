# RFC-0008: Feature Quantization

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: RFC-0001 (Data Matrix)
- **Scope**: Feature binning for histogram-based split finding

## Summary

Feature quantization converts continuous feature values to discrete bin indices. This enables O(bins) split finding instead of O(n log n) sorting. This RFC covers bin boundary computation, quantized storage, and bin lookup.

**Note**: Gradient quantization is a separate concern covered in RFC-0012.

## Overview

```text
ColMatrix<f32>  ──►  quantize()  ──►  QuantizedFeatures
   (features)          (once)          (used throughout training)
```

### Why Quantize Features?

| Aspect | Without Quantization | With Quantization |
|--------|---------------------|-------------------|
| Split finding | Sort per feature: O(n log n) | Histogram scan: O(bins) |
| Storage | 4 bytes per value (f32) | ~1 byte per value |
| Cache | Poor locality | Sequential per feature |

With max_bins=256 and n=1M rows, split finding is ~20× faster.

### Key Insight

Feature quantization is **lossless for split finding**. We don't need exact feature values—only which bin each value falls into. The histogram aggregates gradients by bin, and splits occur at bin boundaries.

## QuantizedFeatures

A single structure holding both bin boundaries (cuts) and quantized data:

```rust
pub struct QuantizedFeatures {
    /// Bin boundaries for all features (CSC-like: values + offsets).
    cut_values: Box<[f32]>,
    cut_offsets: Box<[u32]>,
    
    /// Quantized bin indices, col-major layout.
    bins: Box<[u8]>,
    
    /// Per-feature metadata for adaptive storage.
    features: Box<[FeatureInfo]>,
    
    n_rows: usize,
    n_features: usize,
}

pub struct FeatureInfo {
    pub byte_offset: usize,  // Offset into bins buffer
    pub n_bins: u16,         // Number of bins (cuts + 2)
    pub bits: BinBits,       // Storage width: U4, U8, or U16
}

pub enum BinBits { U4, U8, U16 }
```

### Interface

```rust
impl QuantizedFeatures {
    /// Create from raw features using quantile binning.
    pub fn from_features(features: &ColMatrix<f32>, max_bins: u16) -> Self;
    
    /// Number of features.
    pub fn n_features(&self) -> usize;
    
    /// Number of rows.
    pub fn n_rows(&self) -> usize;
    
    /// Number of bins for a feature.
    pub fn n_bins(&self, feature: usize) -> u16;
    
    /// Maximum bins across all features.
    pub fn max_bins(&self) -> u16;
    
    /// Get cut values for a feature (for prediction/export).
    pub fn cuts(&self, feature: usize) -> &[f32];
    
    /// Get bin index for (feature, row). Used in histogram building.
    #[inline]
    pub fn bin(&self, feature: usize, row: usize) -> u16;
}
```

### Why Combined?

Previously we had separate `BinCuts` and `QuantizedMatrix`:

| Separate | Combined |
|----------|----------|
| `BinCuts` + `QuantizedMatrix` | `QuantizedFeatures` |
| Two allocations | One allocation |
| Must keep in sync | Always consistent |
| Pass two arguments | Pass one argument |

The cuts and bins are **always used together** and have the same lifetime. Combining them simplifies the API and ensures consistency.

## Bin Assignment

For a feature with cuts `[c0, c1, c2]`:

```text
value ≤ c0       → bin 0
c0 < value ≤ c1  → bin 1
c1 < value ≤ c2  → bin 2
value > c2       → bin 3
NaN              → bin 4 (missing bin, always last)

n_bins = n_cuts + 1 (regular) + 1 (missing) = n_cuts + 2
```

## Storage Layout

### Col-Major Bins

Bin indices stored in **col-major** order: features are contiguous.

```text
Memory layout for 3 features, 4 rows:

  Feature 0    Feature 1    Feature 2
  ─────────    ─────────    ─────────
  [row0]       [row0]       [row0]
  [row1]       [row1]       [row1]
  [row2]       [row2]       [row2]
  [row3]       [row3]       [row3]
```

**Why col-major?** Histogram building iterates features-first. Col-major makes each feature column contiguous, maximizing cache efficiency.

### Adaptive Bit Width

Different features need different storage based on bin count:

```text
n_bins ≤ 15   → U4 (4 bits, 2 values per byte)
n_bins ≤ 256  → U8 (8 bits, default)
n_bins > 256  → U16 (16 bits, rare)
```

The `bin()` method handles unpacking transparently:

```text
bin(feature, row):
    info = self.features[feature]
    match info.bits:
        U4:
            byte_idx = info.byte_offset + row / 2
            byte = self.bins[byte_idx]
            if row % 2 == 0: return byte & 0x0F
            else: return byte >> 4
        U8:
            return self.bins[info.byte_offset + row]
        U16:
            idx = info.byte_offset + row * 2
            return self.bins[idx] | (self.bins[idx + 1] << 8)
```

### CSC-like Cut Storage

Cuts stored as concatenated values with offsets:

```text
Feature 0 cuts: [0.5, 1.0]
Feature 1 cuts: [10.0, 20.0, 30.0]
Feature 2 cuts: [100.0]

cut_values:  [0.5, 1.0, 10.0, 20.0, 30.0, 100.0]
cut_offsets: [0, 2, 5, 6]

cuts(feature=1) → cut_values[2..5] = [10.0, 20.0, 30.0]
```

## Algorithms

### Quantization (Construction)

```text
QuantizedFeatures::from_features(features, max_bins):
    n_rows = features.n_rows()
    n_features = features.n_cols()
    
    // Phase 1: Compute cuts for all features
    all_cuts = []
    cut_offsets = [0]
    feature_info = []
    
    for feat in 0..n_features:
        cuts = compute_quantile_cuts(features.col(feat), max_bins)
        all_cuts.extend(cuts)
        cut_offsets.push(all_cuts.len())
        
        n_bins = cuts.len() + 2  // regular bins + missing bin
        bits = select_bit_width(n_bins)
        feature_info.push(FeatureInfo { n_bins, bits, byte_offset: 0 })
    
    // Phase 2: Compute byte offsets
    byte_offset = 0
    for info in feature_info:
        info.byte_offset = byte_offset
        byte_offset += bytes_needed(info.bits, n_rows)
    
    // Phase 3: Quantize values into bins
    bins = allocate(byte_offset)
    
    for feat in 0..n_features:
        cuts = get_cuts(all_cuts, cut_offsets, feat)
        info = feature_info[feat]
        
        for row in 0..n_rows:
            value = features.get(row, feat)
            bin = search_bin(cuts, value)
            write_bin(bins, info, row, bin)
    
    return QuantizedFeatures { 
        cut_values: all_cuts, 
        cut_offsets, 
        bins, 
        features: feature_info,
        n_rows, 
        n_features 
    }
```

### Quantile Cut Computation

Equal-population binning places cuts so each bin has similar sample count:

```text
compute_quantile_cuts(column, max_bins):
    // Filter out NaN values
    values = column.filter(not_nan).sort()
    
    if values.is_empty():
        return []  // All missing → no cuts
    
    n_unique = count_unique(values)
    n_bins = min(n_unique, max_bins)
    
    if n_bins <= 1:
        return []  // Single value → no cuts needed
    
    // Pick quantile cut points
    cuts = []
    for i in 1..n_bins:
        q = i / n_bins
        idx = floor(q * (values.len() - 1))
        cut = values[idx]
        
        // Avoid duplicate cuts
        if cuts.is_empty() or cut > cuts.last():
            cuts.push(cut)
    
    return cuts
```

**Why quantile binning?** Uniform-width bins create many empty bins for skewed distributions. Quantile binning adapts to data density.

### Bin Search

Binary search to find bin for a value:

```text
search_bin(cuts, value):
    if is_nan(value):
        return cuts.len() + 1  // Missing bin (last)
    
    // Binary search: find first cut >= value
    lo, hi = 0, cuts.len()
    while lo < hi:
        mid = (lo + hi) / 2
        if cuts[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    
    return lo  // Bin index
```

## Memory Analysis

```text
Original (f32):     n_rows × n_features × 4 bytes
Quantized (u8):     n_rows × n_features × 1 byte
Quantized (mixed):  ~0.75 bytes/value with U4 packing

Example: 1M rows × 100 features
  Original:  400 MB
  All U8:    100 MB
  Mixed:     ~75 MB (assuming 50% features use U4)
```

## Design Decisions

### DD-1: Combined Structure

**Decision**: Combine cuts and bins into single `QuantizedFeatures` struct.

**Rationale**:

- Cuts and bins always used together
- Single allocation, single lifetime
- Simpler API: pass one object instead of two
- Impossible to have mismatched cuts/bins

### DD-2: Col-Major Layout

**Decision**: Store bins in col-major order.

**Rationale**: Histogram building iterates features-first. Col-major makes each feature column contiguous, maximizing cache efficiency.

### DD-3: Quantile Binning

**Decision**: Use equal-population (quantile) binning.

**Rationale**: Real-world features often have skewed distributions. Equal-width creates many empty bins; quantile adapts to data density.

### DD-4: Dedicated Missing Bin

**Decision**: Reserve the last bin for missing values (NaN).

**Rationale**:

- Clean separation from regular bins
- No sentinel value needed
- Missing value handling in split finding is explicit

### DD-5: Adaptive Storage Width

**Decision**: Support U4/U8/U16 per feature based on bin count.

**Rationale**:

- Many features have few unique values (categoricals, booleans)
- U4 packing saves ~25% memory overall
- U16 handles high-cardinality features (rare)

## Integration

Used by histogram building (RFC-0009):

```text
// In histogram accumulation
for row in rows:
    bin = quantized.bin(feature, row)
    histogram.add(bin, grads.grad(row), grads.hess(row))
```

Used by tree training (RFC-0007):

```text
// Quantization happens once before training
quantized = QuantizedFeatures::from_features(features, max_bins)

// Training uses quantized features
for round in 0..n_rounds:
    tree = grower.grow(quantized, grads, rows)
    // ...
```

## References

- [XGBoost Quantile Sketch](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
- [LightGBM Feature Binning](https://github.com/microsoft/LightGBM/blob/master/src/io/bin.cpp)
