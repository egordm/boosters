```markdown
# RFC-0008: Quantization

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: RFC-0001 (Data Matrix)
- **Scope**: Feature quantization for histogram-based training

## Summary

Quantization converts continuous feature values to discrete bin indices. This enables O(bins) split finding instead of O(n) sorting. This RFC covers bin boundary computation (cuts), quantized matrix storage, and bin lookup.

## Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          Quantizer                                  │
│  Computes bin boundaries from feature data                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           BinCuts                                   │
│  Stores bin boundaries for all features (CSC-like format)           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       QuantizedMatrix                               │
│  Dense matrix of bin indices (col-major, u8 or u16)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```text
Features (col-major f32) ──► Quantizer ──► BinCuts
                                             │
Features + BinCuts ─────────────────────────►│──► QuantizedMatrix (col-major u8/u16)
                                                          │
                                                          ▼
                                              Used by HistogramBuilder (RFC-0009)
```

### Why Quantize?

| Without Quantization | With Quantization |
|---------------------|-------------------|
| Sort per feature: O(n log n) | Bin lookup: O(log bins) |
| 4 bytes per value (f32) | 1 byte per value (u8 for ≤256 bins) |
| Cache misses on random access | Better cache locality |

## Components

### BinCuts

Stores bin boundaries for all features in CSC-like format:

```rust
pub struct BinCuts {
    /// Concatenated cut values (bin upper bounds) for all features.
    values: Box<[f32]>,
    /// Feature offsets into values. Length = n_features + 1.
    /// Cuts for feature f are at values[ptrs[f]..ptrs[f+1]].
    ptrs: Box<[u32]>,
}
```

**Bin assignment** for feature with cuts `[c0, c1, c2]`:

```text
value ≤ c0       → bin 0
c0 < value ≤ c1  → bin 1
c1 < value ≤ c2  → bin 2
value > c2       → bin 3
NaN              → bin 4 (missing bin)
```

Number of bins = number of cuts + 1 (regular bins) + 1 (missing bin).

### QuantizedMatrix

Dense matrix of bin indices, **col-major** layout:

```rust
pub struct QuantizedMatrix {
    /// Bin indices, col-major: index[row + col * n_rows].
    data: QuantizedStorage,
    /// Per-feature storage info for adaptive packing.
    feature_info: Box<[FeatureStorageInfo]>,
    n_rows: usize,
    n_features: usize,
}

/// Per-feature storage metadata for adaptive bin packing.
pub struct FeatureStorageInfo {
    pub offset: usize,    // Byte offset into data
    pub n_bins: u16,      // Number of bins for this feature
    pub bits: BinBits,    // Storage width: U4, U8, or U16
}

/// Bin storage width (const generic parameter for kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinBits {
    U4,   // 4-bit: ≤15 bins, 2 values per byte
    U8,   // 8-bit: ≤256 bins (default)
    U16,  // 16-bit: >256 bins
}

enum QuantizedStorage {
    /// Uniform storage: all features use same width.
    Uniform(UniformStorage),
    /// Adaptive storage: per-feature width (4/8/16-bit).
    Adaptive(Box<[u8]>),
}

enum UniformStorage {
    U8(Box<[u8]>),    // max_bin ≤ 256
    U16(Box<[u16]>),  // max_bin > 256
}

impl QuantizedMatrix {
    /// Get bin for (feature, row). Col-major: feature column is contiguous.
    pub fn get_bin(&self, feature: usize, row: usize) -> u16;
    
    /// Get column accessor for a feature (handles different storage widths).
    pub fn col(&self, feature: usize) -> BinColumn<'_>;
    
    /// Get storage info for a feature.
    pub fn feature_info(&self, feature: usize) -> &FeatureStorageInfo;
}

/// Column accessor that abstracts over storage width.
pub enum BinColumn<'a> {
    U4(&'a [u8], usize),   // Packed 4-bit: (data, n_rows)
    U8(&'a [u8]),
    U16(&'a [u16]),
}

impl BinColumn<'_> {
    /// Get bin value at row index.
    #[inline]
    pub fn get(&self, row: usize) -> u16 {
        match self {
            Self::U4(data, _) => {
                let byte = data[row / 2];
                let shift = (row & 1) * 4;
                ((byte >> shift) & 0x0F) as u16
            }
            Self::U8(data) => data[row] as u16,
            Self::U16(data) => data[row],
        }
    }
}
```

**Why col-major?** Histogram building iterates features-first. Col-major gives contiguous access per feature column. See RFC-0009.

### Quantizer

Computes bin cuts from data:

```rust
pub struct Quantizer {
    pub max_bin: u16,  // Default: 256
}

impl Quantizer {
    pub fn compute_cuts(&self, features: &ColMatrix<f32>) -> BinCuts;
    pub fn quantize(&self, features: &ColMatrix<f32>, cuts: &BinCuts) -> QuantizedMatrix;
}
```

## Algorithms

### Cut Computation

Equal-population (quantile) binning ensures each bin has similar sample count:

```text
compute_cuts(features, max_bin):
  for each feature:
    values = feature_col.filter(not_nan).sort()
    
    if values.empty():
      // All missing → no cuts
      continue
    
    n_bins = min(count_unique(values), max_bin)
    
    // Pick quantile cut points
    for i in 1..n_bins:
      q = i / n_bins
      cut = values[q * (len - 1)]
      add_cut_if_unique(feature, cut)
  
  return BinCuts { values, ptrs }
```

**Why quantile binning?** Uniform binning creates many empty bins for skewed distributions. Quantile binning places more cuts where data is dense.

### Bin Search

Binary search to find bin for a value:

```text
search_bin(feature, value):
  if is_nan(value): return missing_bin(feature)
  cuts = self.cuts(feature)
  return binary_search_upper_bound(cuts, value)
```

### Matrix Quantization

```text
quantize(features, cuts):
  for feature in 0..n_features:
    for row in 0..n_rows:
      value = features.get(row, feature)
      bin = cuts.search_bin(feature, value)
      output.set(feature, row, bin)  // Col-major storage
  
  return QuantizedMatrix { output }
```

Parallelizable by feature (columns are independent).

### Adaptive Bin Packing

4-bit packing stores two bin values per byte for features with ≤15 bins:

```text
pack_4bit(bins: &[u8]) -> Vec<u8>:
  packed = Vec::with_capacity((bins.len() + 1) / 2)
  for i in (0..bins.len()).step_by(2):
    low = bins[i]
    high = if i + 1 < bins.len() { bins[i + 1] } else { 0 }
    packed.push(low | (high << 4))
  return packed

unpack_4bit(row: usize, data: &[u8]) -> u8:
  byte = data[row / 2]
  shift = (row & 1) * 4
  return (byte >> shift) & 0x0F
```

**Storage selection per feature**:

```text
select_bin_bits(n_bins: u16) -> BinBits:
  if n_bins <= 15:
    return U4    // 4-bit: 2× memory savings
  elif n_bins <= 256:
    return U8    // 8-bit: default
  else:
    return U16   // 16-bit: rare, many unique values
```

**Memory savings**:
- 50% of features typically have ≤15 unique values
- 4-bit packing saves 50% memory for those features
- Overall savings: ~25% for typical datasets

## Gradient Quantization

For large datasets (>1M rows), gradient values can be quantized to reduce memory bandwidth during histogram building.

### GradientStorage

```rust
/// Gradient storage with configurable precision.
pub enum GradientStorage {
    /// Full precision: grad f32 + hess f32 per sample.
    F32 { grads: Box<[f32]>, hess: Box<[f32]> },
    /// Quantized: grad i8 + hess i8 packed into i16 per sample.
    Int16 { packed: Box<[i16]>, grad_scale: f32, hess_scale: f32 },
    /// Heavily quantized: grad i4 + hess i4 packed into i8 per sample.
    Int8 { packed: Box<[u8]>, grad_scale: f32, hess_scale: f32 },
}

impl GradientStorage {
    /// Quantize f32 gradients to target precision.
    pub fn quantize(
        grads: &[f32],
        hess: &[f32],
        bits: GradientBits,
    ) -> Self;
    
    /// Get gradient/hessian for a sample (dequantizes if needed).
    pub fn get(&self, idx: usize) -> (f32, f32);
}
```

### Quantization Algorithm

```text
quantize_gradients(grads: &[f32], hess: &[f32], bits: GradientBits) -> GradientStorage:
  // Find max absolute values
  max_grad = max(|grads|)
  max_hess = max(|hess|)
  
  match bits:
    Int16:
      // 8 bits each for grad and hess
      grad_scale = max_grad / 127.0
      hess_scale = max_hess / 255.0  // Hessian always positive
      
      packed = []
      for i in 0..len:
        g = round(grads[i] / grad_scale) as i8   // -127..127
        h = round(hess[i] / hess_scale) as u8    // 0..255
        packed.push((g as i16) | ((h as i16) << 8))
      
      return Int16 { packed, grad_scale, hess_scale }
    
    Int8:
      // 4 bits each
      grad_scale = max_grad / 7.0
      hess_scale = max_hess / 15.0
      
      packed = []
      for i in 0..len:
        g = round(grads[i] / grad_scale) as i8   // -7..7
        h = round(hess[i] / hess_scale) as u8    // 0..15
        packed.push((g & 0x0F) | (h << 4))
      
      return Int8 { packed, grad_scale, hess_scale }
```

### When to Quantize

```text
should_quantize_gradients(n_rows: usize, n_outputs: usize, l2_cache: usize) -> GradientBits:
  gradient_bytes = n_rows * n_outputs * 8  // grad + hess f32
  
  if gradient_bytes < l2_cache:
    return F32  // Fits in cache, no benefit from quantization
  
  if n_rows > 10_000_000:
    return Int8  // Very large: aggressive quantization
  
  if n_rows > 1_000_000:
    return Int16  // Large: moderate quantization
  
  return F32
```

**Tradeoffs**:
- **F32**: Full precision, 8 bytes per sample
- **Int16**: ~0.1% accuracy loss, 2 bytes per sample (4× bandwidth reduction)
- **Int8**: ~0.5% accuracy loss, 1 byte per sample (8× bandwidth reduction)

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Col-major layout | Yes | Features-first histogram iteration (RFC-0009) |
| Quantile binning | Yes | Handles skewed distributions |
| Adaptive storage | u8/u16 based on max_bin | 4x memory savings with u8 |
| Missing bin | Dedicated last bin | Clean handling, no sentinel values |
| CSC-like cuts | values + ptrs arrays | Variable cuts per feature, compact |

## Memory

```text
Original features: n_rows × n_features × 4 bytes (f32)
Quantized (u8):    n_rows × n_features × 1 byte

1M rows × 100 features:
  Original: 400 MB
  Quantized: 100 MB (4x reduction)
```

## References

- [XGBoost Quantile Sketch](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
```
