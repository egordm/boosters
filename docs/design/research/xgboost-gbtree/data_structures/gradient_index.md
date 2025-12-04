# GHistIndexMatrix: Quantized Feature Storage

## Overview

`GHistIndexMatrix` (Gradient Histogram Index Matrix) stores the pre-quantized
feature values. Each cell contains a bin index instead of the original float value.

## Purpose

Converting features to bin indices is expensive (binary search per value).
`GHistIndexMatrix` does this once during construction, enabling fast histogram
building during training.

## Structure

```cpp
class GHistIndexMatrix {
  // Quantized bin indices (dense or CSR format)
  common::Index index;
  
  // For CSR format: row pointers
  std::vector<size_t> row_ptr;
  
  // Cut points used for quantization
  common::HistogramCuts cut;
  
  // Whether data is dense or sparse
  bool isDense_;
  
  // Number of features
  size_t nfeatures_;
  
  // Maximum bins per feature (for selecting index type)
  size_t max_bins_;
};
```

## Index Type Selection

The `Index` type adapts to the number of bins:

```cpp
class Index {
  // Actual storage - one of these is used
  std::vector<uint8_t> data_u8;   // max_bins <= 256
  std::vector<uint16_t> data_u16; // max_bins <= 65536
  std::vector<uint32_t> data_u32; // otherwise
  
  // Type flag
  enum class Type { kUint8, kUint16, kUint32 };
  Type type_;
};
```

## Dense Layout

For dense data without missing values:

```text
Row-major storage: [row0_feat0, row0_feat1, ..., row1_feat0, ...]

index[row * n_features + feature] = bin_idx
```

## CSR Layout

For sparse data or data with missing values:

```text
row_ptr: [0, nnz_row0, nnz_row0+nnz_row1, ...]
index: [bin_indices for all non-missing values]
feature_idx: [feature indices for each value in index]
```

## Building GHistIndexMatrix

### From Dense Data

```cpp
void Init(DMatrix* p_fmat, int32_t max_bins) {
  // Step 1: Build cut points from data
  cut.Build(p_fmat, max_bins);
  
  // Step 2: Allocate index storage
  size_t total_elements = n_rows * n_features;
  index.Resize(total_elements);
  
  // Step 3: Quantize each value
  for (size_t row = 0; row < n_rows; row++) {
    auto row_data = p_fmat->GetRow(row);
    for (size_t f = 0; f < n_features; f++) {
      float value = row_data[f];
      uint32_t bin = cut.SearchBin(value, f);
      index.Set(row * n_features + f, bin);
    }
  }
}
```

### Parallel Construction

```cpp
void InitParallel(DMatrix* p_fmat, int32_t max_bins) {
  // Build cuts (may require full pass over data)
  cut.Build(p_fmat, max_bins);
  
  // Parallel quantization
  #pragma omp parallel for schedule(static)
  for (size_t row = 0; row < n_rows; row++) {
    auto row_data = p_fmat->GetRow(row);
    for (size_t f = 0; f < n_features; f++) {
      uint32_t bin = cut.SearchBin(row_data[f], f);
      index.Set(row * n_features + f, bin);
    }
  }
}
```

## Accessing Data

### Row Access

```cpp
// Get bin indices for a row
std::span<const uint32_t> GetRow(size_t row_idx) const {
  if (isDense_) {
    size_t start = row_idx * nfeatures_;
    return index.Slice(start, start + nfeatures_);
  } else {
    return index.Slice(row_ptr[row_idx], row_ptr[row_idx + 1]);
  }
}
```

### Single Value Access

```cpp
uint32_t GetBin(size_t row_idx, size_t feature_idx) const {
  if (isDense_) {
    return index[row_idx * nfeatures_ + feature_idx];
  } else {
    // Need to search for feature in CSR row
    for (size_t i = row_ptr[row_idx]; i < row_ptr[row_idx + 1]; i++) {
      if (feature_indices[i] == feature_idx) {
        return index[i];
      }
    }
    return kMissingBin;  // Feature not present
  }
}
```

## Use in Histogram Building

The histogram building loop accesses GHistIndexMatrix row-by-row:

```cpp
void BuildHist(std::vector<RowIdx> const& rows,
               std::vector<GradientPair> const& gpair,
               GHistRow hist) {
  for (RowIdx row : rows) {
    GradientPair g = gpair[row];
    
    // Iterate over features in this row
    auto bins = gmat.GetRow(row);
    for (size_t f = 0; f < bins.size(); f++) {
      uint32_t global_bin = cut.feature_offset(f) + bins[f];
      hist[global_bin] += g;
    }
  }
}
```

## booste-rs Design

```rust
/// Quantized feature matrix
pub struct GHistIndexMatrix {
    /// Quantized bin indices
    index: QuantizedIndex,
    
    /// For CSR format: row pointers
    row_ptr: Option<Vec<usize>>,
    
    /// Bin boundaries
    cuts: HistogramCuts,
    
    /// Number of rows
    n_rows: usize,
    
    /// Number of features
    n_features: usize,
}

/// Type-adaptive index storage
pub enum QuantizedIndex {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl GHistIndexMatrix {
    /// Get bin index for (row, feature)
    pub fn get_bin(&self, row: usize, feature: usize) -> u32 {
        match &self.index {
            QuantizedIndex::U8(data) => {
                data[row * self.n_features + feature] as u32
            }
            QuantizedIndex::U16(data) => {
                data[row * self.n_features + feature] as u32
            }
            QuantizedIndex::U32(data) => {
                data[row * self.n_features + feature]
            }
        }
    }
    
    /// Get global bin index (for histogram)
    pub fn get_global_bin(&self, row: usize, feature: usize) -> u32 {
        let local_bin = self.get_bin(row, feature);
        self.cuts.global_bin(feature, local_bin)
    }
    
    /// Iterate over bins in a row
    pub fn row_bins(&self, row: usize) -> impl Iterator<Item = (usize, u32)> {
        (0..self.n_features).map(move |f| (f, self.get_bin(row, f)))
    }
}
```

## Memory Layout Comparison

| Layout | Memory | Access Pattern | Best For |
|--------|--------|----------------|----------|
| Dense row-major | n × d | Sequential per row | Dense data |
| Dense col-major | n × d | Sequential per feature | Column-wise ops |
| CSR | nnz + n + nnz | Random per row | Sparse data |

## Trade-offs

**Dense advantages:**
- Simple indexing
- Cache-friendly row access
- No indirection

**Dense disadvantages:**
- Wastes memory for sparse data
- Missing values need sentinel bin

**CSR advantages:**
- Memory efficient for sparse data
- Natural missing value handling

**CSR disadvantages:**
- More complex access
- Worse cache behavior

## booste-rs Recommendation

Start with **dense layout** for simplicity:

1. Most real datasets are dense or low sparsity
2. Simpler implementation
3. Better cache behavior for histogram building
4. Can add CSR later if needed

## Source Code References

| Component | XGBoost Source |
|-----------|----------------|
| GHistIndexMatrix | `src/data/gradient_index.h` |
| Index | `src/common/index.h` |
| Construction | `src/data/gradient_index.cc` |
