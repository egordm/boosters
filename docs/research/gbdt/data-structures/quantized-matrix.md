# Quantized Feature Matrix

## Overview

A quantized feature matrix stores pre-computed bin indices for all (sample, feature) pairs. Given histogram cuts computed during preprocessing, the quantization function:

$$Q(x_{i,j}) = \text{SearchBin}(x_{i,j}, \text{cuts}_j)$$

maps each raw feature value to its bin index. This transformation is performed once during data loading, amortizing the O(log b) binary search cost across all subsequent histogram operations.

## Key Benefits

| Benefit | Explanation |
|---------|-------------|
| **Memory reduction** | 1 byte (uint8) vs 4 bytes (float32) per value |
| **Cache efficiency** | 4x more data fits in cache |
| **Integer operations** | Bin comparisons are integer ops (faster than float) |
| **SIMD friendly** | Pack 32 uint8 values into one AVX-256 register |
| **Deterministic** | Integer comparisons are exact across platforms |

## The Core Trade-off

Quantization represents a fundamental space-time trade-off:

```
Traditional approach:
  - Store raw floats
  - Every histogram build: O(n * log(bins)) for bin lookups
  
Quantized approach:
  - Pre-compute bins once: O(n * log(bins))
  - Store bin indices (smaller)
  - Every histogram build: O(n) direct indexing
```

Since histogram building happens many times (once per node candidate, per tree, per iteration), the one-time quantization cost is quickly amortized.

## Conceptual Data Layouts

### Dense Layout (Row-Major)

For data without missing values, store bin indices in a simple 2D array:

```
Row-major storage (samples x features):
+-------------------------------------------+
| Row 0: [bin_f0, bin_f1, bin_f2, ..., bin_fn] |
| Row 1: [bin_f0, bin_f1, bin_f2, ..., bin_fn] |
| Row 2: [bin_f0, bin_f1, bin_f2, ..., bin_fn] |
| ...                                          |
+-------------------------------------------+

Access pattern: index[row * n_features + feature]
```

**When to use**: Dense data, no missing values, primarily row-wise iteration.

### Sparse Layout (CSR)

For sparse data or data with many missing values, use Compressed Sparse Row format:

```
CSR (Compressed Sparse Row):
  row_ptr: [0, 2, 3, 6]    <- Row boundaries
  col_idx: [0, 2, 1, 0, 1, 2]  <- Feature indices  
  values:  [b0, b1, b2, b3, b4, b5]  <- Bin values

Row 0: 2 values at features 0, 2
Row 1: 1 value at feature 1
Row 2: 3 values at features 0, 1, 2
```

Missing values are implicit (not stored), saving memory for sparse datasets.

**When to use**: Sparse data (>50% zeros/missing), memory-constrained environments.

### GPU Layout (Padded/ELLPACK)

GPUs require different memory layouts for coalesced memory access:

```
ELLPACK (padded dense format):
  row_stride = max_features_per_row (e.g., 3)
  
  Row 0: [bin0, bin1, NULL]
  Row 1: [bin0, NULL, NULL]  
  Row 2: [bin0, bin1, bin2]

All rows padded to same width -> predictable memory access
```

**Why GPU needs this**: GPU threads in a warp must access adjacent memory addresses for efficient coalesced reads. Variable-length rows break this pattern.

## XGBoost: GHistIndexMatrix

XGBoost's primary CPU structure for quantized training data:

```cpp
// From xgboost/src/data/gradient_index.h
class GHistIndexMatrix {
    common::Index index;           // Quantized bin indices
    std::vector<size_t> row_ptr;   // Row pointers (for sparse)
    HistogramCuts cut;             // Bin boundaries
    bool isDense_;                 // Dense or sparse layout
    size_t max_bins_;              // Determines index type
};
```

### Type-Adaptive Storage

XGBoost adapts the storage type based on the number of bins:

| max_bins | Storage Type | Bytes per Value |
|----------|--------------|-----------------|
| <= 256 | uint8_t | 1 |
| <= 65536 | uint16_t | 2 |
| > 65536 | uint32_t | 4 |

This is why `max_bins=256` is the sweet spot: each value fits in a single byte.

> **Reference**: `xgboost/src/data/gradient_index.h`, class `GHistIndexMatrix`

### GPU: EllpackPage

For GPU training, XGBoost uses the ELLPACK format:

```cpp
// Conceptual structure from xgboost/src/data/ellpack_page.cuh
struct EllpackPage {
    size_t row_stride;                  // Max features per row
    common::CompressedBuffer gidx_buffer;  // Bit-packed bins
};
```

Bit-packing further reduces memory: if you have 256 bins, you need 8 bits per value, but multiple values can be packed into a single 32-bit or 64-bit word.

## LightGBM: Dataset and Bin Storage

LightGBM organizes data by **feature groups** for cache efficiency:

```cpp
// Conceptual structure from LightGBM/include/LightGBM/bin.h
template<typename VAL_T>
class DenseBin {
    std::vector<VAL_T> data_;  // data_[row] = bin_idx
};

template<typename VAL_T>
class SparseBin {
    std::vector<VAL_T> vals_;         // Non-zero bins
    std::vector<data_size_t> deltas_; // Row index deltas (delta encoding)
};
```

### Feature Groups

LightGBM groups features that will be processed together:

```
Feature Group 0: [feature 0, feature 1, feature 2]
Feature Group 1: [feature 3, feature 4]
...
```

Each group is stored contiguously, improving cache locality when histogram building processes one group at a time.

> **Reference**: `LightGBM/src/io/bin.cpp`, `LightGBM/include/LightGBM/bin.h`

## CPU vs GPU Layout Comparison

| Aspect | CPU (GHistIndexMatrix) | GPU (EllpackPage) |
|--------|------------------------|-------------------|
| **Format** | CSR sparse or dense | Padded dense (ELLPACK) |
| **Access pattern** | Sequential per row | Strided across rows |
| **Missing values** | Implicit (not stored) | Explicit (sentinel value) |
| **Memory efficiency** | High (skip missing) | Lower (padding overhead) |
| **Thread model** | Few threads, large work chunks | Many threads, lockstep execution |

### Why the Difference?

**CPU threading**: Each thread processes many rows independently. Variable-length rows are fine because each thread manages its own iteration.

**GPU threading**: Thousands of threads execute in lockstep (warps of 32). All threads in a warp should access adjacent memory locations simultaneously for coalesced memory access.

```
GPU warp reading feature 0 from 32 consecutive samples:
  Thread 0: row[0].feat[0]  --+
  Thread 1: row[1].feat[0]    |-- Coalesced read (one memory transaction)
  Thread 2: row[2].feat[0]    |
  ...                       --+
```

## Quantization Process

### Conceptual Algorithm

```
ALGORITHM: BuildQuantizedMatrix(raw_data, max_bins)
---------------------------------------------------
1. cuts <- BuildHistogramCuts(raw_data, max_bins)
2. n_rows, n_features <- SHAPE(raw_data)
3. 
4. // Choose storage type based on max_bins
5. IF max_bins <= 256:
       index <- ALLOCATE(n_rows * n_features, type=uint8)
   ELSE IF max_bins <= 65536:
       index <- ALLOCATE(n_rows * n_features, type=uint16)
   ELSE:
       index <- ALLOCATE(n_rows * n_features, type=uint32)
6. 
7. // Quantize each value (parallelizable by row)
8. PARALLEL FOR row FROM 0 TO n_rows:
       FOR feat FROM 0 TO n_features:
           value <- raw_data[row, feat]
           index[row * n_features + feat] <- SearchBin(value, cuts[feat])
9. 
10. RETURN QuantizedMatrix(index, cuts)
```

### Parallelization

Quantization is embarrassingly parallel:

- Each (row, feature) cell can be quantized independently
- Row-wise parallelism is typically used (better cache locality)
- GPU construction uses per-row threads

## Memory Layout Comparison

| Layout | Memory Size | Access Pattern | Best For |
|--------|-------------|----------------|----------|
| Dense row-major | n *d* sizeof(bin) | O(1) per cell | Dense data, row iteration |
| Dense col-major | n *d* sizeof(bin) | O(1) per cell | Column iteration, split finding |
| CSR | nnz *(sizeof(bin) + sizeof(col)) + n* sizeof(ptr) | O(k) per row | Sparse data (>50% sparse) |
| ELLPACK | n *row_stride* bits | Strided | GPU, moderate sparsity |

### Memory Estimates

For 1 million samples, 100 features:

```
Dense (uint8):        100 MB
Dense (uint16):       200 MB
CSR (50% sparse):     ~75 MB (with feature indices)
ELLPACK (bit-packed): Depends on row_stride and bit width
```

## When to Use Each Format

### Dense Row-Major (Simple Case)

**Use when:**

- Data is dense (no or few missing values)
- Simple implementation needed
- Primarily row-wise access during histogram building

**Avoid when:**

- Very sparse data (wasted memory)
- Need column-wise access for split finding

### CSR Sparse (Memory-Constrained)

**Use when:**

- Significant sparsity (>50% missing/zero)
- Memory is constrained
- Missing values are common

**Avoid when:**

- Dense data (overhead of column indices)
- GPU training (layout not suitable)

### ELLPACK (GPU)

**Use when:**

- GPU training or inference
- Dense or moderately sparse data
- High sample count (amortizes padding)

**Avoid when:**

- Very sparse data (padding explodes memory)
- CPU-only workloads

## Access Patterns

### Row Access (Histogram Building)

During histogram building, iterate over rows assigned to a node:

```
ALGORITHM: BuildHistogram(row_indices, gradients, quantized_matrix)
-------------------------------------------------------------------
1. histogram <- ZEROS(total_bins)
2. 
3. FOR row IN row_indices:
       grad <- gradients[row]
       FOR feat FROM 0 TO n_features:
           bin <- quantized_matrix.get_bin(row, feat)
           global_bin <- feature_offset[feat] + bin
           histogram[global_bin] += grad
4. 
5. RETURN histogram
```

### Column Access (Split Finding)

For split finding, sometimes column-major access is preferred:

```
// Transpose can be pre-computed for efficient column iteration
column_matrix <- TRANSPOSE(row_matrix)

FOR feat FROM 0 TO n_features:
    FOR row IN column_matrix.rows_for_feature(feat):
        bin <- column_matrix.get_bin(feat, row)
        // Process bin for split evaluation
```

Some implementations maintain both row-major and column-major views for optimal access patterns.

## Summary

| Aspect | Description |
|--------|-------------|
| **Purpose** | Pre-computed bin indices for fast histogram building |
| **Key insight** | One-time O(log b) lookup vs per-histogram lookup |
| **CPU format** | GHistIndexMatrix (CSR or dense) |
| **GPU format** | EllpackPage (padded dense, bit-packed) |
| **Type selection** | uint8 for <=256 bins, uint16 for <=65536, uint32 otherwise |
| **Benefits** | 4x memory, cache efficiency, SIMD-friendly |

### Key References

- `xgboost/src/data/gradient_index.h` — `GHistIndexMatrix` class
- `xgboost/src/data/ellpack_page.cuh` — GPU ELLPACK format
- `LightGBM/include/LightGBM/bin.h` — `DenseBin`, `SparseBin` classes
- `LightGBM/src/io/dataset.cpp` — Dataset construction and feature grouping
