# Feature Quantization

Feature quantization (also called binning or discretization) converts continuous feature
values into discrete bin indices. This is a fundamental preprocessing step in modern
histogram-based gradient boosting.

---

## Overview

Quantization maps continuous features to integer bin indices:

```text
Feature value: 0.35 → Bin boundaries: [0.0, 0.25, 0.5, 0.75, 1.0]
Search: 0.35 falls in [0.25, 0.5) → Bin index: 1
```

This enables:

- **Histogram-based training**: O(bins) split finding instead of O(n)
- **Memory efficiency**: u8 indices instead of f32 values (4x reduction)
- **Cache efficiency**: Small working set fits in cache

Given feature values $x_1, x_2, \ldots, x_n$ and $k$ bins, we compute bin boundaries
$b_0, b_1, \ldots, b_k$ where $b_0 = -\infty$ and $b_k = +\infty$.

For any value $x$, the bin index is:
$$\text{bin}(x) = \max\{j : b_j \le x\}$$

The choice of boundaries affects model quality. Two main strategies:

1. **Uniform**: Equal-width bins
   $$b_j = x_{\min} + j \cdot \frac{x_{\max} - x_{\min}}{k}$$

2. **Quantile**: Equal-frequency bins (each bin has ~n/k samples)
   $$b_j = Q(j/k)$$ where $Q$ is the empirical quantile function

---

## Uniform vs Quantile Binning

### Uniform Binning

Divides the range into equal-width intervals:

```text
Values: [0, 1, 2, 3, 100]  max_bins=4
Boundaries: [0, 25, 50, 75, 100]
Bins: [0, 0, 0, 0, 3]  ← All small values in same bin!
```

**Problem**: Outliers or skewed distributions waste bins on sparse regions.

### Quantile Binning

Divides the range so each bin has roughly equal samples:

```text
Values: [0, 1, 2, 3, 100]  max_bins=4
Boundaries: [0, 1, 2, 3, 100]  (approximate quantiles)
Bins: [0, 1, 2, 3, 4]  ← Each value gets its own bin
```

**Advantage**: Bins are placed where data exists, maximizing split resolution.

### Library Approaches

| Library | Default | Notes |
|---------|---------|-------|
| XGBoost (`hist`) | Quantile | Hessian-weighted quantiles for better splits |
| LightGBM | Quantile | Greedy binning with distinct value handling |
| XGBoost (`approx`) | Quantile | Per-node sketch (slower, more accurate) |

---

## Quantile Sketching

For large datasets, computing exact quantiles requires O(n log n) sorting per feature.
Streaming sketches provide approximate quantiles with bounded error.

### Greenwald-Khanna Sketch

Used by XGBoost. Properties:

- **Space**: O(1/ε log εn) for ε-approximate quantiles
- **Time**: O(log(1/ε)) per insertion
- **Mergeable**: Can combine sketches from distributed workers

### Hessian-Weighted Quantiles

XGBoost weights samples by their hessian during quantile computation:

```text
For loss L(y, f), the optimal split considers:
  - Gradient: g_i = ∂L/∂f
  - Hessian: h_i = ∂²L/∂f²

Weighted quantile: use h_i as weight for sample i
```

**Why?** Samples with larger hessian (higher curvature) benefit more from precise splits.

**Note**: For constant-hessian objectives like MSE, this reduces to standard quantiles.

---

## Bin Lookup

Given quantized features, we need efficient bin lookup during training.

### Binary Search

Standard approach for variable-width bins:

```rust
fn find_bin(value: f32, cuts: &[f32]) -> u8 {
    cuts.binary_search_by(|c| c.partial_cmp(&value).unwrap())
        .unwrap_or_else(|i| i.saturating_sub(1)) as u8
}
```

Time: O(log bins) per value.

### Linear Scan for Few Bins

For small bin counts, linear scan may be faster due to branch prediction:

```rust
fn find_bin_linear(value: f32, cuts: &[f32]) -> u8 {
    for (i, &cut) in cuts.iter().enumerate() {
        if value < cut {
            return i as u8;
        }
    }
    cuts.len() as u8
}
```

### Precomputed Lookup Table

For integer features with known range, direct indexing is O(1):

```rust
// Precompute: lookup_table[int_value] = bin_index
let bin = lookup_table[value as usize];
```

---

## Handling Special Values

### Missing Values

Both XGBoost and LightGBM reserve a special bin for missing values:

| Library | Missing Bin |
|---------|-------------|
| XGBoost | Bin 0 (default) or last bin |
| LightGBM | Separate missing indicator |

During split finding, missing values are tried in both directions to learn the best
default.

### Distinct Values

If a feature has fewer distinct values than max_bins, use one bin per value:

```text
Feature has values: {1.0, 2.0, 3.0}  max_bins=256
Use 3 bins, not 256
```

This preserves exact splits and saves memory.

---

## Bin Count Selection

### Default: 256 bins (u8 index)

Most libraries default to 256 bins:

- Fits in u8 (1 byte per feature per row)
- Sufficient resolution for most features
- Good balance of accuracy vs memory

### When to Use More Bins

| Situation | Recommendation |
|-----------|----------------|
| Many distinct values (>1000) | Consider 512-1024 bins |
| High-precision features | More bins may help |
| Large dataset | Can afford more bins |
| Overfitting concerns | Fewer bins act as regularization |

### Type Selection

| max_bins | Type | Bytes |
|----------|------|-------|
| ≤256 | u8 | 1 |
| ≤65536 | u16 | 2 |
| >65536 | u32 | 4 |

---

## Global vs Local Indexing

### Feature-Local Indexing

Each feature has bins 0, 1, 2, ..., k-1:

```text
Feature 0: bins 0-255
Feature 1: bins 0-255
Feature 2: bins 0-127 (fewer distinct values)
```

**Pro**: Maximum resolution per feature.
**Con**: Need feature offset for histogram access.

### Global Indexing

Bins are numbered globally across all features:

```text
Feature 0: bins 0-255
Feature 1: bins 256-511
Feature 2: bins 512-639
```

**Pro**: Direct histogram indexing without offset lookup.
**Con**: Maximum bin count limited by total.

---

## Quantization Pipeline

### Training-Time Quantization

```text
1. Compute bin boundaries (quantile sketch or exact)
   - Store as HistogramCuts structure
   
2. Transform raw data to bin indices
   - For each (row, feature): bin_index = find_bin(value, cuts[feature])
   - Store as GHistIndexMatrix (XGBoost) or Dataset (LightGBM)
   
3. Training uses bin indices exclusively
   - Histogram building: increment histogram[bin]
   - Split finding: scan histograms, not raw data
```

### Inference-Time Options

Two approaches:

1. **Quantize at inference time**: Transform input features using stored cuts
2. **Float comparison**: Compare against split thresholds directly

Most libraries use float comparison for inference (simpler, no cuts dependency).

---

## Memory Analysis

For a dataset with n rows, d features, and b bins:

| Storage | Size | Notes |
|---------|------|-------|
| Raw data (f32) | 4nd bytes | Original |
| Quantized (u8) | nd bytes | 4x reduction |
| Quantized (u16) | 2nd bytes | 2x reduction |
| Bin boundaries | O(db) | Per-feature cuts |

**Example**: 1M rows × 100 features

- Raw: 400 MB
- Quantized (u8): 100 MB
- Savings: 300 MB

---

## Implementation Considerations

### Parallel Quantization

Bin finding is embarrassingly parallel:

- Each row can be processed independently
- Each feature can be processed independently (if global cuts precomputed)

### Cache Efficiency

For row-major raw data:

- Process one row at a time across features
- Output row-major quantized data

For column-major:

- Process one feature at a time across rows
- Better for quantile computation (needs sorted feature)

### 4-bit Packing

LightGBM supports 4-bit bins (≤15 bins per feature):

```rust
// Pack two bins per byte
let packed = (bin1 << 4) | bin2;

// Unpack
let bin1 = (packed >> 4) & 0xf;
let bin2 = packed & 0xf;
```

Useful for low-cardinality categoricals or heavily binned features.

---

## References

### XGBoost

- `src/common/quantile.cc` — Streaming quantile sketch
- `src/common/hist_util.cc` — HistogramCuts builder
- `src/data/gradient_index.cc` — GHistIndexMatrix

### LightGBM

- `src/io/bin.cpp` — BinMapper, bin finding
- `include/LightGBM/bin.h` — Bin structures
- `src/io/dataset.cpp` — Dataset with binned features

### Academic

- Greenwald & Khanna (2001): Space-Efficient Online Computation of Quantile Summaries
- Chen & Guestrin (2016): XGBoost §3.3 on weighted quantile sketch
