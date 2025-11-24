# Storage Layouts: CPU (GHistIndexMatrix) vs GPU (EllpackPage)

After `QuantileDMatrix` computes bin boundaries, quantized data must be stored for their respective hardware.

## The trade-off: CSR sparse vs padded dense

- **GHistIndexMatrix (CPU)**: CSR format. Only stores non-missing features per row → compact, cache-efficient. Like a sparse matrix.
- **EllpackPage (GPU)**: Padded dense format. All rows have the same width (filled with nulls if needed) → predictable memory access, warp-friendly. Like a regular dataframe.

---

## Side-by-side comparison

| Aspect | GHistIndexMatrix (CPU) | EllpackPage (GPU) |
|--------|------------------------|-------------------|
| **Format** | CSR sparse (row_ptr + index) | Padded dense (fixed row_stride) |
| **Mental model** | Catalog cards with only existing features | Fixed-length shelves with all slots (nulls for missing) |
| **Storage** | High efficiency—skips missing values | Lower efficiency—pads with nulls, but bit-compresses well |
| **Memory access** | Sequential per row or per column (transpose) | Strided across rows (warp-friendly) |
| **Best for** | Sparse data (variable features per row) | Dense/balanced data; many samples |
| **Worst for** | Ultra-dense (CSR overhead) | Ultra-sparse (padding explodes) |

---

## Why each exists

**GHistIndexMatrix**: CPU histogram loops read the same books repeatedly.
- **CSR saves cache**: Only non-missing features consume bandwidth.
- **Column-major transpose**: Histogram builders rearrange into column-major to stream all books' feature-0 bins sequentially.

**EllpackPage**: GPU warps must process rows in lockstep.
- **Padding is necessary**: Fixed-length rows prevent warp divergence (all threads stay synchronized).
- **Coalesced memory**: Aligned reads across threads → GPU efficiently bundles them into cache transactions.

---

## Visual example

### Input
```
book0: page_count=150, weight=0.8
book1: page_count=350
book2: page_count=120, weight=1.2, author_id=5
```

### After QuantileDMatrix binning
```
book0: page_count→bin 12, weight→bin 3
book1: page_count→bin 14
book2: page_count→bin 10, weight→bin 5, author_id→bin 2
```

### GHistIndexMatrix (CSR storage)
```
row_ptr: [0, 2, 3, 6]
index:   [12, 3, 14, 10, 5, 2]
         └─book0─┘ └book1┘ └──book2──┘

Storage: Only the 6 bins + 4 row pointers = very compact
```
Each row stores only the features that exist.

### EllpackPage (padded dense)
```
row_stride = 3 (max features per row)

book0: [12,  3,  null]
book1: [14, null, null]
book2: [10,  5,   2]

packed: [12, 3, null, 14, null, null, 10, 5, 2]
(bit-compressed: fewer bits per symbol)

Storage: 9 slots (padded) = less compact, but GPU-friendly
```
Every row has 3 slots, even if most are empty.

---

## When to use

**GHistIndexMatrix**:
- CPU training (`tree_method=hist`).
- Sparse data (not all rows have all features).
- Memory-constrained environments.

**EllpackPage**:
- GPU training/inference (`tree_method=gpu_hist`, GPU predictor).
- Dense or balanced data.
- Many samples (amortizes padding overhead).

**Avoid**:
- GHistIndexMatrix for GPU (needs padded format).
- EllpackPage for ultra-sparse, high-dimensional data (padding explodes).

---

## Implementation pointers

**GHistIndexMatrix**:
- Core: `src/data/gradient_index.{h,cc}`
- Column transpose: `src/common/column_matrix.h`

**EllpackPage**:
- Layout: `src/data/ellpack_page.cuh`
- Kernels: `src/data/ellpack_page.cu`

---

## Deep dive (optional)

### Construction: GHistIndexMatrix

```
1. QuantileDMatrix supplies HistogramCuts
2. PushBatch: for each (row, feature, value):
     bin_id = SearchBin(cuts, feature, value)
     append bin_id to index[]
3. row_ptr tracks where each row's data starts
4. ColumnMatrix transposes for feature-major loops
```

### Construction: EllpackPage

```
1. Input: CSR data or GHistIndexMatrix
2. Compute row_stride = max(features per row)
3. For each (row, feature_slot):
     bin_id = value exists ? SearchBin(cuts, feature, value) : NullValue()
     write bin_id to gidx_buffer[row * row_stride + slot]
4. Bit-pack buffer (use minimum bits/feature)
```

### CPU training loop (GHistIndexMatrix)
```cpp
col_matrix = ghist.Transpose();  // column-major view
for (each feature) {
  for (each book) {
    bin_id = col_matrix[feature][book];
    histogram[bin_id]++;
  }
}
```
Pattern: Stream through all books' bins for one feature at a time.

### GPU kernel (EllpackPage)
```cpp
__global__ void histogram_kernel(EllpackPage page) {
  int book = blockIdx.x * blockDim.x + threadIdx.x;
  for (int slot = 0; slot < page.RowStride(); ++slot) {
    int bin = page.GetBinIndex(book, slot);
    if (bin != NullValue()) atomicAdd(&hist[bin], 1);
  }
}
```
Pattern: All threads read slot `i` together → coalesced memory access.

---

## Practical notes

- **GHistIndexMatrix**: Missing values are implicit (not stored). Row order matters (CSR layout).
- **EllpackPage**: Missing values are explicit (sentinel `NullValue()`). Can be constructed from CSR or GHistIndexMatrix.
- Both use the same bin boundaries from `QuantileDMatrix`; they differ only in storage layout.
