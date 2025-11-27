# RFC-0004: DMatrix and Data Input

- **Status**: Accepted
- **Created**: 2024-11-24
- **Scope**: Data input abstraction, feature storage, quantization transforms

## Summary

This RFC defines the data input layer for xgboost-rs:

1. **`DataMatrix` trait**: Common interface for feature matrices
2. **Dense and Sparse storage**: Leveraging existing crates where possible
3. **Quantization as transformation**: Converting floats to bins, not a separate matrix type
4. **Arrow integration**: Zero-copy interop for Python/pandas/polars workflows
5. **Categorical feature support**: Native handling without one-hot encoding

## Motivation

XGBoost's `DMatrix` serves multiple purposes:

- Feature storage (sparse/dense)
- Metadata (labels, weights, groups)
- Quantization for histogram-based training
- Device portability (CPU/GPU)

For xgboost-rs, we need:

1. Ergonomic APIs for Rust users (slices, iterators)
2. Zero-copy interop with Python ecosystem (Arrow, PyArrow, pandas/polars)
3. Efficient quantization for `hist` tree method
4. Extensibility for GPU backends and external memory (future)

### Design Philosophy

- **Use existing crates** where mature solutions exist (Arrow, sparse matrices)
- **Quantization is a transform**, not a special matrix type
- **NaN is missing** — no custom sentinel complexity
- **Metadata is separate** from feature storage

## Design

### Type Hierarchy

```text
                         +-------------------------+
                         |   DataMatrix (trait)    |
                         |   - num_rows()          |
                         |   - num_features()      |
                         |   - row(i) -> RowView   |
                         +-----------+-------------+
                                     |
          +--------------------------+-------------------------+
          |                          |                         |
          v                          v                         v
  +---------------+          +---------------+         +---------------+
  |  DenseMatrix  |          | SparseMatrix  |         |  ArrowMatrix  |
  |  (row-major)  |          | (CSR, sprs?)  |         | (RecordBatch) |
  +---------------+          +---------------+         +---------------+
          |                          |                         |
          +--------------------------+-------------------------+
                                     |
                                     v
                             +---------------+
                             |   quantize()  |  <- Transform, not type
                             |  -> Matrix<B> |
                             +---------------+
```

### DataMatrix Trait

```rust
/// Core trait for feature matrix access
pub trait DataMatrix {
    type Element: Copy;
    type RowView<'a>: RowView<'a, Element = Self::Element> where Self: 'a;
    
    fn num_rows(&self) -> usize;
    fn num_features(&self) -> usize;
    fn row(&self, i: usize) -> Self::RowView<'_>;
    fn get(&self, row: usize, col: usize) -> Option<Self::Element>;
    fn is_dense(&self) -> bool { false }
    
    // Required by RFC-0003 (Visitor/Traversal):
    
    /// Copy row into dense buffer (for block traversal)
    fn copy_row(&self, i: usize, buf: &mut [Self::Element]);
    
    /// Check if matrix contains missing values (NaN for f32)
    fn has_missing(&self) -> bool;
    
    /// Density ratio: nnz / (rows × features). Used to choose block vs row traversal.
    fn density(&self) -> f64;
}

/// View of a single row (sparse or dense)
pub trait RowView<'a>: IntoIterator<Item = (usize, Self::Element)> {
    type Element: Copy;
    fn nnz(&self) -> usize;
    fn get(&self, feature_idx: usize) -> Option<Self::Element>;
}
```

### Dense Matrix

Row-major storage, generic over storage type (owned, borrowed, mmap).

```rust
pub struct DenseMatrix<T = f32, S: AsRef<[T]> = Box<[T]>> {
    data: S,
    num_rows: usize,
    num_features: usize,
}
```

Missing values represented as `f32::NAN`.

### Sparse Matrix — Use Existing Crate?

**Question**: Use `sprs` crate or implement our own CSR?

| Option | Pros | Cons |
|--------|------|------|
| `sprs` | Mature, tested | Extra dependency |
| Custom | Full control | More code to maintain |

**Recommendation**: Start with `sprs`, wrap to implement `DataMatrix`.

### Arrow Integration

For Python interop (pandas, polars, numpy via PyArrow).

```rust
pub struct ArrowMatrix {
    batch: RecordBatch,
    feature_columns: Vec<usize>,
}
```

Arrow is columnar; for repeated row access, `to_dense()` may be faster.

**Note**: Arrow lacks native sparse support. Sparse data from Python needs COO/CSR conversion at boundary.

### Quantization as Transform

Quantization converts floats to bin indices. Result is a matrix with integer element type — no special `QuantizedMatrix` type needed.

```rust
pub struct HistogramCuts {
    pub cut_ptrs: Box<[u32]>,    // Offsets per feature
    pub cut_values: Box<[f32]>,  // Sorted cut boundaries  
    pub min_vals: Box<[f32]>,    // Min value per feature
}

/// Quantize: f32 -> bin indices
pub fn quantize<M: DataMatrix<Element = f32>>(
    matrix: &M,
    cuts: &HistogramCuts,
) -> DenseMatrix<u16>;
```

### Categorical Features

XGBoost supports native categorical splits (partition-based, not one-hot).

```rust
pub enum FeatureType {
    Numeric,
    Categorical { num_categories: u32 },
}

pub struct FeatureInfo {
    pub types: Vec<FeatureType>,
    pub names: Option<Vec<String>>,
}
```

Categorical values stored as f32 (category index). Tree nodes use bitsets for categorical splits ("go left" categories).

### Metadata

Separate from feature matrix.

```rust
pub struct DatasetMeta {
    pub labels: Option<Box<[f32]>>,
    pub weights: Option<Box<[f32]>>,
    pub group_ptr: Option<Box<[u32]>>,  // Ranking groups
    pub base_margin: Option<Box<[f32]>>,
}
```

### Complete Dataset

```rust
pub struct Dataset<M: DataMatrix> {
    pub features: M,
    pub meta: DatasetMeta,
    pub feature_info: FeatureInfo,
}
```

## Design Decisions

### DD-1: Use Existing Sparse Matrix Crate **[OPEN]**

**Options**: A) `sprs` — mature, less code. B) Custom — full control.

Leaning toward A unless `sprs` API proves problematic.

### DD-2: Quantization is a Transform **[DECIDED]**

Quantization is a function returning matrix with integer element type.

**Rationale**: Cleaner model, reuses existing types.

### DD-3: NaN-Only Missing Values **[DECIDED]**

Use `f32::NAN` exclusively.

**Rationale**: Modern standard, simplifies implementation.

### DD-4: Bin Index Type **[OPEN]**

**Options**: A) Always u16 — simple. B) Generic `B: BinIndex` — flexible.

Leaning toward A unless profiling shows benefit from u8.

### DD-5: Arrow for Python Interop **[DECIDED]**

Use Arrow as Python bridge for zero-copy with PyArrow/pandas/polars.

### DD-6: Categorical Handling **[DECIDED]**

Categorical values as f32 indices, `FeatureInfo` marks categorical features. Tree nodes use bitsets for partition-based splits.

## Integration with RFC-0003

RFC-0003 (Visitor and Traversal) references `FeatureMatrix` in its `Predictor` API. The `DataMatrix` trait defined here fulfills that role:

| RFC-0003 expects | RFC-0004 provides |
|------------------|-------------------|
| `num_rows()` | `DataMatrix::num_rows()` |
| `row(idx) -> &[f32]` | `DataMatrix::row(idx) -> RowView` |
| `copy_row(idx, &mut [f32])` | `DataMatrix::copy_row()` |
| `has_missing() -> bool` | `DataMatrix::has_missing()` |
| `density() -> f64` | `DataMatrix::density()` |

Type alias for clarity: `type FeatureMatrix = dyn DataMatrix<Element = f32>;`

## Future Extensions

- **External memory / streaming**: Iterator-based input for out-of-core datasets (design for extensibility)
- **GPU matrices**: Device-side storage with transfer primitives
- **Memory-mapped matrices**: For large datasets

## References

- XGBoost `DMatrix`: `include/xgboost/data.h`
- `sprs` crate: <https://docs.rs/sprs>
- Apache Arrow Rust: <https://docs.rs/arrow>
