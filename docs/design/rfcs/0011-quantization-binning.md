# RFC-0011: Quantization & Binning

- **Status**: Draft
- **Created**: 2024-11-30
- **Updated**: 2024-11-30
- **Depends on**: RFC-0010 (Matrix Layouts)
- **Scope**: Feature value quantization and bin index storage for histogram-based training

## Summary

This RFC defines how continuous feature values are discretized into bins for histogram-based
gradient boosting. It covers:

1. **Bin boundaries (cuts)**: How to compute quantile-based thresholds
2. **Quantized storage**: The `QuantizedMatrix` that stores bin indices
3. **Binning API**: How raw features become bin indices

## Motivation

Histogram-based training (used by XGBoost "hist" and LightGBM) requires discretizing continuous
features into a small number of bins (typically 256). This:

- **Reduces split search**: O(n_bins) instead of O(n_samples) for finding best split
- **Enables histogram aggregation**: Sum gradients per bin, not per unique value
- **Improves cache locality**: u8 bin indices fit more data in cache
- **Enables histogram subtraction**: Parent - sibling = current node's histogram

The quantization strategy directly impacts:

- Memory usage (bytes per cell)
- Training speed (histogram build time)
- Model quality (bin resolution affects split precision)

## Design

### Overview

```
Raw Features (f32)           Cuts (bin boundaries)         Quantized (u8)
┌─────────────────┐         ┌─────────────────────┐       ┌─────────────┐
│ 0.15, 2.3, 1.7  │  ──▶    │ [0.0, 0.5, 1.0,     │  ──▶  │  0, 4, 3    │
│ 0.82, 1.1, 0.3  │  bin()  │  1.5, 2.0, 2.5]     │       │  1, 2, 0    │
│ ...             │         │ (6 bins for feat 0) │       │  ...        │
└─────────────────┘         └─────────────────────┘       └─────────────┘
     (n × m)                  (per-feature)                  (n × m)
```

### Bin Cuts Structure

```rust
/// Bin boundaries for all features
/// 
/// For feature `f`, the bin boundaries are `cut_values[cut_ptrs[f]..cut_ptrs[f+1]]`.
/// A value `v` maps to the bin index where `cuts[bin] <= v < cuts[bin+1]`.
pub struct BinCuts {
    /// All cut values concatenated, sorted per feature
    cut_values: Box<[f32]>,
    
    /// Offsets into cut_values: cut_ptrs[f] is start of feature f's cuts
    /// Length: num_features + 1
    cut_ptrs: Box<[u32]>,
    
    /// Number of bins per feature (max 256 for u8 storage)
    /// bins_per_feature[f] = cut_ptrs[f+1] - cut_ptrs[f]
    num_features: u32,
}

impl BinCuts {
    /// Get bin boundaries for a specific feature
    pub fn feature_cuts(&self, feature: u32) -> &[f32] {
        let start = self.cut_ptrs[feature as usize] as usize;
        let end = self.cut_ptrs[feature as usize + 1] as usize;
        &self.cut_values[start..end]
    }
    
    /// Number of bins for a feature
    pub fn num_bins(&self, feature: u32) -> u8 {
        let start = self.cut_ptrs[feature as usize];
        let end = self.cut_ptrs[feature as usize + 1];
        (end - start) as u8
    }
    
    /// Total bins across all features (for histogram allocation)
    pub fn total_bins(&self) -> u32 {
        self.cut_values.len() as u32
    }
    
    /// Map a single value to its bin index
    /// Uses binary search: O(log num_bins)
    #[inline]
    pub fn bin_value(&self, feature: u32, value: f32) -> u8 {
        let cuts = self.feature_cuts(feature);
        // Binary search for the bin
        match cuts.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
            Ok(idx) => idx as u8,           // Exact match
            Err(idx) => idx.saturating_sub(1) as u8,  // Insert position - 1
        }
    }
}
```

### Quantized Feature Storage

```rust
/// Quantized feature matrix storing bin indices
/// 
/// Stored in column-major order for efficient histogram building
/// (iterate rows for a single feature = contiguous memory access).
/// 
/// Type parameter `B` controls bin index width (u8, u16, u32).
/// Default u8 supports up to 256 bins per feature.
pub struct QuantizedMatrix<B: BinIndex = u8> {
    /// Bin indices in column-major layout: index[col * num_rows + row]
    /// Column-major enables contiguous access when building histograms
    index: Box<[B]>,
    
    /// Number of rows
    num_rows: u32,
    
    /// Number of features
    num_features: u32,
    
    /// Reference to the bin cuts used
    cuts: Arc<BinCuts>,
    
    /// Missing value indicator (typically 0 or max_bin)
    missing_bin: u8,
}

impl<B: BinIndex> QuantizedMatrix<B> {
    /// Get bin index for a specific cell
    #[inline]
    pub fn get(&self, row: u32, feature: u32) -> B {
        let idx = (feature as usize) * (self.num_rows as usize) + (row as usize);
        self.index[idx]
    }
    
    /// Get all bin indices for a feature (contiguous slice)
    #[inline]
    pub fn feature_column(&self, feature: u32) -> &[B] {
        let start = (feature as usize) * (self.num_rows as usize);
        let end = start + self.num_rows as usize;
        &self.index[start..end]
    }
    
    /// Iterate over rows for histogram building
    pub fn iter_rows_for_feature(&self, feature: u32, rows: &[u32]) -> impl Iterator<Item = B> + '_ {
        let col = self.feature_column(feature);
        rows.iter().map(move |&row| col[row as usize])
    }
}
```

### Quantile Sketch for Cut Discovery

```rust
/// Strategy for computing bin boundaries
pub trait CutFinder {
    /// Compute bin cuts from feature data
    fn find_cuts(
        &self,
        data: &impl ColumnAccess<f32>,
        max_bins: u8,
    ) -> BinCuts;
}

/// Exact quantile computation (for small datasets)
/// Sorts each feature and picks evenly-spaced quantiles
pub struct ExactQuantileCuts {
    /// Minimum samples per bin
    pub min_samples_per_bin: u32,
}

impl CutFinder for ExactQuantileCuts {
    fn find_cuts(&self, data: &impl ColumnAccess<f32>, max_bins: u8) -> BinCuts {
        let num_features = data.num_cols();
        let num_rows = data.num_rows();
        
        let mut cut_values = Vec::new();
        let mut cut_ptrs = vec![0u32];
        
        for feat in 0..num_features {
            // Collect non-missing values
            let mut values: Vec<f32> = data.column(feat)
                .filter(|v| !v.is_nan())
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Deduplicate
            values.dedup();
            
            // Pick quantile points
            let num_bins = max_bins.min(values.len() as u8);
            let step = values.len() / (num_bins as usize);
            
            for i in 0..num_bins {
                let idx = (i as usize * step).min(values.len() - 1);
                cut_values.push(values[idx]);
            }
            
            cut_ptrs.push(cut_values.len() as u32);
        }
        
        BinCuts {
            cut_values: cut_values.into_boxed_slice(),
            cut_ptrs: cut_ptrs.into_boxed_slice(),
            num_features: num_features as u32,
        }
    }
}

/// Streaming quantile sketch (for large datasets)
/// Uses Greenwald-Khanna or similar algorithm
pub struct SketchQuantileCuts {
    /// Error tolerance for quantile approximation
    pub epsilon: f32,
    /// Number of samples to buffer before merging
    pub sketch_size: usize,
}

impl CutFinder for SketchQuantileCuts {
    fn find_cuts(&self, data: &impl ColumnAccess<f32>, max_bins: u8) -> BinCuts {
        // Implementation uses streaming quantile sketch
        // Each feature maintains a sketch, then we query quantiles
        todo!("Implement GK sketch or use existing crate")
    }
}
```

### Quantization Builder

```rust
/// Builder for creating GHistIndexMatrix from raw features
pub struct Quantizer {
    cuts: Arc<BinCuts>,
}

impl Quantizer {
    /// Create from pre-computed cuts
    pub fn new(cuts: BinCuts) -> Self {
        Self { cuts: Arc::new(cuts) }
    }
    
    /// Create by computing cuts from data
    pub fn from_data<C: CutFinder>(
        data: &impl ColumnAccess<f32>,
        cut_finder: &C,
        max_bins: u8,
    ) -> Self {
        let cuts = cut_finder.find_cuts(data, max_bins);
        Self::new(cuts)
    }
    
    /// Quantize a feature matrix
    pub fn quantize(&self, data: &impl ColumnAccess<f32>) -> GHistIndexMatrix {
        let num_rows = data.num_rows() as u32;
        let num_features = data.num_cols() as u32;
        
        // Allocate column-major storage
        let mut index = vec![0u8; (num_rows * num_features) as usize];
        
        // Parallel quantization per feature
        index.par_chunks_mut(num_rows as usize)
            .enumerate()
            .for_each(|(feat, col)| {
                for (row, bin) in col.iter_mut().enumerate() {
                    let value = data.get(row, feat);
                    *bin = if value.is_nan() {
                        0 // Missing value bin
                    } else {
                        self.cuts.bin_value(feat as u32, value)
                    };
                }
            });
        
        GHistIndexMatrix {
            index: index.into_boxed_slice(),
            num_rows,
            num_features,
            cuts: Arc::clone(&self.cuts),
            missing_bin: 0,
        }
    }
}
```

### Memory Layout Visualization

```
BinCuts layout for 3 features with varying bins:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cut_ptrs:    [0, 5, 8, 12]  (offsets)
              │  │  │  │
              ▼  │  │  │
cut_values:  [0.1, 0.3, 0.5, 0.7, 0.9,   ← Feature 0: 5 bins
                  ▼  │  │
              1.0, 2.0, 3.0,              ← Feature 1: 3 bins  
                     ▼  │
              0.0, 0.25, 0.5, 0.75]       ← Feature 2: 4 bins
                        ▼
                       (end)

GHistIndexMatrix layout (column-major):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For 4 rows × 3 features:

index: [r0f0, r1f0, r2f0, r3f0,   ← Feature 0 column (contiguous)
        r0f1, r1f1, r2f1, r3f1,   ← Feature 1 column
        r0f2, r1f2, r2f2, r3f2]   ← Feature 2 column

Access pattern for histogram building:
  iterate rows in feature column = sequential memory access ✓
```

### Integration with Multi-output Trees

For multi-output (multi-class) training, the quantization is the same—only the gradients differ.
Design hooks:

```rust
impl GHistIndexMatrix {
    /// The quantized features are class-agnostic
    /// Histogram building uses different gradient dimensions
    pub fn with_gradients<G: GradientStorage>(&self, grads: &G) -> QuantizedWithGradients<'_, G> {
        QuantizedWithGradients { index: self, grads }
    }
}
```

### Integration with Feature Bundling (EFB)

The cuts structure supports bundled features:

```rust
/// Extended cuts with optional bundling information
pub struct BinCutsWithBundles {
    /// Base cuts structure
    cuts: BinCuts,
    
    /// Feature → bundle mapping (None if no bundling)
    /// bundle_map[feature] = bundle_id
    bundle_map: Option<Box<[u32]>>,
    
    /// Bundle → features mapping
    /// bundles[bundle_id] = [feat_1, feat_2, ...]
    bundles: Option<Vec<Vec<u32>>>,
    
    /// Bin offset within bundle for each feature
    /// feature's bin in bundle = base_offset[feature] + raw_bin
    bin_offsets: Option<Box<[u8]>>,
}
```

## Design Decisions

### DD-1: Column-Major Storage for GHistIndexMatrix

**Context**: Need efficient access pattern for histogram building.

**Options considered**:

1. **Row-major**: Natural for row-wise access, matches input data
2. **Column-major**: Contiguous access when iterating feature columns

**Decision**: Column-major storage.

**Rationale**:

- Histogram building iterates all rows for each feature
- Column-major makes `feature_column(f)` a contiguous slice
- Enables SIMD vectorization of histogram accumulation
- XGBoost and LightGBM both use column-major for quantized data

### DD-2: Configurable Bin Index Type via Generic

**Context**: Bin index storage type affects memory and bin count.

**Options considered**:

1. **u8 only**: 256 max bins, 1 byte per cell
2. **u16 only**: 65536 max bins, 2 bytes per cell  
3. **Generic parameter**: `QuantizedMatrix<B: BinIndex>` where `BinIndex: u8 | u16 | u32`

**Decision**: Use generic parameter for bin index type.

**Rationale**:

- LightGBM supports 8/16/32-bit bin indices via templates (`SparseBin<uint8_t>`, etc.)
- 256 bins (u8) is sufficient for most cases, but some need more
- Generic enables zero-cost abstraction with compile-time specialization
- Default to u8, allow u16/u32 via type parameter

```rust
pub trait BinIndex: Copy + Into<usize> + TryFrom<usize> {
    const MAX_BINS: usize;
}

impl BinIndex for u8  { const MAX_BINS: usize = 256; }
impl BinIndex for u16 { const MAX_BINS: usize = 65536; }
```

### DD-3: Missing Value Handling

**Context**: Need to handle NaN/missing values in quantized data.

**Options considered**:

1. **Reserved bin 0**: First bin always means missing
2. **Reserved max bin**: Last bin (255) means missing
3. **Separate mask**: BitVec tracking missing positions

**Decision**: Reserved bin 0 for missing values.

**Rationale**:

- Simple: just check `bin == 0` for missing
- Matches XGBoost's convention
- No extra storage for missing mask
- Histogram bin 0 accumulates "missing" gradients for split decision

### DD-4: Quantile Computation Strategy

**Context**: Need to compute bin boundaries from data.

**Options considered**:

1. **Exact quantile**: Sort and pick evenly-spaced values
2. **Streaming sketch**: GK sketch or T-Digest for large data
3. **Sampling**: Subsample then exact quantile

**Decision**: Provide both exact and sketch implementations behind trait.

**Rationale**:

- Small data (< 1M rows): Exact is fast enough, more precise
- Large data: Sketch avoids O(n log n) sort per feature
- Trait allows users to plug in preferred algorithm
- XGBoost uses weighted quantile sketch, LightGBM uses sampling

## Integration

| Component | Integration Point | Notes |
|-----------|-------------------|-------|
| RFC-0010 (Matrix Layouts) | `ColumnAccess` trait | Input data access |
| RFC-0012 (Histograms) | `QuantizedMatrix` | Consumed by histogram builder |
| RFC-0019 (Feature Bundling) | `BinCutsWithBundles` | Extended cuts structure |

### Integration with Existing Code

- **`src/data/traits.rs`**: `ColumnAccess` trait provides input to `CutFinder`
- **`src/data/dense.rs`**: `DenseMatrix` implements `ColumnAccess` for quantization input
- **New module**: `src/training/quantize.rs` for `BinCuts`, `QuantizedMatrix`, `Quantizer`

## Open Questions

1. ~~**Weighted quantiles**~~: **Yes** — Support sample weights in cut computation. Added to Feature Overview.

2. **Incremental updates**: Both XGBoost and LightGBM recompute cuts from scratch. For simplicity, we'll do the same initially.

3. **Sparse data optimization**: Lower priority. Sparse `QuantizedMatrix` variant is a separate feature.

## Future Work

- [ ] Implement Greenwald-Khanna streaming quantile sketch
- [ ] Add weighted quantile support
- [ ] Sparse quantized matrix variant
- [ ] GPU-friendly quantization path

## References

- [XGBoost Quantile Sketch](https://github.com/dmlc/xgboost/blob/master/src/common/hist_util.h)
- [LightGBM Binning](https://github.com/microsoft/LightGBM/blob/master/src/io/bin.cpp)
- [Greenwald-Khanna Paper](https://doi.org/10.1145/375663.375670)
- [Feature Overview](../FEATURE_OVERVIEW.md) - Priority and design context

## Changelog

- 2024-11-30: Initial draft
