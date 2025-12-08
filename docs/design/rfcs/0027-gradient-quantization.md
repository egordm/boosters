# RFC-0027: Gradient Quantization

**Status**: Approved  
**Created**: 2024-12-02  
**Depends on**: RFC-0012 (Histogram Building), RFC-0025 (Row-Parallel Histograms)

## Summary

Quantize gradients and hessians from f32 to int16 to reduce memory bandwidth
during histogram building. This is LightGBM's key optimization for histogram-based
training, providing 1.5-2x speedup on memory-bound workloads.

## Motivation

Histogram building is memory-bandwidth limited:

```text
For each row in node:
    for each feature:
        bin = quantized_features[row, feature]   // 1 byte read
        hist[bin].grad += gradients[row]         // 4 byte read
        hist[bin].hess += hessians[row]          // 4 byte read
```

**Memory traffic per row**: 1 + 4 + 4 = 9 bytes per feature

With quantized gradients (int16, SoA):

```text
For each row in node:
    for each feature:
        bin = quantized_features[row, feature]   // 1 byte read
        hist[bin].grad += quant_gradients[row]   // 2 byte read
        hist[bin].hess += quant_hessians[row]    // 2 byte read
```

**Memory traffic per row**: 1 + 2 + 2 = 5 bytes per feature (44% reduction)

### Performance Analysis

For a typical dataset (1M rows, 100 features, 8 threads):

| Metric | f32 grad/hess | int16 SoA | Improvement |
|--------|---------------|-----------|-------------|
| Gradient buffer size | 8 MB | 4 MB | 2x |
| Memory bandwidth/node | ~900 MB | ~500 MB | 1.8x |
| Expected speedup | Baseline | 1.5-2x | |

**When it matters most**:

- Large datasets (memory-bound)
- Many features (more gradient reads per row)
- Row-parallel building (each thread reads all gradients)

## Design Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Gradient Quantization Pipeline                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Loss::compute_gradients()                                  ││
│  │  → GradientBuffer (f32 grads, f32 hess - SoA)               ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  QuantizedGradientBuffer::from_f32()                        ││
│  │                                                             ││
│  │  1. Find min/max of gradients and hessians                  ││
│  │  2. Compute scale factors                                   ││
│  │  3. Quantize to i16/u16 (SoA: separate arrays)              ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  HistogramBuilder::build_quantized()                        ││
│  │                                                             ││
│  │  Accumulate i64/u64 sums (no per-sample dequantization)     ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  SplitFinder (dequantize once per histogram)                ││
│  │                                                             ││
│  │  sum_grad_f32 = sum_grad_int * scale + n * offset           ││
│  │  → Use f32 sums for gain calculation                        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Design

### Quantization Scheme

We use **SoA layout** (matching existing `GradientBuffer`) with separate i16/u16 arrays:

```rust
/// Quantization parameters for a batch of gradients.
#[derive(Clone, Copy, Debug)]
pub struct GradientQuantParams {
    /// Minimum gradient value (for asymmetric quantization)
    pub grad_offset: f32,
    /// Scale factor: grad_f32 = grad_int * grad_scale + grad_offset
    pub grad_scale: f32,
    /// Scale factor: hess_f32 = hess_int * hess_scale (symmetric, min=0)
    pub hess_scale: f32,
}

impl GradientQuantParams {
    /// Compute quantization parameters from f32 gradients/hessians.
    pub fn from_f32(grads: &[f32], hess: &[f32]) -> Self {
        let (grad_min, grad_max) = min_max(grads);
        let hess_max = hess.iter().cloned().fold(0.0f32, f32::max);
        
        let grad_range = grad_max - grad_min;
        let grad_scale = if grad_range > 0.0 {
            grad_range / (i16::MAX as f32)
        } else {
            1.0
        };
        
        let hess_scale = if hess_max > 0.0 {
            hess_max / (u16::MAX as f32)
        } else {
            1.0
        };
        
        Self { grad_offset: grad_min, grad_scale, hess_scale }
    }
    
    /// Dequantize gradient sum for split finding.
    #[inline]
    pub fn dequantize_grad_sum(&self, sum: i64, count: u32) -> f32 {
        sum as f32 * self.grad_scale + count as f32 * self.grad_offset
    }
    
    /// Dequantize hessian sum.
    #[inline]
    pub fn dequantize_hess_sum(&self, sum: u64) -> f32 {
        sum as f32 * self.hess_scale
    }
}
```

### SoA Quantized Gradient Storage

Matching our existing `GradientBuffer` SoA layout:

```rust
/// Quantized gradient buffer using SoA layout.
/// 
/// This parallels `GradientBuffer` but with i16/u16 instead of f32.
/// The SoA layout enables SIMD-friendly accumulation and matches
/// our histogram building code structure.
pub struct QuantizedGradientBuffer {
    /// Quantized gradients (signed), one per sample.
    grads: Box<[i16]>,
    /// Quantized hessians (unsigned), one per sample.
    hess: Box<[u16]>,
    /// Quantization parameters for dequantization.
    params: GradientQuantParams,
}

impl QuantizedGradientBuffer {
    /// Quantize f32 gradients from GradientBuffer.
    pub fn from_gradient_buffer(buffer: &GradientBuffer) -> Self {
        let (grads_f32, hess_f32) = buffer.as_slices();
        let params = GradientQuantParams::from_f32(grads_f32, hess_f32);
        
        let grads: Box<[i16]> = grads_f32.iter()
            .map(|&g| ((g - params.grad_offset) / params.grad_scale).round() as i16)
            .collect();
        
        let hess: Box<[u16]> = hess_f32.iter()
            .map(|&h| (h / params.hess_scale).round() as u16)
            .collect();
        
        Self { grads, hess, params }
    }
    
    #[inline]
    pub fn grad(&self, idx: usize) -> i16 { self.grads[idx] }
    
    #[inline]
    pub fn hess(&self, idx: usize) -> u16 { self.hess[idx] }
    
    pub fn params(&self) -> &GradientQuantParams { &self.params }
}
```

### Integration with Existing Histogram Building

We add a parallel type for quantized histograms:

```rust
/// Feature histogram with quantized accumulators.
/// 
/// Parallels `FeatureHistogram` but with integer sums.
/// Uses i64/u64 to prevent overflow.
pub struct QuantizedFeatureHistogram {
    pub sum_grads: Box<[i64]>,  // SoA layout
    pub sum_hess: Box<[u64]>,
    pub counts: Box<[u32]>,
}
```

### Histogram Builder Integration

The accumulation loop structure is identical to f32, just with different types:

```rust
impl HistogramBuilder {
    /// Build histogram using quantized gradients.
    pub fn build_quantized<B: BinIndex>(
        &self,
        hist: &mut QuantizedNodeHistogram,
        quantized: &QuantizedMatrix<B>,
        grads: &QuantizedGradientBuffer,
        rows: &[u32],
    ) {
        // Same structure as build(), different types
        for &row in rows {
            let row = row as usize;
            let g = grads.grad(row) as i64;
            let h = grads.hess(row) as u64;
            
            for feat_idx in 0..self.num_features {
                let bin = quantized.get_bin(row, feat_idx).as_usize();
                hist.features[feat_idx].sum_grads[bin] += g;
                hist.features[feat_idx].sum_hess[bin] += h;
                hist.features[feat_idx].counts[bin] += 1;
            }
        }
    }
}
```

### Split Finding Integration

The key difference: dequantize running sums once per bin, not per sample:

```rust
impl SplitFinder {
    pub fn find_best_split_quantized(...) -> Option<SplitCandidate> {
        // Accumulate integer sums (cheap)
        for bin_idx in 0..num_bins {
            left_grad_int += hist.sum_grads[bin_idx];
            left_hess_int += hist.sum_hess[bin_idx];
            left_count += hist.counts[bin_idx];
            
            // Only dequantize when evaluating gain
            let left_grad = params.dequantize_grad_sum(left_grad_int, left_count);
            let left_hess = params.dequantize_hess_sum(left_hess_int);
            // ... gain calculation unchanged
        }
    }
}
```

## Compatibility with RFC-0025

RFC-0025 (Row-Parallel Histograms) defines parallelism strategies and
`RowParallelScratch`. Gradient quantization is orthogonal:

| Strategy | f32 Gradients | Quantized (i16) |
|----------|---------------|-----------------|
| Sequential | `build()` | `build_quantized()` |
| Feature-Parallel | `build_feature_parallel()` | `build_quantized_feature_parallel()` |
| Row-Parallel | `build_row_parallel()` | `build_quantized_row_parallel()` |

**Shared infrastructure**:

- Both use `kernel.rs` for the inner accumulation loop
- `RowParallelScratch` becomes generic over accumulator type
- Pool stores either f32 or i64 sums

**Generic Accumulator**:

```rust
pub trait Accumulator {
    type Grad: Copy + Default + AddAssign;
    type Hess: Copy + Default + AddAssign;
}

pub struct F32Accumulator;
impl Accumulator for F32Accumulator {
    type Grad = f32;
    type Hess = f32;
}

pub struct QuantizedAccumulator;
impl Accumulator for QuantizedAccumulator {
    type Grad = i64;
    type Hess = u64;
}

/// RowParallelScratch becomes generic.
pub struct RowParallelScratch<A: Accumulator> { ... }
```

## Design Decisions

### DD-1: SoA vs Packed (AoS) Storage

**Context**: LightGBM packs grad+hess into single u32. Should we?

**Options considered**:

1. **Packed AoS**: `Box<[u32]>` where `u32 = (i16 grad << 16) | u16 hess`
2. **SoA** (chosen): Separate `Box<[i16]>` and `Box<[u16]>` arrays

**Decision**: SoA layout.

**Rationale**:

- **Consistency**: Matches existing `GradientBuffer` (separate grads/hess arrays)
- **Consistency**: Matches histogram SoA layout
- **SIMD-friendly**: Process all grads with full SIMD width
- **Simpler code**: No bit manipulation
- **Flexibility**: Different precisions per component possible

**Trade-off**: Two reads vs one, but same bytes (2+2=4) and better for SIMD.

### DD-2: Quantization Precision

**Decision**: i16 gradient, u16 hessian (16-bit each).

**Rationale**:

- Matches LightGBM's proven approach
- 16 bits sufficient for GBDT (empirically validated)
- 2x memory reduction vs f32

### DD-3: Requantization Frequency

**Decision**: Once per boosting round.

**Rationale**:

- Gradient distribution changes each round
- Quantization is O(n) — cheap vs histogram building

### DD-4: Integer Accumulator Width

**Decision**: i64 for grad sum, u64 for hess sum.

**Rationale**:

- Max sum: 2^31 rows × 2^15 = 2^46 < 2^63 ✓
- No overflow possible

### DD-5: Code Duplication Avoidance

**Decision**: Generic `HistogramValueTypes` trait + shared kernel.

```rust
// Shared kernel for both f32 and quantized
pub fn accumulate_row_range<V: HistogramValueTypes, B: BinIndex>(
    hist: &mut impl HistogramMut<V>,
    quantized: &QuantizedMatrix<B>,
    grads: &impl GradientSource<V>,
    rows: &[u32],
) {
    // Same loop, generic over types
}
```

## Code Organization

```text
src/training/gbtree/histogram/
├── mod.rs                    # Re-exports
├── types.rs                  # FeatureHistogram (f32)
├── quantized_types.rs        # NEW: QuantizedFeatureHistogram (i64)
├── kernel.rs                 # Generic accumulate kernel
├── builder.rs                # HistogramBuilder
└── ...

src/training/
├── buffer.rs                 # GradientBuffer (f32)
├── quantized_buffer.rs       # NEW: QuantizedGradientBuffer (i16)
└── ...
```

## Performance Expectations

| Dataset | f32 | Quantized | Speedup |
|---------|-----|-----------|---------|
| 100K × 100 | Baseline | 1.2-1.4x | Cache fits |
| 1M × 100 | Baseline | 1.5-2x | Memory-bound |
| 10M × 100 | Baseline | 1.8-2.5x | Very memory-bound |

## Testing Strategy

1. **Quantization round-trip**: `quantize → dequantize ≈ original` within tolerance
2. **Histogram equivalence**: f32 histogram ≈ dequantized quantized histogram
3. **Split finding**: Same best split (or within tolerance)
4. **End-to-end accuracy**: Similar model quality

## Open Questions

1. **SIMD quantization?** — Vectorize quantization loop

## Future Work

- [ ] SIMD-optimized quantization
- [ ] GPU integration (RFC-0028)

## References

- [LightGBM gradient discretization](https://github.com/microsoft/LightGBM/blob/master/src/io/dataset.cpp)
- RFC-0025: Row-Parallel Histograms

## Changelog

- 2024-12-02: Initial draft
- 2024-12-02: Changed from packed AoS to SoA layout to match existing patterns
- 2024-12-02: Added DD-5 on code duplication avoidance with generic kernel
- 2024-12-02: Clarified integration with RFC-0025
- 2024-12-02: Updated to use `Loss` trait (existing name)
- 2024-12-02: Simplified split finding to focus on key difference
- 2024-12-03: Renamed `Accumulator` → `HistogramValueTypes` to clarify distinction from RFC-0025's `HistogramAccumulator`
- 2024-12-03: Removed hessian-free mode (not worth complexity for edge cases)
- 2024-12-03: Status → Approved
- 2024-12-03: Renamed `Accumulator` → `HistogramValueTypes` to clarify distinction from RFC-0025's `HistogramAccumulator`
- 2024-12-03: Removed hessian-free mode (not worth complexity for edge cases)
- 2024-12-03: Status → Approved
