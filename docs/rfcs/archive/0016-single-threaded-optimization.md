# RFC-0016: Single-Threaded Training Optimization

- **Status**: Draft
- **Created**: 2025-12-27
- **Updated**: 2025-12-27
- **Depends on**: RFC-0004 (Binning and Histograms), RFC-0005 (Tree Growing)
- **Scope**: Histogram building, memory layout, cache optimization

## Summary

This RFC proposes a series of optimizations to close the 6x single-threaded performance gap between boosters and LightGBM. The optimizations focus on histogram building—the dominant cost in tree training—through quantized gradient accumulation, software prefetching, packed gradient storage, and multi-feature histogram building.

## Motivation

### Problem Statement

Benchmarks show boosters is **6x slower than LightGBM** in single-threaded mode:

| Mode | Boosters | LightGBM | Ratio |
| --- | --- | --- | --- |
| Single-threaded | 0.46s | 0.08s | **6.06x slower** |
| Multi-threaded (8 cores) | 0.26s | 0.46s | **1.7x faster** |

While multi-threaded performance is excellent, single-threaded performance matters for:

1. **MLOps pipelines**: Training models per-location/per-segment is common in production
2. **Cloud cost**: Single-threaded efficiency reduces vCPU requirements
3. **Latency-sensitive**: Faster single-tree training means faster iteration
4. **Multi-threaded scaling**: Single-threaded improvements often translate to parallel speedups

### Root Cause Analysis

After analyzing LightGBM's codebase, the performance gap stems from several key differences:

| Area | Boosters | LightGBM | Impact |
| --- | --- | --- | --- |
| **Histogram bins** | `(f64, f64)` = 16 bytes | Packed `i32` or `i16` = 4-8 bytes | 2-4x bandwidth |
| **Gradient storage** | Separate `f32` grad/hess | Packed `i16` grad+hess | 2x loads |
| **Accumulation** | 2 separate f64 adds | 1 packed integer add | 2x stores |
| **Prefetching** | None | Software prefetch | ~30% speedup |
| **Feature bundling** | Per-feature loops | Multi-feature groups | Better cache |

### Why Parallel Mode Compensates

Our parallel mode is fast because:

1. **Feature-parallel histogram building** utilizes multiple cores effectively
2. **Memory bandwidth scales** with cores on modern CPUs
3. **Rayon work-stealing** keeps all cores busy

But single-threaded mode exposes the raw per-operation inefficiency.

## Design

### Overview

We propose a phased approach with four optimization levels:

1. **Phase 1 (Quick Wins)**: Software prefetching + packed GradsTuple (estimated 1.3-1.5x)
2. **Phase 2 (Medium Effort)**: Quantized 16-bit histogram mode (estimated 2-3x)
3. **Phase 3 (Larger Effort)**: Multi-feature histogram building (estimated 1.2-1.5x)
4. **Phase 4 (Advanced)**: SIMD vectorization for contiguous ranges (estimated 1.1-1.3x)

Combined, these should achieve 3-5x single-threaded speedup.

### Phase 1: Prefetching and Packed Gradients

#### 1a. Software Prefetching in Histogram Loops

Add explicit prefetch instructions to hide memory latency when accessing bin indices.

```rust
// Current code (no prefetching)
fn build_u8_gathered(
    bins: &[u8],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    for i in 0..indices.len() {
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

// Proposed: With prefetching
fn build_u8_gathered_prefetch(
    bins: &[u8],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    const PREFETCH_OFFSET: usize = 64; // Cache line lookahead
    
    let len = indices.len();
    let prefetch_end = len.saturating_sub(PREFETCH_OFFSET);
    
    // Prefetch loop
    for i in 0..prefetch_end {
        let pf_row = unsafe { *indices.get_unchecked(i + PREFETCH_OFFSET) } as usize;
        prefetch_read(bins.as_ptr().add(pf_row));
        
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
    
    // Tail without prefetch
    for i in prefetch_end..len {
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        let slot = unsafe { histogram.get_unchecked_mut(bin) };
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        slot.0 += gh.grad as f64;
        slot.1 += gh.hess as f64;
    }
}

/// Prefetch for read access (architecture-specific)
#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_READ, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
}
```

#### 1b. Ensure GradsTuple is Optimally Packed

Our current `GradsTuple` is already `#[repr(C)]` which is good, but we should verify the compiler treats it as a single 64-bit load:

```rust
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(8))]  // Ensure 8-byte alignment for single load
pub struct GradsTuple {
    pub grad: f32,
    pub hess: f32,
}
```

### Phase 2: Quantized Histogram Mode

The biggest opportunity is using integer histograms when precision allows. LightGBM packs gradient+hessian into 16 or 32 bits per sample.

#### 2a. Quantized Gradient Representation

```rust
/// Quantized gradient representation for fast histogram building.
/// 
/// Gradients and hessians are scaled to fit in signed integers:
/// - 8-bit mode: [-128, 127] for each
/// - 16-bit mode: [-32768, 32767] for each
#[derive(Clone, Copy)]
#[repr(C)]
pub struct QuantizedGrads {
    /// Packed gradient and hessian.
    /// For 16-bit: high 16 bits = gradient, low 16 bits = hessian
    /// For 8-bit: high 8 bits = gradient, low 8 bits = hessian
    packed: i32,
}

impl QuantizedGrads {
    /// Create from f32 gradient and hessian with scaling.
    #[inline]
    pub fn from_f32_16bit(grad: f32, hess: f32, grad_scale: f32, hess_scale: f32) -> Self {
        let g = (grad * grad_scale).clamp(-32768.0, 32767.0) as i16;
        let h = (hess * hess_scale).clamp(-32768.0, 32767.0) as i16;
        Self {
            packed: ((g as i32) << 16) | ((h as i32) & 0xFFFF),
        }
    }
    
    /// Create from f32 gradient and hessian in 8-bit mode.
    #[inline]
    pub fn from_f32_8bit(grad: f32, hess: f32, grad_scale: f32, hess_scale: f32) -> Self {
        let g = (grad * grad_scale).clamp(-128.0, 127.0) as i8;
        let h = (hess * hess_scale).clamp(-128.0, 127.0) as i8;
        Self {
            packed: ((g as i32) << 8) | ((h as i32) & 0xFF),
        }
    }
}
```

#### 2b. Quantized Histogram Bins

```rust
/// 32-bit packed histogram bin (16-bit gradient sum, 16-bit hessian sum).
pub type QuantizedHistBin32 = i64;  // Room for accumulated 16-bit values

/// 16-bit packed histogram bin (8-bit gradient sum, 8-bit hessian sum).
pub type QuantizedHistBin16 = i32;  // Room for accumulated 8-bit values

/// Fast histogram accumulation with single integer add.
#[inline]
fn build_u8_gathered_quantized(
    bins: &[u8],
    ordered_grads: &[QuantizedGrads],
    histogram: &mut [QuantizedHistBin32],
    indices: &[u32],
) {
    const PREFETCH_OFFSET: usize = 64;
    let len = indices.len();
    let prefetch_end = len.saturating_sub(PREFETCH_OFFSET);
    
    for i in 0..prefetch_end {
        let pf_row = unsafe { *indices.get_unchecked(i + PREFETCH_OFFSET) } as usize;
        prefetch_read(bins.as_ptr().add(pf_row));
        
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let bin = unsafe { *bins.get_unchecked(row) } as usize;
        let packed = unsafe { ordered_grads.get_unchecked(i).packed };
        
        // Extend i32 to i64 preserving the 16-bit halves
        let extended = extend_packed_16_to_32(packed);
        unsafe {
            *histogram.get_unchecked_mut(bin) += extended;
        }
    }
    
    // Tail loop...
}

/// Extend 16-bit packed values to 32-bit to prevent overflow during accumulation.
#[inline]
fn extend_packed_16_to_32(packed: i32) -> i64 {
    let grad = (packed >> 16) as i16 as i64;
    let hess = (packed & 0xFFFF) as i16 as i64;
    (grad << 32) | (hess & 0xFFFFFFFF)
}
```

#### 2c. Automatic Mode Selection

Choose quantized mode based on dataset size and gradient range:

```rust
/// Histogram precision mode selection.
pub enum HistogramMode {
    /// Full precision: f64 accumulation (current behavior).
    Full,
    /// 16-bit quantized: ~1-2% accuracy loss for 2-3x speedup.
    Quantized16,
    /// 8-bit quantized: ~2-5% accuracy loss for 3-4x speedup.
    Quantized8,
}

impl HistogramMode {
    /// Select mode based on dataset characteristics.
    pub fn auto_select(n_samples: usize, max_grad: f32, max_hess: f32) -> Self {
        // For small datasets, full precision is fast enough
        if n_samples < 10_000 {
            return Self::Full;
        }
        
        // Check if gradients can be accurately represented in 16-bit
        // LightGBM uses a threshold based on the gradient range
        let grad_range = max_grad.abs() * 2.0;
        let hess_range = max_hess.abs() * 2.0;
        
        // 16-bit can represent ~4 significant digits
        if grad_range < 10000.0 && hess_range < 10000.0 {
            Self::Quantized16
        } else {
            Self::Full
        }
    }
}
```

### Phase 3: Multi-Feature Histogram Building

Instead of iterating over features separately, build histograms for multiple features in a single pass over the data.

#### 3a. Row-Major Feature Group Storage

For features stored together (row-major layout), we can process multiple features per row access:

```rust
/// Build histograms for a feature group in a single data pass.
fn build_group_gathered(
    group_bins: &[u8],          // Row-major: [row0_f0, row0_f1, ..., row1_f0, ...]
    n_features: usize,
    ordered_grad_hess: &[GradsTuple],
    group_histogram: &mut [HistogramBin],  // Concatenated per-feature histograms
    feature_offsets: &[usize],  // Offset into histogram for each feature
    indices: &[u32],
) {
    const PREFETCH_OFFSET: usize = 32;
    let len = indices.len();
    
    for i in 0..len {
        let row = unsafe { *indices.get_unchecked(i) } as usize;
        let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
        let grad = gh.grad as f64;
        let hess = gh.hess as f64;
        
        // Prefetch next row's feature bins
        if i + PREFETCH_OFFSET < len {
            let pf_row = unsafe { *indices.get_unchecked(i + PREFETCH_OFFSET) } as usize;
            prefetch_read(group_bins.as_ptr().add(pf_row * n_features));
        }
        
        // Process all features for this row in one go
        let row_start = row * n_features;
        for f in 0..n_features {
            let bin = unsafe { *group_bins.get_unchecked(row_start + f) } as usize;
            let hist_idx = feature_offsets[f] + bin;
            let slot = unsafe { group_histogram.get_unchecked_mut(hist_idx) };
            slot.0 += grad;
            slot.1 += hess;
        }
    }
}
```

#### 3b. Exclusive Feature Bundling (EFB)

LightGBM's EFB bundles mutually exclusive sparse features. This is already tracked in RFC-0011, but the key insight for performance is:

- Features with disjoint non-zero patterns can share bin storage
- One data pass builds histograms for all bundled features
- Critical for sparse datasets (e.g., one-hot encoded categoricals)

### Phase 4: SIMD Vectorization

For contiguous row ranges (root node, some child nodes), SIMD can provide additional speedup.

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-vectorized histogram building for contiguous ranges.
#[cfg(target_arch = "x86_64")]
fn build_u8_contiguous_simd(
    bins: &[u8],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    start_row: usize,
) {
    // For contiguous ranges, we can use SIMD gather operations
    // This is a simplified sketch - full implementation would use
    // AVX2/AVX-512 gather intrinsics
    
    let n_rows = ordered_grad_hess.len();
    let bins = &bins[start_row..start_row + n_rows];
    
    // Process 8 rows at a time with AVX2
    let chunks = n_rows / 8;
    
    for chunk in 0..chunks {
        let base = chunk * 8;
        
        // Load 8 bin indices
        let bin_indices = unsafe {
            _mm256_loadu_si256(bins.as_ptr().add(base) as *const __m256i)
        };
        
        // Convert u8 to u32 indices for gather
        let indices_32 = unsafe { _mm256_cvtepu8_epi32(_mm256_extracti128_si256(bin_indices, 0)) };
        
        // Load gradients and hessians (would need proper gather implementation)
        // ...
        
        // Scatter-add to histogram (complex with variable indices)
        // ...
    }
    
    // Scalar tail for remaining rows
    // ...
}
```

Note: SIMD histogram building is complex due to variable scatter patterns. The main wins come from phases 1-3.

### Data Structures Summary

```rust
// Phase 1: Minimal changes
#[repr(C, align(8))]
pub struct GradsTuple { pub grad: f32, pub hess: f32 }

// Phase 2: New quantized types
#[repr(C)]
pub struct QuantizedGrads { packed: i32 }
pub type QuantizedHistBin32 = i64;
pub type QuantizedHistBin16 = i32;

// Phase 3: Feature group info
pub struct FeatureGroupMeta {
    pub n_features: usize,
    pub feature_offsets: Vec<usize>,  // Into histogram
    pub bin_counts: Vec<usize>,
}
```

## Design Decisions

### DD-1: Quantization Bit Width

**Context**: How many bits to use for quantized gradients?

**Options considered**:

1. **8-bit**: 4x memory reduction, but limited precision
2. **16-bit**: 2x memory reduction, good precision
3. **Adaptive**: Choose based on gradient range

**Decision**: Start with 16-bit as the primary quantized mode.

**Consequences**:

- 16-bit provides 2-3x speedup with minimal accuracy loss
- 8-bit can be added later for specific use cases
- Scale factors must be computed per-tree iteration

### DD-2: Prefetch Distance

**Context**: How far ahead should we prefetch?

**Options considered**:

1. **Fixed distance**: 64 elements (LightGBM's approach)
2. **Cache-line based**: 64 bytes / sizeof(element)
3. **Dynamic**: Tune based on cache size

**Decision**: Use fixed 64-element lookahead initially.

**Consequences**:

- Simple implementation
- May not be optimal for all cache hierarchies
- Can tune later based on benchmarks

### DD-3: Full Precision Fallback

**Context**: When to use full precision vs quantized?

**Options considered**:

1. **Always quantized**: Maximum speed, some accuracy risk
2. **Auto-select**: Based on gradient range and dataset size
3. **User-controlled**: Let user choose precision mode

**Decision**: Auto-select with user override option.

**Consequences**:

- Safe default behavior
- Power users can force modes for specific needs
- Need to compute gradient statistics for auto mode

## Integration

| Component | Integration Point | Notes |
| --- | --- | --- |
| RFC-0004 (Histograms) | HistogramBuilder | Add quantized kernels |
| RFC-0005 (Tree Growing) | Grower | Select histogram mode |
| RFC-0011 (Feature Bundling) | EFB | Multi-feature build |
| Training Config | TrainerParams | Precision mode option |

## Benchmarking Plan

Each phase should be benchmarked independently:

```text
Dataset: Covertype (581k samples, 54 features)
Metric: Single-tree training time, single-threaded
Target: Match or beat LightGBM single-threaded performance

Phase 1 target: 0.35s (from 0.46s = 1.3x speedup)
Phase 2 target: 0.15s (from 0.35s = 2.3x speedup)
Phase 3 target: 0.12s (from 0.15s = 1.25x speedup)
Overall target: 0.08-0.12s (match LightGBM's 0.08s)
```

## Open Questions

1. **Quantization accuracy impact**: Need to measure accuracy degradation on real datasets.
   - Run accuracy benchmarks on standard datasets
   - Compare RMSE/AUC with and without quantization

2. **Apple Silicon optimization**: ARM NEON prefetch behavior may differ.
   - Benchmark prefetch on M1/M2
   - May need different prefetch distances

3. **Multi-feature vs feature-parallel trade-off**: For parallel mode, is feature-parallel still better?
   - Current: Feature-parallel is 2.8x faster than row-parallel
   - Multi-feature might change this balance

## Future Work

- [ ] AVX-512 SIMD kernels for x86_64
- [ ] NEON SIMD kernels for ARM64
- [ ] GPU histogram building (CUDA/Metal)
- [ ] Adaptive prefetch distance based on runtime measurement
- [ ] Exclusive Feature Bundling (RFC-0011) for sparse data

## References

- [LightGBM dense_bin.hpp](https://github.com/microsoft/LightGBM/blob/master/src/io/dense_bin.hpp) - Prefetching and quantization
- [LightGBM feature_histogram.hpp](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) - Histogram operations
- [LightGBM dataset.cpp](https://github.com/microsoft/LightGBM/blob/master/src/io/dataset.cpp) - Gradient ordering and group histogram building
- [LightGBM multi_val_dense_bin.hpp](https://github.com/microsoft/LightGBM/blob/master/src/io/multi_val_dense_bin.hpp) - Multi-feature histogram building
- Internal benchmarks: `tmp/development_review_2025-01-24.md`

## Changelog

- 2025-12-27: Initial draft based on LightGBM comparison analysis
