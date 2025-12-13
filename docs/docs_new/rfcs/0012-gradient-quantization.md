# RFC-0012: Gradient Quantization

- **Status**: Draft
- **Created**: 2024-12-05
- **Depends on**: RFC-0009 (Histogram Building)
- **Scope**: Optional gradient compression for bandwidth-bound training

## Summary

Gradient quantization reduces gradient/hessian storage from 8 bytes (f32 + f32) to 2 bytes (i8 + u8) per sample. This is a **user-controlled optimization** that trades ~0.1% accuracy for 4× memory bandwidth reduction. Useful for very large datasets (>1M rows) on bandwidth-limited hardware.

## Motivation

During histogram building, we iterate over gradients for each selected row:

```text
for row in rows:
    hist.add(bin, grads[row], hess[row])  // Memory access
```

For large datasets, gradient access becomes bandwidth-bound. Quantizing gradients reduces memory traffic at the cost of precision.

**Key difference from feature quantization (RFC-0008)**:
- Feature quantization is **lossless** for split finding (we only need bin indices)
- Gradient quantization is **lossy** (affects split quality)

## When to Use

| Dataset Size | Platform | Recommendation |
|--------------|----------|----------------|
| < 500k rows | Any | Don't use (no benefit) |
| 500k - 5M | x86_64 | Consider if training is slow |
| 500k - 5M | Apple Silicon | Usually not needed (high bandwidth) |
| > 5M rows | x86_64 | Likely beneficial |
| > 5M rows | Apple Silicon | Benchmark to decide |

## Design

### User Configuration

```rust
/// User-facing enum for gradient precision.
#[derive(Default, Clone, Copy)]
pub enum GradientPrecision {
    #[default]
    F32,    // Full precision (default)
    I16,  // 8-bit grad + 8-bit hess, ~0.1% accuracy loss
}
```

Added to training parameters:

```rust
pub struct GBTreeParams {
    // ... other params ...
    pub gradient_precision: GradientPrecision,
}
```

### Gradients Trait

All gradient storage types implement a common trait for zero-cost generic accumulation:

```rust
/// Trait for gradient access. Implementations are monomorphized.
pub trait Gradients {
    /// Get dequantized gradient at index. Inlined.
    fn grad(&self, idx: usize) -> f32;
    /// Get dequantized hessian at index. Inlined.
    fn hess(&self, idx: usize) -> f32;
    /// Number of samples.
    fn len(&self) -> usize;
}
```

### Storage Variants

Each precision level is a named struct:

```rust
/// Full precision gradients (default).
pub struct GradientsF32 {
    pub grads: Box<[f32]>,
    pub hess: Box<[f32]>,
}

impl Gradients for GradientsF32 {
    #[inline(always)]
    fn grad(&self, idx: usize) -> f32 { self.grads[idx] }
    #[inline(always)]
    fn hess(&self, idx: usize) -> f32 { self.hess[idx] }
    fn len(&self) -> usize { self.grads.len() }
}

/// Quantized gradients: 8-bit each, dequantized on access.
pub struct GradientsI16 {
    pub grads: Box<[i8]>,     // Signed: -127..127
    pub hess: Box<[u8]>,      // Unsigned: 0..255
    pub grad_scale: f32,
    pub hess_scale: f32,
}

impl Gradients for GradientsI16 {
    #[inline(always)]
    fn grad(&self, idx: usize) -> f32 { 
        self.grads[idx] as f32 * self.grad_scale 
    }
    #[inline(always)]
    fn hess(&self, idx: usize) -> f32 { 
        self.hess[idx] as f32 * self.hess_scale 
    }
    fn len(&self) -> usize { self.grads.len() }
}
```

### GradientStorage Enum

Runtime dispatch wrapper:

```rust
pub enum GradientStorage {
    F32(GradientsF32),
    I16(GradientsI16),
}

impl GradientStorage {
    pub fn from_f32(grads: Box<[f32]>, hess: Box<[f32]>) -> Self {
        Self::F32(GradientsF32 { grads, hess })
    }
    
    pub fn quantize_i16(grads: &[f32], hess: &[f32]) -> Self {
        Self::I16(GradientsI16::quantize(grads, hess))
    }
}
```

## Quantization Algorithm

```text
GradientsI16::quantize(grads, hess):
    // Compute scales from data range
    max_grad = max(abs(grads))
    max_hess = max(hess)  // Hessian always positive
    
    grad_scale = max_grad / 127.0
    hess_scale = max_hess / 255.0
    
    // Quantize (can be parallelized)
    q_grads = []
    q_hess = []
    for i in 0..len:
        q_grads.push(round(grads[i] / grad_scale) as i8)
        q_hess.push(round(hess[i] / hess_scale) as u8)
    
    return GradientsI16 { 
        grads: q_grads, 
        hess: q_hess, 
        grad_scale, 
        hess_scale 
    }
```

## Usage in Histogram Building

Histogram accumulation is generic over `Gradients`:

```rust
impl HistogramBuilder {
    pub fn build<G: Gradients>(
        &mut self,
        histogram: &mut NodeHistogram,
        quantized: &QuantizedMatrix,
        grads: &G,
        rows: &[u32],
    ) {
        // ... accumulation using grads.grad(row), grads.hess(row)
    }
}
```

Dispatch happens once at the training loop level:

```text
// In training loop
match &grad_storage {
    GradientStorage::F32(g) => builder.build(hist, quantized, g, rows),
    GradientStorage::Int16(g) => builder.build(hist, quantized, g, rows),
}
```

The trait methods are inlined, so `GradientsInt16` fuses dequantization (multiply by scale) directly into the accumulation loop. No per-element dispatch overhead.

## Memory Layout

Both variants use **SoA** (Struct of Arrays):

```text
GradientsF32:
  grads: [f32; n]  // 4 bytes × n
  hess:  [f32; n]  // 4 bytes × n
  Total: 8 bytes per sample

GradientsI16:
  grads: [i8; n]   // 1 byte × n
  hess:  [u8; n]   // 1 byte × n
  scales: 8 bytes  // Fixed overhead
  Total: 2 bytes per sample + 8 bytes
```

SoA is preferred over AoS because:
1. Matches histogram iteration pattern (grad and hess accessed together per row)
2. Better vectorization potential
3. Consistent with feature quantization layout

## Future: Int8 (4-bit packing)

For even more aggressive compression, pack two values per byte:

```rust
/// 4-bit grad + 4-bit hess per sample.
pub struct GradientsInt8 {
    pub grads: Box<[u8]>,  // Packed: low nibble = sample 2i, high = sample 2i+1
    pub hess: Box<[u8]>,
    pub grad_scale: f32,
    pub hess_scale: f32,
}
```

Deferred because:
- More complex unpacking logic
- Higher accuracy impact (~0.5%)
- Int16 already provides 4× bandwidth reduction

## Design Decisions

### DD-1: Separate SoA Arrays (Not Packed)

**Decision**: Store `grads` and `hess` in separate arrays.

**Rationale**: 
- Simpler implementation
- Better alignment
- Easier to vectorize
- Histogram accumulation accesses both together anyway

### DD-2: Per-Round Quantization

**Decision**: Quantize gradients fresh each round (not cached).

**Rationale**:
- Gradients change every round
- Scale factors change with gradient distribution
- Quantization is O(n), cheap compared to histogram building

### DD-3: User-Controlled, Not Auto-Enabled

**Decision**: Require explicit opt-in via `gradient_precision` parameter.

**Rationale**:
- Affects model quality (lossy)
- Benefit is hardware-dependent
- Users should make informed tradeoff

## References

- [LightGBM Gradient Compression](https://github.com/microsoft/LightGBM/blob/master/src/io/dense_bin.hpp)
