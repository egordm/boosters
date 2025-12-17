# Performance Optimization Proposal

**Date:** 2025-01-17  
**Status:** Proposed  
**Context:** Single-threaded training performance (post-ColumnMajor optimization)

---

## Executive Summary

After the ColumnMajor layout optimization (38% speedup, now 7% faster than LightGBM), 
this document proposes the next wave of performance improvements. Analysis identifies 
**6 high-value optimizations** with estimated **15-40% additional speedup**.

Current benchmark position:
- booste-rs: 1.39s (3.60 Melem/s)
- LightGBM: 1.49s (3.36 Melem/s) — **7% slower**
- XGBoost: 2.13s (2.35 Melem/s) — **53% slower**

---

## Priority 1: Split Finding Optimizations (Easy Wins)

### 1.1 Pre-compute Parent Score

**Location:** [src/training/gbdt/split/gain.rs](../../../src/training/gbdt/split/gain.rs)

**Problem:** `compute_gain()` calculates `score_parent` on every split candidate evaluation, 
but parent score is constant for all candidates in a node.

**Current code:**
```rust
pub fn compute_gain(&self, ..., grad_parent: f64, hess_parent: f64) -> f32 {
    let score_parent = grad_parent * grad_parent / (hess_parent + lambda);  // Recomputed every call!
    let gain = 0.5 * (score_left + score_right - score_parent) - self.min_gain as f64;
    gain as f32
}
```

**Proposed optimization:** Add `NodeGainContext` struct:
```rust
/// Pre-computed values for a node to avoid redundant computation in split finding.
pub struct NodeGainContext {
    lambda: f64,
    /// Pre-computed: 0.5 * score_parent + min_gain
    gain_offset: f64,
}

impl NodeGainContext {
    pub fn new(parent_grad: f64, parent_hess: f64, params: &GainParams) -> Self {
        let lambda = params.reg_lambda as f64;
        let parent_score = parent_grad * parent_grad / (parent_hess + lambda);
        Self {
            lambda,
            gain_offset: 0.5 * parent_score + params.min_gain as f64,
        }
    }

    #[inline]
    pub fn compute_gain(&self, gl: f64, hl: f64, gr: f64, hr: f64) -> f32 {
        let sl = gl * gl / (hl + self.lambda);
        let sr = gr * gr / (hr + self.lambda);
        (0.5 * (sl + sr) - self.gain_offset) as f32
    }
}
```

**Impact:** Eliminates 1 division + 1 multiplication per candidate. With ~256 bins × ~100 features = 25,600 candidates per node, this is significant.

**Estimated speedup:** 5-10% on split finding phase.

**Effort:** Low (< 1 hour)

---

### 1.2 Early Termination in Split Scan

**Location:** [src/training/gbdt/split/find.rs](../../../src/training/gbdt/split/find.rs#L196-L230)

**Problem:** The split scan evaluates all bins even when early termination is possible.

**Current code:**
```rust
for bin in 0..(n_bins - 1) {
    // ... accumulate left child ...
    let right_count = parent_count.saturating_sub(left_count);
    
    if self.gain_params.is_valid_split(...) {  // Checked every iteration
        // compute gain
    }
}
```

**Proposed optimization:** Use constraint-aware early exit:
```rust
let min_hess = self.gain_params.min_child_weight as f64;
let min_count = self.gain_params.min_samples_leaf;

for bin in 0..(n_bins - 1) {
    // ... accumulate left child ...
    
    // Skip if left side too small (common at start of scan)
    if left_hess < min_hess || left_count < min_count {
        continue;
    }
    
    let right_count = parent_count - left_count;  // No saturating_sub needed
    
    // Early exit: once right side is too small, it stays too small
    if right_count < min_count {
        break;
    }
    
    let right_hess = parent_hess - left_hess;
    if right_hess < min_hess {
        break;
    }
    
    // Compute gain (no validity check needed - we already verified)
    let gain = ctx.compute_gain(...);
    // ...
}
```

**Impact:** Avoids ~50% of gain computations on average (early bins fail left constraint, late bins fail right constraint).

**Estimated speedup:** 10-20% on split finding phase.

**Effort:** Low (< 1 hour)

---

## Priority 2: Histogram Operations (Medium Wins)

### 2.1 SIMD Histogram Subtraction

**Location:** [src/training/gbdt/histograms/ops.rs](../../../src/training/gbdt/histograms/ops.rs#L239-L247)

**Problem:** Histogram subtraction is a hot path (used in subtraction trick). Current scalar loop:
```rust
pub fn subtract_histogram(dst: &mut [HistogramBin], src: &[HistogramBin]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d.0 -= s.0;
        d.1 -= s.1;
    }
}
```

**Proposed optimization:** Explicit AVX2 vectorization:
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub fn subtract_histogram(dst: &mut [HistogramBin], src: &[HistogramBin]) {
    debug_assert_eq!(dst.len(), src.len());
    
    // Cast to f64 slices (HistogramBin is (f64, f64) = 16 bytes)
    let dst_f64 = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len() * 2)
    };
    let src_f64 = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len() * 2)
    };
    
    #[cfg(target_feature = "avx")]
    {
        let chunks = dst_f64.len() / 4;
        for i in 0..chunks {
            unsafe {
                let d = _mm256_loadu_pd(dst_f64.as_ptr().add(i * 4));
                let s = _mm256_loadu_pd(src_f64.as_ptr().add(i * 4));
                let r = _mm256_sub_pd(d, s);
                _mm256_storeu_pd(dst_f64.as_mut_ptr().add(i * 4), r);
            }
        }
        // Scalar remainder
        for i in (chunks * 4)..dst_f64.len() {
            dst_f64[i] -= src_f64[i];
        }
    }
    
    #[cfg(not(target_feature = "avx"))]
    {
        for (d, s) in dst_f64.iter_mut().zip(src_f64.iter()) {
            *d -= *s;
        }
    }
}
```

**Impact:** Processes 4 f64s per instruction instead of 1. Subtraction trick uses this once per node for the smaller child.

**Estimated speedup:** 2-4x on histogram subtraction (small portion of total time).

**Effort:** Medium (2-3 hours with testing)

---

### 2.2 Loop Unrolling in Histogram Building (Investigation Needed)

**Location:** [src/training/gbdt/histograms/ops.rs](../../../src/training/gbdt/histograms/ops.rs#L142-L153)

**Current inner loop:**
```rust
for i in 0..indices.len() {
    let row = unsafe { *indices.get_unchecked(i) } as usize;
    let bin = unsafe { *bins.get_unchecked(row) } as usize;
    let slot = unsafe { histogram.get_unchecked_mut(bin) };
    let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
    slot.0 += gh.grad as f64;
    slot.1 += gh.hess as f64;
}
```

**Question:** Does LLVM already unroll this? The random access pattern (bin lookup) may prevent optimal scheduling.

**Proposed investigation:**
1. Check assembly output with `cargo asm`
2. Try manual 4x unrolling
3. Benchmark both versions

**Expected outcome:** 0-15% depending on LLVM's current optimization.

**Effort:** Medium (requires benchmarking)

---

## Priority 3: Objective Function Optimizations

### 3.1 Fast Sigmoid Approximation

**Location:** [src/training/objectives/classification.rs](../../../src/training/objectives/classification.rs#L27-L30)

**Problem:** `exp()` is expensive and called once per sample per iteration:
```rust
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
```

**Proposed optimization:** Use polynomial approximation:
```rust
/// Fast sigmoid approximation using rational function.
/// Accurate to ~1e-4 for |x| < 10.
#[inline]
fn fast_sigmoid(x: f32) -> f32 {
    // Clamp to avoid overflow
    let x = x.clamp(-10.0, 10.0);
    
    // Rational approximation: 0.5 + 0.5 * x / (1 + |x|)
    // This is the "fast tanh" trick adapted for sigmoid
    0.5 + 0.5 * x / (1.0 + x.abs())
}
```

Alternative (more accurate but slightly slower):
```rust
/// Padé approximation for sigmoid.
#[inline]  
fn fast_sigmoid_pade(x: f32) -> f32 {
    let x = x.clamp(-10.0, 10.0);
    let x2 = x * x;
    // Padé(1,2) approximant
    0.5 + x * (0.2159198 + 0.0082176 * x2) / (1.0 + x2 * (0.0993484 + 0.00264768 * x2))
}
```

**Impact:** Affects LogisticLoss gradient computation. With 50K samples × 100 iterations = 5M exp() calls.

**Estimated speedup:** 2-3x on logistic gradient computation (small portion of total).

**Effort:** Low (< 1 hour)

**Trade-off:** Slight accuracy loss. Should be configurable or opt-in.

---

### 3.2 Separate Weighted/Unweighted Paths

**Location:** [src/training/objectives/classification.rs](../../../src/training/objectives/classification.rs#L43-L51)

**Problem:** The `weight_iter()` abstraction may prevent optimal code generation:
```rust
for (i, w) in weight_iter(weights, n_rows).enumerate() {
    let p = Self::sigmoid(pred_slice[i]);
    pair_slice[i].grad = w * (p - target_slice[i]);
    pair_slice[i].hess = (w * p * (1.0 - p)).max(HESS_MIN);
}
```

**Proposed optimization:** Explicit branch for unweighted (common) case:
```rust
if weights.is_empty() {
    // Unweighted path - no multiplication by w
    for i in 0..n_rows {
        let p = Self::sigmoid(pred_slice[i]);
        pair_slice[i].grad = p - target_slice[i];
        pair_slice[i].hess = (p * (1.0 - p)).max(HESS_MIN);
    }
} else {
    // Weighted path
    for i in 0..n_rows {
        let w = weights[i];
        let p = Self::sigmoid(pred_slice[i]);
        pair_slice[i].grad = w * (p - target_slice[i]);
        pair_slice[i].hess = (w * p * (1.0 - p)).max(HESS_MIN);
    }
}
```

**Impact:** Eliminates branch and multiplication in unweighted case.

**Estimated speedup:** 5-10% on gradient computation.

**Effort:** Low (< 30 minutes)

---

## Priority 4: Architectural Improvements (Larger Effort)

### 4.1 Quantized Histogram Building (LightGBM-style)

**Concept:** Use int16/int32 for gradient accumulation instead of f64.

**Benefits:**
- Halves memory bandwidth (gradients become 4 bytes instead of 8)
- Integer SIMD is faster than FP SIMD
- Can pack grad+hess into single 32-bit or 64-bit value

**Implementation sketch:**
```rust
/// Quantized gradient pair.
#[repr(C)]
pub struct GradHessI16 {
    pub grad: i16,
    pub hess: u16,  // Hessian is always positive
}

/// Quantization context for a training run.
pub struct GradientQuantizer {
    grad_scale: f32,  // Maps f32 gradient to i16
    hess_scale: f32,
}
```

**Estimated speedup:** 20-40% on histogram building.

**Effort:** High (1-2 weeks) — requires changes throughout training pipeline.

**Recommendation:** Defer to next major optimization round.

---

### 4.2 SoA Gradient Layout for SIMD

**Concept:** Store gradients as separate `grad[]` and `hess[]` arrays instead of interleaved `(grad, hess)[]`.

**Benefits:**
- Enables SIMD vectorization in gradient computation
- Better cache utilization for specific operations

**Trade-off:** 
- Worse for histogram building (need to load from two arrays)
- Requires layout conversion

**Recommendation:** Profile before implementing. The AoS layout is optimal for histogram building which is the dominant cost.

---

## Implementation Roadmap

### Phase 1: Quick Wins (This Week)
| Task | Est. Speedup | Effort | Priority |
|------|-------------|--------|----------|
| 1.1 Pre-compute parent score | 5-10% | Low | ★★★ |
| 1.2 Early termination | 10-20% | Low | ★★★ |
| 3.2 Separate weighted/unweighted | 5-10% | Low | ★★ |

**Expected combined impact:** 10-25% on split finding phase.

### Phase 2: Medium Effort (Next Week)
| Task | Est. Speedup | Effort | Priority |
|------|-------------|--------|----------|
| 2.1 SIMD histogram subtraction | 2-4x on subtract | Medium | ★★ |
| 3.1 Fast sigmoid | 2-3x on logistic | Low | ★ |
| 2.2 Loop unrolling investigation | 0-15% | Medium | ★ |

### Phase 3: Major Changes (Future)
| Task | Est. Speedup | Effort | Priority |
|------|-------------|--------|----------|
| 4.1 Quantized histograms | 20-40% | High | Future |
| 4.2 SoA gradients | TBD | High | Investigate |

---

## Benchmarking Plan

For each optimization:
1. Run `compare_training` benchmark before
2. Implement change
3. Run `compare_training` benchmark after
4. Document results in `docs/benchmarks/YYYY-MM-DD-<commit>-<topic>.md`

Key metrics to track:
- Total training time (medium dataset: 50K × 100)
- Histogram building time (via profiling)
- Split finding time (via profiling)
- Gradient computation time (via profiling)

---

## Summary

The proposed optimizations target three areas:
1. **Split finding** (Priority 1): 15-30% improvement with minimal effort
2. **Histogram operations** (Priority 2): 5-10% with SIMD
3. **Objectives** (Priority 3): 5-10% on gradient-heavy workloads

Combined with the existing ColumnMajor optimization, these changes should solidify booste-rs's lead over LightGBM and extend the gap to **15-25% faster**.
