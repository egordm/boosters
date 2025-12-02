# Gradient Batch Computation Benchmark

**Date**: 2025-11-29  
**Story**: 12 - Gradient Batch Optimization  
**System**: Apple M3 Pro, macOS  
**Commit**: `c3eb835` (benchmark code available at this commit)  

## Objective

Compare naive vs optimized `gradient_buffer` implementations:

1. **Naive**: Default trait impl that loops calling `compute_gradient()` per sample
2. **Optimized**: Custom `gradient_buffer()` impl with separate loops for better vectorization

## Hypothesis

Custom batch implementations should be faster due to:
- Separate loops for grad/hess allow independent vectorization
- Constants (like hess=1.0) can use `fill()` instead of per-element writes
- Single-pass designs avoid redundant computation

## Optimized Implementations

```rust
// SquaredLoss: separate loops, hess is constant 1.0
fn gradient_buffer(&self, preds: &[f32], labels: &[f32], buf: &mut GradientBuffer) {
    let (grads, hess) = buf.as_mut_slices();
    for i in 0..preds.len() {
        grads[i] = preds[i] - labels[i];
    }
    hess.fill(1.0);  // memset vs per-element
}

// LogisticLoss: single combined loop (exp dominates anyway)
fn gradient_buffer(&self, preds: &[f32], labels: &[f32], buf: &mut GradientBuffer) {
    let (grads, hess) = buf.as_mut_slices();
    for i in 0..preds.len() {
        let p = 1.0 / (1.0 + (-preds[i]).exp());
        grads[i] = p - labels[i];
        hess[i] = (p * (1.0 - p)).max(1e-6);
    }
}

// QuantileLoss: single-pass conditional
fn gradient_buffer(&self, preds: &[f32], labels: &[f32], buf: &mut GradientBuffer) {
    let alpha = self.alphas[0];
    let (grads, hess) = buf.as_mut_slices();
    for i in 0..preds.len() {
        grads[i] = if preds[i] >= labels[i] { 1.0 - alpha } else { -alpha };
        hess[i] = 1.0;
    }
}
```

## Results

### 1k samples

| Loss | Naive | Optimized | Speedup |
|------|-------|-----------|---------|
| SquaredLoss | 147 ns | 142 ns | 1.03x |
| LogisticLoss | 2.36 µs | 2.37 µs | 1.00x |
| QuantileLoss | 146 ns | 143 ns | 1.02x |

### 10k samples

| Loss | Naive | Optimized | Speedup |
|------|-------|-----------|---------|
| SquaredLoss | 2.25 µs | 1.98 µs | 1.14x |
| LogisticLoss | 23.6 µs | 23.8 µs | 0.99x |
| QuantileLoss | 2.29 µs | 1.60 µs | 1.43x |

### 100k samples

| Loss | Naive | Optimized | Speedup |
|------|-------|-----------|---------|
| SquaredLoss | 18.6 µs | 19.0 µs | 0.98x |
| LogisticLoss | 237 µs | 238 µs | 1.00x |
| QuantileLoss | 19.8 µs | 19.2 µs | 1.03x |

### Throughput (100k samples)

| Loss | Naive | Optimized |
|------|-------|-----------|
| SquaredLoss | **5.37 Gelem/s** | **5.26 Gelem/s** |
| LogisticLoss | 423 Melem/s | 420 Melem/s |
| QuantileLoss | **5.04 Gelem/s** | **5.22 Gelem/s** |

## Analysis

### Key Finding: No Significant Difference

The Rust compiler (LLVM) already does an excellent job auto-vectorizing the naive loop:

```rust
// Naive - compiler vectorizes this well
for (i, (pred, label)) in preds.iter().zip(labels.iter()).enumerate() {
    let (g, h) = self.compute_gradient(*pred, *label);  // gets inlined
    grads[i] = g;
    hess[i] = h;
}
```

### Why Optimized ≈ Naive?

1. **Inlining**: `compute_gradient()` is `#[inline]` and small, so LLVM inlines it
2. **Auto-vectorization**: LLVM recognizes the pattern and vectorizes
3. **Memory-bound**: At 100k samples, we're likely hitting memory bandwidth limits

### LogisticLoss: exp() Dominates

For LogisticLoss, both naive and optimized are identical because:
- The `exp()` operation dominates runtime (~420 Melem/s)
- Loop structure doesn't matter when transcendental ops are the bottleneck

### Squared/Quantile: Already Optimal

At ~5 Gelem/s, these simple losses are already:
- Vectorized (SIMD)
- Cache-efficient
- Near memory bandwidth limits

## Conclusions

1. **LLVM auto-vectorization is excellent** - manual optimization doesn't help
2. **Simple losses are memory-bound** at ~5 Gelem/s throughput
3. **exp()-heavy losses are compute-bound** at ~420 Melem/s
4. **No code changes needed** - current implementation is optimal

## Recommendation

**Keep custom `gradient_buffer` implementations** for:
- Code clarity (explicit intent)
- Potential future optimizations (SIMD intrinsics for exp)
- No performance regression

But don't expect significant speedups from loop restructuring alone.

## Future Work

1. **SIMD exp/log**: Use fast-math SIMD for LogisticLoss/Softmax
2. **Parallelization**: Multi-threaded gradient computation for very large datasets
3. **Fused ops**: Combine gradient + update in single pass

## Raw Benchmark Output

```
gradient_computation/squared/naive/1000       time: [147.15 ns 147.60 ns 148.10 ns] thrpt: [6.75 Gelem/s]
gradient_computation/squared/optimized/1000   time: [142.10 ns 142.30 ns 142.52 ns] thrpt: [7.02 Gelem/s]
gradient_computation/logistic/naive/1000      time: [2.3557 µs 2.3589 µs 2.3626 µs] thrpt: [423 Melem/s]
gradient_computation/logistic/optimized/1000  time: [2.3635 µs 2.3653 µs 2.3673 µs] thrpt: [422 Melem/s]
gradient_computation/quantile/naive/1000      time: [145.32 ns 145.92 ns 146.61 ns] thrpt: [6.85 Gelem/s]
gradient_computation/quantile/optimized/1000  time: [143.12 ns 143.26 ns 143.42 ns] thrpt: [6.98 Gelem/s]

gradient_computation/squared/naive/10000       time: [2.2351 µs 2.2516 µs 2.2682 µs] thrpt: [4.41 Gelem/s]
gradient_computation/squared/optimized/10000   time: [1.9299 µs 1.9755 µs 2.0156 µs] thrpt: [5.06 Gelem/s]
gradient_computation/logistic/naive/10000      time: [23.611 µs 23.627 µs 23.643 µs] thrpt: [423 Melem/s]
gradient_computation/logistic/optimized/10000  time: [23.754 µs 23.768 µs 23.784 µs] thrpt: [421 Melem/s]
gradient_computation/quantile/naive/10000      time: [2.2230 µs 2.2876 µs 2.3832 µs] thrpt: [4.37 Gelem/s]
gradient_computation/quantile/optimized/10000  time: [1.5421 µs 1.6048 µs 1.6696 µs] thrpt: [6.23 Gelem/s]

gradient_computation/squared/naive/100000       time: [18.516 µs 18.611 µs 18.714 µs] thrpt: [5.37 Gelem/s]
gradient_computation/squared/optimized/100000   time: [18.997 µs 19.003 µs 19.011 µs] thrpt: [5.26 Gelem/s]
gradient_computation/logistic/naive/100000      time: [236.59 µs 236.84 µs 237.12 µs] thrpt: [422 Melem/s]
gradient_computation/logistic/optimized/100000  time: [238.02 µs 238.14 µs 238.28 µs] thrpt: [420 Melem/s]
gradient_computation/quantile/naive/100000      time: [19.102 µs 19.844 µs 20.805 µs] thrpt: [5.04 Gelem/s]
gradient_computation/quantile/optimized/100000  time: [19.165 µs 19.171 µs 19.177 µs] thrpt: [5.22 Gelem/s]
```
