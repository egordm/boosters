# 2025-11-29: Gradient Storage Layout — SoA Migration

## Goal

Evaluate Structure-of-Arrays (SoA) gradient storage for linear model training and
migrate from Array-of-Structures (AoS) to SoA-only implementation.

**Hypothesis**: Separate `grads: Vec<f32>` and `hess: Vec<f32>` arrays (SoA) will be:

1. **Faster** — better cache utilization, auto-vectorization friendly
2. **Cleaner code** — no `gradient_stride` hacks for multiclass/multi-quantile
3. **More ergonomic** — natural `[n_samples, n_outputs]` shape for multi-output

## Summary

Benchmarks showed SoA and AoS have **identical performance** for coordinate descent
training. The workload is memory-bound, so storage layout doesn't affect throughput.
Given no performance difference, we adopted SoA for its significant **code quality
benefits**: cleaner API, no stride hacks, and natural multi-output indexing.

**Outcome**: AoS (`GradientPair`) removed entirely. SoA (`GradientBuffer`) is now
the only gradient storage method.

## Environment

| Property | Value |
|----------|-------|
| CPU | Apple M3 Pro (11-core) |
| RAM | 18GB |
| OS | macOS 15.1 |
| Rust | 1.82.0 |
| Commit | Story 13 (gblinear/13) |

## Results

### Single-Output Training (SquaredLoss)

| Configuration | Samples | Time | Throughput | Notes |
|---------------|---------|------|------------|-------|
| AoS (removed) | 1,000 | 1.87 ms | 26.767 Melem/s | Baseline |
| SoA | 1,000 | 1.88 ms | 26.604 Melem/s | -0.6% |
| AoS (removed) | 10,000 | 18.81 ms | 26.581 Melem/s | Baseline |
| SoA | 10,000 | 18.79 ms | 26.602 Melem/s | +0.1% |

Performance is **identical** within measurement noise (±0.6%).

### Multiclass Training (SoftmaxLoss, K=5)

| Configuration | Samples | Time | Throughput | Notes |
|---------------|---------|------|------------|-------|
| AoS (removed) | 1,000 | 11.27 ms | 4.438 Melem/s | Baseline |
| SoA | 1,000 | 11.30 ms | 4.424 Melem/s | -0.3% |
| AoS (removed) | 10,000 | 111.98 ms | 4.465 Melem/s | Baseline |
| SoA | 10,000 | 112.88 ms | 4.430 Melem/s | -0.8% |

Performance is **identical** within measurement noise (±0.8%).

## Analysis

### Why No Performance Difference?

1. **Memory-bound workload**: Coordinate descent iterates over all samples per feature
   update. The bottleneck is memory bandwidth, not compute or cache efficiency.

2. **Same memory footprint**: Both layouts store the same 8 bytes per sample
   (4 bytes grad + 4 bytes hess), so total memory traffic is identical.

3. **Access pattern**: During weight updates, we access both `grad` and `hess` for
   each sample sequentially. Neither layout provides a clear cache advantage for
   this interleaved access pattern.

4. **No SIMD benefit**: The inner loop computes `weight_update += grad * x[i]` and
   accumulates hess separately. Both layouts require similar memory loads.

### Current Storage Layout (SoA Only)

```rust
struct GradientBuffer {
    grads: Vec<f32>,  // [g0, g1, g2, ...]
    hess: Vec<f32>,   // [h0, h1, h2, ...]
    n_samples: usize,
    n_outputs: usize,
}
```

### Code Quality Improvements (After Migration)

1. **Removed `gradient_stride` parameter**: No more stride arithmetic for multiclass.
   SoA uses natural `[n_samples, n_outputs]` shape with explicit dimensions.

2. **Simplified API**: `trainer.train(&data, &labels, &loss)` for single-output,
   `trainer.train_multiclass(&data, &labels, &loss)` for multiclass. No `num_groups`
   parameter needed.

3. **Cleaner index arithmetic**: `buffer.grad(sample, output)` with bounds-checked
   accessors instead of raw index computation.

4. **Unified loss trait interface**: Both `Loss` and `MulticlassLoss` now have
   `gradient_buffer()` method that fills a `GradientBuffer` directly.

### What Was Removed

- `GradientPair` struct and `src/training/gradient.rs` module
- `train()` / `train_multiclass()` methods with `num_groups` parameter
- `update_round()` / `update_bias()` / `update_bias_multiclass()` AoS methods
- `gradient_stride` parameter from updater functions
- AoS vs SoA benchmark comparisons

## Conclusions

| Claim | Result |
|-------|--------|
| **Faster** | ❌ No difference — memory-bound |
| **Cleaner code** | ✅ Significant improvement |
| **More ergonomic** | ✅ Significant improvement |

**Decision**: Removed AoS entirely and migrated to SoA-only implementation:

- No performance penalty from migration
- Cleaner API eliminates `gradient_stride` complexity
- Unified `gradient_buffer()` method on loss traits
- Better foundation for future optimizations (SIMD, GPU)

## Reproducing

```bash
# Training benchmarks (SoA only)
cargo bench --bench linear_training -- "training_format"
cargo bench --bench linear_training -- "multiclass_training"

# Core operation benchmarks
cargo bench --bench linear_training -- "gradient_column_sum"
```
