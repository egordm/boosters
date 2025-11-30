# GBLinear Optimizations

What performance optimizations does XGBoost apply for GBLinear?

## Summary

GBLinear has **fewer optimizations** than GBTree because:
1. Linear models are inherently simpler — just dot products
2. The workload is memory-bound, not compute-bound
3. SIMD provides minimal benefit for sparse data

## Inference Optimizations

### Thread Parallelism

XGBoost parallelizes over **rows** during batch prediction:

```cpp
// From gblinear.cc - PredictBatchInternal
common::ParallelFor(batch.Size(), ctx_->Threads(), [&](omp_ulong i) {
    for (int gid = 0; gid < ngroup; ++gid) {
        Pred(batch[i], &preds[ridx * ngroup], gid, margin);
    }
});
```

Each row's prediction is independent, so this parallelizes well.

### No SIMD

GBLinear does **not** use explicit SIMD for inference. Why?

1. **Sparse data**: Most GBLinear use cases have sparse features. SIMD requires
   contiguous data, but sparse iteration jumps around randomly.

2. **Memory-bound**: The bottleneck is loading feature values and weights from
   memory, not the multiply-add operations. SIMD doesn't help memory bandwidth.

3. **Compiler auto-vectorization**: For dense data, compilers can auto-vectorize
   the simple multiply-add loop. No manual SIMD needed.

### No GPU Inference

GBLinear has **no GPU inference path**. The prediction code is CPU-only.

Why not GPU?
1. **Data transfer overhead**: Copying sparse data to GPU often costs more than
   the computation itself
2. **Low arithmetic intensity**: Just one multiply-add per non-zero feature.
   GPUs need high compute/memory ratios to shine.
3. **Small models**: Linear models are tiny (just weights). No benefit from
   GPU's parallel processing power.

## Training Optimizations

### Thread Parallelism

Training has parallelism at multiple levels:

**1. Gradient accumulation** (parallel over rows):
```cpp
// GetGradientParallel - accumulates gradient stats for one feature
common::ParallelFor(ndata, ctx->Threads(), [&](size_t j) {
    sum_grad_tloc[tid] += p.GetGrad() * v;
    sum_hess_tloc[tid] += p.GetHess() * v * v;
});
```

**2. Bias residual update** (parallel over rows):
```cpp
common::ParallelFor(ndata, ctx->Threads(), [&](auto i) {
    gpair[i * num_group + group_idx] += GradientPair(hess * dbias, 0);
});
```

**3. Shotgun updater** (parallel over features):
```cpp
// All features updated simultaneously with lock-free writes
common::ParallelFor(nfeat, ctx->Threads(), [&](auto i) {
    // Update weight for feature i
    // Race conditions in gradient updates are tolerated
});
```

### GPU Training

GBLinear **does** have a GPU training path: `gpu_coord_descent`

However, it's **deprecated** as of XGBoost 2.0:
```cpp
if (param_.updater == "gpu_coord_descent") {
    LOG(FATAL) << error::DeprecatedFunc("gpu_coord_descent", "2.0.0", ...);
}
```

The modern approach uses device dispatch:
```cpp
auto name = ctx_->DispatchDevice(
    [] { return "coord_descent"; },      // CPU
    [] { return "gpu_coord_descent"; }); // CUDA (still uses old impl)
```

#### How GPU Training Works

The GPU updater uses Thrust for parallel reductions:

```cuda
// Gradient computation on GPU
GradientPair GetGradient(int group_idx, int num_group, int fidx) {
    auto f = [=] __device__(size_t idx) {
        auto entry = d_col[idx];
        auto g = d_gpair[entry.index * num_group + group_idx];
        return GradientPair{
            g.GetGrad() * entry.fvalue,
            g.GetHess() * entry.fvalue * entry.fvalue
        };
    };
    return dh::SumReduction(multiply_iterator, col_size);
}
```

**Key observation**: The GPU version still processes **one feature at a time**
(sequential coordinate descent). It just parallelizes the gradient summation
for each feature across all rows.

### CSC Data Format

Training uses Column-Sparse-Compressed (CSC) format:

```
Column 0: [(row_2, 0.5), (row_7, 1.2), (row_9, 0.8)]
Column 1: [(row_0, 0.3), (row_5, 2.1)]
...
```

Why CSC?
- Coordinate descent iterates over features (columns)
- CSC gives O(nnz_in_column) access to all non-zeros
- Row-major would require O(total_nnz) scan per feature

### Thread-Local Buffers

Gradient accumulation uses thread-local buffers to avoid atomics:

```cpp
std::vector<double> sum_grad_tloc(num_threads, 0.0);
std::vector<double> sum_hess_tloc(num_threads, 0.0);

ParallelFor(..., [&](size_t j) {
    auto tid = omp_get_thread_num();
    sum_grad_tloc[tid] += ...;  // No contention
});

// Single-threaded reduce at end
double sum_grad = std::accumulate(sum_grad_tloc.begin(), sum_grad_tloc.end(), 0.0);
```

## Why So Few Optimizations?

### Linear Models Are Simple

The prediction loop is just:
```cpp
for (auto& entry : sparse_row) {
    sum += entry.fvalue * weights[entry.index];
}
```

There's not much to optimize. The compiler handles it well.

### Memory-Bound Workload

| Operation | Compute | Memory |
|-----------|---------|--------|
| Load feature value | 0 | 1 read |
| Load weight | 0 | 1 read |
| Multiply-add | 2 ops | 0 |

The arithmetic intensity is ~2 ops per 8 bytes loaded = 0.25 ops/byte.
Modern CPUs can do 10+ ops per byte of memory bandwidth.
This workload is **heavily memory-bound**.

### Diminishing Returns

For GBLinear, the best optimizations are:
1. ✅ Thread parallelism (XGBoost does this)
2. ✅ CSC format for training (XGBoost does this)
3. ❌ SIMD (not worth it for sparse)
4. ❌ GPU inference (data transfer overhead)

## Comparison: GBLinear vs GBTree Optimizations

| Optimization | GBTree | GBLinear |
|--------------|--------|----------|
| Thread parallelism | ✅ Yes | ✅ Yes |
| Block/batch processing | ✅ Yes | ❌ No (not beneficial) |
| SIMD | ❌ No (gather issues) | ❌ No (sparse) |
| GPU inference | ✅ Yes | ❌ No |
| GPU training | ✅ Yes | ⚠️ Deprecated |
| Array layout (SoA) | ✅ Yes | N/A (just weights) |
| Histogram tricks | ✅ Yes (training) | N/A |

GBTree has more optimizations because tree traversal has more structure to exploit.
