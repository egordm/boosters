# GPU Training in XGBoost

## Overview

XGBoost fully supports GPU-accelerated training via CUDA. The GPU implementation
(`grow_gpu_hist`) provides significant speedups over CPU training, especially for
large datasets.

## Key Components

### Tree Updater: `updater_gpu_hist.cu`

The main GPU tree updater (`GPUHistMaker`) mirrors the CPU histogram-based approach
but with GPU-optimized kernels:

```cpp
class GPUHistMaker : public TreeUpdater {
  GPUHistMakerDevice* p_scimpl_;  // Scalar tree implementation
  MultiTargetHistMaker* p_mtimpl_;  // Multi-target implementation
};
```

### Data Format: ELLPACK

GPU training uses ELLPACK format instead of CSR for better GPU memory access:

```text
ELLPACK format:
  - Fixed-width rows (padded to max non-zeros)
  - Coalesced memory access for GPU threads
  - Feature indices stored in column-major order
```

```cpp
class EllpackPage {
  // Quantized feature values in ELLPACK format
  common::Span<uint32_t> gidx_buffer;
  // Number of non-zero entries per row (fixed for dense)
  size_t row_stride;
};
```

## GPU-Specific Optimizations

### 1. Gradient Quantization

GPU uses quantized (int64) gradients to enable efficient atomic operations:

```cpp
class GradientQuantiser {
  // Convert float gradients to fixed-point for atomic add
  GradientPairInt64 ToFixedPoint(GradientPair const& gpair);
  GradientPair ToFloatingPoint(GradientPairInt64 const& gpair);
};
```

**Why**: Atomic float operations are slow on GPU. Int64 atomics are faster and
avoid floating-point precision issues in parallel accumulation.

### 2. Feature Groups

Features are grouped for shared memory histogram building:

```cpp
class FeatureGroups {
  // Group features that fit together in shared memory
  // Each group's histogram fits in one shared memory block
  std::vector<int> feature_segments;  // [group_start, ...]
  size_t max_group_bins;  // Max bins per group
};
```

**Strategy**:
1. Divide features into groups that fit in shared memory
2. Build histogram for one group at a time
3. Use shared memory for fast atomic accumulation

### 3. Histogram Building Kernel

The GPU histogram kernel uses shared memory for fast accumulation:

```cpp
template <typename GradientSumT>
__global__ void SharedMemHistKernel(
    EllpackAccessor matrix,
    common::Span<GradientPair const> gpair,
    common::Span<uint32_t const> ridx,
    common::Span<GradientSumT> histogram,
    FeatureGroupsAccessor feature_groups
) {
  // Shared memory histogram for this feature group
  extern __shared__ char smem[];
  GradientSumT* smem_hist = reinterpret_cast<GradientSumT*>(smem);
  
  // Initialize shared memory to zero
  for (int i = threadIdx.x; i < n_bins_group; i += blockDim.x) {
    smem_hist[i] = GradientSumT{};
  }
  __syncthreads();
  
  // Accumulate into shared memory
  for (auto row : rows_for_this_block) {
    auto grad = gpair[row];
    auto bin = matrix.GetBin(row, feature);
    atomicAdd(&smem_hist[bin - group_start_bin], grad);
  }
  __syncthreads();
  
  // Write shared memory to global histogram
  for (int i = threadIdx.x; i < n_bins_group; i += blockDim.x) {
    atomicAdd(&histogram[group_start_bin + i], smem_hist[i]);
  }
}
```

### 4. Row Partitioner

GPU row partitioning uses parallel scan for efficient partitioning:

```cpp
class RowPartitioner {
  // Row indices stored on device
  dh::device_vector<RowIndexT> ridx_;
  dh::device_vector<RowIndexT> ridx_tmp_;
  
  // Segment info per node
  std::vector<Segment> segments_;
  
  void UpdatePositionBatch(
      Context const* ctx,
      std::vector<bst_node_t> const& nidx,
      std::vector<bst_node_t> const& left_nidx,
      std::vector<bst_node_t> const& right_nidx,
      std::vector<NodeSplitData> const& split_data,
      GoLeftOp op
  );
};
```

**Algorithm**:
1. Compute `go_left` flag for each row
2. Parallel prefix sum on flags
3. Scatter rows to new positions based on prefix sum

```text
Input:     [A B C D E F]
go_left:   [1 0 1 1 0 1]
scan:      [1 1 2 3 3 4]
scatter:   [A C D F] [B E]  (left | right)
```

### 5. Split Evaluation

GPU split evaluation parallelizes across features:

```cpp
void EvaluateSplits(
    Context const* ctx,
    std::vector<int> const& nidx,
    bst_feature_t max_active_features,
    common::Span<EvaluateSplitInputs> inputs,
    EvaluateSplitSharedInputs shared_inputs,
    common::Span<GPUExpandEntry> out_entries
) {
  // One thread block per node
  // Threads within block evaluate different features
  // Reduction to find best split
}
```

### 6. Histogram Subtraction

GPU subtraction is element-wise parallel:

```cpp
bool SubtractionTrick(Context const* ctx, 
                      bst_node_t nidx_parent,
                      bst_node_t nidx_histogram, 
                      bst_node_t nidx_subtraction) {
  auto d_parent = GetNodeHistogram(nidx_parent);
  auto d_built = GetNodeHistogram(nidx_histogram);
  auto d_sibling = GetNodeHistogram(nidx_subtraction);
  
  dh::LaunchN(d_parent.size(), ctx->CUDACtx()->Stream(),
    [=] __device__(size_t idx) {
      d_sibling[idx] = d_parent[idx] - d_built[idx];
    });
  return true;
}
```

## Memory Management

### Histogram Storage

```cpp
class DeviceHistogramStorage {
  // Main cache for histograms
  dh::device_vector<int64_t> data_;
  
  // Overflow for when cache is full
  dh::device_vector<int64_t> overflow_;
  
  // Node ID -> histogram offset
  std::map<int, size_t> nidx_map_;
  
  // Maximum cached histogram memory
  size_t stop_growing_size_;
};
```

**Strategy**:
- Cache histograms up to `stop_growing_size_`
- Use overflow buffer for additional nodes
- Reuse histogram memory when possible

### External Memory Support

GPU training supports external memory (data larger than GPU memory):

```cpp
for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, batch)) {
  // Process one page at a time
  this->BuildHist(page, k, nidx);
}
```

Pages are streamed from disk and processed sequentially.

## Key Differences from CPU

| Aspect | CPU | GPU |
|--------|-----|-----|
| Data format | CSR/Dense | ELLPACK |
| Gradients | float32 | int64 (quantized) |
| Histogram build | Row-wise loop | Shared memory kernel |
| Atomics | Not needed (thread-local) | Required (shared/global) |
| Row partition | Block-based + prefix sum | Parallel scan |
| Memory | System RAM | GPU VRAM (limited) |

## Performance Considerations

### When GPU is Faster

- Large datasets (100K+ rows)
- Many features (100+)
- Deep trees (high max_depth)
- Many boosting rounds

### When CPU May Be Better

- Small datasets
- Very sparse data
- Limited GPU memory
- Multi-socket CPU systems

### Memory Constraints

GPU memory is typically 8-80GB vs 100GB+ system RAM:

```text
Memory per sample ≈ n_features × 1 byte (quantized) + 8 bytes (gradient)
For 1M samples, 100 features:
  Quantized: 100 MB
  Gradients: 8 MB
  Histograms: ~10 MB per node batch
```

## Multi-GPU Support

XGBoost supports distributed GPU training:

1. **Data-parallel**: Each GPU holds subset of rows
2. **AllReduce**: Synchronize histograms across GPUs
3. **Row split**: Partition rows across GPUs
4. **Column split**: Partition features across GPUs (for very wide data)

## Relevance for booste-rs

### What We Can Learn

1. **Quantized gradients**: Int64 enables efficient atomic operations
2. **Feature grouping**: Fit histogram in shared memory
3. **Parallel scan**: Efficient row partitioning
4. **ELLPACK format**: Better memory access patterns

### What's Not Applicable

1. **CUDA specifics**: We'd need different GPU backend (wgpu, metal, vulkan)
2. **Shared memory tricks**: GPU-specific optimization
3. **Int64 atomics**: Rust atomics work differently

### Future GPU Support for booste-rs

If we add GPU support later, consider:

1. **wgpu**: Cross-platform GPU compute (WebGPU)
2. **rust-cuda**: CUDA bindings for Rust (NVIDIA only)
3. **metal-rs**: Apple Metal (macOS/iOS)

For now, focus on CPU with good vectorization (SIMD).

## Source Code References

| Component | XGBoost Source |
|-----------|----------------|
| GPU tree updater | `src/tree/updater_gpu_hist.cu` |
| GPU histogram builder | `src/tree/gpu_hist/histogram.cuh` |
| GPU row partitioner | `src/tree/gpu_hist/row_partitioner.cuh` |
| GPU split evaluator | `src/tree/gpu_hist/evaluate_splits.cuh` |
| Feature groups | `src/tree/gpu_hist/feature_groups.cuh` |
| Gradient quantizer | `src/tree/gpu_hist/quantiser.cuh` |
| ELLPACK page | `src/data/ellpack_page.cuh` |
