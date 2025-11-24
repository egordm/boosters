# XGBoost C++ Data Structures — Relationships & Optimizations

This file provides a compact hierarchy/diagram of the key XGBoost C++ data structures, what they represent, and which inference/training optimizations they enable. It includes mapping to suggested Rust names and types.

Legend:
- `->`: "contains" or "references".
- `=>`: supports or enables.
- [CPU] / [GPU]: where the optimization or structure is primarily used.


## High-level diagram (ASCII)

GBTreeModel (model)
├─ trees[] -> RegTree (per-tree, AoS)
│  ├─ nodes[] -> Node (AoS struct: parent, cleft, cright, sindex, info)
│  ├─ TreeParam (size_leaf_vector, group mapping)
│  └─ is_vector_leaf? => Multi-target support
├─ base_score / meta
└─ feature_meta -> Quantile/Category cuts

DMatrix (data)
├─ SparsePage (CSR) => general CPU flow; sparse traversal
├─ QuantileDMatrix -> GHistIndexMatrix => quantized bins for histogram training
└─ EllpackPage (dense binned) [GPU] => high-throughput GPU inference/training

Predictor Views & Ephemeral Layouts
├─ HostModel / ScalarTreeView / MultiTargetTreeView => read-only views over RegTree
├─ DeviceModel (Device mem copy) [GPU] => persistent device-side model
├─ ArrayTreeLayout (transient) [CPU] => unrolled top-k levels (SoA-like hot path)
├─ ThreadTmp / FVec => per-thread staging/feature buffers for block traversal
└─ Bitvectors (Column-split) => distributed / column-split inference (two-pass)

CPU Inference Flow (simplified)
Input DMatrix -> ShouldUseBlock ?
  - per-row traversal: traverse RegTree via views -> accumulate
  - block traversal: stage rows into FVec (ThreadTmp) -> ProcessArrayTree -> traverse in array layout -> accumulate

GPU Inference Flow (simplified)
Input EllpackPage / SparsePage -> loaders (EllpackLoader / SparsePageLoader) -> shared mem staging -> kernel (PredictKernel / PredictLeafKernel)
- DeviceModel used to avoid repeated host->device copies
- Predict by bitvector for column-split cases


## Component Annotations & Optimizations

- `RegTree::Node` (AoS)
  - Role: canonical node layout for both training and inference.
  - Optimization: compact memory footprint; layout-friendly for traversal when contiguous.
  - Limitations: AoS might not be ideal for SIMD; pointer chasing increases cache misses.

- `HostModel / ScalarTreeView / MultiTargetTreeView`
  - Role: lightweight read-only views used by predictors to avoid copying the full model.
  - Optimization: zero-copy read-only access and minimal preparation for prediction.

- `ArrayTreeLayout` / `ProcessArrayTree` [CPU]
  - Role: transient array layout that unrolls the top `k` levels of trees.
  - Optimization: improves branch predictability, cache locality, fewer pointer dereferences.
  - Use case: `PredictBatchByBlockKernel` in `cpu_predictor.cc` (block traversal).

- `ThreadTmp` + `FVec`
  - Role: per-thread buffers that stage feature values for a block of rows.
  - Optimization: avoid repeated allocations; keep per-sample computation reusing CPU cache lines.

- `GHistIndexMatrix` / `QuantileDMatrix` (quantized bins)
  - Role: quantized representation mapping floats -> bins for histogram-based training and fast comparisons.
  - Optimization: smaller memory, faster integer-based comparisons, better cache utilization.
  - Use case: training, and CPU inference when quantized DMatrix is available.

- `EllpackPage` (GPU)
  - Role: dense padded binned representation, designed for GPU kernels.
  - Optimization: predictable memory access, compact representation, coalesced reads.
  - Use case: GPU training and inference; kernel-friendly indexing.

- `DeviceModel` / Device-side tree variants [GPU]
  - Role: persistent device copy of the model for kernel use.
  - Optimization: avoids host->device copies per prediction; enables efficient device traversal.

- `PredictKernel` and `Block` processing (GPU)
  - Role: kernels that assume either thread-per-row (wide parallelism) or block-level staging (shared mem).
  - Optimization: maximize occupancy, shared-memory reuse, vectorized/hot loops.

- `Bitvectors` (column-split distributed inference)
  - Role: compact encode per-row decisions per tree-level, enabling two-pass allreduce across split columns.
  - Optimization: reduces per-worker communication by shipping bitmasks rather than full feature vectors.

## Possible Mapping to Rust (Naming & Types)
- `RegTree` (AoS) -> `TrainTree` / `TrainForest` (for training types; `Vec<Node>`)
- `ArrayTreeLayout` (AoS->SoA hot path) -> `SoAForest` / `SoATree` (persistent SoA storage)
- `HostModel` / `DeviceModel` -> `ForestView` / `DeviceForest` with `Arc<[T]>`/`DeviceBuf<T>` ownership
- `ThreadTmp` -> `ThreadLocal<Buffer>` pool for `BlockVisitor`
- `FVec` -> `FeatureBlock` / `QuantizedBlock` typed arrays
- `EllpackPage` -> `PackedGpuForest<DeviceBuf<u32>>` (or `EllpackBuffer` type)
- `Bitvector` -> `BitVec` / `Bitmap` crates or compact `u8/packed` arrays
- `ScalarTreeView` / `MultiTargetTreeView` -> Rust `Visitor` trait generic over `LeafKind` (scalar vs vector)


## Use Case Matrix (quick reference)

- Fast single-row latency: per-row traversal + persistent `DeviceModel` for GPU or `SoAForest` optimized CPU sequential path.
- Batch throughput: block traversal (`PredictBatchByBlockKernel`), `ArrayTreeLayout`, `ThreadTmp` reuse.
- Memory-constrained: quantized `GHistIndexMatrix` / `Ellpack`.
- Distributed column-split: bitvector two-pass (minimal network traffic, reduced read-set).
- Multi-target: vector leaves (`size_leaf_vector`) and `MultiTargetTreeView` or vectorized `LeafVecForest` in Rust.


## Notes / Caveats
- XGBoost favors reuse of `RegTree` across training/inference with ephemeral layout transforms; the Rust design chooses a tradeoff: convert-on-load to persistent SoA/packed layouts for inference performance at the cost of conversion time and memory.
- The Rust `SoAForest` / `PackedForest` approach can simplify visitor code and reduce per-block transformations.
- GPU implementation choices depend on the target backend (CUDA vs Vulkan/wgpu), but the high-level concepts (Ellpack, device model, shared staging) are portable.