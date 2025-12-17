# Lessons Learned from XGBoost Research

## Overview

Key takeaways from studying XGBoost's implementation that inform booste-rs design.

## Algorithmic Lessons

### 1. Histogram-based Training is Essential

XGBoost's histogram approach provides:

- O(bins) split finding instead of O(n) per feature
- Better cache efficiency (small working set)
- Natural sparsity handling (missing values default to bin 0 or 255)

**booste-rs decision**: Implement histogram-based training from the start.

### 2. Quantile Sketching Enables Scalability

For large datasets, exact quantiles require O(n log n) sort per feature. XGBoost's
streaming sketch provides:

- O(1/ε²) memory regardless of dataset size
- Single-pass quantile computation
- Distributed-friendly (mergeable sketches)

**booste-rs decision**: Start with exact quantiles for correctness, add sketching later.

### 3. Histogram Subtraction Halves Work

The insight that `child = parent - sibling` reduces histogram builds:

- From 2^d builds at depth d
- To ~1.5 × 2^(d-1) builds (build smaller child, derive larger)

**booste-rs decision**: Implement subtraction trick from the start.

### 4. Block-based Partitioning Improves Cache Usage

XGBoost's block-based row partitioning:

- Processes rows in 2048-row blocks
- Computes block counts first (parallel prefix sum)
- Writes to final positions in parallel

**booste-rs decision**: Use block-based partitioning with Rayon.

## Implementation Lessons

### 5. Separate Training and Inference Representations

XGBoost uses different tree formats for training vs inference:

- Training: mutable, parent-child pointers, split info
- Inference: immutable, array-based, compact

**booste-rs decision**: Build mutable tree during training, convert to SoA for inference.

### 6. Gradient Precision Matters

XGBoost uses float32 for gradients by default:

- 2x memory savings over float64
- Sufficient precision for most tasks
- Can overflow with very large datasets (use float64 then)

**booste-rs decision**: Use f32 by default, support f64 via feature flag.

### 7. Bin Index Size is Configurable

XGBoost adapts bin index type to max_bins:

- max_bins ≤ 256: uint8_t
- max_bins ≤ 65536: uint16_t
- Otherwise: uint32_t

**booste-rs decision**: Use u8 for default 256 bins, support larger via enum/generics.

### 8. Missing Values Need Learned Direction

XGBoost learns `default_left` for each split:

- Try both directions during split evaluation
- Choose direction with higher gain
- Store direction for inference

**booste-rs decision**: Support missing values with learned direction.

## Architecture Lessons

### 9. Decouple Components

XGBoost's modular design:

- `Learner` orchestrates training
- `GBTree` manages ensemble
- `TreeUpdater` builds individual trees
- `DMatrix` holds data

**booste-rs decision**: Similar layering for testability and flexibility.

### 10. Objective is Pluggable

XGBoost's objective system:

- Defines GetGradient(predictions, labels) → gradients
- Defines PredTransform for raw → probability conversion
- Allows custom objectives

**booste-rs decision**: Define `Objective` trait with gradient and transform methods.

### 11. Early Stopping Requires Validation

XGBoost's early stopping:

- Requires validation set
- Tracks best iteration
- Stops when no improvement for N rounds

**booste-rs decision**: Support optional validation set with early stopping.

## Performance Lessons

### 12. Parallelism at Multiple Levels

XGBoost parallelizes:

- Features: split finding (independent per feature)
- Rows: histogram building (partition + reduce)
- Nodes: depth-wise processes all nodes at same level
- Trees: can build trees in parallel (independent)

**booste-rs decision**: Use Rayon for all parallelism, profile to find bottlenecks.

### 13. Memory Reuse Matters

XGBoost reuses memory:

- Histogram ring buffer (only keep 2 levels)
- Row index buffers (swap between iterations)
- Thread-local histograms (reuse across trees)

**booste-rs decision**: Design for memory reuse, avoid allocations in hot loops.

### 14. SIMD is Nice-to-Have

XGBoost's SIMD optimizations:

- Histogram building with AVX2
- Prediction with vectorized tree traversal

But: Most gains come from algorithmic optimizations (histogram, subtraction).

**booste-rs decision**: Focus on algorithms first, add SIMD later if needed.

## API Lessons

### 15. Builder Pattern for Configuration

XGBoost's many parameters benefit from:

- Sensible defaults (max_depth=6, learning_rate=0.3)
- Named parameters for clarity
- Validation of parameter combinations

**booste-rs decision**: Builder pattern with defaults and validation.

### 16. Progress Reporting is Useful

XGBoost's progress:

- Prints iteration and evaluation metrics
- Supports custom callbacks
- Silent mode for embedding

**booste-rs decision**: Optional progress callback, default silent.

## Summary Table

| Area | XGBoost Approach | booste-rs Decision |
|------|------------------|-------------------|
| Split finding | Histogram-based | Histogram-based |
| Quantization | Streaming sketch | Exact, then sketch |
| Histogram subtraction | Yes | Yes |
| Row partitioning | Block-based | Block-based |
| Tree format | Separate train/infer | Separate |
| Gradient precision | f32 default | f32 default |
| Bin index | u8/u16/u32 | u8 default |
| Missing values | Learned direction | Learned direction |
| Parallelism | OpenMP | Rayon |
| SIMD | Some paths | Later |
| API | Params dict | Builder pattern |

## What We Won't Copy

Some XGBoost features are out of scope or will be simplified:

1. **Distributed training** — Start single-machine only
2. **GPU support** — CPU-only initially
3. **External memory** — In-memory only initially
4. **Approximate tree method** — Exact histogram only
5. **Monotonic constraints** — Not in initial version
6. **Feature interaction constraints** — Not in initial version
7. **Custom splits** — Not in initial version

## Next Steps

With this research complete, the next phase is to:

1. Write RFC for GBTree training architecture
2. Break down into epics and stories
3. Implement quantization foundation
4. Build histogram infrastructure
5. Implement core training loop
6. Add optimizations incrementally
