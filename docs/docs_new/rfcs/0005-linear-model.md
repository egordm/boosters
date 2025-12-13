# RFC-0005: Linear Model

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: RFC-0001 (Data Matrix)
- **Scope**: Linear model structure, storage, and prediction

## Summary

The LinearModel stores feature weights and biases for gradient boosted linear models. This RFC defines the weight layout, prediction algorithms, and the design tradeoffs between training and inference efficiency. The same model struct serves both training (where weights are updated) and inference (where predictions are computed).

## Overview

### Model Structure

```text
LinearModel
├── weights: Box<[f32]>    ← All weights + biases in output-major layout
├── n_features: usize
└── n_outputs: usize

Memory Layout (output-major):
┌─────────────────────────────────────────────────────────────────┐
│  w[0,0] w[1,0] w[2,0] ... w[F-1,0] bias[0]  │  output 0         │
│  w[0,1] w[1,1] w[2,1] ... w[F-1,1] bias[1]  │  output 1         │
│  ...                                         │  ...              │
│  w[0,K-1] ... w[F-1,K-1] bias[K-1]          │  output K-1       │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```text
Training (RFC-0004):                    Inference:
┌─────────────────────┐                ┌─────────────────────┐
│ GBLinearTrainer     │                │ User Code           │
│  └► model.add_weight│                │  └► model.predict() │
│  └► model.add_bias  │                └─────────────────────┘
└─────────────────────┘                         │
         │                                      ▼
         ▼                             ┌─────────────────────┐
┌─────────────────────┐                │ features: ColMatrix │
│   LinearModel       │◄───────────────│ (n_samples × n_feat)│
└─────────────────────┘                └─────────────────────┘
         │                                      │
         └──────────────────────────────────────┘
                          │
                          ▼
                 predictions: ColMatrix
                 (n_samples × n_outputs)
```

## Components

### LinearModel

```rust
/// Linear model with weights and bias for each output.
///
/// Weight layout: output-major (weights for one output are contiguous).
pub struct LinearModel {
    /// All weights and biases.
    /// Layout: [output_0_weights..., output_0_bias, output_1_weights..., ...]
    weights: Box<[f32]>,
    n_features: usize,
    n_outputs: usize,
}
```

### Weight Layout Diagram

For `n_features = 3` and `n_outputs = 2`:

```text
Index:    0     1     2     3     4     5     6     7
         ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
         │w0,0 │w1,0 │w2,0 │b0   │w0,1 │w1,1 │w2,1 │b1   │
         └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
         │◄── output 0 (stride = 4)──►│◄── output 1 ──►│

Indexing:
  weight(f, k) = weights[k * (n_features + 1) + f]
  bias(k)      = weights[k * (n_features + 1) + n_features]
  stride       = n_features + 1
```

### Core Interface

```rust
impl LinearModel {
    // Construction
    pub fn new(weights: Box<[f32]>, n_features: usize, n_outputs: usize) -> Self;
    pub fn zeros(n_features: usize, n_outputs: usize) -> Self;
    
    // Dimensions
    pub fn n_features(&self) -> usize;
    pub fn n_outputs(&self) -> usize;
    
    // Read access
    pub fn weight(&self, feature: usize, output: usize) -> f32;
    pub fn bias(&self, output: usize) -> f32;
    
    /// All weights + bias for one output (contiguous slice).
    pub fn output_weights(&self, output: usize) -> &[f32];  // length = n_features + 1
    
    /// Raw storage access.
    pub fn weights_slice(&self) -> &[f32];
    
    // Write access (for training)
    pub fn set_weight(&mut self, feature: usize, output: usize, value: f32);
    pub fn set_bias(&mut self, output: usize, value: f32);
    pub fn add_weight(&mut self, feature: usize, output: usize, delta: f32);
    pub fn add_bias(&mut self, output: usize, delta: f32);
}
```

### Prediction Interface

```rust
impl LinearModel {
    /// Predict for a single sample.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32>;
    pub fn predict_row_into(&self, features: &[f32], output: &mut [f32]);
    
    /// Predict for batch (column-major input and output).
    pub fn predict_batch<S: AsRef<[f32]>>(&self, data: &ColMatrix<f32, S>) -> ColMatrix<f32>;
    pub fn predict_into<S: AsRef<[f32]>>(&self, data: &ColMatrix<f32, S>, output: &mut ColMatrix<f32>);
    
    /// Parallel batch prediction.
    pub fn par_predict_into<S: AsRef<[f32]> + Sync>(&self, data: &ColMatrix<f32, S>, output: &mut ColMatrix<f32>);
}
```

## Algorithms

### Single-Row Prediction

```text
predict_row_into(features, output):
  for k in 0..n_outputs:
    sum = bias(k)
    for f in 0..n_features:
      sum += features[f] * weight(f, k)
    output[k] = sum
```

### Batch Prediction (Column-Wise)

With column-major input and output, use **column-wise accumulation**:

```text
predict_into(data, output):
  n_samples = data.n_rows()
  
  // 1. Initialize output with bias
  for k in 0..n_outputs:
    output_col = output.col_slice_mut(k)
    bias_k = bias(k)
    for row in 0..n_samples:
      output_col[row] = bias_k
  
  // 2. Accumulate feature contributions
  for f in 0..n_features:
    feature_col = data.col_slice(f)          // contiguous
    for k in 0..n_outputs:
      w = weight(f, k)
      output_col = output.col_slice_mut(k)   // contiguous
      for row in 0..n_samples:
        output_col[row] += feature_col[row] * w
```

**Why column-wise?**

| Approach | Memory Access Pattern | Cache Behavior |
|----------|----------------------|----------------|
| Row-wise | Strided reads across features | Cache misses |
| Column-wise | Contiguous reads per column | Cache friendly |

Column-wise accumulation reads each feature column once (contiguous), writes each output column once (contiguous). Inner loop is SIMD-friendly.

### Parallel Batch Prediction

Partition **features** across threads (columns are independent):

```text
par_predict_into(data, output):
  // Initialize bias (parallel by output column)
  output.cols_par().for_each(|(k, col)| col.fill(bias(k)))
  
  // Feature loop in parallel chunks
  features.par_chunks().for_each(|chunk| {
    for f in chunk:
      feature_col = data.col_slice(f)
      for k in 0..n_outputs:
        // Needs atomic add or thread-local accumulation
        ...
  })
```

Alternative: partition by **samples** (rows). Each thread owns a row range, no synchronization needed:

```text
par_predict_into(data, output):
  (0..n_samples).into_par_iter().chunks(CHUNK_SIZE).for_each(|rows| {
    for row in rows:
      for k in 0..n_outputs:
        sum = bias(k)
        for f in 0..n_features:
          sum += data[f][row] * weight(f, k)
        output[k][row] = sum
  })
```

Sample-parallel has strided access but no synchronization. Profile to determine which is faster.

## Design Decisions

### DD-1: Output-Major Weight Layout

**Context**: How to lay out weights in memory?

**Alternatives**:

1. **Output-major**: `[w0_all, bias0, w1_all, bias1, ...]` - one output's weights contiguous
2. **Feature-major**: `[w_f0_all_outputs, w_f1_all_outputs, ...]` - one feature's weights across outputs contiguous

**Decision**: Output-major layout.

**Rationale**:

- **Prediction**: `output_weights(k)` returns contiguous slice for one output
- **Serialization**: Single contiguous array per output
- **Training access**: Coordinate descent updates `(feature, output)` pairs—either layout works equally well
- **Bias inline**: Bias at end of each output's block keeps it together

### DD-2: Bias Stored Inline

**Context**: Store bias separately or with weights?

**Decision**: Store bias at index `n_features` within each output's block.

**Rationale**:

```text
output_weights(k) = weights[k*(F+1) .. (k+1)*(F+1)]
                  = [w0, w1, ..., w_{F-1}, bias]
```

- **Single allocation**: No separate bias array
- **Clean serialization**: All parameters in one array
- **Prediction simplicity**: One slice contains everything for an output

### DD-3: Same Model for Training and Inference

**Context**: Should we have separate representations?

**Decision**: Single `LinearModel` struct for both.

**Rationale**:

| GBTree | GBLinear |
|--------|----------|
| Training: histogram-friendly | Training: weights array |
| Inference: traversal-optimized | Inference: same weights array |
| May benefit from conversion | No conversion needed |

Linear models are simple: `y = Wx + b`. The same array works efficiently for both updating weights (training) and computing predictions (inference).

### DD-4: predict_into Pattern

**Context**: Should prediction allocate or write to buffer?

**Decision**: Both APIs, prefer `predict_into` for performance.

**Rationale**:

- **Training hot path**: `predict_into` reuses buffer across rounds
- **Convenience**: `predict_batch` allocates for one-off inference
- **Zero allocation**: Training loop never allocates after setup
- **Consistent with RFC-0002**: Same pattern as gradient computation

### DD-5: No Separate Base Score

**Context**: Should base_score be stored separately?

**Decision**: No. Bias already includes base_score.

**Rationale**:

Training initialization (RFC-0004):
```text
base_score = objective.base_score_vec(labels, weights)
model.set_bias(k, base_score[k])
```

- **Self-contained model**: Predict correctly without external state
- **Simpler API**: No "don't forget to add base_score" errors
- **Clean serialization**: Model is complete

### DD-6: Column-Wise Prediction Algorithm

**Context**: How to implement batch prediction with column-major data?

**Decision**: Column-wise accumulation.

**Rationale**:

```text
// This code:
for f in 0..n_features:
  for k in 0..n_outputs:
    for row in 0..n_samples:
      output[k][row] += data[f][row] * weight(f, k)

// Has these access patterns:
data[f][row]    → sequential (col-major)
output[k][row]  → sequential (col-major)
weight(f, k)    → random but cached (small)
```

The inner loop reads/writes contiguous memory, enabling:

- **Cache efficiency**: Sequential access
- **SIMD**: Vectorized multiply-accumulate
- **Prefetching**: Predictable access pattern

Row-wise would stride across both input and output.

### DD-7: Column-Major Output

**Context**: What layout for prediction output?

**Decision**: `ColMatrix<f32>` (column-major).

**Rationale**:

- **Consistent with pipeline**: Features, labels, gradients all column-major
- **No conversion**: Output feeds directly into gradient computation
- **Training loop**: `objective.compute_gradients(predictions, ...)`

For pure inference with user-facing API, consider future `predict_rows_to_rows()` that outputs row-major.

## Integration

| Component | How Model is Used |
|-----------|-------------------|
| RFC-0004 (Training) | `add_weight()`, `add_bias()`, `predict_into()` |
| RFC-0006 (Evaluation) | `predict_into()` for eval set predictions |
| Serialization | `weights_slice()` + dimensions |

### Training Integration

```text
// Per round in GBLinearTrainer:
for output_idx in 0..n_outputs:
  // Coordinate descent updates
  for feature in selector:
    delta = compute_delta(...)
    model.add_weight(feature, output_idx, delta)  // ◄─ Model mutation

// After all outputs updated:
model.predict_into(features, &mut predictions)    // ◄─ Model used for prediction
```

## Memory Usage

| n_features | n_outputs | Model Size |
|------------|-----------|------------|
| 100 | 1 | 404 bytes |
| 100 | 10 | 4,040 bytes |
| 1,000 | 1 | 4,004 bytes |
| 1,000 | 10 | 40,040 bytes |
| 10,000 | 100 | 4 MB |

Formula: `(n_features + 1) × n_outputs × 4 bytes`

## Future Work

- [ ] SIMD-optimized prediction kernels
- [ ] Quantized weights (int8) for compressed models
- [ ] Sparse weight representation (many zeros after L1)
- [ ] Row-major output for inference-only use cases

## References

- [XGBoost Linear Booster](https://xgboost.readthedocs.io/en/latest/tutorials/linear.html)
