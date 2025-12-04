# XGBoost vs LightGBM: Comparative Analysis

## Executive Summary

This document compares XGBoost and LightGBM implementations to inform design decisions
for booste-rs. **Goal**: Take the best from both libraries.

## Feature Comparison Matrix

| Feature | XGBoost | LightGBM | Winner | Notes |
|---------|---------|----------|--------|-------|
| **Tree Growth** | Depth-wise | Leaf-wise | LightGBM | More efficient, but needs depth limit |
| **Categorical Features** | ❌ Manual encoding | ✅ Native | LightGBM | Huge advantage |
| **Gradient Quantization** | GPU only | CPU + GPU | LightGBM | Memory + speed win |
| **Data Sampling** | Random subsample | GOSS | LightGBM | Gradient-aware sampling |
| **Histogram Build** | Row-wise | Adaptive | LightGBM | Cache-aware selection |
| **GPU Training** | ✅ Mature | ✅ Good | XGBoost | ELLPACK is well-optimized |
| **Monotonic Constraints** | ✅ Full | ✅ Full | Tie | Both support |
| **Interaction Constraints** | ✅ Full | ✅ Full | Tie | Both support |
| **Missing Values** | ✅ Learned | ✅ Learned | Tie | Both learn default direction |
| **Sparse Data** | ✅ Optimized | ✅ Optimized | Tie | Both efficient |
| **Feature Groups** | ❌ | ✅ EFB | LightGBM | Memory optimization |
| **Documentation** | Excellent | Good | XGBoost | More detailed docs |
| **Code Quality** | Good | Complex | XGBoost | Easier to follow |

## Tree Growth Strategy

### XGBoost: Depth-wise

```text
Level 0:    [root]
               │
Level 1:  [L1]  [L2]    ← Split ALL at level
             │      │
Level 2: [a][b] [c][d]  ← Split ALL at level
```

- Balanced trees
- Level-parallel possible
- Better for small datasets

### LightGBM: Leaf-wise

```text
[root] → [best leaf] → [best leaf] → ...
```

- Lower loss with same #leaves
- More efficient
- Risk of overfitting small data

### booste-rs: Tree Growth

**Implement both strategies** with leaf-wise as default:

```rust
enum TreeGrowthStrategy {
    LeafWise,   // Default (LightGBM style)
    DepthWise,  // XGBoost style
}
```

## Categorical Feature Handling

### XGBoost: No Native Support

No native support. Users must:

1. One-hot encode (memory explosion)
2. Label encode (loses category relationships)
3. Target encode (leakage risk)

### LightGBM: Native Categorical

Native support with gradient-based sorting:

```python
# LightGBM: just declare categorical
params = {'categorical_feature': [0, 3, 7]}
# Finds optimal binary split in O(k log k)
```

### booste-rs: Categorical Features

**Must implement LightGBM's categorical handling**:

1. One-hot strategy for low cardinality (≤4)
2. Gradient-sorted strategy for high cardinality
3. Store category sets in split nodes

```rust
enum SplitCondition {
    Numerical { threshold: f32, default_left: bool },
    Categorical { categories_left: Vec<u32> },
}
```

## Data Sampling

### XGBoost: Random Subsample

Random subsampling:

```python
params = {
    'subsample': 0.8,        # Row sampling
    'colsample_bytree': 0.8  # Feature sampling
}
```

### LightGBM: GOSS

Gradient-based one-side sampling:

- Keep top 20% by gradient
- Random sample 10% from rest
- Rescale sampled gradients

### booste-rs: Sampling Strategy

**Implement both**:

1. Random sampling (simpler, good baseline)
2. GOSS (for large datasets)

```rust
enum SamplingStrategy {
    None,
    Random { subsample_rate: f32 },
    Goss { top_rate: f32, other_rate: f32 },
}
```

## Histogram Building

### XGBoost: Row-wise Default

- Row-wise by default
- GPU: ELLPACK format
- Gradient quantization GPU-only

### LightGBM: Adaptive Strategy

- Adaptive col-wise vs row-wise
- CPU gradient quantization
- Feature groups (EFB)

### booste-rs: Histogram Strategy

**Take LightGBM's approach**:

1. Adaptive strategy based on cache analysis
2. CPU gradient quantization support
3. Consider feature groups for memory

## Gradient Quantization

### XGBoost: GPU-only Quantization

```cpp
// GPU only
int32_t gradient : 16;
int32_t hessian  : 16;
```

### LightGBM: CPU + GPU Quantization

```cpp
// Adaptive bit width
if (leaf_size < threshold) {
    use_16bit();  // grad:8 + hess:8
} else {
    use_32bit();  // grad:16 + hess:16
}
```

### booste-rs: Quantization Strategy

**Implement CPU quantization** like LightGBM:

- 16-bit packed histograms for small leaves
- 32-bit for larger leaves
- Significant memory + cache benefits

## Missing Value Handling

Both libraries learn the best direction for missing values:

```text
If feature is missing:
  - XGBoost: stores default_left in node
  - LightGBM: stores default_left in node
```

### booste-rs: Missing Values

**Same approach** - learn and store default direction.

## Feature Groups (EFB)

### LightGBM: Exclusive Feature Bundling

Exclusive Feature Bundling groups mutually exclusive features:

```text
If feature A != 0 implies feature B == 0:
  Bundle them → Reduce histogram memory
```

### booste-rs: Feature Groups

**Defer to later** - nice optimization but not critical.

## GPU Training

### XGBoost: ELLPACK

- ELLPACK format
- Gradient quantization
- Mature implementation

### LightGBM: GPU Support

- Good GPU support
- Different memory layout

### booste-rs: GPU Strategy

**Start with CPU, add GPU later** following XGBoost's ELLPACK design.

## What to Copy from Each

### From XGBoost

1. **ELLPACK format** for GPU
2. **Clear documentation** style
3. **JSON model format** compatibility
4. **Monotonic constraint** implementation

### From LightGBM

1. **Leaf-wise growth** (default)
2. **Categorical feature** handling
3. **GOSS sampling**
4. **Gradient quantization** on CPU
5. **Adaptive histogram** strategy

## Implementation Priority

| Feature | Priority | Source |
|---------|----------|--------|
| Leaf-wise growth | P0 | LightGBM |
| Histogram subtraction | P0 | Both |
| Categorical features | P1 | LightGBM |
| Gradient quantization (CPU) | P1 | LightGBM |
| GOSS sampling | P2 | LightGBM |
| Depth-wise growth | P2 | XGBoost |
| GPU training | P3 | XGBoost |
| Feature groups (EFB) | P4 | LightGBM |

## API Design Implications

### Training Configuration

```rust
pub struct TreeConfig {
    // Growth strategy (LightGBM default)
    pub growth_strategy: TreeGrowthStrategy,
    pub num_leaves: u32,
    pub max_depth: Option<u32>,
    
    // Sampling (LightGBM's GOSS optional)
    pub sampling: SamplingStrategy,
    
    // Histogram (LightGBM's quantization)
    pub use_quantized_grad: bool,
    pub num_grad_quant_bins: u32,
}

pub struct DataConfig {
    // Categorical (LightGBM)
    pub categorical_features: Vec<usize>,
    pub max_cat_to_onehot: u32,
}
```

### Model Compatibility

Support both formats:

```rust
pub enum ModelFormat {
    XGBoostJson,   // For XGBoost compatibility
    LightGbmText,  // For LightGBM compatibility
    Native,        // Our own format
}
```

## Conclusion

booste-rs should primarily follow **LightGBM's design** for training, while maintaining
**XGBoost JSON compatibility** for inference. Key differentiators to implement:

1. ✅ Leaf-wise tree growth
2. ✅ Native categorical features
3. ✅ GOSS sampling
4. ✅ CPU gradient quantization
5. ✅ XGBoost model loading

This combination gives us the best training efficiency while maintaining ecosystem compatibility.
