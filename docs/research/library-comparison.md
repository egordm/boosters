# XGBoost vs LightGBM: Feature Comparison

This document compares XGBoost and LightGBM implementations across key features.
Use this as a reference when understanding trade-offs between different approaches.

---

## Feature Matrix

| Feature | XGBoost | LightGBM | Notes |
|---------|---------|----------|-------|
| **Tree Growth** | Depth-wise (default) | Leaf-wise | LightGBM more efficient, but needs depth limit |
| **Categorical Features** | Approximate (1.5+) | Native | LightGBM's O(k log k) is more efficient |
| **Gradient Quantization** | GPU only | CPU + GPU | LightGBM saves memory on CPU too |
| **Data Sampling** | Random subsample | GOSS | GOSS is gradient-aware |
| **Histogram Build** | Row-wise default | Adaptive col/row | LightGBM auto-selects strategy |
| **Feature Bundling** | ❌ | EFB | LightGBM bundles sparse features |
| **GPU Training** | ELLPACK | Custom | Both mature implementations |
| **Monotonic Constraints** | ✅ | ✅ | Both support |
| **Interaction Constraints** | ✅ | ✅ | Both support |
| **Missing Values** | Learned direction | Learned direction | Same approach |
| **Sparse Data** | Optimized | Optimized | Both efficient |
| **Documentation** | Excellent | Good | XGBoost more detailed |

---

## Tree Growth Strategy

### XGBoost: Depth-wise (Level-wise)

```
Level 0:    [root]
               │
Level 1:  [L1]   [L2]    ← Split ALL nodes at level
             │      │
Level 2: [a] [b] [c] [d]  ← Split ALL nodes at level
```

- Produces balanced trees
- Can parallelize entire level
- Better for small datasets (less overfitting risk)
- Default for `tree_method=hist`

### LightGBM: Leaf-wise (Best-first)

```
[root] → [best leaf] → [best leaf] → ...
```

- Always splits the leaf with highest gain
- Lower loss with same number of leaves
- More efficient (less wasted work on low-gain nodes)
- Risk of overfitting on small data (mitigate with `max_depth`)

### When to Use Each

| Situation | Recommended |
|-----------|-------------|
| Large dataset (>100k rows) | Leaf-wise |
| Small dataset (<10k rows) | Depth-wise |
| Deep trees needed | Leaf-wise |
| Regularization priority | Depth-wise |
| Training speed priority | Leaf-wise |

---

## Categorical Feature Handling

### XGBoost (v1.5+)

- Approximate categorical support via `enable_categorical=True`
- Uses partition-based splits
- Requires category codes as integers

### LightGBM

Native support with two strategies:

1. **One-hot strategy** (low cardinality, ≤4 categories)
   - Try each category vs all others: O(k)
   
2. **Gradient-sorted strategy** (high cardinality)
   - Sort categories by gradient statistics
   - Cumulative split search: O(k log k)

```python
# LightGBM: just declare categorical columns
params = {'categorical_feature': [0, 3, 7]}
```

**Advantage**: LightGBM finds optimal binary partitions, not just one-vs-rest.

---

## GOSS: Gradient-based One-Side Sampling

LightGBM's GOSS keeps instances with large gradients and randomly samples small-gradient instances:

```
Algorithm:
1. Sort instances by |gradient|
2. Keep top a% (e.g., 20%)
3. Random sample b% from remaining (e.g., 10%)
4. Amplify sampled gradients by (1-a)/b to correct bias
```

**Why it works**: Large gradients indicate poorly-fit instances that need more attention.

**Parameters**:
- `top_rate`: Fraction to keep (default: 0.2)
- `other_rate`: Fraction to sample (default: 0.1)

**XGBoost Alternative**: Random subsampling (`subsample` parameter) - simpler but less efficient.

---

## Histogram Building

### XGBoost

- Row-wise by default
- GPU uses ELLPACK format
- Gradient quantization GPU-only

### LightGBM

- **Adaptive strategy**: Auto-selects row-wise vs column-wise based on cache analysis
- CPU gradient quantization (16/32-bit packed histograms)
- Feature groups via EFB

### Histogram Subtraction

Both libraries use the subtraction trick:

```
child_histogram = parent_histogram - sibling_histogram
```

This halves histogram building work by only computing for the smaller child.

---

## Gradient Quantization

### XGBoost

GPU-only quantization:
```cpp
int32_t gradient : 16;
int32_t hessian  : 16;
```

### LightGBM

CPU + GPU quantization with adaptive precision:
```cpp
// Small leaves: 16-bit (grad:8 + hess:8)
// Large leaves: 32-bit (grad:16 + hess:16)
```

**Benefits**:
- 2-4x memory reduction
- Better cache utilization
- Minimal accuracy impact

---

## Exclusive Feature Bundling (EFB)

LightGBM-only optimization that bundles mutually exclusive features:

```
If feature A ≠ 0 implies feature B = 0:
  → Bundle them into single feature
  → Reduces histogram memory and computation
```

Useful for:
- One-hot encoded features
- Sparse feature matrices
- High-dimensional data

---

## Model Formats

### XGBoost

- **JSON**: Primary format, fully specified schema
- **Binary**: Legacy format (deprecated)
- **UBJSON**: Compact binary JSON

### LightGBM

- **Text**: Human-readable format
- **Binary**: Faster loading

### Compatibility

| Format | XGBoost | LightGBM |
|--------|---------|----------|
| XGBoost JSON | ✅ Native | ❌ |
| LightGBM Text | ❌ | ✅ Native |
| ONNX | Via converter | Via converter |
| Treelite | ✅ | ✅ |

---

## API Comparison

### XGBoost (Python)

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
}
model = xgb.train(params, dtrain, num_boost_round=100)
```

### LightGBM (Python)

```python
import lightgbm as lgb

dtrain = lgb.Dataset(X_train, label=y_train)
params = {
    'num_leaves': 31,
    'learning_rate': 0.1,
    'objective': 'regression',
}
model = lgb.train(params, dtrain, num_boost_round=100)
```

### Key Differences

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Tree size control | `max_depth` | `num_leaves` (+ optional `max_depth`) |
| Default growth | Depth-wise | Leaf-wise |
| Data container | `DMatrix` | `Dataset` |
| Categorical syntax | `enable_categorical` | `categorical_feature` |

---

## Performance Characteristics

### Training Speed

| Dataset Size | XGBoost | LightGBM | Winner |
|--------------|---------|----------|--------|
| Small (<10k) | Fast | Fast | Tie |
| Medium (10k-100k) | Good | Better | LightGBM |
| Large (100k-1M) | Good | Better | LightGBM |
| Very Large (>1M) | Good | Better | LightGBM (especially with GOSS) |

### Memory Usage

| Configuration | XGBoost | LightGBM |
|---------------|---------|----------|
| Dense data | Similar | Similar |
| Sparse data | Good | Better (EFB) |
| Many features | Higher | Lower (bundling) |
| Gradient storage | f32/f64 | Quantized (int8/16) |

---

## Distributed Training

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Data parallel | ✅ | ✅ |
| Feature parallel | ✅ | ✅ |
| Voting parallel | ❌ | ✅ |
| Communication | Rabit/Dask/Spark | MPI/Dask/Spark |

---

## Summary: When to Use Which

### Choose XGBoost when:
- You need excellent documentation
- Compatibility with existing XGBoost models
- Working with small datasets
- You prefer depth-wise tree growth

### Choose LightGBM when:
- Training speed is critical
- Working with large datasets (>100k rows)
- You have many categorical features
- Memory efficiency matters
- Using GOSS for very large datasets

### What We Take from Each

**From XGBoost**:
- JSON model format compatibility
- Clear documentation style
- Monotonic constraint implementation

**From LightGBM**:
- Leaf-wise growth (default)
- Native categorical handling
- GOSS sampling
- Gradient quantization on CPU
