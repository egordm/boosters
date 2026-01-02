# RFC-0008: GBDT Training

**Status**: Implemented  
**Created**: 2025-12-01  
**Updated**: 2026-01-02  
**Scope**: Gradient boosted decision tree training pipeline

## Summary

GBDT training combines objective-driven gradient computation, histogram-based
split finding, and tree growing into an iterative boosting loop. The design
separates orchestration (`GBDTTrainer`) from tree mechanics (`TreeGrower`).

## Why Histogram-Based Training?

Traditional exact split finding scans all samples for every feature at every
node—O(n × m × depth) per tree. Histogram-based training bins features upfront:

| Aspect | Exact | Histogram-Based |
| ------ | ----- | --------------- |
| Split candidates | All unique values | Up to 256 bins |
| Memory per node | Full sample refs | Histogram only |
| Parallelism | Limited | Easy: each feature independent |

XGBoost, LightGBM, and CatBoost all use histogram-based training.

## Layers

### High Level

Users call `GBDTModel::train`:

```rust
let model = GBDTModel::train(&dataset, Some(&eval_set), config, seed)?;
```

This constructs `GBDTTrainer` internally and runs the boosting loop.

### Quick Start

```rust
use boosters::{GBDTModel, GBDTConfig, Dataset};

let dataset = Dataset::from_array(features.view(), Some(targets.view()), None)?;
let config = GBDTConfig::default();  // 100 trees, lr=0.3, max_depth=6
let model = GBDTModel::train(&dataset, None, config, 42)?;
let preds = model.predict(&dataset, 4);  // 4 threads
```

### Medium Level (Trainer)

```rust
pub struct GBDTTrainer<O: ObjectiveFn, M: MetricFn> {
    objective: O,
    metric: M,
    params: GBDTParams,
}

impl<O, M> GBDTTrainer<O, M> {
    pub fn train<W, T>(
        &self,
        dataset: &BinnedDataset,
        targets: T,
        weights: W,
        eval_set: Option<(&Dataset, T)>,
        parallelism: Parallelism,
    ) -> Result<Forest<ScalarLeaf>, TrainingError>;
}
```

**Boosting loop (conceptual)**:
1. Compute gradients from objective: `objective.gradients(preds, targets, grads)`
2. Grow tree from gradients: `grower.grow(dataset, grads)`
3. Update predictions: `preds += learning_rate * tree_outputs`
4. Evaluate and check early stopping
5. Repeat for `n_trees` rounds

### Medium Level (Grower)

```rust
pub struct TreeGrower {
    params: GrowerParams,
    histogram_pool: HistogramPool,
    partitioner: RowPartitioner,
    tree_builder: MutableTree<ScalarLeaf>,
    histogram_builder: HistogramBuilder,
    // ... feature metadata, samplers
}

impl TreeGrower {
    pub fn grow(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &Gradients,
        parallelism: Parallelism,
    ) -> Tree<ScalarLeaf>;
}
```

### Low Level (Split Finding)

Split finding evaluates every feature at every bin boundary to find the
optimal split point.

#### Gain Formula

Split gain uses the XGBoost formula with regularization:

$$\text{gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_P^2}{H_P + \lambda} \right] - \gamma$$

Where:

- $G_L, G_R, G_P$ = gradient sums for left, right, parent
- $H_L, H_R, H_P$ = hessian sums for left, right, parent
- $\lambda$ = L2 regularization (`reg_lambda`)
- $\gamma$ = minimum gain threshold (`min_split_gain`)

**Optimization**: Parent score is precomputed once per node, reducing from
3 divisions to 2 per candidate split.

#### Leaf Weight Formula

$$w = -\frac{\text{sign}(G) \cdot \max(0, |G| - \alpha)}{H + \lambda}$$

Where $\alpha$ = L1 regularization (`reg_alpha`). When $\alpha = 0$, this
simplifies to the Newton step: $w = -G/(H + \lambda)$.

#### Splitter API

```rust
pub struct GreedySplitter {
    gain_params: GainParams,
    max_onehot_cats: u32,
    parallelism: Parallelism,
}

impl GreedySplitter {
    pub fn find_split(
        &self,
        histogram: HistogramView<'_>,
        parent_stats: GradsTuple,
        feature_indices: &[u32],
    ) -> Option<SplitInfo>;
}
```

Scan strategies:
- **Numerical**: Bidirectional scan for optimal missing value handling
- **Categorical one-hot**: Each category as singleton left partition
- **Categorical sorted**: Sort by grad/hess ratio, scan partition point

## Key Design Decisions

### DD-1: Subtraction Trick

When parent and one child histogram exist, compute sibling by subtraction:
`sibling = parent - child`. Reduces histogram builds by ~50%.

```text
    Parent (computed)
    ├── Left (computed)  ← Smaller child
    └── Right = Parent - Left  ← Subtraction
```

Always compute histogram for smaller child (fewer samples to aggregate).

### DD-2: Growth Strategies

```rust
pub enum GrowthStrategy {
    DepthWise { max_depth: u32 },   // XGBoost-style: level by level
    LeafWise { max_leaves: u32 },   // LightGBM-style: best-gain first
}
```

Both produce equivalent trees given same hyperparameters; leaf-wise often
converges faster but risks overfitting without early stopping.

### DD-3: Row Partitioning

Samples are partitioned into node-specific ranges as tree grows. Benefits:
- Gradient gathering is sequential (cache-friendly)
- Histogram building accesses contiguous memory
- Child counts known for subtraction trick

Partitioner uses double-buffer swap to avoid allocation per split.

### DD-4: Ordered Gradients

Before histogram building, gradients are gathered into contiguous buffers per
node, ordered by sample index within that node. This enables vectorized
histogram kernels.

### DD-5: LRU Histogram Cache

Large trees may exceed memory if all histograms are kept. `HistogramPool` uses
LRU eviction, keeping only recently used histograms for the subtraction trick.

### DD-6: Multi-Output via Tree Groups

For K-class classification, we train K trees per round (one per class).
Each tree sees class-specific gradients. Trees are grouped in the forest.
This matches XGBoost/LightGBM behavior.

## Objective and Metric Traits

```rust
pub trait ObjectiveFn: Send + Sync {
    fn n_outputs(&self) -> usize;
    fn init_predictions(&self, targets: &[f32], out: &mut [f32]);
    fn gradients(&self, preds: &[f32], targets: &[f32], grads: &mut [GradsTuple]);
}

pub trait MetricFn: Send + Sync {
    fn name(&self) -> &str;
    fn score(&self, preds: &[f32], targets: &[f32]) -> f64;
    fn higher_is_better(&self) -> bool;
}
```

Built-in objectives: `SquaredError`, `LogLoss`, `Softmax`.
Built-in metrics: `RMSE`, `MAE`, `LogLoss`, `AUC`, `Accuracy`.

## Sampling

### Row Sampling

```rust
pub enum RowSamplingParams {
    None,
    Uniform { subsample: f32 },
    GOSS { top_rate: f32, other_rate: f32 },  // Gradient-based
}
```

GOSS (Gradient-based One-Side Sampling) keeps all high-gradient samples and
subsamples low-gradient ones. From LightGBM, improves quality under sampling.

### Column Sampling

```rust
pub enum ColSamplingParams {
    None,
    ByTree { colsample: f32 },
    ByLevel { colsample: f32 },
    ByNode { colsample: f32 },
}
```

## Parameters

```rust
pub struct GBDTParams {
    pub n_trees: u32,              // Boosting rounds (default: 100)
    pub learning_rate: f32,        // Shrinkage (default: 0.3)
    pub growth_strategy: GrowthStrategy,
    pub gain: GainParams,          // Regularization
    pub row_sampling: RowSamplingParams,
    pub col_sampling: ColSamplingParams,
    pub cache_size: usize,         // Histogram cache slots
    pub early_stopping_rounds: u32,
    pub verbosity: Verbosity,
    pub seed: u64,
    pub linear_leaves: Option<LinearLeafConfig>,
}
```

**DART (Dropout Trees)**: Not currently implemented. DART adds dropout regularization
by randomly dropping trees during training. Deferred to future work.

### Early Stopping

```rust
// In boosting loop
for round in 0..n_trees {
    // ... train tree ...
    if let Some(eval) = &eval_set {
        let score = metric.compute(preds, targets);
        if early_stopper.should_stop(round, score) {
            break;  // Stop training, keep best model
        }
    }
}
```

Early stopping monitors validation metric and stops when no improvement for
`early_stopping_rounds` consecutive rounds.

```rust
pub struct GainParams {
    pub reg_lambda: f32,      // L2 regularization
    pub reg_alpha: f32,       // L1 regularization (pruning)
    pub min_child_weight: f32,
    pub min_samples_leaf: u32,
    pub min_split_gain: f32,
}
```

## Testing Strategy

Training correctness is validated through:

| Category | Location |
| -------- | -------- |
| Unit tests | Inline in `training/gbdt/*.rs` modules |
| Integration tests | `tests/training/gbdt.rs` |
| Quality benchmarks | `packages/boosters-eval/` (vs XGBoost/LightGBM) |
| Reference models | `tests/test-cases/xgboost/` (prediction comparison) |

## Files

| Path | Contents |
| ---- | -------- |
| `training/gbdt/trainer.rs` | `GBDTTrainer`, `GBDTParams`, boosting loop |
| `training/gbdt/grower.rs` | `TreeGrower`, `GrowerParams` |
| `training/gbdt/split/` | `GreedySplitter`, `GainParams`, split algorithms |
| `training/gbdt/expansion.rs` | `GrowthStrategy`, `GrowthState` |
| `training/gbdt/histograms/` | `HistogramBuilder`, `HistogramPool` |
| `training/gbdt/partition.rs` | `RowPartitioner` |
| `training/objectives.rs` | `ObjectiveFn` trait, implementations |
| `training/metrics.rs` | `MetricFn` trait, implementations |
| `training/sampling/` | Row and column samplers |
