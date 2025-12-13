```markdown
# RFC-0010: Optimization Profiles

- **Status**: Draft
- **Created**: 2024-12-05
- **Updated**: 2024-12-05
- **Depends on**: RFC-0008 (Quantization), RFC-0009 (Histogram Building), RFC-0007 (GBTree Training)
- **Scope**: Auto-selection of optimization strategies based on data characteristics

## Summary

This RFC defines **optimization profiles**—a framework for automatically selecting the best algorithmic strategies based on data characteristics. Rather than exposing dozens of knobs, we detect data shape (rows, features, sparsity) and cache properties at runtime, then select optimal strategies. Expert users can override with const generics for compile-time optimization.

## Motivation

XGBoost and LightGBM have many performance optimizations:

| Optimization | XGBoost | LightGBM | Condition |
|--------------|---------|----------|-----------|
| Row-wise histogram | ✓ | | Fits L2 cache |
| Col-wise histogram | ✓ | ✓ | Large histogram |
| 4-bit bin packing | | ✓ | ≤15 bins |
| Gradient quantization | | ✓ | Large data |
| GOSS sampling | | ✓ | Large data |
| EFB bundling | | ✓ | Sparse features |
| Prefetching | ✓ | ✓ | Random access |

These optimizations are **conditional**—they help in some scenarios, hurt in others. Our goal: **auto-select the best combination** while keeping code maintainable.

## Design Philosophy

### Layered Optimization

```text
Layer 3: User-Facing Config
         ┌──────────────────────────────────────┐
         │ "I have 10M rows, optimize for that" │
         │ → OptimizationProfile::LargeData     │
         └──────────────────────────────────────┘
                          │
                          ▼
Layer 2: Strategy Selection (Runtime)
         ┌──────────────────────────────────────┐
         │ DataCharacteristics { rows, cols, .. }│
         │ → Select: GOSS, ColWise, Prefetch    │
         └──────────────────────────────────────┘
                          │
                          ▼
Layer 1: Const Generic Kernels (Compile-Time)
         ┌──────────────────────────────────────┐
         │ accumulate::<PREFETCH=true, BITS=4>()│
         │ → Specialized, inlined code          │
         └──────────────────────────────────────┘
```

### Const Generics for Zero-Cost Abstraction

The inner loops use const generics to eliminate runtime branches:

```rust
// Const generic kernel - all branches optimized away
fn accumulate_inner<const PREFETCH: bool, const BITS: BinBits>(
    bins: &BinStorage<BITS>,
    grads: &[f32],
    hess: &[f32],
    rows: &[u32],
    hist: &mut FeatureHistogram,
) {
    if PREFETCH {
        // Prefetch code included only when PREFETCH=true
    }
    // BinBits::U4, U8, U16 specialized at compile time
}
```

### Runtime Dispatch to Const Generic Kernels

```rust
// Runtime dispatch - happens once per feature, not per row
fn accumulate_feature(config: &StrategyConfig, ...) {
    match (config.prefetch, config.bin_bits) {
        (true, BinBits::U4) => accumulate_inner::<true, BinBits::U4>(...),
        (true, BinBits::U8) => accumulate_inner::<true, BinBits::U8>(...),
        (false, BinBits::U4) => accumulate_inner::<false, BinBits::U4>(...),
        // ... 6 combinations total
    }
}
```

This gives us the best of both worlds: runtime flexibility for strategy selection, compile-time optimization for inner loops.

## Components

### DataCharacteristics

Detected at training start:

```rust
/// Data shape and properties detected from input.
pub struct DataCharacteristics {
    pub n_rows: usize,
    pub n_features: usize,
    pub n_outputs: usize,
    
    // Sparsity metrics (computed during quantization)
    pub sparsity_rate: f32,           // Fraction of missing/zero values
    pub sparse_features: usize,       // Features with >70% missing
    pub max_bins: u16,                // Maximum bins across all features
    pub features_with_few_bins: usize, // Features with ≤15 bins
    
    // System properties (detected at runtime)
    pub l2_cache_bytes: usize,
    pub n_threads: usize,
}

impl DataCharacteristics {
    /// Detect from quantized data.
    pub fn from_quantized(qm: &QuantizedMatrix, cuts: &BinCuts) -> Self;
    
    /// Histogram memory for one feature.
    pub fn hist_size_bytes(&self) -> usize {
        self.max_bins as usize * 12 // 3 × f32 per bin (grad, hess, count)
    }
    
    /// Total histogram memory for all features.
    pub fn total_hist_bytes(&self) -> usize {
        self.n_features * self.hist_size_bytes()
    }
    
    /// Does full histogram fit in L2 cache?
    pub fn hist_fits_l2(&self) -> bool {
        self.total_hist_bytes() <= self.l2_cache_bytes
    }
}
```

### OptimizationProfile

User-facing preset:

```rust
/// High-level optimization profile.
#[derive(Debug, Clone, Copy, Default)]
pub enum OptimizationProfile {
    /// Auto-detect based on data characteristics.
    #[default]
    Auto,
    /// Optimized for small datasets (<100k rows). Minimal overhead.
    SmallData,
    /// Optimized for medium datasets (100k-1M rows). Balanced.
    MediumData,
    /// Optimized for large datasets (>1M rows). Maximum throughput.
    LargeData,
    /// Optimized for wide datasets (>1000 features). Feature-parallel.
    WideData,
    /// Optimized for sparse datasets (>50% missing). EFB enabled.
    SparseData,
}

impl OptimizationProfile {
    /// Convert to concrete strategy config based on data characteristics.
    pub fn resolve(&self, data: &DataCharacteristics) -> StrategyConfig {
        match self {
            Self::Auto => Self::auto_detect(data),
            Self::SmallData => StrategyConfig::small_data(),
            Self::LargeData => StrategyConfig::large_data(data),
            // ...
        }
    }
    
    fn auto_detect(data: &DataCharacteristics) -> StrategyConfig {
        let mut config = StrategyConfig::default();
        
        // Row sampling: GOSS for large data
        if data.n_rows > 100_000 {
            config.row_sampling = RowSamplingStrategy::Goss { 
                top_rate: 0.2, 
                other_rate: 0.1 
            };
        }
        
        // Histogram algorithm: row-wise if fits L2
        config.hist_algorithm = if data.hist_fits_l2() {
            HistogramAlgorithm::RowWise
        } else {
            HistogramAlgorithm::ColWise
        };
        
        // Prefetching: always for random access patterns
        config.prefetch = true;
        
        // Gradient quantization: for very large data
        if data.n_rows > 1_000_000 {
            config.gradient_bits = GradientBits::Int16;
        }
        
        // Bin packing: 4-bit if many low-cardinality features
        if data.features_with_few_bins > data.n_features / 2 {
            config.bin_packing = BinPacking::Adaptive;
        }
        
        config
    }
}
```

### StrategyConfig

Concrete strategy selection:

```rust
/// Concrete optimization strategies selected for training.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    // Histogram building (RFC-0009)
    pub hist_algorithm: HistogramAlgorithm,
    pub accumulation: AccumulationStrategy,
    pub prefetch: bool,
    pub prefetch_distance: usize,
    
    // Quantization (RFC-0008)
    pub bin_packing: BinPacking,
    pub gradient_bits: GradientBits,
    
    // Row sampling (RFC-0007)
    pub row_sampling: RowSamplingStrategy,
    
    // Feature handling
    pub sparse_threshold: f32,  // When to use sparse bin storage
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            hist_algorithm: HistogramAlgorithm::ColWise,
            accumulation: AccumulationStrategy::Serial,
            prefetch: true,
            prefetch_distance: 8,
            bin_packing: BinPacking::Adaptive,
            gradient_bits: GradientBits::F32,
            row_sampling: RowSamplingStrategy::None,
            sparse_threshold: 0.7,
        }
    }
}
```

### Strategy Enums

```rust
/// Histogram building algorithm.
#[derive(Debug, Clone, Copy, Default)]
pub enum HistogramAlgorithm {
    /// Column-wise: iterate features, then rows. Better for large histograms.
    #[default]
    ColWise,
    /// Row-wise: iterate rows, then features. Better when histogram fits L2.
    RowWise,
}

/// Accumulation parallelism strategy.
#[derive(Debug, Clone, Copy, Default)]
pub enum AccumulationStrategy {
    #[default]
    Serial,
    /// Parallelize over features (no synchronization needed).
    FeatureParallel,
    /// Parallelize over rows (requires histogram merge).
    RowParallel,
    /// Blocked iteration for multi-node (depth-wise).
    BlockedMultiNode,
}

/// Bin storage packing strategy.
#[derive(Debug, Clone, Copy, Default)]
pub enum BinPacking {
    /// Always use u8 (max 256 bins).
    U8,
    /// Use u16 when needed.
    U16,
    /// Per-feature adaptive: u4 for ≤15 bins, u8 for ≤256, u16 otherwise.
    #[default]
    Adaptive,
}

/// Gradient storage precision.
#[derive(Debug, Clone, Copy, Default)]
pub enum GradientBits {
    /// Full precision float32.
    #[default]
    F32,
    /// Quantized to int16 (gradient + hessian packed).
    Int16,
    /// Quantized to int8 (very large data only).
    Int8,
}

/// Row sampling strategy.
#[derive(Debug, Clone, Copy, Default)]
pub enum RowSamplingStrategy {
    #[default]
    None,
    /// Uniform random sampling.
    Random { subsample: f32 },
    /// Gradient-based one-side sampling (LightGBM).
    Goss { top_rate: f32, other_rate: f32 },
}
```

## Auto-Selection Heuristics

### Histogram Algorithm Selection

```text
select_hist_algorithm(data: DataCharacteristics) -> HistogramAlgorithm:
  hist_bytes = data.n_features * data.max_bins * 12
  
  if hist_bytes <= data.l2_cache_bytes:
    // Histogram fits L2: row-wise gives better gradient locality
    return RowWise
  else:
    // Large histogram: col-wise keeps feature column in cache
    return ColWise
```

**Why this works**:
- **Row-wise**: Process all features for one row, then next row. Histogram scattered access, but gradient sequential.
- **Col-wise**: Process all rows for one feature, then next feature. Feature column sequential, gradient scattered.
- When histogram fits L2, scattered histogram access is fast → row-wise wins.
- When histogram exceeds L2, col-wise keeps working set small.

### Accumulation Strategy Selection

```text
select_accumulation(data: DataCharacteristics) -> AccumulationStrategy:
  if data.n_rows < 10_000:
    return Serial  // Parallelism overhead > benefit
  
  if data.n_features <= 64 and data.n_rows > 100_000:
    return RowParallel  // Few features, many rows
  
  if data.n_features >= 4 * data.n_threads:
    return FeatureParallel  // Enough features to parallelize
  
  return Serial
```

### Row Sampling Selection

```text
select_row_sampling(data: DataCharacteristics, learning_rate: f32) -> RowSamplingStrategy:
  if data.n_rows < 100_000:
    return None  // Sampling overhead > benefit
  
  if data.n_rows > 500_000:
    return Goss { top_rate: 0.2, other_rate: 0.1 }
    // ~30% of data, unbiased with gradient weighting
  
  return Random { subsample: 0.8 }
```

### Gradient Quantization Selection

```text
select_gradient_bits(data: DataCharacteristics) -> GradientBits:
  gradient_bytes = data.n_rows * data.n_outputs * 8  // grad + hess f32
  
  if gradient_bytes > data.l2_cache_bytes * 10:
    // Gradients much larger than cache: quantize
    return Int16
  
  return F32
```

## Integration

### Training Setup

```rust
impl<O: Objective> GBTreeTrainer<O> {
    pub fn new(
        objective: O,
        params: GBTreeParams,
        profile: OptimizationProfile,
    ) -> Self;
    
    pub fn train(&mut self, features: &ColMatrix<f32>, labels: &ColMatrix<f32>) -> TreeEnsemble {
        // 1. Quantize features
        let (quantized, cuts) = self.quantize(features);
        
        // 2. Detect data characteristics
        let data = DataCharacteristics::from_quantized(&quantized, &cuts);
        
        // 3. Resolve optimization profile
        let config = self.profile.resolve(&data);
        
        // 4. Create optimized components
        let histogram_builder = HistogramBuilder::with_config(&cuts, &config);
        let row_sampler = RowSampler::with_config(config.row_sampling);
        
        // 5. Training loop uses config-selected strategies
        self.train_with_config(quantized, labels, config)
    }
}
```

### Const Generic Kernel Dispatch

```rust
impl HistogramBuilder {
    pub fn accumulate(
        &mut self,
        node_id: u32,
        quantized: &QuantizedMatrix,
        grads: &GradientStorage,  // F32 or quantized
        rows: &[u32],
    ) {
        // Dispatch based on config (once per call, not per row)
        match (&self.config.gradient_bits, self.config.prefetch) {
            (GradientBits::F32, true) => 
                self.accumulate_inner::<GradF32, true>(node_id, quantized, grads, rows),
            (GradientBits::F32, false) => 
                self.accumulate_inner::<GradF32, false>(node_id, quantized, grads, rows),
            (GradientBits::Int16, true) => 
                self.accumulate_inner::<GradInt16, true>(node_id, quantized, grads, rows),
            // ...
        }
    }
    
    // Inner function is const-generic - fully specialized
    fn accumulate_inner<G: GradientFormat, const PREFETCH: bool>(
        &mut self,
        node_id: u32,
        quantized: &QuantizedMatrix,
        grads: &G::Storage,
        rows: &[u32],
    ) {
        // All PREFETCH branches optimized away at compile time
        // G::Storage type specialized at compile time
    }
}
```

## Default Strategies

These are **always-on** because they have no downside:

| Strategy | Default | Rationale |
|----------|---------|-----------|
| Histogram subtraction | ✓ | 50% less work, no overhead |
| LRU histogram pool | ✓ | Avoids allocation, no overhead |
| Col-major layout | ✓ | Optimal for feature iteration |
| SoA histograms | ✓ | Better vectorization |
| Missing bin | ✓ | Clean handling, negligible memory |

## Toggleable Strategies

These are **selected by profile** based on data:

| Strategy | When Enabled | Const Generic |
|----------|--------------|---------------|
| Prefetching | Always (configurable distance) | `PREFETCH: bool` |
| Row-wise histogram | Histogram fits L2 | `ALGORITHM: HistAlgo` |
| GOSS sampling | n_rows > 100k | Runtime config |
| Gradient quantization | n_rows > 1M | `GRAD: GradientFormat` |
| 4-bit bin packing | Features with ≤15 bins | `BITS: BinBits` |
| Feature-parallel | n_features > 4 × threads | Runtime config |
| Row-parallel | n_features < 64, n_rows > 100k | Runtime config |

## Performance Impact

Expected improvements over naive implementation:

| Optimization | Speedup | When |
|--------------|---------|------|
| Histogram subtraction | ~2× | Always |
| Prefetching | 1.2-1.5× | Random access |
| Row-wise histogram | 1.3-1.5× | Small histograms |
| GOSS | 2-3× | Large data |
| Gradient quantization | 1.5-2× | Memory-bound |
| 4-bit packing | 1.2× | Low-cardinality features |

## Design Decisions

### DD-1: Profile-Based Rather Than Knob-Based

**Context**: XGBoost/LightGBM expose many parameters. Should we?

**Decision**: Expose high-level profiles, auto-detect specifics.

**Rationale**:
- Most users don't understand `tree_method='hist'` vs `tree_method='approx'`
- Wrong settings can hurt performance significantly
- Profiles express intent ("I have big data"), not mechanism
- Expert override via `StrategyConfig` still possible

### DD-2: Const Generics for Inner Loops

**Context**: How to avoid runtime branches in hot loops?

**Decision**: Use const generics, dispatch once per call.

**Rationale**:
- Inner loops run millions of times per tree
- Const generics eliminate branches completely
- Dispatch overhead (one match per feature) is negligible
- ~6-12 kernel combinations is manageable

### DD-3: Detect at Quantization Time

**Context**: When to detect data characteristics?

**Decision**: After quantization, before training.

**Rationale**:
- Quantization reveals true bin counts, sparsity
- Raw features don't show binned structure
- One-time cost, amortized over all rounds

### DD-4: L2 Cache Size Detection

**Context**: How to detect L2 cache size?

**Decision**: Platform-specific detection with fallback.

```rust
fn detect_l2_cache() -> usize {
    #[cfg(target_os = "macos")]
    { /* sysctl hw.l2cachesize */ }
    
    #[cfg(target_os = "linux")]
    { /* /sys/devices/system/cpu/cpu0/cache/index2/size */ }
    
    // Fallback: assume 256KB (conservative)
    256 * 1024
}
```

## Future Work

- GPU optimization profiles
- Distributed training profiles
- Auto-tuning via micro-benchmarks
- Profile serialization for reproducibility

## References

- [XGBoost Tree Methods](https://xgboost.readthedocs.io/en/latest/treemethod.html)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
```
