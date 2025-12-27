# RFC-0015: Evaluation Framework

- **Status**: Ready for Implementation
- **Created**: 2025-12-27
- **Updated**: 2025-12-27
- **Depends on**: RFC-0014 (Python Bindings)
- **Scope**: `boosters-eval` quality evaluation and benchmarking package

## Summary

A comprehensive evaluation framework for comparing gradient boosting libraries (boosters,
XGBoost, LightGBM, CatBoost) across diverse tasks and configurations. The framework serves
two purposes: (1) benchmarking for development and documentation, and (2) quality regression
testing for release pipelines.

## Quick Start

```bash
# Install with all comparison libraries
pip install boosters-eval[all]

# === REGRESSION TESTING (most common) ===
# Quick check during development (~30s)
boosters-eval quick

# Full regression check for releases (~5min)
boosters-eval full

# === LIBRARY COMPARISON ===
# Compare boosters against XGBoost and LightGBM
boosters-eval compare --dataset california --seeds 3

# === PERFORMANCE BENCHMARKING ===
# Measure training/prediction time and memory
boosters-eval compare --dataset california --timing-mode --measure-memory
```

**Expected output** (compare command):

```text
california/gbdt (5 seeds)
┌──────────┬────────────────┬────────────────┬────────────────┐
│ Library  │ RMSE           │ MAE            │ Train Time (s) │
├──────────┼────────────────┼────────────────┼────────────────┤
│ boosters │ 0.4521 ± 0.012 │ 0.3234 ± 0.008 │ 0.42 ± 0.03    │
│ xgboost  │ 0.4518 ± 0.011 │ 0.3231 ± 0.007 │ 0.38 ± 0.02    │
│ lightgbm │ 0.4525 ± 0.013 │ 0.3238 ± 0.009 │ **0.21 ± 0.01**│
└──────────┴────────────────┴────────────────┴────────────────┘
Note: Only Train Time shows bold - it's the only statistically significant difference.
```

```python
# Programmatic usage
from boosters_eval import compare

results = compare(
    datasets=["california", "breast_cancer"],
    libraries=["boosters", "xgboost", "lightgbm"],
    seeds=5,
)
print(results.to_markdown())
```

### Troubleshooting

**XGBoost/LightGBM not installed:**

```bash
# Install without comparison libraries (boosters only)
pip install boosters-eval

# Install with all comparison libraries
pip install boosters-eval[all]

# Or install specific libraries
pip install boosters-eval xgboost lightgbm
```

When comparison libraries aren't installed, `boosters-eval` gracefully degrades:

- `compare` command warns and only runs available libraries
- `run --library xgboost` fails with helpful error message
- `list libraries` shows which are actually available

**Baseline check failed (exit code 1):**

```text
❌ Regression detected in 2 configs:
  california/gbdt: RMSE 0.4821 > baseline 0.4521 (+6.6%, tolerance 2%)
  breast_cancer/gbdt: LogLoss 0.1234 > baseline 0.1150 (+7.3%, tolerance 2%)
```

Common causes: algorithm changes, hyperparameter drift, or genuine bugs.

## Motivation

As the boosters library matures, we need systematic quality evaluation:

1. **Competitive benchmarking**: Compare boosters against XGBoost, LightGBM, CatBoost
   on standardized datasets to demonstrate quality parity or improvements
2. **Algorithm comparison**: Evaluate different booster types (GBDT, GBLinear, linear trees)
   within the same framework with fair, comparable configurations
3. **Quality regression testing**: Detect quality regressions before releases by comparing
   against recorded baseline results
4. **Coverage**: Ensure evaluation across regression, binary classification, multiclass
   classification, and ranking tasks

### Goals

- Unified configuration system for fair library comparisons
- Extensible dataset and runner abstractions
- Automated baseline recording and regression detection
- Rich reporting (markdown tables, JSON, CSV, CI integration)
- Support for boosters-specific algorithms (GBLinear, linear trees)
- Fault-tolerant execution with partial result saving

### Non-Goals

- Performance/speed benchmarking (separate concern, use criterion)
- Hyperparameter tuning or AutoML
- Production deployment scenarios
- Distributed training evaluation

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI Layer (Typer)                             │
│   boosters-eval compare   boosters-eval baseline   boosters-eval check  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Benchmark Suite                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Configs    │  │ Execution  │  │ Results    │  │ Reporting  │        │
│  │ (YAML/Py)  │──│ Engine     │──│ Collector  │──│ Engine     │        │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│  Datasets  │ │  Runners   │ │  Metrics   │ │ Baselines  │
│ (Loaders)  │ │ (Adapters) │ │ (sklearn)  │ │ (JSON)     │
└────────────┘ └────────────┘ └────────────┘ └────────────┘
```

---

## Core Concepts

### Tasks

The framework supports these ML task types:

| Task | Objective | Primary Metric | Secondary Metrics |
| ---- | --------- | -------------- | ----------------- |
| `regression` | Predict continuous values | RMSE | MAE, R² |
| `binary` | Binary classification | LogLoss | Accuracy, AUC-ROC |
| `multiclass` | Multi-class classification | Multi-LogLoss | Accuracy, Macro-F1 |
| `ranking` | Learning to rank (future) | NDCG | MAP |

### Booster Types

Different gradient boosting algorithms to compare:

| Booster Type | Description | Supported Libraries |
| ------------ | ----------- | ------------------- |
| `gbdt` | Standard decision tree boosting | boosters, XGBoost, LightGBM, CatBoost |
| `gblinear` | Linear model boosting | boosters, XGBoost |
| `linear_trees` | GBDT with linear leaf models | boosters, LightGBM |

### Fair Configuration Mapping

A key challenge is configuring different libraries fairly. The framework uses
canonical parameter names that map to library-specific equivalents:

```python
class GrowthStrategy(Enum):
    DEPTHWISE = "depthwise"  # XGBoost default, boosters supports
    LEAFWISE = "leafwise"    # LightGBM default, boosters supports


@dataclass
class TrainingConfig:
    """Canonical training parameters - consistent semantics across libraries."""
    
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    n_leaves: int = 31
    min_samples_leaf: int = 20
    l1: float = 0.0
    l2: float = 1.0
    subsample: float = 1.0
    colsample: float = 1.0
    n_threads: int = 1
    seed: int = 42
    growth_strategy: GrowthStrategy = GrowthStrategy.DEPTHWISE
```

Each runner translates these to library-specific parameters:

| Canonical | XGBoost | LightGBM | boosters |
| --------- | ------- | -------- | -------- |
| `n_estimators` | `num_boost_round` | `n_estimators` | `n_estimators` |
| `learning_rate` | `eta` | `learning_rate` | `learning_rate` |
| `max_depth` | `max_depth` | `max_depth` | `max_depth` |
| `n_leaves` | (derived from depth) | `num_leaves` | `n_leaves` |
| `min_samples_leaf` | `min_child_weight` | `min_data_in_leaf` | `min_samples_leaf` |
| `l1` | `alpha` | `lambda_l1` | `l1` |
| `l2` | `lambda` | `lambda_l2` | `l2` |

**Tree growth strategy considerations:**

Different libraries have different default growth strategies:

| Library | Default Strategy | Alternative |
| ------- | ---------------- | ----------- |
| XGBoost | Depthwise | - |
| LightGBM | Leafwise | Depthwise (via `boosting_type`) |
| boosters | Depthwise | Leafwise |
| CatBoost | Depthwise | - |

**Fair comparison requires testing both strategies:**

1. **DEPTHWISE suite**: All libraries use depthwise growth (LightGBM configured explicitly)
2. **LEAFWISE suite**: Only libraries supporting leafwise (boosters, LightGBM)

The built-in `COMPARISON_DEPTHWISE` and `COMPARISON_LEAFWISE` suites handle this:

```python
# Depthwise comparison - all libraries
COMPARISON_DEPTHWISE = BenchmarkSuite(
    name="comparison-depthwise",
    configs=[...],  # growth_strategy=DEPTHWISE for all
)

# Leafwise comparison - boosters vs LightGBM only
COMPARISON_LEAFWISE = BenchmarkSuite(
    name="comparison-leafwise",
    configs=[...],  # growth_strategy=LEAFWISE, libraries=["boosters", "lightgbm"]
)
```

**Recommendation**: Run both suites and report results separately. Don't mix strategies
in the same comparison table.

---

## Configuration System

### Benchmark Configuration

Benchmarks are defined declaratively:

```python
@dataclass
class BenchmarkConfig:
    """Single benchmark configuration."""
    
    name: str
    dataset: DatasetConfig
    task: Task
    booster_type: BoosterType
    training: TrainingConfig
    libraries: list[str]
    
    # Evaluation settings
    validation_fraction: float = 0.2
    seeds: list[int] = field(default_factory=lambda: [42, 1379, 2716, 4053, 5390])
```

**Seed derivation**: When `--seeds N` is passed via CLI, seeds are derived deterministically
as `[42 + i * 1337 for i in range(N)]`. This ensures reproducibility across runs while
providing sufficient spread. The formula uses a large prime multiplier to avoid correlated
random sequences.

### Benchmark Suites

Suites group related benchmarks with appropriate seed counts:

```python
# QUICK: Minimal smoke test - fewer seeds, small datasets only
QUICK_SUITE = BenchmarkSuite(
    name="quick",
    default_seeds=3,  # Fewer seeds for speed
    configs=[
        # One dataset per task type, small sizes only
        BenchmarkConfig(
            name="california/gbdt",
            dataset=DATASETS["california"],  # ~20K samples
            task=Task.REGRESSION,
            booster_type=BoosterType.GBDT,
            training=TrainingConfig(n_estimators=50, max_depth=4),  # Reduced
            libraries=["boosters", "xgboost", "lightgbm"],
        ),
        BenchmarkConfig(
            name="breast_cancer/gbdt",
            dataset=DATASETS["breast_cancer"],  # ~500 samples
            task=Task.BINARY,
            booster_type=BoosterType.GBDT,
            training=TrainingConfig(n_estimators=50, max_depth=4),
            libraries=["boosters", "xgboost", "lightgbm"],
        ),
    ],
)

# FULL: Complete evaluation - more seeds, all datasets
FULL_SUITE = BenchmarkSuite(
    name="full",
    default_seeds=5,  # More seeds for reliable statistics
    configs=[
        # All datasets × all compatible boosters
        # Regression: california, synthetic_reg_small, synthetic_reg_medium
        # Binary: breast_cancer, synthetic_bin_small, synthetic_bin_medium
        # Multiclass: iris, synthetic_multi_small, synthetic_multi_medium
        ...
    ],
)
```

**Suite configuration summary:**

| Suite | Seeds | Datasets | Estimators | Depth | Purpose |
| ----- | ----- | -------- | ---------- | ----- | ------- |
| `quick` | 3 | 2 (small) | 50 | 4 | Development, PR checks (~30s) |
| `full` | 5 | 9 (all) | 100 | 6 | Release validation (~5min) |

**Expected runtimes** (single-threaded, M1 Mac):

| Suite | Approximate Runtime | Use Case |
| ----- | ------------------- | -------- |
| QUICK_SUITE | ~30 seconds | Development, PR checks |
| FULL_SUITE | ~5 minutes | Release validation |

Add 20-30% buffer for CI environments. First run may be slower due to lazy imports.

### Code-Based Configuration

Benchmarks are defined in Python code rather than YAML/JSON files. This approach:

1. **Avoids serialization overhead** - No parsing or validation at runtime
2. **Stays in sync with code** - No separate file to maintain
3. **Enables IDE support** - Autocompletion, type checking, refactoring
4. **Allows dynamic configuration** - Programmatic suite generation

```python
# boosters_eval/suites/ablation.py
from boosters_eval import BenchmarkConfig, TrainingConfig, GrowthStrategy

ABLATION_GROWTH_SUITE = BenchmarkSuite(
    name="ablation-growth",
    configs=[
        BenchmarkConfig(
            name="california/boosters-depthwise",
            dataset=DATASETS["california"],
            task=Task.REGRESSION,
            booster_type=BoosterType.GBDT,
            training=TrainingConfig(
                n_estimators=100,
                growth_strategy=GrowthStrategy.DEPTHWISE,
            ),
            libraries=["boosters"],  # Single library for ablation
        ),
        BenchmarkConfig(
            name="california/boosters-leafwise",
            dataset=DATASETS["california"],
            task=Task.REGRESSION,
            booster_type=BoosterType.GBDT,
            training=TrainingConfig(
                n_estimators=100,
                growth_strategy=GrowthStrategy.LEAFWISE,
            ),
            libraries=["boosters"],
        ),
    ],
)
```

### Ablation Study Patterns

Ablation suites compare settings within a single library, useful for:

1. **Growth strategy comparison** (leafwise vs depthwise)
2. **Device comparison** (CPU vs GPU, future)
3. **Threading modes** (single vs multi-threaded)
4. **Algorithm variants** (GBDT vs GBLinear vs linear_trees)

```python
# Helper to generate ablation configs
def create_ablation_suite(
    name: str,
    dataset: str,
    library: str,
    variants: dict[str, TrainingConfig],
) -> BenchmarkSuite:
    """Generate ablation suite from named variants."""
    configs = [
        BenchmarkConfig(
            name=f"{dataset}/{library}-{variant_name}",
            dataset=DATASETS[dataset],
            task=DATASETS[dataset].task,
            booster_type=BoosterType.GBDT,
            training=training_config,
            libraries=[library],
        )
        for variant_name, training_config in variants.items()
    ]
    return BenchmarkSuite(name=name, configs=configs)

# Example: Compare threading modes
ABLATION_THREADING = create_ablation_suite(
    name="ablation-threading",
    dataset="california",
    library="boosters",
    variants={
        "single": TrainingConfig(n_threads=1),
        "multi-4": TrainingConfig(n_threads=4),
        "multi-auto": TrainingConfig(n_threads=-1),  # All cores
    },
)

# Future: GPU vs CPU comparison
# ABLATION_DEVICE = create_ablation_suite(
#     name="ablation-device",
#     dataset="higgs",
#     library="boosters",
#     variants={
#         "cpu": TrainingConfig(device="cpu"),
#         "gpu": TrainingConfig(device="cuda"),
#     },
# )
```

---

## Dataset System

### DatasetConfig

Datasets are defined with metadata and a loader function:

```python
@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    name: str
    task: Task
    loader: Callable[[], tuple[NDArray, NDArray]]
    n_classes: int | None = None
    n_features: int | None = None
    n_samples: int | None = None
    subsample: int | None = None  # Cap for large datasets
    
    # Metadata for reporting
    description: str = ""
    source: str = ""
```

### Built-in Datasets

| Category | Datasets | Purpose |
| -------- | -------- | ------- |
| **Regression** | California Housing, Synthetic Regression | Standard regression tasks |
| **Binary** | Breast Cancer, Adult Income, Synthetic Binary | Binary classification |
| **Multiclass** | Iris, Covertype, Synthetic Multiclass | Multi-class scenarios |
| **Ranking** | MSLR-Web10K subset (future) | LTR evaluation |

**Dataset caching:** Large datasets (e.g., `cover_type` with 581K samples) are cached after
first load using `functools.lru_cache`. This ensures fast iteration during development—the
first run pays the load cost, subsequent runs are instant.

**Small dataset caveat:** Results on very small datasets (e.g., Iris with 150 samples) should
be interpreted with caution due to high train/test split variance. For robust evaluation on
small datasets, use K-fold cross-validation (v1.1 feature).

### Custom Dataset Registration

```python
from boosters_eval import register_dataset, DatasetConfig, Task

def load_my_dataset():
    """Load custom dataset."""
    df = pd.read_parquet("my_dataset.parquet")
    X = df.drop("target", axis=1).values
    y = df["target"].values
    return X.astype(np.float32), y.astype(np.float32)

register_dataset(
    DatasetConfig(
        name="my_dataset",
        task=Task.BINARY,
        loader=load_my_dataset,
        description="Custom binary classification dataset",
    )
)
```

---

## Runner System

### Runner Protocol

Each library implements the Runner protocol:

```python
class Runner(Protocol):
    """Protocol for library-specific runners."""
    
    name: str
    
    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if this runner supports the given configuration."""
        ...
    
    @classmethod
    def run(
        cls,
        config: BenchmarkConfig,
        x_train: NDArray[np.float32],
        y_train: NDArray[np.float32],
        x_valid: NDArray[np.float32],
        y_valid: NDArray[np.float32],
        seed: int,
    ) -> BenchmarkResult:
        """Train model and return results with metrics."""
        ...
```

### Built-in Runners

| Runner | Library | Supported Boosters |
| ------ | ------- | ------------------ |
| `BoostersRunner` | boosters | gbdt, gblinear, linear_trees |
| `XGBoostRunner` | xgboost | gbdt, gblinear |
| `LightGBMRunner` | lightgbm | gbdt, linear_trees |
| `CatBoostRunner` | catboost | gbdt |

### Boosters Runner Implementation

```python
def _task_to_objective(task: Task, n_classes: int | None) -> bst.Objective:
    """Map evaluation task to boosters objective."""
    import boosters as bst
    
    if task == Task.REGRESSION:
        return bst.Objective.mse()
    elif task == Task.BINARY:
        return bst.Objective.log_loss()
    elif task == Task.MULTICLASS:
        return bst.Objective.multi_log_loss(n_classes)
    elif task == Task.RANKING:
        return bst.Objective.lambdarank()
    else:
        raise ValueError(f"Unknown task: {task}")


class BoostersRunner(Runner):
    """Runner for the boosters library."""
    
    name = "boosters"
    
    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        return config.booster_type in (
            BoosterType.GBDT, 
            BoosterType.GBLINEAR, 
            BoosterType.LINEAR_TREES,
        )
    
    @classmethod
    def run(cls, config, x_train, y_train, x_valid, y_valid, seed) -> BenchmarkResult:
        import boosters as bst
        
        task = config.dataset.task
        tc = config.training
        
        # Build objective from task
        objective = _task_to_objective(task, config.dataset.n_classes)
        
        if config.booster_type == BoosterType.GBLINEAR:
            model_config = bst.GBLinearConfig(
                n_estimators=tc.n_estimators,
                learning_rate=tc.learning_rate,
                l1=tc.l1,
                l2=tc.l2,
                objective=objective,
                seed=seed,
            )
            model = bst.GBLinearModel(model_config)
        else:
            model_config = bst.GBDTConfig(
                n_estimators=tc.n_estimators,
                learning_rate=tc.learning_rate,
                max_depth=tc.max_depth,
                n_leaves=tc.n_leaves,
                min_samples_leaf=tc.min_samples_leaf,
                l1=tc.l1,
                l2=tc.l2,
                subsample=tc.subsample,
                colsample=tc.colsample,
                objective=objective,
                seed=seed,
            )
            model = bst.GBDTModel(model_config)
        
        train_ds = bst.Dataset(x_train, y_train)
        valid_ds = bst.Dataset(x_valid, y_valid)
        
        start = time.perf_counter()
        model.fit(train_ds, valid=[bst.EvalSet(valid_ds, "valid")])
        train_time = time.perf_counter() - start
        
        y_pred = model.predict(valid_ds)
        metrics = compute_metrics(task, y_valid, y_pred)
        
        return BenchmarkResult(
            config_name=config.name,
            library=cls.name,
            seed=seed,
            task=task,
            metrics=metrics,
            train_time_s=train_time,
        )
```

---

## Metrics System

### Metric Computation

Metrics are computed via sklearn for consistency:

```python
def compute_metrics(
    task: Task,
    y_true: NDArray,
    y_pred: NDArray,
    n_classes: int | None = None,
) -> dict[str, float]:
    """Compute task-appropriate metrics."""
    
    if task == Task.REGRESSION:
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
    elif task == Task.BINARY:
        y_class = (y_pred >= 0.5).astype(int)
        return {
            "logloss": log_loss(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_class),
            "auc_roc": roc_auc_score(y_true, y_pred),
        }
    else:  # MULTICLASS
        y_class = np.argmax(y_pred, axis=1)
        return {
            "mlogloss": log_loss(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_class),
        }
```

### Metric Direction

For reporting and regression detection:

```python
METRIC_DIRECTION = {
    "rmse": "lower_better",
    "mae": "lower_better",
    "r2": "higher_better",
    "logloss": "lower_better",
    "mlogloss": "lower_better",
    "accuracy": "higher_better",
    "auc_roc": "higher_better",
    "train_time_s": "lower_better",
    "predict_time_s": "lower_better",
    "peak_memory_mb": "lower_better",
}
```

---

## Timing and Memory Measurement

### Timing Methodology

For fair performance comparisons:

```python
def measure_timing(
    runner: Runner,
    config: BenchmarkConfig,
    warmup: int = 1,
) -> tuple[float, float]:
    """Measure train and predict times with warmup."""
    
    # Warmup runs (not timed) - let JIT/caches stabilize
    for _ in range(warmup):
        model = runner.train(config)
        _ = model.predict(x_valid)
    
    # Timed training
    gc.collect()  # Reduce GC noise
    start = time.perf_counter()
    model = runner.train(config)
    train_time = time.perf_counter() - start
    
    # Timed prediction (median of 3 runs)
    predict_times = []
    for _ in range(3):
        gc.collect()
        start = time.perf_counter()
        _ = model.predict(x_valid)
        predict_times.append(time.perf_counter() - start)
    predict_time = np.median(predict_times)
    
    return train_time, predict_time
```

**Fair timing requires:**

- Same `n_threads` across libraries (or `auto` for all)
- Warmup runs to eliminate cold-start effects (2-3 warmups is standard practice)
- GC collection before timing
- Same hardware/environment

**Thread count in timing mode:**

When `--timing-mode` is enabled, `n_threads=auto` means:

- Uses all available physical cores (`psutil.cpu_count(logical=False)`)
- Can be overridden via `BOOSTERS_EVAL_THREADS` environment variable
- All libraries receive the same thread count for fair comparison

### Memory Measurement

Peak memory tracking via `tracemalloc`:

**Important limitation:** `tracemalloc` only tracks Python memory allocations. Native Rust
allocations in boosters (tree structures, gradients) are not tracked. For accurate boosters
memory profiling, use system tools:

- **macOS:** Activity Monitor, Instruments
- **Linux:** `/proc/PID/status`, `valgrind --tool=massif`
- **Cross-platform:** `memory_profiler` with `@profile` decorator

```python
def measure_memory(
    runner: Runner,
    config: BenchmarkConfig,
) -> tuple[float, float]:
    """Measure peak memory and model size."""
    import tracemalloc
    import sys
    
    tracemalloc.start()
    
    model = runner.train(config)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_memory_mb = peak / (1024 * 1024)
    
    # Estimate model size (library-specific)
    model_size_mb = _estimate_model_size(model) / (1024 * 1024)
    
    return peak_memory_mb, model_size_mb
```

**Memory measurement caveats:**

- `tracemalloc` only tracks Python allocations
- Native library memory (C++/Rust) may not be fully captured
- Model size estimation is approximate
- Memory varies with input size and tree depth

---

## Results and Reporting

### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    
    config_name: str
    library: str
    seed: int
    task: Task
    booster_type: str
    dataset_name: str
    metrics: dict[str, float]
    
    # Timing statistics
    train_time_s: float | None = None
    predict_time_s: float | None = None
    
    # Memory tracking
    peak_memory_mb: float | None = None
    model_size_mb: float | None = None
    
    # Optional metadata
    model_params: dict[str, Any] | None = None
    n_trees_trained: int | None = None
```

### ResultCollection

```python
class ResultCollection:
    """Collection of benchmark results with aggregation."""
    
    results: list[BenchmarkResult]
    errors: list[BenchmarkError]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        ...
    
    def filter(
        self,
        library: str | None = None,
        dataset: str | None = None,
        booster: str | None = None,
        task: Task | None = None,
    ) -> ResultCollection:
        """Filter results by criteria."""
        ...
    
    def best(
        self, 
        metric: str,
        significance_level: float = 0.05,
    ) -> pd.DataFrame:
        """Get best library per (dataset, booster) for given metric.
        
        Uses confidence intervals to determine if lead is statistically
        significant. If intervals overlap, marks as 'tie' rather than
        falsely highlighting a winner.
        """
        ...
    
    def is_significant(
        self,
        library_a: str,
        library_b: str,
        metric: str,
        alpha: float = 0.05,
    ) -> tuple[bool, float]:
        """Test if difference between libraries is statistically significant.
        
        Uses Welch's t-test across seeds. Returns (is_significant, p_value).
        """
        ...
    
    def summary(
        self, 
        group_by: list[str] = ["dataset", "booster", "library"],
    ) -> pd.DataFrame:
        """Aggregate with mean ± std."""
        ...
    
    def to_markdown(
        self,
        highlight_best: bool = True,
        require_significance: bool = True,
        alpha: float = 0.05,
    ) -> str:
        \"\"\"Generate markdown report.
        
        Args:
            highlight_best: Bold the best result per metric
            require_significance: Only highlight if statistically significant
            alpha: Significance level for Welch's t-test
        
        When require_significance=True, a library is only marked as "best" if:
        1. It has the best mean value
        2. Its confidence interval doesn't overlap with the second-best
        3. Welch's t-test p-value < alpha
        
        If the lead is not significant, both are shown without highlighting,
        indicating a statistical tie.
        \"\"\"
        ...
    
    def to_json(self) -> str:
        """Export as JSON for baselines."""
        ...
    
    def to_csv(self) -> str:
        """Export as CSV."""
        ...
```

### Report Formatting

**Significance-aware highlighting example:**

```text
california/gbdt (5 seeds)
┌──────────┬────────────────┬────────────────┬────────────────┐
│ Library  │ RMSE           │ MAE            │ Train Time (s) │
├──────────┼────────────────┼────────────────┼────────────────┤
│ boosters │ 0.4521 ± 0.012 │ 0.3234 ± 0.008 │ 0.42 ± 0.03    │
│ xgboost  │ 0.4518 ± 0.011 │ 0.3231 ± 0.007 │ 0.38 ± 0.02    │
│ lightgbm │ 0.4525 ± 0.013 │ 0.3238 ± 0.009 │ **0.21 ± 0.01**│
└──────────┴────────────────┴────────────────┴────────────────┘
                           ↑ No highlight: RMSE/MAE differences not significant
                                                      ↑ Bold: LightGBM significantly faster
```

---

## Report Generation

### Machine Fingerprinting

To ensure comparable results, reports include machine fingerprint:

```python
@dataclass
class MachineInfo:
    """Machine fingerprint for result comparability."""
    cpu_model: str       # e.g., "Apple M1 Pro"
    cpu_cores: int       # Physical cores
    cpu_threads: int     # Logical threads  
    memory_gb: float     # Total RAM
    os: str              # e.g., "macOS 14.2"
    blas_backend: str | None = None  # e.g., "OpenBLAS", "MKL" (optional)
    build_type: str | None = None    # e.g., "release", "debug" (optional)
    
    @classmethod
    def collect(cls) -> MachineInfo:
        import platform
        import psutil
        
        # Fallback for platform.processor() which returns empty on some Linux
        cpu_model = platform.processor()
        if not cpu_model:
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":")[1].strip()
                            break
            except FileNotFoundError:
                cpu_model = "Unknown"
        
        return cls(
            cpu_model=cpu_model or "Unknown",
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            cpu_threads=psutil.cpu_count(logical=True) or 1,
            memory_gb=psutil.virtual_memory().total / (1024**3),
            os=f"{platform.system()} {platform.release()}",
            blas_backend=_detect_blas_backend(),
            build_type=_detect_build_type(),
        )

def _detect_blas_backend() -> str | None:
    """Attempt to detect BLAS backend (best effort)."""
    try:
        import numpy as np
        config = np.__config__
        if hasattr(config, 'blas_ilp64_opt_info'):
            return str(config.blas_ilp64_opt_info)
        return None
    except Exception:
        return None

def _detect_build_type() -> str | None:
    """Detect if boosters was built in release or debug mode."""
    try:
        import boosters
        return getattr(boosters, '__build_type__', None)
    except Exception:
        return None
```

### Report Types

| Type | Focus | Includes |
| ---- | ----- | -------- |
| `quality` | Model accuracy | Metrics per task (RMSE, LogLoss, etc.), no timing |
| `performance` | Speed/memory | Train time, predict time, memory usage |
| `comparison` | Full analysis | Both quality and performance, with statistical tests |

### Full Report to docs/benchmarks/

The framework generates dated, versioned reports for `docs/benchmarks/`:

**Example output files:**

```text
docs/benchmarks/2025-12-27-abc1234-quality-report.md
docs/benchmarks/2025-12-27-abc1234-quality-report.json
```

**Output directory conventions:**

| Directory | Purpose | Reason |
| --------- | ------- | ------ |
| `docs/benchmarks/` | Reports | Documentation, human-readable, versioned |
| `tests/baselines/` | Baselines | Test fixtures, machine-readable, CI-checked |

The output directory is auto-created if it doesn't exist.

```python
def generate_report(
    results: ResultCollection,
    report_type: str = "quality",  # "quality", "performance", "comparison"
    output_dir: Path = Path("docs/benchmarks"),
) -> tuple[Path, Path]:
    """Generate full markdown report and structured JSON.
    
    Returns paths to (markdown_report, json_baseline).
    """
    # Auto-create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metadata
    metadata = ReportMetadata(
        git_sha=get_git_sha(),
        git_branch=get_git_branch(),
        timestamp=datetime.now(UTC),
        machine=MachineInfo.collect(),
        rust_version=get_rust_version(),
        python_version=get_python_version(),
        boosters_version=get_boosters_version(),
    )
    
    # Generate filename: YYYY-MM-DD-<sha>-<type>-report.md
    date_str = metadata.timestamp.strftime("%Y-%m-%d")
    filename = f"{date_str}-{metadata.git_sha[:7]}-{report_type}-report"
    
    md_path = output_dir / f"{filename}.md"
    json_path = output_dir / f"{filename}.json"
    
    # Render markdown from template
    md_content = render_report_template(results, metadata, report_type)
    md_path.write_text(md_content)
    
    # Export structured JSON for regression checking
    json_content = results.to_json(include_metadata=True, metadata=metadata)
    json_path.write_text(json_content)
    
    return md_path, json_path
```

### Report Template

Reports follow the existing `docs/benchmarks/TEMPLATE.md` format:

```markdown
# {date}: {report_type} Report

## Environment

| Property | Value |
|----------|-------|
| Commit | {git_sha} ({git_branch}) |
| Date | {timestamp} |
| Machine | {machine.cpu} |
| Rust | {rust_version} |
| Python | {python_version} |
| boosters | {boosters_version} |

## Configuration

**Seeds**: {n_seeds} ({seed_list})
**Growth Strategy**: {growth_strategy}

## Results

### REGRESSION
{regression_tables}

### BINARY  
{binary_tables}

### MULTICLASS
{multiclass_tables}

## Benchmark Configuration

{config_table}

## Reproducing

\```bash
boosters-eval run --suite {suite_name} --seeds {n_seeds}
\```
```

### Dual Output: Human-Readable + Machine-Readable

| Output | Format | Purpose | Location |
| ------ | ------ | ------- | -------- |
| Full Report | Markdown | Human review, documentation | `docs/benchmarks/{date}-{sha}-{type}-report.md` |
| Structured Data | JSON | Regression checking, CI | `docs/benchmarks/{date}-{sha}-{type}-report.json` |
| Baseline | JSON | Release baselines | `tests/baselines/{version}.json` |

---

## Programmatic API

The CLI wraps a Python API that can be used directly:

```python
from boosters_eval import run_suite, compare, ResultCollection

# Run a predefined suite
results = run_suite("quick", seeds=3, verbose=True)
print(results.to_markdown())

# Custom comparison
results = compare(
    datasets=["california", "breast_cancer"],
    libraries=["boosters", "xgboost"],
    booster="gbdt",
    seeds=5,
)

# Explore results
df = results.to_dataframe()
print(df.head())

# Filter and analyze
boosters_results = results.filter(library="boosters")
regression_results = results.filter(task=Task.REGRESSION)

# Find best library per dataset
print(results.best(metric="rmse"))

# Export
results.to_csv("results.csv")
results.to_json("results.json")
```

### Jupyter Notebook Support

```python
from boosters_eval import compare
from rich.jupyter import print as rprint

# Rich progress bars work in Jupyter
results = compare(
    datasets=["california"],
    libraries=["boosters", "xgboost"],
    verbose=True,  # Shows progress bar
)

# Display formatted results
rprint(results.summary())
```

### Markdown Report Format

```markdown
## california (gbdt)

| Library   | rmse ↓          | mae ↓           | r² ↑            |
|-----------|-----------------|-----------------|-----------------|
| boosters  | **0.4521 ± 0.01** | **0.3012 ± 0.01** | 0.8234 ± 0.02   |
| xgboost   | 0.4532 ± 0.01   | 0.3024 ± 0.01   | **0.8245 ± 0.02** |
| lightgbm  | 0.4528 ± 0.01   | 0.3018 ± 0.01   | 0.8239 ± 0.02   |

**Runtime Statistics:**
- boosters: avg 1.234s, total 6.170s
- xgboost: avg 1.456s, total 7.280s
- lightgbm: avg 0.987s, total 4.935s
```

---

## Baseline and Regression Testing

### Baseline Recording

Baselines are recorded for regression detection:

```bash
# Record current results as baseline
boosters-eval baseline record --output baselines/v0.1.0.json

# Compare against baseline
boosters-eval baseline check --baseline baselines/v0.1.0.json
```

**Baseline file location convention:**

| Use Case | Location | Rationale |
| -------- | -------- | --------- |
| Release baselines | `tests/baselines/` | Version-controlled with tests |
| Development snapshots | `tmp/baselines/` | Temporary, gitignored |
| CI artifacts | `$CI_ARTIFACTS/baselines/` | CI system managed |

The CLI defaults to `tests/baselines/` when no path is specified:

```bash
boosters-eval baseline record  # -> tests/baselines/baseline.json
```

### Baseline Format

Baselines use a versioned JSON schema for forward compatibility:

```json
{
  "schema_version": 1,
  "boosters_version": "0.1.0",
  "recorded_at": "2025-12-27T10:30:00Z",
  "git_sha": "abc123def",
  "config": {
    "seeds": [42, 1379, 2716, 4053, 5390],
    "suite": "full"
  },
  "results": [
    {
      "config": "california/gbdt",
      "library": "boosters",
      "task": "regression",
      "primary_metric": "rmse",
      "metrics": {
        "rmse": {"mean": 0.4521, "std": 0.0102, "n": 5},
        "mae": {"mean": 0.3012, "std": 0.0087, "n": 5}
      }
    }
  ]
}
```

### Baseline Validation

Baselines are validated using Pydantic for schema enforcement:

```python
from pydantic import BaseModel, field_validator
from datetime import datetime

class MetricStats(BaseModel):
    mean: float
    std: float
    n: int

class BaselineResult(BaseModel):
    config: str
    library: str
    task: str
    primary_metric: str
    metrics: dict[str, MetricStats]

class Baseline(BaseModel):
    schema_version: int
    boosters_version: str
    recorded_at: datetime
    git_sha: str | None = None
    config: dict[str, Any]
    results: list[BaselineResult]
    
    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: int) -> int:
        if v > 1:
            raise ValueError(f"Unsupported schema version: {v}")
        return v

def load_baseline(path: Path) -> Baseline:
    """Load and validate baseline from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return Baseline.model_validate(data)
```

**Schema version notes:**

- `schema_version: 1` - Initial format (v0.1.0+)
- Future versions will include migration utilities

### Regression Definition

A **regression** is defined precisely as:

> A degradation in the **primary metric** for a (config, library) pair that exceeds
> the configured tolerance threshold.

**Primary metrics by task:**

| Task | Primary Metric | Direction |
| ---- | -------------- | --------- |
| regression | RMSE | lower is better |
| binary | LogLoss | lower is better |
| multiclass | Multi-LogLoss | lower is better |
| ranking | NDCG@10 | higher is better |

**Regression detection logic:**

```python
def is_regression(
    baseline_mean: float,
    current_mean: float,
    tolerance: float,
    lower_is_better: bool,
) -> bool:
    """Check if current result is a regression from baseline."""
    if lower_is_better:
        # Regression if current is worse (higher) by more than tolerance
        threshold = baseline_mean * (1 + tolerance)
        return current_mean > threshold
    else:
        # Regression if current is worse (lower) by more than tolerance
        threshold = baseline_mean * (1 - tolerance)
        return current_mean < threshold
```

**Limitations of tolerance-based detection:**

- Does not account for measurement variance (std)
- 2% threshold is arbitrary; may need tuning per-metric
- For high-precision requirements, consider statistical tests (future work)

**Tolerance rationale:**

The default 2% tolerance was chosen empirically:

- **1%** is too tight—natural variance in tree splitting and sampling causes false positives
- **5%** is too loose—real regressions (e.g., wrong hyperparameters) slip through
- **2%** catches meaningful degradations while allowing for normal measurement noise

Ensemble models have inherent variance from:

- Random feature selection for splits
- Bootstrap sampling (bagging)
- Tie-breaking in split selection

For tighter guarantees, use more seeds (reduces standard error) or statistical regression detection (v1.1).

### Baseline Comparison Edge Cases

| Scenario | Behavior |
| -------- | -------- |
| Config in baseline but not in current run | Warning: "Skipped config X (not in current run)" |
| Config in current run but not in baseline | Info: "New config X (no baseline)" - not a failure |
| Library crashed in current run | Error: counted as regression if baseline exists |
| All libraries crashed for a config | Error: report failure but continue |
| Baseline file invalid/corrupt | Error: exit with validation error |
| Schema version higher than supported | Error: exit with upgrade notice |

### Error Handling and Partial Results

Benchmark execution is fault-tolerant:

```python
@dataclass
class BenchmarkError:
    """Error during benchmark execution."""
    config_name: str
    library: str
    seed: int
    error_type: str
    error_message: str
    traceback: str | None = None

class ResultCollection:
    results: list[BenchmarkResult]
    errors: list[BenchmarkError]
    
    def save_partial(self, path: Path) -> None:
        """Save results collected so far (for crash recovery)."""
        ...
    
    @classmethod
    def load_partial(cls, path: Path) -> ResultCollection:
        """Resume from partial results."""
        ...
```

When a library crashes or times out:

1. Log the error with full context
2. Continue with remaining benchmarks
3. Include error summary in the final report
4. Exit with non-zero code if errors occurred (unless `--continue-on-error`)

### CI Integration

```yaml
# .github/workflows/quality.yml
name: Quality Regression Check

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: |
          uv pip install -e "packages/boosters-python[dev]"
          uv pip install -e "packages/boosters-eval[all]"
      
      - name: Run quality benchmarks
        run: |
          boosters-eval baseline check \
            --baseline baselines/main.json \
            --tolerance 0.02 \
            --fail-on-regression \
            --output results.json
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: results.json
```

---

## CLI Design

### Commands

```bash
# === QUICK REGRESSION TEST SHORTCUTS ===
# Fast check during development
boosters-eval quick   # ~30 seconds: 3 seeds, 2 datasets, 50 estimators

# Complete regression check for releases  
boosters-eval full    # ~5 minutes: 5 seeds, 9 datasets, 100 estimators

# These are shortcuts for:
boosters-eval baseline check --suite quick --baseline tests/baselines/quick.json
boosters-eval baseline check --suite full --baseline tests/baselines/full.json

# === COMPARISON COMMANDS ===
# Compare libraries on datasets
boosters-eval compare \
  --dataset california breast_cancer \
  --library boosters xgboost lightgbm \
  --booster gbdt \
  --seeds 5 \
  --output report.md \
  --format markdown

# Run comprehensive suite
boosters-eval run \
  --suite full \
  --output results.json \
  --format json

# Quick development check (boosters only if others not installed)
boosters-eval run --suite quick --library boosters

# === TIMING/BENCHMARK MODE ===
# Timing mode: multi-threaded with warmup runs for performance comparison
boosters-eval compare \
  --dataset california \
  --timing-mode \
  --warmup 2 \
  --measure-memory

# === ABLATION STUDIES ===
# Compare growth strategies within boosters
boosters-eval run --suite ablation-growth

# Compare boosters settings (custom ablation)
boosters-eval compare \
  --dataset california \
  --library boosters \
  --variants "depthwise:growth=depthwise" "leafwise:growth=leafwise"

# === REPORT GENERATION ===
# Generate full markdown report to docs/benchmarks/
boosters-eval report --suite full --type quality

# Generate comparison report (quality + performance)
boosters-eval report --suite full --type comparison

# Preview report without writing files
boosters-eval report --suite full --type quality --dry-run

# Auto-open generated report in browser/editor
boosters-eval report --suite full --type quality --open

# Custom output directory
boosters-eval report --suite full --output ./custom-reports/

# === BASELINE MANAGEMENT ===
# Record baseline
boosters-eval baseline record \
  --suite full \
  --seeds 5 \
  --output baselines/v0.2.0.json

# Check against baseline
boosters-eval baseline check \
  --baseline baselines/v0.1.0.json \
  --tolerance 0.02 \
  --fail-on-regression

# === UTILITIES ===
# List available datasets/libraries/suites
boosters-eval list datasets
boosters-eval list libraries
boosters-eval list suites

# Control verbosity
boosters-eval run --suite quick --quiet      # Minimal output (CI)
boosters-eval run --suite quick --verbose    # Detailed progress
```

### CLI Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `--output`, `-o` | Output file path | stdout |
| `--format`, `-f` | Output format: `markdown`, `json`, `csv` | `markdown` |
| `--seeds`, `-s` | Number of random seeds | 5 |
| `--timing-mode` | Enable timing mode (implies `n_threads=auto`, `warmup=1`) | False |
| `--measure-memory` | Track peak memory usage via tracemalloc | False |
| `--warmup` | Number of warmup runs to discard (excluded from timing) | 0 |
| `--verbose`, `-v` | Detailed progress with Rich progress bars | False |
| `--quiet`, `-q` | Minimal output (CI-friendly) | False |
| `--continue-on-error` | Continue if a library crashes | False |
| `--highlight-significant` | Only highlight winners with significant lead (p<0.05) | True |
| `--open` | Auto-open generated report in browser/editor | False |
| `--dry-run` | Preview output without writing files | False |

### Command Groups

| Group | Commands | Description |
| ----- | -------- | ----------- |
| **Quick Checks** | `quick`, `full` | Shortcuts for regression testing (~30s / ~5min) |
| **Execution** | `run`, `compare` | Run benchmarks with flexible configuration |
| **Baselines** | `baseline record`, `baseline check` | Record and verify baseline results |
| **Reports** | `report` | Generate full reports to `docs/benchmarks/` |
| **Utilities** | `list` | Discover available datasets, libraries, suites |

### Exit Codes

| Code | Meaning | Example |
| ---- | ------- | ------- |
| 0 | Success | All checks passed |
| 1 | Regression detected | Quality degraded beyond tolerance |
| 2 | Execution error | Library crash, missing dependency |
| 3 | Configuration error | Invalid baseline file, unknown dataset |

### Configuration Precedence

Configuration is resolved in order (first wins):

1. **CLI arguments** — `--seeds 5`, `--tolerance 0.02`
2. **Environment variables** — `BOOSTERS_EVAL_THREADS`, `BOOSTERS_EVAL_SEEDS`
3. **Suite defaults** — Defined in code (see Suite Configuration)

**Environment variables:**

| Variable | Description |
| -------- | ----------- |
| `BOOSTERS_EVAL_THREADS` | Override thread count for timing mode |
| `BOOSTERS_EVAL_SEEDS` | Default seed count |
| `BOOSTERS_EVAL_BASELINE_DIR` | Default baseline directory |

### CLI Implementation

```python
import typer
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(name="boosters-eval", help="Evaluate and compare gradient boosting libraries.")

@app.command()
def compare(
    datasets: list[str] = typer.Option(["california"], "-d", "--dataset"),
    libraries: list[str] = typer.Option(None, "-l", "--library"),
    booster: str = typer.Option("gbdt", "-b", "--booster"),
    seeds: int = typer.Option(5, "-s", "--seeds"),
    output: Path | None = typer.Option(None, "-o", "--output"),
    format: str = typer.Option("markdown", "-f", "--format"),
    timing_mode: bool = typer.Option(False, "--timing-mode"),
    warmup: int = typer.Option(0, "--warmup"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Compare libraries on selected datasets."""
    ...

@app.command()
def run(
    suite: str = typer.Option("quick", help="Benchmark suite to run."),
    output: Path | None = typer.Option(None, "-o", "--output"),
    format: str = typer.Option("markdown", "-f", "--format"),
) -> None:
    """Run a predefined benchmark suite."""
    ...

baseline_app = typer.Typer()
app.add_typer(baseline_app, name="baseline")

@baseline_app.command("record")
def baseline_record(
    suite: str = typer.Option("full"),
    seeds: int = typer.Option(5),
    output: Path = typer.Argument(...),
) -> None:
    """Record benchmark results as a baseline."""
    ...

@baseline_app.command("check")
def baseline_check(
    baseline: Path = typer.Argument(...),
    tolerance: float = typer.Option(0.02),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression"),
) -> None:
    """Check current results against a baseline."""
    ...
```

---

## Package Structure

```text
packages/boosters-eval/
├── pyproject.toml
├── README.md
├── baselines/                  # Committed baseline files
│   ├── main.json               # Latest stable baseline
│   └── v0.1.0.json             # Version-specific baselines
└── src/boosters_eval/
    ├── __init__.py
    ├── cli.py                  # Typer CLI application
    ├── config.py               # Configuration dataclasses
    ├── datasets/
    │   ├── __init__.py
    │   ├── registry.py         # Dataset registration
    │   ├── sklearn_datasets.py # sklearn-based loaders
    │   └── synthetic.py        # Synthetic data generators
    ├── runners/
    │   ├── __init__.py
    │   ├── base.py             # Runner protocol
    │   ├── boosters_runner.py
    │   ├── xgboost_runner.py
    │   ├── lightgbm_runner.py
    │   └── catboost_runner.py
    ├── metrics.py              # Metric computation
    ├── results.py              # Result dataclasses
    ├── baseline.py             # Baseline recording/checking
    ├── suite.py                # Benchmark suite orchestration
    └── reports/
        ├── __init__.py
        ├── markdown.py         # Markdown report generation
        └── json.py             # JSON export
```

---

## Design Decisions

### DD-1: Canonical Parameter Names

**Decision**: Use a single set of parameter names (boosters naming) as canonical,
with each runner translating to library-specific equivalents.

**Rationale**: Ensures fair comparison by making configuration explicit. Parameter
semantic differences (e.g., XGBoost's `min_child_weight` vs LightGBM's `min_data_in_leaf`)
are documented in the translation layer.

### DD-2: sklearn for Metrics

**Decision**: Use sklearn for all metric computation rather than library-specific metrics.

**Rationale**: Ensures metrics are computed identically across libraries. Library-specific
metrics might have subtle implementation differences.

### DD-3: JSON Baselines

**Decision**: Store baselines as JSON files committed to the repository.

**Rationale**: Enables version control, diffing, and history. JSON is human-readable
and easily parsed by CI tools.

### DD-4: Tolerance-Based Regression Detection

**Decision**: Use percentage-based tolerance (default 2%) rather than statistical tests.

**Rationale**: Simple and interpretable. Statistical tests require many seeds to be
reliable, which increases benchmark time. 2% tolerance balances sensitivity and stability.

**Justification for 2% default**: Empirical testing with 5 seeds shows GBDT training
typically exhibits 0.5-1% variance in metric values due to random initialization. A 2%
threshold is 2-4× the typical variance, preventing false positives while catching
meaningful regressions. For high-precision needs, reduce tolerance and increase seeds.

### DD-5: Multiple Seeds with Aggregation

**Decision**: Run each benchmark with multiple seeds and report mean ± std.

**Rationale**: Reduces noise from random initialization. Standard deviation helps
identify unstable configurations.

### DD-6: Boosters Runner First-Class

**Decision**: Include a dedicated boosters runner, not just compare third-party libraries.

**Rationale**: The primary purpose is evaluating boosters quality. Having boosters as
a first-class citizen ensures the API and runner are well-tested.

### DD-7: Optional Dependencies for Runners

**Decision**: Each library (xgboost, lightgbm, catboost) is an optional dependency.

**Rationale**: Users can install only the libraries they want to compare. The CLI
gracefully handles missing libraries by skipping unavailable runners.

### DD-8: Depth-Mode Configuration Default

**Decision**: Use `max_depth` as the primary tree constraint for cross-library fairness.

**Rationale**: Depth-wise constraints are more portable across libraries. LightGBM's
leaf-wise growth with `num_leaves` is powerful but not directly comparable to
XGBoost's depth-wise approach. Using depth as the primary constraint gives
more consistent semantics, though it may not achieve each library's optimal
performance.

---

## Future Work

### v1.1

- [ ] HTML report generation with interactive charts
- [ ] K-fold cross-validation option for small datasets
- [ ] Sklearn GradientBoosting runner for reference comparisons
- [ ] Parallel benchmark execution across seeds
- [ ] Statistical regression detection (`--statistical` flag using t-test instead of tolerance)

### v1.2

- [ ] Ranking task support (NDCG@k, MAP metrics)
- [ ] Feature importance comparison across libraries
- [ ] SHAP value comparison
- [ ] GPU vs CPU ablation support
- [ ] Separation of BenchmarkExecutor and ReportGenerator classes

### v2.0

- [ ] Remote dataset support (download on demand)
- [ ] Cloud benchmark execution
- [ ] Historical trend analysis with visualization
- [ ] Integration with MLflow/W&B for tracking
- [ ] Plugin system for third-party runners via entry points

---

## CI Workflow Examples

### GitHub Actions: Full Workflow

```yaml
# .github/workflows/quality-regression.yml
name: Quality Regression Tests

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly full suite

jobs:
  quick-check:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install boosters-eval[all]
      
      - name: Run quick benchmarks
        run: |
          boosters-eval baseline check \
            --baseline tests/baselines/quick.json \
            --suite quick \
            --tolerance 0.02 \
            --continue-on-error
      
      - name: Upload results on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark-results.json

  full-suite:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install boosters-eval[all]
      
      - name: Run full benchmarks
        run: |
          boosters-eval baseline check \
            --baseline tests/baselines/full.json \
            --suite full \
            --tolerance 0.02
```

---

## Testing Strategy

### Built-in Test Suites

| Suite | Datasets | Boosters | Libraries | Seeds | Purpose |
| ----- | -------- | -------- | --------- | ----- | ------- |
| `minimal` | california, breast_cancer | gbdt | boosters | 1 | PR checks (<1 min) |
| `quick` | 3 datasets | gbdt | boosters, xgboost, lightgbm | 3 | Development |
| `full` | All datasets | All boosters | All libraries | 5 | Release validation |

### Unit Tests

- Configuration dataclass validation
- Parameter translation accuracy for each runner
- Metric computation correctness (vs sklearn reference)
- Baseline JSON parsing and validation
- Regression detection logic

### Integration Tests

- End-to-end benchmark execution (minimal suite)
- CLI command testing (all subcommands)
- Report generation validation (markdown, json, csv)
- Baseline recording and checking workflow
- Error handling when library crashes

### Report Tests

```python
def test_report_snapshot():
    """Report markdown output matches golden file."""
    results = load_fixture("sample_results.json")
    actual = render_report_template(results, mock_metadata(), "quality")
    expected = Path("tests/fixtures/expected_report.md").read_text()
    assert actual == expected

def test_report_json_schema():
    """Report JSON validates against schema."""
    results = run_minimal_benchmark()
    json_output = results.to_json(include_metadata=True)
    jsonschema.validate(json.loads(json_output), REPORT_SCHEMA)

def test_dry_run_no_write(tmp_path):
    """Dry-run mode does not create files."""
    result = runner.invoke(cli, ["report", "--suite", "minimal", "--dry-run"])
    assert result.exit_code == 0
    assert not list(tmp_path.iterdir())  # No files created
```

**Golden file updates:** When report format changes intentionally, update golden files:

```bash
# Update golden files for report tests
pytest tests/test_reports.py --update-golden

# Or manually regenerate
python -m boosters_eval.tests.fixtures generate
```

### Determinism Tests

```python
def test_determinism():
    """Running same config twice with same seed produces identical results."""
    r1 = run_benchmark(config, seed=42)
    r2 = run_benchmark(config, seed=42)
    assert r1.metrics == r2.metrics, "Results should be deterministic"
```

### Quality Sanity Tests

- Verify boosters achieves reasonable quality vs XGBoost/LightGBM
- Ensure regression detection catches intentional 5% degradations
- Verify tolerance threshold prevents flaky tests with natural variance

### Flaky Test Prevention

Benchmark tests use **deterministic seeding** to prevent flakiness:

```python
# Seeds are derived from config hash, not random
def derive_seed(base_seed: int, config_name: str, library: str) -> int:
    """Derive deterministic but uncorrelated seed."""
    return hash((base_seed, config_name, library)) % (2**32)
```

This ensures:

- Same config always uses same seeds across runs
- Different configs get uncorrelated seeds (no systematic bias)
- Reproducible results without storing all random states

**CI timeout guidance:** Quick suite should complete in ~30s. If it hangs beyond 5 minutes,
likely causes are: library installation issues, network problems (dataset download), or
infinite loops in custom code. CI uses 10-minute timeout as safety margin.

---

## Changelog

- 2025-12-27: Initial draft based on existing implementation
- 2025-12-27: Round 1 - Added schema versioning, regression definition, error handling
- 2025-12-27: Round 2 - Added fair comparison caveats, CLI options, test suites
- 2025-12-27: Round 3 - Added programmatic API, result filtering, baseline validation, edge cases
- 2025-12-27: Round 4 - Added Quick Start section, objective mapping function, timing-mode clarification, tolerance justification
- 2025-12-27: Round 5 - Added config file loading, seed derivation formula, baseline file conventions, updated future work
- 2025-12-27: Round 6 - Added CI workflow examples, expected runtimes, expected CLI output, final polish
- 2025-12-27: Revision - Code-based config (no YAML), statistical significance for highlighting, ablation suites, growth strategy testing, timing/memory measurement, `quick`/`full` CLI shortcuts
- 2025-12-27: Round 7 - Added report generation (`generate_report()`), machine fingerprinting (`MachineInfo`), dual output (markdown + JSON), `report` CLI command, report types documentation
- 2025-12-27: Round 8 - Added `--open` flag, CLI command groups table, runtime estimates in help, enhanced MachineInfo with BLAS/build detection, report snapshot tests, statistical regression as future work
- 2025-12-27: Round 9 - Added tolerance rationale, exit codes table, configuration precedence, environment variables, thread count documentation, memory measurement limitations, troubleshooting section, flaky test prevention
- 2025-12-27: Round 10 - Added example output paths, output directory conventions table, auto-creation note, dataset caching/small dataset caveats, golden file update mechanism, final status update
