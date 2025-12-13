# RFC-0006: Evaluation

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: RFC-0001 (Data Matrix), RFC-0003 (Metrics)
- **Scope**: Training evaluation, logging, and early stopping

## Summary

Evaluation tracks model quality during training for monitoring, early stopping, and model selection. This RFC defines the Evaluator component that coordinates metric computation, early stopping logic, and result reporting. The evaluator returns **actions** that tell the training loop what to do, keeping evaluation logic separate from model management.

## Overview

### Component Hierarchy

```text
Evaluator
├── metric: EvalMetric               ← Metric for evaluation (RFC-0003)
├── eval_period: usize               ← Evaluate every N rounds
├── early_stopping: EarlyStoppingConfig
│   ├── enabled: bool
│   ├── patience: usize              ← Rounds without improvement
│   └── min_delta: f64               ← Minimum improvement threshold
└── state: EvaluatorState
    ├── best_value: f64
    ├── best_round: usize
    └── rounds_without_improvement: usize

EvalAction                            ← Returned by evaluate()
├── Skip                              ← Not an evaluation round
├── Continue { results }              ← Evaluated, no special action
├── NewBest { results }               ← New best score → checkpoint model
└── Stop { results }                  ← Early stopping triggered → break

EvalSet<'a>
├── dataset: &'a Dataset             ← Features, labels, weights
└── name: Option<String>             ← "train", "valid", custom

RoundEvalResults
├── round: usize
└── results: Vec<EvalResult>
    ├── name: String                 ← Dataset name
    ├── metric: String               ← Metric name
    ├── value: f64                   ← Computed value
    └── higher_is_better: bool       ← For interpretation
```

### Data Flow

```text
Training Loop
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  for round in 0..n_rounds:                                      │
│    train_step(...)                                              │
│                                                                 │
│    match evaluator.evaluate(round, &predictions):               │
│      Skip => { }                                                │
│      Continue { results } => { log(results) }                   │
│      NewBest { results } => { log(results); best = model.clone() }
│      Stop { results } => { log(results); break }                │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
best.unwrap_or(model) ──► Final Model
```

### Why Action-Based Design?

The evaluator returns an action enum instead of requiring multiple method calls:

```rust
// ❌ Old: Multiple calls, easy to forget steps
if let Some(results) = evaluator.maybe_evaluate(round, ...) {
    log(results);
    if evaluator.improved() {      // Easy to forget!
        best_model = model.clone();
    }
    if evaluator.should_stop() {   // Separate check
        break;
    }
}

// ✅ New: Single call, action tells you what to do
match evaluator.evaluate(round, &predictions) {
    EvalAction::Skip => {}
    EvalAction::Continue { results } => log(results),
    EvalAction::NewBest { results } => { log(results); best = model.clone(); }
    EvalAction::Stop { results } => { log(results); break; }
}
```

Benefits:

- **Can't forget**: Compiler ensures you handle all cases
- **Clear flow**: Action encodes both "what happened" and "what to do"
- **Separation of concerns**: Evaluator doesn't touch model, trainer handles cloning

## Components

### Evaluator

```rust
pub struct Evaluator {
    pub metric: EvalMetric,
    pub eval_period: usize,
    pub early_stopping: EarlyStoppingConfig,
    state: EvaluatorState,
}

pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub patience: usize,
    pub min_delta: f64,
}

struct EvaluatorState {
    best_value: f64,
    best_round: usize,
    rounds_without_improvement: usize,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            patience: 10,
            min_delta: 0.0,
        }
    }
}
```

### EvalAction

The action returned by `evaluate()` tells the training loop what to do:

```rust
pub enum EvalAction {
    /// Not an evaluation round (round % eval_period != 0).
    Skip,
    
    /// Evaluation completed, no special action needed.
    Continue { results: RoundEvalResults },
    
    /// New best score achieved. Caller should checkpoint the model.
    NewBest { results: RoundEvalResults },
    
    /// Early stopping triggered. Caller should break the training loop.
    /// Note: This is also a new best if it's the first evaluation.
    Stop { results: RoundEvalResults },
}

impl EvalAction {
    /// Get results if evaluation happened (not Skip).
    pub fn results(&self) -> Option<&RoundEvalResults> {
        match self {
            Self::Skip => None,
            Self::Continue { results } 
            | Self::NewBest { results } 
            | Self::Stop { results } => Some(results),
        }
    }
    
    /// Whether this action indicates a new best score.
    pub fn is_new_best(&self) -> bool {
        matches!(self, Self::NewBest { .. })
    }
    
    /// Whether training should stop.
    pub fn should_stop(&self) -> bool {
        matches!(self, Self::Stop { .. })
    }
}
```

### Core Interface

```rust
impl Evaluator {
    pub fn new(metric: EvalMetric, eval_period: usize, early_stopping: EarlyStoppingConfig) -> Self;
    
    /// Evaluate and return an action indicating what the caller should do.
    ///
    /// - `Skip`: Not an evaluation round
    /// - `Continue`: Evaluated, continue training
    /// - `NewBest`: New best score, caller should `model.clone()` for checkpoint
    /// - `Stop`: Early stopping triggered, caller should break
    pub fn evaluate<S: AsRef<[f32]>>(
        &mut self,
        round: usize,
        predictions: &[(&ColMatrix<f32>, &EvalSet<S>)],
    ) -> EvalAction;
    
    /// Round with best metric value.
    pub fn best_round(&self) -> usize;
    
    /// Best metric value seen so far.
    pub fn best_value(&self) -> f64;
    
    /// Reset state for new training run.
    pub fn reset(&mut self);
}
```

### EvalSet

```rust
/// Named dataset for evaluation.
pub struct EvalSet<'a, S: AsRef<[f32]> = Box<[f32]>> {
    pub dataset: &'a Dataset<'a, S>,
    pub name: Option<String>,
}

impl<'a, S: AsRef<[f32]>> EvalSet<'a, S> {
    pub fn new(dataset: &'a Dataset<'a, S>) -> Self {
        Self { dataset, name: None }
    }
    
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}
```

### Evaluation Results

```rust
/// Result of evaluating a single dataset.
pub struct EvalResult {
    pub name: String,
    pub metric: String,
    pub value: f64,
    pub higher_is_better: bool,
}

/// Results for one round across all datasets.
pub struct RoundEvalResults {
    pub round: usize,
    pub results: Vec<EvalResult>,
}
```

## Algorithms

### Evaluation Algorithm

```text
evaluate(round, predictions_and_eval_sets) -> EvalAction:
  // Skip if not evaluation round
  if round % eval_period != 0:
    return EvalAction::Skip
  
  // Compute metrics for all eval sets
  results = []
  for (predictions, eval_set) in predictions_and_eval_sets:
    value = metric.compute(
      predictions,
      eval_set.dataset.labels,
      eval_set.dataset.weights
    )
    
    results.push(EvalResult {
      name: eval_set.name.unwrap_or_else(|| auto_name(index)),
      metric: metric.name(),
      value,
      higher_is_better: metric.higher_is_better(),
    })
  
  round_results = RoundEvalResults { round, results }
  
  // Update early stopping with first eval set (typically validation)
  if !results.is_empty():
    improved = update_early_stopping(round, results[0].value)
  else:
    improved = false
  
  // Determine action
  if early_stopping.enabled && rounds_without_improvement >= patience:
    return EvalAction::Stop { results: round_results }
  else if improved:
    return EvalAction::NewBest { results: round_results }
  else:
    return EvalAction::Continue { results: round_results }
```

### Early Stopping Update

```text
update_early_stopping(round, value):
  improved = is_improvement(value, best_value, min_delta, higher_is_better)
  
  if improved:
    best_value = value
    best_round = round
    rounds_without_improvement = 0
  else:
    rounds_without_improvement += 1

is_improvement(new, best, min_delta, higher_is_better) -> bool:
  if higher_is_better:
    return new > best + min_delta
  else:
    return new < best - min_delta

should_stop() -> bool:
  return early_stopping.enabled && rounds_without_improvement >= patience
```

## Design Decisions

### DD-1: Action-Based Return Type

**Context**: How should the evaluator communicate what happened and what the caller should do?

**Decision**: Return `EvalAction` enum instead of separate `improved()` / `should_stop()` methods.

**Rationale**:

```rust
// Evaluator returns action, caller handles it
match evaluator.evaluate(round, &predictions) {
    EvalAction::Skip => {}
    EvalAction::Continue { results } => log(results),
    EvalAction::NewBest { results } => {
        log(results);
        best_model = Some(model.clone());  // Caller does the clone
    }
    EvalAction::Stop { results } => {
        log(results);
        break;
    }
}
```

- **Exhaustive**: `match` ensures caller handles all cases
- **Atomic**: One call returns both "what happened" and "what to do"
- **No forgotten steps**: Can't forget to check `improved()` or `should_stop()`
- **Clean separation**: Evaluator doesn't know about models, just returns actions

### DD-2: Checkpointing is Caller's Responsibility

**Context**: Who stores the best model checkpoint?

**Decision**: Training loop handles `model.clone()` when `NewBest` is returned.

**Rationale**:

- **Simple**: No `ModelCheckpointer` component needed
- **Type-safe**: Evaluator doesn't need to be generic over model type
- **Flexible**: Caller decides clone vs serialize vs nothing
- **Memory efficient**: Models are typically small enough to clone in-memory

```rust
// Trainer (generic over model type M: Clone)
let mut best_model: Option<M> = None;

for round in 0..n_rounds {
    // ... train ...
    match evaluator.evaluate(round, &predictions) {
        EvalAction::NewBest { .. } => best_model = Some(model.clone()),
        EvalAction::Stop { .. } => break,
        _ => {}
    }
}

best_model.unwrap_or(model)
```

### DD-3: First Eval Set for Early Stopping

**Context**: Which dataset's metric should drive early stopping?

**Decision**: Use the first eval set's metric.

**Rationale**:

- **User controls order**: Caller decides `[valid, train]` ordering
- **Typical pattern**: Validation first → early stopping uses validation
- **No special "validation" flag**: Simple positional semantics
- **Explicit**: User sees exactly which metric triggers stopping

```rust
// User controls early stopping dataset by ordering:
let eval_sets = [
    EvalSet::new(&valid_dataset).with_name("valid"),  // ← Used for early stopping
    EvalSet::new(&train_dataset).with_name("train"),
];
```

### DD-4: Simple eval_period Integer

**Context**: How to configure evaluation frequency?

**Decision**: Simple `eval_period: usize` (default 1).

**Rationale**:

| Alternative | Complexity | Benefit |
|-------------|------------|---------|
| `eval_period: usize` | Low | Covers 99% of use cases |
| `EvalSchedule` enum | Medium | Exponential backoff, etc. |
| `Fn(round) -> bool` | High | Full flexibility |

Simple integer works well:

- `eval_period = 1`: Every round (default, small datasets)
- `eval_period = 10`: Every 10 rounds (large datasets)

### DD-5: Structured Results, Caller Handles Logging

**Context**: Should Evaluator log directly?

**Decision**: Return `RoundEvalResults` in the action, let caller decide formatting.

**Rationale**:

```rust
// Caller controls logging via action matching:
match evaluator.evaluate(round, &predictions) {
    EvalAction::Skip => {}
    action => {
        if let Some(results) = action.results() {
            // Plain text
            for r in &results.results {
                println!("[{}] {}: {:.6}", results.round, r.metric, r.value);
            }
            // Or JSON, TensorBoard, etc.
        }
    }
}
```

- **Separation of concerns**: Evaluation ≠ presentation
- **Flexibility**: Plain text, JSON, TensorBoard, etc.
- **Testing**: Results are data, easy to assert on

### DD-6: Explicit Reset Method

**Context**: Can Evaluator be reused across training runs?

**Decision**: Yes, via `reset()` method.

**Rationale**:

```rust
evaluator.reset();  // Clear best_value, rounds_without_improvement
trainer.train(&mut evaluator);
```

- **Reusable**: Same evaluator for cross-validation folds
- **Explicit**: User calls `reset()` when starting new run
- **State visibility**: `best_round()`, `best_value()` reflect current run

### DD-7: Predictions Paired with EvalSet

**Context**: How to pass predictions for multiple eval sets?

**Decision**: Pass `&[(&ColMatrix, &EvalSet)]` pairs.

**Rationale**:

- **Explicit pairing**: Predictions and labels match by construction
- **Flexible**: Different prediction buffers per eval set
- **Training integration**: Trainer computes predictions per eval set

## Integration

### Training Loop Pattern

```rust
// Setup
let eval_sets = [
    (&valid_preds, EvalSet::new(&valid_ds).with_name("valid")),
    (&train_preds, EvalSet::new(&train_ds).with_name("train")),
];
let mut evaluator = Evaluator::new(EvalMetric::Rmse, 1, early_stopping);
let mut best_model: Option<M> = None;

// Training loop
for round in 0..n_rounds {
    train_step(&mut model, ...);
    
    // Update predictions for evaluation
    model.predict_into(&valid_ds.features, &mut valid_preds);
    model.predict_into(&train_ds.features, &mut train_preds);
    
    // Evaluate and act
    match evaluator.evaluate(round, &eval_sets) {
        EvalAction::Skip => {}
        EvalAction::Continue { results } => {
            log_results(&results);
        }
        EvalAction::NewBest { results } => {
            log_results(&results);
            best_model = Some(model.clone());
        }
        EvalAction::Stop { results } => {
            log_results(&results);
            break;
        }
    }
}

// Return best model
best_model.unwrap_or(model)
```

### Integration Points

| Component | Integration |
|-----------|-------------|
| RFC-0003 (Metrics) | `EvalMetric` for `metric.compute()` |
| RFC-0004 (GBLinear) | Calls `evaluate()` in training loop |
| RFC-0007 (GBTree) | Calls `evaluate()` in training loop |
| User Code | Matches on `EvalAction` for logging/checkpointing |

## Example Usage

```rust
// Configure evaluator with early stopping
let mut evaluator = Evaluator::new(
    EvalMetric::Auc,
    5,  // eval_period: every 5 rounds
    EarlyStoppingConfig {
        enabled: true,
        patience: 20,
        min_delta: 0.001,
    },
);

// Setup eval sets (validation first for early stopping)
let valid_eval = EvalSet::new(&valid_dataset).with_name("valid");
let train_eval = EvalSet::new(&train_dataset).with_name("train");

// Train with evaluation
let model = trainer.train(
    &dataset,
    &[valid_eval, train_eval],
    &mut evaluator,
)?;

// Query final state
println!("Best AUC {:.4} at round {}", 
    evaluator.best_value(),
    evaluator.best_round()
);
```

### Logging Helper

```rust
fn log_results(results: &RoundEvalResults) {
    print!("[{:4}]", results.round);
    for r in &results.results {
        print!("  {}: {:.6}", r.name, r.value);
    }
    println!();
}

// Output:
// [   0]  valid: 0.892341  train: 0.901234
// [   5]  valid: 0.923456  train: 0.945678
// ...
```

## Future Work

- [ ] Callback hooks for custom evaluation logic
- [ ] Multi-metric early stopping (stop when ANY metric plateaus)
- [ ] Metric history tracking for plotting
- [ ] Warm-starting from previous training runs

## References

- [XGBoost Early Stopping](https://xgboost.readthedocs.io/en/latest/python/callbacks.html)
- [LightGBM Early Stopping](https://lightgbm.readthedocs.io/en/latest/Parameters.html#early_stopping_round)
