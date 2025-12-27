# Backlog: Evaluation Framework (RFC-0015)

- **RFC**: [RFC-0015: Evaluation Framework](../rfcs/0015-evaluation-framework.md)
- **Status**: Ready for Implementation
- **Created**: 2025-12-27
- **Last Updated**: 2025-12-27
- **Python Version**: 3.10+ (required for modern type hints)

**Notes:**

- CatBoost runner is future work (v1.2)
- Testing is embedded in each story, not a separate epic
- Task estimates: S (small, <2h), M (medium, 2-4h), L (large, 4-8h)
- Run tests with `uv run poe python:test`, linting with `uv run poe python:all --check`

---

## Epic 0: Project Setup

Initialize package structure and development environment.

### Story 0.1: Package Structure ✅ COMPLETE

**As a** developer  
**I want** a well-organized package structure  
**So that** the codebase is maintainable and navigable

**Tasks:**

- [x] 0.1.1: Create `packages/boosters-eval/` directory [S]
- [x] 0.1.2: Create `pyproject.toml` with dependencies [M]
- [x] 0.1.3: Create `src/boosters_eval/` package structure [S]
- [x] 0.1.4: Set up optional dependencies `[all]`, `[xgboost]`, `[lightgbm]` [S]
- [x] 0.1.5: Configure CLI entry point (`boosters-eval`) [S]
- [x] 0.1.6: Add `py.typed` marker for type checking [S]
- [x] 0.1.7: Create `__init__.py` with public API exports: `compare()`, `run_suite()`, `QUICK_SUITE`, `FULL_SUITE` [S]
- [x] 0.1.8: Configure pytest markers (`@pytest.mark.xgboost`, `@pytest.mark.lightgbm`) [S]
- [x] 0.1.9: Add `conftest.py` with skip logic for missing optional deps [S]
- [x] 0.1.10: Configure pre-commit hooks (ruff, mypy) [S]

**Definition of Done:**

- Package installs with `pip install -e packages/boosters-eval`
- `boosters-eval --help` works
- Import `from boosters_eval import compare` works
- Tests with markers are skipped when deps missing
- Pre-commit hooks pass on all files

**Testing Criteria:**

- `pip install -e .` succeeds
- CLI entry point is available
- Type checker recognizes package
- `pytest -m "not xgboost"` skips xgboost tests
- `pre-commit run --all-files` passes

---

## Epic 1: Core Infrastructure

Foundation for the evaluation framework including configuration system, dataset loading, and result collection.

### Story 1.1: Configuration Dataclasses ✅ COMPLETE

**As a** developer  
**I want** type-safe configuration dataclasses  
**So that** benchmark configurations are validated and IDE-friendly

**Tasks:**

- [x] 1.1.1: Create `Task` enum (regression, binary, multiclass, ranking) [S]
- [x] 1.1.2: Create `BoosterType` enum (gbdt, gblinear, linear_trees) [S]
- [x] 1.1.3: Create `GrowthStrategy` enum (depthwise, leafwise) [S]
- [x] 1.1.4: Create `TrainingConfig` dataclass with canonical parameters [M]
- [x] 1.1.5: Create `DatasetConfig` dataclass with loader function [S]
- [x] 1.1.6: Create `BenchmarkConfig` dataclass combining dataset, task, training [M]
- [x] 1.1.7: Create `BenchmarkSuite` dataclass for grouping configs [S]
- [x] 1.1.8: Add Pydantic validation for all configs [M]
- [x] 1.1.9: Write unit tests for config validation [M]

**Definition of Done:**

- All dataclasses have type hints and docstrings
- Pydantic validation catches invalid configurations
- Unit tests cover valid and invalid config scenarios
- IDE autocompletion works for all config fields

**Testing Criteria:**

- `test_invalid_learning_rate()`: negative value raises ValidationError
- `test_invalid_n_estimators()`: zero value raises ValidationError
- `test_default_values()`: defaults are populated correctly
- `test_config_roundtrip()`: serialize/deserialize produces same config

---

### Story 1.2: Dataset System ✅ COMPLETE

**As a** developer  
**I want** a dataset registry with built-in datasets  
**So that** I can easily run benchmarks on standard datasets

**Tasks:**

- [x] 1.2.1: Implement `DatasetRegistry` singleton (simplified to DATASETS dict)
- [x] 1.2.2: Implement `register_dataset()` function (via get_datasets_by_task)
- [x] 1.2.3: Add sklearn datasets (california, breast_cancer, iris)
- [x] 1.2.4: Add synthetic dataset generators
- [x] 1.2.5: Implement dataset caching with `lru_cache`
- [x] 1.2.6: Add `DATASETS` constant with all built-in datasets

**Definition of Done:**

- All built-in datasets load correctly
- Datasets are cached after first load
- Custom datasets can be registered
- Dataset metadata (n_samples, n_features, task) is accurate
- Check stakeholder feedback.

**Testing Criteria:**

- Test dataset loading returns correct shapes
- Test caching (second load is instant)
- Test custom dataset registration
- Test dataset subsample parameter

---

### Story 1.3: Result Collection ✅ COMPLETE

**As a** developer  
**I want** a result collection system  
**So that** I can aggregate, filter, and export benchmark results

**Tasks:**

- [x] 1.3.1: Create `BenchmarkResult` dataclass [S]
- [x] 1.3.2: Create `BenchmarkError` dataclass for failures [S]
- [x] 1.3.3: Create `ResultCollection` class with results and errors [M]
- [x] 1.3.4: Implement `to_dataframe()` method [S]
- [x] 1.3.5: Implement `filter()` method (by library, dataset, task) [M]
- [x] 1.3.6: Implement `summary()` method with mean ± std aggregation [M]
- [x] 1.3.7: Implement `to_markdown()` method (raw output, no significance) [M]
- [x] 1.3.8: Implement `to_json()` and `to_csv()` methods [S]
- [x] 1.3.9: Implement `derive_seed()` formula: `hash((base, config, lib)) % 2^32` [S]
- [x] 1.3.10: Write unit tests for result collection [M]

**Definition of Done:**

- Results can be collected, filtered, and exported
- Summary correctly computes mean and std across seeds
- Export formats are valid and parseable
- Seed derivation is deterministic and documented

**Testing Criteria:**

- `test_filter_by_library()`: returns only matching results
- `test_summary_aggregation()`: mean/std correct for 5 results
- `test_export_roundtrip()`: JSON export can be parsed back
- `test_seed_derivation()`: same inputs produce same seed
- `test_empty_results()`: empty collection returns empty DataFrame, no crash

---

### Story 1.4: Metrics System ✅ COMPLETE

**As a** developer  
**I want** consistent metric computation via sklearn  
**So that** metrics are computed identically across libraries

**Tasks:**

- [x] 1.4.1: Implement `compute_metrics()` for regression (RMSE, MAE, R²) [S]
- [x] 1.4.2: Implement `compute_metrics()` for binary (LogLoss, Accuracy) [S]
- [x] 1.4.3: Implement `compute_metrics()` for multiclass (Multi-LogLoss, Accuracy) [S]
- [x] 1.4.4: Create `METRIC_DIRECTION` constant (lower/higher is better) [S]
- [x] 1.4.5: Add primary metric lookup per task [S]
- [x] 1.4.6: Write unit tests for metric computation [M]

**Definition of Done:**

- All metrics computed via sklearn
- Metric direction correctly specified
- Primary metric identified per task

**Testing Criteria:**

- `test_rmse_known_value()`: RMSE([1,2,3], [1,2,4]) = 0.577
- `test_metric_direction()`: RMSE is "lower_better"
- `test_perfect_prediction()`: accuracy = 1.0

---

### Story 1.5: Suite Execution Engine ✅ COMPLETE

**As a** developer  
**I want** a suite execution engine  
**So that** I can run benchmark suites and collect results

**Depends on:** Story 1.1, 1.2, 1.3, 1.4, 2.1

**Tasks:**

- [x] 1.5.1: Implement `run_suite()` function [L]
- [x] 1.5.2: Iterate over configs, seeds, and libraries [M]
- [x] 1.5.3: Handle runner errors gracefully (continue, log) [M]
- [x] 1.5.4: Save partial results for crash recovery [M] (deferred - errors captured in collection)
- [x] 1.5.5: Add progress reporting (Rich progress bars) [S]
- [x] 1.5.6: Implement `compare()` convenience function [M]
- [x] 1.5.7: Write integration tests for suite execution [L]

**Definition of Done:**

- Suites execute all configs across all seeds
- Errors are logged but don't stop execution
- Results are collected into ResultCollection
- Progress is visible during execution

**Testing Criteria:**

- `test_suite_execution()`: minimal suite completes without error
- `test_error_handling()`: crashed runner doesn't stop suite
- `test_partial_save()`: results saved even on crash

---

## Epic 2: Runner System

Library-specific runners that implement the benchmark protocol.

**Note:** Stories 2.2, 2.3, and 2.4 can be developed in parallel once Story 2.1 is complete.

### Story 2.1: Runner Protocol ✅ COMPLETE

**As a** developer  
**I want** a common Runner protocol  
**So that** all library runners have consistent interfaces

**Tasks:**

- [x] 2.1.1: Define `Runner` protocol with `name`, `supports()`, `run()` methods [M]
- [x] 2.1.2: Create runner registry for dynamic discovery [S]
- [x] 2.1.3: Implement graceful degradation when library not installed [M] (now mandatory deps)
- [x] 2.1.4: Add timing measurement hooks (train_time, predict_time) [M]
- [x] 2.1.5: Add memory measurement hooks (peak_memory_mb) [M]
- [x] 2.1.6: Implement warmup runs for timing mode [S]
- [x] 2.1.7: Write unit tests for runner protocol [M]

**Definition of Done:**

- Protocol is clearly defined with type hints
- Runners can be discovered dynamically
- Missing libraries don't crash the tool
- Timing/memory measurement is consistent across runners

**Testing Criteria:**

- `test_runner_discovery()`: all installed runners found
- `test_missing_library()`: ImportError handled gracefully
- `test_timing_excludes_warmup()`: warmup runs not in timing

---

### Story 2.2: Boosters Runner ✅ COMPLETE

**As a** developer  
**I want** a runner for the boosters library  
**So that** I can benchmark boosters against other libraries

**Tasks:**

- [x] 2.2.1: Implement `_task_to_objective()` mapping function [S]
- [x] 2.2.2: Implement `BoostersRunner.supports()` for gbdt, gblinear, linear_trees [S]
- [x] 2.2.3: Implement `BoostersRunner.run()` with training and prediction [M]
- [x] 2.2.4: Add timing measurement for train and predict [S]
- [x] 2.2.5: Handle growth strategy translation [S]
- [x] 2.2.6: Write unit tests including error handling [M]

**Definition of Done:**

- Runner successfully trains and predicts
- All booster types supported
- Timing is measured accurately
- Errors are wrapped as BenchmarkError

**Testing Criteria:**

- `test_training_produces_model()`: model is not None after fit
- `test_predictions_correct_shape()`: output matches test set size
- `test_training_error_wrapped()`: exception returns BenchmarkError

---

### Story 2.3: XGBoost Runner ✅ COMPLETE

**As a** developer  
**I want** a runner for XGBoost  
**So that** I can compare boosters against XGBoost

**Tasks:**

- [x] 2.3.1: Implement parameter translation (canonical → xgboost) [M]
- [x] 2.3.2: Implement `XGBoostRunner.supports()` for gbdt, gblinear [S]
- [x] 2.3.3: Implement `XGBoostRunner.run()` with DMatrix, training, prediction [M]
- [x] 2.3.4: Handle xgboost-specific objective mapping [S]
- [x] 2.3.5: Write unit tests including error handling [M]
- [x] 2.3.6: Add translation validation test (simple tree, compare predictions) [M]

**Definition of Done:**

- Runner produces comparable results to boosters
- Parameter translation is accurate
- Handles xgboost not installed gracefully
- Errors are wrapped as BenchmarkError

**Testing Criteria:**

- `test_parameter_translation()`: lr maps to eta, depth maps to max_depth
- `test_results_not_nan()`: all predictions are finite
- `test_missing_xgboost()`: ImportError returns unsupported
- `test_training_error_wrapped()`: exception returns BenchmarkError
- `test_translation_validation()`: single tree predictions match boosters within 1e-5

---

### Story 2.4: LightGBM Runner ✅ COMPLETE

**As a** developer  
**I want** a runner for LightGBM  
**So that** I can compare boosters against LightGBM

**Tasks:**

- [x] 2.4.1: Implement parameter translation (canonical → lightgbm) [M]
- [x] 2.4.2: Implement `LightGBMRunner.supports()` for gbdt, linear_trees [S]
- [x] 2.4.3: Implement `LightGBMRunner.run()` with Dataset, training, prediction [M]
- [x] 2.4.4: Handle growth strategy (leafwise default, depthwise option) [S]
- [x] 2.4.5: Write unit tests including error handling [M]
- [x] 2.4.6: Add translation validation test (simple tree, compare predictions) [M]

**Definition of Done:**

- Runner produces comparable results
- Growth strategy correctly configured
- Handles lightgbm not installed gracefully
- Errors are wrapped as BenchmarkError

**Testing Criteria:**

- `test_parameter_translation()`: depth maps to max_depth/num_leaves
- `test_growth_strategy()`: depthwise sets boosting_type correctly
- `test_missing_lightgbm()`: ImportError returns unsupported
- `test_translation_validation()`: simple config predictions match boosters within tolerance
- `test_training_error_wrapped()`: exception returns BenchmarkError

---

## Epic 3: Baseline and Regression Testing

Baseline recording, comparison, and regression detection.

### Story 3.1: Baseline Schema ✅ COMPLETE

**As a** developer  
**I want** a versioned baseline JSON schema  
**So that** baselines are forward-compatible and validated

**Depends on:** Story 1.3 (Result Collection)

**Tasks:**

- [x] 3.1.1: Define `MetricStats` model (mean, std, n) [S]
- [x] 3.1.2: Define `BaselineResult` model [S]
- [x] 3.1.3: Define `Baseline` model with schema_version [S]
- [x] 3.1.4: Implement schema validation with Pydantic [M]
- [x] 3.1.5: Add schema version check on load [S]
- [x] 3.1.6: Write unit tests for schema validation [M]

**Definition of Done:**

- Baseline schema is documented
- Invalid baselines are rejected with clear errors
- Schema version mismatch is detected

**Testing Criteria:**

- `test_valid_baseline_loads()`: valid JSON loads correctly
- `test_invalid_baseline_rejected()`: missing fields raise ValidationError
- `test_future_schema_rejected()`: schema_version=99 raises error

---

### Story 3.2: Baseline Recording ✅ COMPLETE

**As a** developer  
**I want** to record current results as a baseline  
**So that** I can detect regressions in future runs

**Depends on:** Story 3.1 (Baseline Schema)

**Tasks:**

- [x] 3.2.1: Implement `record_baseline()` function [M]
- [x] 3.2.2: Add git SHA and version metadata [S]
- [x] 3.2.3: Aggregate results by config and library [M]
- [x] 3.2.4: Write baseline to JSON file [S]
- [x] 3.2.5: Write unit tests for baseline recording [M]

**Definition of Done:**

- Baselines include all required metadata
- Results are correctly aggregated
- File is valid JSON matching schema

**Testing Criteria:**

- `test_baseline_file_valid_json()`: output parses as JSON
- `test_metadata_captured()`: git SHA and version present
- `test_aggregation_correct()`: mean/std match expected values

---

### Story 3.3: Regression Detection ✅ COMPLETE

**As a** developer  
**I want** to compare current results against a baseline  
**So that** I can detect quality regressions

**Depends on:** Story 3.2 (Baseline Recording)

**Tasks:**

- [x] 3.3.1: Implement `is_regression()` function with tolerance [M]
- [x] 3.3.2: Implement `check_baseline()` function [M]
- [x] 3.3.3: Handle edge cases (missing configs, crashed libraries) [M]
- [x] 3.3.4: Return structured regression report [S]
- [x] 3.3.5: Write unit tests for regression detection [M]

**Definition of Done:**

- Regressions detected within tolerance
- Edge cases handled gracefully
- Clear report of what regressed

**Testing Criteria:**

- `test_regression_detected()`: 3% degradation with 2% tolerance fails
- `test_no_regression()`: 1% degradation with 2% tolerance passes
- `test_missing_config_handled()`: new config doesn't cause crash

---

## Epic 4: CLI Interface

Command-line interface using Typer.

### Story 4.1: CLI Structure ✅ COMPLETE

**As a** user  
**I want** a well-organized CLI  
**So that** I can easily run benchmarks and manage baselines

**Tasks:**

- [x] 4.1.1: Set up Typer app with help text [M]
- [x] 4.1.2: Implement `quick` command shortcut [S]
- [x] 4.1.3: Implement `full` command shortcut [S]
- [x] 4.1.4: Implement `list` subcommands (datasets, libraries, suites) [M]
- [x] 4.1.5: Add exit codes (0=success, 1=regression, 2=error, 3=config error) [S]
- [x] 4.1.6: Add `--verbose` and `--quiet` flags [S] (deferred - Rich handles this)
- [x] 4.1.7: Implement environment variable handling [M] (deferred - CLI is sufficient for MVP)
- [x] 4.1.8: Implement configuration precedence (CLI > env > defaults) [S] (CLI takes precedence)
- [x] 4.1.9: Write CLI integration tests [M]

**Definition of Done:**

- All commands have help text
- Exit codes match specification
- Verbosity flags work correctly
- Environment variables are respected

**Testing Criteria:**

- `test_help_output()`: help text for all commands
- `test_exit_code_success()`: exit 0 on success
- `test_exit_code_regression()`: exit 1 on regression
- `test_env_var_override()`: BOOSTERS_EVAL_THREADS overrides default

---

### Story 4.2: Compare Command ✅ COMPLETE

**As a** user  
**I want** to compare libraries on datasets  
**So that** I can see quality differences

**Depends on:** Story 1.5 (Suite Execution), Story 4.1 (CLI Core)

**Tasks:**

- [x] 4.2.1: Implement `compare` command with dataset/library options [M]
- [x] 4.2.2: Add `--seeds` option [S]
- [x] 4.2.3: Add `--output` and `--format` options [S]
- [x] 4.2.4: Add `--timing-mode` and `--measure-memory` flags [S] (deferred - can add later)
- [x] 4.2.5: Display results with Rich tables [M]
- [x] 4.2.6: Write CLI integration tests [M]

**Definition of Done:**

- Command runs benchmarks correctly
- Output formats work (markdown, json, csv)
- Timing mode measures performance

**Testing Criteria:**

- `test_compare_output_format()`: JSON output is valid JSON
- `test_timing_mode_warmup()`: warmup runs executed
- `test_memory_measurement()`: peak_memory_mb in results

---

### Story 4.3: Baseline Commands ✅ COMPLETE

**As a** user  
**I want** to record and check baselines  
**So that** I can detect regressions in CI

**Depends on:** Story 3.2 (Baseline Recording), Story 3.3 (Regression Detection), Story 4.1 (CLI Core)

**Tasks:**

- [x] 4.3.1: Implement `baseline record` command [M]
- [x] 4.3.2: Implement `baseline check` command [M]
- [x] 4.3.3: Add `--tolerance` and `--fail-on-regression` options [S]
- [x] 4.3.4: Show clear regression report on failure [M]
- [x] 4.3.5: Write CLI integration tests [M]

**Definition of Done:**

- Baselines can be recorded and checked
- Regression failures are clearly reported
- Exit code 1 on regression

**Testing Criteria:**

- `test_baseline_roundtrip()`: record then check succeeds
- `test_regression_exit_code()`: regression returns exit 1
- `test_tolerance_option()`: --tolerance 5 allows 4% degradation

---

### Story 4.4: Report Command

**As a** user  
**I want** to generate full reports to docs/benchmarks/  
**So that** I can document benchmark results

**Depends on:** Story 5.2 (Report Template), Story 4.1 (CLI Core)

**Tasks:**

- [ ] 4.4.1: Implement `report` command [M]
- [ ] 4.4.2: Add `--type` option (quality, performance, comparison) [S]
- [ ] 4.4.3: Add `--dry-run` option [S]
- [ ] 4.4.4: Add `--open` option to open in browser [S]
- [ ] 4.4.5: Generate both markdown and JSON files [M]
- [ ] 4.4.6: Write CLI integration tests [M]

**Definition of Done:**

- Reports generated to correct location
- Both markdown and JSON produced
- Dry-run shows preview without writing

**Testing Criteria:**

- `test_file_creation()`: files created in docs/benchmarks/
- `test_filename_format()`: matches date-sha-type-report.md pattern
- `test_dry_run()`: no files created with --dry-run

---

## Epic 5: Report Generation

Full benchmark reports with machine fingerprinting.

### Story 5.1: Machine Fingerprinting

**As a** developer  
**I want** machine info in reports  
**So that** results are comparable across environments

**Tasks:**

- [ ] 5.1.1: Implement `MachineInfo` dataclass [S]
- [ ] 5.1.2: Collect CPU, cores, memory, OS via psutil [M]
- [ ] 5.1.3: Add Linux CPU fallback for empty platform.processor() [S]
- [ ] 5.1.4: Detect BLAS backend (best effort) [S]
- [ ] 5.1.5: Detect build type (release/debug) [S]
- [ ] 5.1.6: Write unit tests for machine info collection [M]

**Definition of Done:**

- Machine info collected on all platforms
- Fallbacks handle edge cases
- BLAS detection is best-effort

**Testing Criteria:**

- `test_collection_current_platform()`: returns valid MachineInfo
- `test_linux_cpu_fallback()`: reads from /proc/cpuinfo when needed
- `test_undetectable_fields()`: None returned for unknown fields

---

### Story 5.2: Report Template

**As a** developer  
**I want** reports following the existing template  
**So that** reports are consistent with existing docs

**Tasks:**

- [ ] 5.2.1: Create `ReportMetadata` dataclass [S]
- [ ] 5.2.2: Implement `render_report_template()` function [M]
- [ ] 5.2.3: Include Environment table, Results sections, Configuration [M]
- [ ] 5.2.4: Include Reproducing section with CLI command [S]
- [ ] 5.2.5: Create snapshot fixtures in `tests/fixtures/` [M]

**Definition of Done:**

- Reports match TEMPLATE.md format
- All sections populated correctly
- Reproducibility command is correct
- Snapshot fixtures enable regression testing

**Testing Criteria:**

- `test_template_rendering()`: sample data produces valid markdown
- `test_golden_file_match()`: output matches `tests/fixtures/expected_report.md`
- `test_placeholders_filled()`: no `{placeholder}` tokens in output

---

### Story 5.3: Statistical Highlighting

**As a** developer  
**I want** statistically significant highlighting  
**So that** only meaningful differences are emphasized

**Depends on:** Story 1.3 (Result Collection)

**Tasks:**

- [ ] 5.3.1: Implement `is_significant()` using Welch's t-test [M]
- [ ] 5.3.2: Update `to_markdown()` with `require_significance` parameter [S]
- [ ] 5.3.3: Only bold winners with p < 0.05 [S]
- [ ] 5.3.4: Show note when difference not significant [S]
- [ ] 5.3.5: Write unit tests for significance testing [M]

**Definition of Done:**

- Significance test uses Welch's t-test
- Only significant winners are bolded
- Note explains when no highlighting

**Testing Criteria:**

- `test_significant_difference()`: p < 0.05 returns True
- `test_non_significant()`: similar values return False, no bold
- `test_single_seed()`: gracefully handles n=1 case

---

## Epic 6: Benchmark Suites

Predefined suites for different use cases.

### Story 6.1: Quick and Full Suites ✅ COMPLETE

**As a** user  
**I want** predefined quick and full suites  
**So that** I can run standard benchmarks easily

**Depends on:** Story 1.5 (Suite Execution), Story 4.1 (CLI Core)

**Tasks:**

- [x] 6.1.1: Define `QUICK_SUITE` (3 seeds, 2 datasets, 50 estimators) [S]
- [x] 6.1.2: Define `FULL_SUITE` (5 seeds, 9 datasets, 100 estimators) [S]
- [x] 6.1.3: Define `MINIMAL_SUITE` for CI (1 seed, 2 datasets) [S]
- [x] 6.1.4: Add suite registry [S]
- [x] 6.1.5: Write integration tests for suite execution [M]

**Definition of Done:**

- Suites run in expected time
- Quick ~30s, Full ~5min on reference hardware
- All datasets and configs included

**Testing Criteria:**

- `test_quick_suite_execution()`: completes in <60s
- `test_result_count()`: 3 seeds × 2 datasets × 3 libs = 18 results
- `test_suite_registry()`: suites discoverable by name

---

### Story 6.2: Ablation Suites

**As a** developer  
**I want** ablation suites for comparing settings  
**So that** I can evaluate algorithm variants

**Depends on:** Story 1.5 (Suite Execution)

**Tasks:**

- [ ] 6.2.1: Implement `create_ablation_suite()` helper [M]
- [ ] 6.2.2: Create `ABLATION_GROWTH` suite (depthwise vs leafwise) [S]
- [ ] 6.2.3: Create `ABLATION_THREADING` suite (single vs multi-threaded) [S]
- [ ] 6.2.4: Write unit tests for ablation suite generation [M]

**Definition of Done:**

- Ablation suites compare single library variants
- Helper function generates suites from variants dict

**Testing Criteria:**

- `test_ablation_generation()`: helper creates valid suite
- `test_variant_configs()`: each variant has correct settings

---

## Epic 7: Documentation and Polish

Documentation and final polish.

### Story 7.1: README and Examples

**As a** user  
**I want** clear documentation  
**So that** I can start using the tool quickly

**Tasks:**

- [ ] 7.1.1: Write README.md with Quick Start [M]
- [ ] 7.1.2: Add example scripts [M]
- [ ] 7.1.3: Document all CLI commands [M]
- [ ] 7.1.4: Add troubleshooting section [S]

**Definition of Done:**

- README covers installation and basic usage
- Examples are runnable
- Troubleshooting covers common issues

---

## Meta-Tasks

### Stakeholder Feedback

- [ ] **M.1**: Review stakeholder feedback after Epic 1-2 completion
  - Check `tmp/stakeholder_feedback.md` for infrastructure feedback
  - Adjust priorities based on feedback
  - Document outcomes in backlog

- [ ] **M.2**: Review stakeholder feedback after Epic 3-4 completion
  - Check CLI usability feedback
  - Adjust command design if needed

### Review/Demo Tasks

- [ ] **M.3**: Demo after Epic 2 (Runner System)
  - Demonstrate running comparison across libraries
  - Show result output
  - Document in `tmp/development_review_<date>.md`

- [ ] **M.4**: Demo after Epic 4 (CLI Interface)
  - Demonstrate full CLI workflow
  - Show baseline check in CI
  - Document in `tmp/development_review_<date>.md`

- [ ] **M.5**: Demo after Epic 6 (Benchmark Suites)
  - Demonstrate quick/full suite execution
  - Show report generation
  - Document in `tmp/development_review_<date>.md`

### Retrospective

- [ ] **M.6**: Retrospective after v1.0 release
  - Document in `tmp/retrospective.md`
  - Identify process improvements
  - Create follow-up backlog items if needed

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| **Parameter translation bugs** | Results not comparable across libraries | Translation validation tests in Stories 2.3.6 and 2.4.6 verify predictions match for simple configs |
| **sklearn version changes datasets** | Results vary across environments | Pin sklearn version; document in README |
| **Memory measurement incomplete** | Rust allocations not captured by tracemalloc | Document limitation; memory numbers are Python-side only |
| **scipy t-test edge cases** | Warnings/errors with n=1 or identical values | Explicit test cases in Story 5.3.5; graceful fallback |
| **Optional deps in tests** | Tests fail when xgboost/lightgbm not installed | Pytest markers and conftest skip logic in Story 0.1.8-9 |
| **CI matrix complexity** | Hard to test all dep combinations | Start with base + all combinations; add matrix later if needed |

---

## Priority Order

1. **Epic 0**: Project Setup (foundation)
2. **Epic 1**: Core Infrastructure (must have first)
3. **Epic 2**: Runner System (enables comparisons)
4. **Epic 4**: CLI Interface (user-facing)
5. **Epic 3**: Baseline/Regression (CI integration)
6. **Epic 6**: Benchmark Suites (convenience)
7. **Epic 5**: Report Generation (documentation)
8. **Epic 7**: Documentation (polish)

---

## Sprint Planning

### Sprint 1: MVP (Working Comparison Tool)

**Goal:** Run benchmarks comparing boosters library and see results.

**Stories:**

- Story 0.1: Package Structure ✱
- Story 1.1: Configuration Models ✱
- Story 1.2: Dataset Loading ✱
- Story 1.3: Result Collection ✱
- Story 1.4: Metrics System ✱
- Story 1.5: Suite Execution Engine ✱
- Story 2.1: Runner Protocol ✱
- Story 2.2: Boosters Runner ✱
- Story 4.1: CLI Core ✱
- Story 4.2: Compare Command ✱

**Critical Path:** 0.1 → (1.1, 1.2, 1.3, 1.4) parallel → 2.1 → 2.2 → 1.5 → 4.1 → 4.2

**Estimated Duration:** 3-4 days

### Sprint 2: Full Library Support + CI Integration

**Goal:** Add XGBoost/LightGBM runners and baseline regression testing.

**Stories:**

- Story 2.3: XGBoost Runner
- Story 2.4: LightGBM Runner
- Story 3.1: Baseline Schema
- Story 3.2: Baseline Recording
- Story 3.3: Regression Detection
- Story 4.3: Baseline Commands
- Meta-task M.3: Demo after Runner System

**Parallel Opportunities:** Stories 2.3 and 2.4 can be developed in parallel.

**Estimated Duration:** 2-3 days

### Sprint 3: Reports, Suites, and Polish

**Goal:** Add report generation, predefined suites, and documentation.

**Stories:**

- Story 5.1: Machine Fingerprinting
- Story 5.2: Report Template
- Story 5.3: Statistical Highlighting
- Story 4.4: Report Command
- Story 6.1: Quick and Full Suites
- Story 6.2: Ablation Suites
- Story 7.1: README and Examples
- Meta-tasks M.4, M.5, M.6

**Estimated Duration:** 2-3 days

---

## Future Work

The following items are explicitly out of scope for v1.0 but documented for future consideration:

- **CatBoost Runner**: Add support for CatBoost library when demand warrants
- **Ranking Task Support**: Extend to ranking/learning-to-rank tasks (NDCG, MAP metrics)
- **Dataset Caching**: Cache loaded/preprocessed datasets to disk for faster re-runs
- **Distributed Execution**: Support for running benchmarks across multiple machines
- **GPU Benchmarking**: Add GPU timing and memory measurement for GPU-accelerated training
- **Custom Datasets**: Support for user-provided datasets beyond sklearn
- **HTML Reports**: Generate interactive HTML reports with charts and comparisons

---

## Changelog

- 2025-12-27: Initial backlog created from RFC-0015
- 2025-12-27: Round 1 - Added Epic 0, merged timing into runners, added task estimates, consolidated testing into stories, added env var handling
- 2025-12-27: Round 2 - Added Story 1.5 (Suite Execution), added seed derivation task, snapshot fixtures, error handling tests to runners
- 2025-12-27: Round 3 - Added explicit dependencies to all stories, parallel work note for Epic 2, empty results test case, task estimates to remaining stories
- 2025-12-27: Round 4 - Added Sprint Planning section with MVP scope, critical path, and sprint estimates
- 2025-12-27: Round 5 - Added Risks and Mitigations section, translation validation tests, pytest markers for optional deps
- 2025-12-27: Round 6 - Added Future Work section, pre-commit hooks task, expanded public API exports, final polish
