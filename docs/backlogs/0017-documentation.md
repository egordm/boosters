# Backlog: RFC-0017 Documentation and READMEs

**RFC**: [docs/rfcs/0017-documentation.md](../rfcs/0017-documentation.md)  
**Created**: 2026-01-13  
**Status**: Draft

---

## Overview

Implement comprehensive documentation infrastructure for boosters:

- **Sphinx docs** (Python-centric): Main documentation with tutorials, getting started, API reference, explanations, and embedded research/RFCs
- **Rustdoc** (Rust-centric): Rust API reference documentation
- **READMEs**: Package-level quick starts linking to main docs
- **GitHub Actions**: Automated builds and GitHub Pages deployment

### Success Metrics

- Documentation builds without warnings (`sphinx-build -W` passes)
- All 9 tutorials execute without error
- Zero broken internal links
- API coverage: every public symbol in `boosters.__all__` documented
- Sphinx search returns relevant results for key terms
- Local development workflow functional (`make serve`)

### Out of Scope

- Video tutorials (future enhancement)
- Multi-version documentation (deferred to v1.0)
- Internationalization (i18n)
- Read the Docs integration (using GitHub Pages)

---

## Milestones

| Milestone | Epics | Effort | Description |
|-----------|-------|--------|-------------|
| M1: Infrastructure | Epic 1 | L | Sphinx setup, GitHub Actions, local dev workflow |
| M2: Core Content | Epics 2, 3 | XL | Getting started, API reference, explanations |
| M3: Tutorials | Epic 4 | XL | All 9 Jupyter notebook tutorials |
| M4: Polish | Epics 5, 6 | L | How-to guides, READMEs, final review |

### Parallelization Opportunities

- **Epic 2** (Getting Started + API) and **Epic 3** (Explanations) can run in parallel after Epic 1
- **Epic 4** (Tutorials) can start once Epic 1 is complete
- **Epic 5** (How-To Guides) can run in parallel with Epic 4
- **Epic 6** (READMEs) depends on core content being stable

### Effort Size Reference

| Size | Duration | Examples |
|------|----------|----------|
| S | 1-2 hours | Add single page, update config, fix links |
| M | 2-4 hours | Write tutorial notebook, create how-to guide |
| L | 4-8 hours | Set up Sphinx infrastructure, write explanations section |
| XL | 1-2 days | Complete all tutorials, full API documentation |

---

## Epic 1: Documentation Infrastructure

Set up Sphinx, GitHub Actions, and local development workflow.

### Story 1.1: Sphinx Project Setup [L]

Create the Sphinx documentation structure and configuration.

**Tasks**:

- [x] 1.1.1: Create `docs/pyproject.toml` for boosters-docs package with dependencies:
  - sphinx, pydata-sphinx-theme, nbsphinx, myst-parser
  - sphinx-design, sphinx-autobuild, sphinx-autodoc-typehints
- [x] 1.1.2: Create `docs/pyproject.toml` for boosters-docs package (see RFC for dependencies)
- [x] 1.1.3: Add `docs` to workspace members in root `pyproject.toml`
- [x] 1.1.4: Create `docs/conf.py` with:
  - pydata-sphinx-theme configuration
  - Extensions: autodoc, autosummary, napoleon, intersphinx, nbsphinx, myst_parser
  - Intersphinx mappings: numpy, pandas, sklearn, scipy
  - MathJax for LaTeX rendering
- [x] 1.1.5: Create `docs/index.rst` landing page with feature cards (sphinx-design)
- [x] 1.1.6: Add version notice banner for pre-1.0 status
- [x] 1.1.7: Create `docs/_static/` directory with placeholder logo
- [x] 1.1.8: Create `docs/_templates/` directory
- [x] 1.1.9: Create directory structure: `getting-started/`, `tutorials/`, `howto/`, `explanations/`, `api/`, `design/`, `contributing/`
- [x] 1.1.10: Verify `uv sync --package boosters-docs` installs dependencies
- [x] 1.1.11: Verify `uv run poe docs:build` builds without errors

**Definition of Done**:

- Sphinx builds successfully with pydata-sphinx-theme
- Landing page renders with feature cards
- Pre-1.0 notice visible
- Intersphinx links to external docs functional
- Directory structure matches RFC specification

### Story 1.2: Local Development Workflow [M]

Create poe tasks for local docs development.

**Tasks**:

- [x] 1.2.1: Add `docs:build` poe task: `sphinx-build -W -j auto docs docs/_build/html`
- [x] 1.2.2: Add `docs:serve` poe task: build then serve with `python -m http.server -d docs/_build/html 8000`
- [x] 1.2.3: Add `docs:watch` poe task using sphinx-autobuild for auto-rebuild on changes
- [x] 1.2.4: Add `docs:clean` poe task: `rm -rf docs/_build`
- [x] 1.2.5: Add `docs:linkcheck` poe task: `sphinx-build -b linkcheck docs docs/_build/linkcheck`
- [x] 1.2.6: Add `docs:coverage` poe task: `sphinx-build -b coverage docs docs/_build/coverage`
- [x] 1.2.7: Create `docs/.gitignore` for `_build/`, `*.pyc`, `__pycache__/`
- [x] 1.2.8: Verify `uv run poe docs:serve` starts local server at localhost:8000
- [x] 1.2.9: Verify `uv run poe docs:watch` auto-reloads on file changes

**Definition of Done**:

- Local development workflow via poe tasks functional
- `uv run poe docs:watch` enables rapid iteration with auto-reload
- No Makefile used (poe tasks only)

### Story 1.3: GitHub Actions Workflow [L]

Set up CI/CD for documentation builds and deployment.

**Tasks**:

- [x] 1.3.1: Create `.github/workflows/docs.yml` with build-sphinx and build-rustdoc jobs
- [x] 1.3.2: Add comprehensive caching:
  - uv cache: `~/.cache/uv`
  - cargo cache: `~/.cargo/registry`, `~/.cargo/git`, `target/`
  - Sphinx doctree cache: `docs/_build/.doctrees`
  - nbsphinx cache: pre-executed notebooks
- [x] 1.3.3: Add concurrency settings to cancel in-progress builds
- [x] 1.3.4: Configure GitHub Pages deployment (on main branch only)
- [x] 1.3.5: Add PR validation (build only, no deploy)
- [x] 1.3.6: Add link validation step (`docs/scripts/validate_links.py`)
- [ ] 1.3.7: Consider notebook pre-execution strategy (cache executed notebooks, only re-run on change)
- [ ] 1.3.8: Test workflow on a feature branch
- [ ] 1.3.9: Enable GitHub Pages in repository settings

**Definition of Done**:

- Docs build on every PR
- Docs deploy to GitHub Pages on main merge
- Build failures block PR merge
- Build time under 5 minutes with warm cache

### Story 1.4: RFC and Research Embedding [M]

Set up mechanism to embed existing RFCs and research docs.

**Tasks**:

- [x] 1.4.1: Create `docs/scripts/validate_links.py` for RFC cross-reference validation
- [x] 1.4.2: Create `docs/design/index.rst` with RFC toctree
- [x] 1.4.3: Create include wrappers for key RFCs (using myst `{include}` directive)
- [x] 1.4.4: Verify RFC math formulas render correctly (LaTeX via mathjax)
- [ ] 1.4.5: Test link validation script catches broken references

**Definition of Done**:

- RFCs accessible from Sphinx documentation
- Math formulas ($, $$) render correctly
- Cross-references validated in CI

### Story 1.5: Review and Demo (Epic 1) [S]

**Tasks**:

- [ ] 1.5.1: Demo: local docs build, GitHub Pages preview, RFC embedding
- [ ] 1.5.2: Verify pydata-sphinx-theme styling matches OpenSTEF aesthetic
- [ ] 1.5.3: Document any configuration changes needed

**Definition of Done**:

- Infrastructure demo completed
- Theme and styling approved

---

## Epic 2: Getting Started and API Reference

Create onboarding documentation and API reference.

### Story 2.1: Installation Page [M]

Document all installation methods.

**Tasks**:

- [x] 2.1.1: Create `docs/getting-started/index.rst` with overview
- [x] 2.1.2: Create `docs/getting-started/installation.rst` with:
  - pip installation
  - Building from source (maturin)
  - Troubleshooting section
  - Verification snippet
- [x] 2.1.3: Add platform-specific notes (GLIBC, Rust toolchain)
- [ ] 2.1.4: Test installation instructions on clean environment

**Definition of Done**:

- Installation page covers all methods
- Instructions verified to work

### Story 2.2: Python Quickstart [M]

Create Python quickstart guide.

**Tasks**:

- [x] 2.2.1: Create `docs/getting-started/quickstart-python.rst`
- [x] 2.2.2: Include 5-line GBDT training example (matches main README) with syntax highlighting (`.. code-block:: python`)
- [x] 2.2.3: Include sklearn estimator example
- [x] 2.2.4: Add "Next Steps" links to tutorials
- [ ] 2.2.5: Verify code examples execute correctly
- [ ] 2.2.6: Add examples as doctests (verified in CI)
- [x] 2.2.7: Ensure consistent code style (imports at top, snake_case variables)

**Definition of Done**:

- Quickstart is under 10 minutes to complete
- Code examples match README exactly
- Examples verified via doctest

### Story 2.3: Rust Quickstart [S]

Create Rust quickstart guide.

**Tasks**:

- [x] 2.3.1: Create `docs/getting-started/quickstart-rust.rst`
- [x] 2.3.2: Include cargo dependency snippet
- [x] 2.3.3: Include basic training and prediction example
- [x] 2.3.4: Link to Rustdoc for detailed API

**Definition of Done**:

- Rust quickstart is concise (<10 lines of code)
- Links to Rustdoc work correctly

### Story 2.4: Python API Reference [L]

Set up autodoc for Python API.

**Tasks**:

- [x] 2.4.1: Create `docs/api/index.rst` with API overview
- [x] 2.4.2: Create `docs/api/python/index.rst` with autosummary
- [x] 2.4.3: Configure autodoc to generate stubs for all public modules
- [x] 2.4.4: Integrate with existing `rust:build:python` task (stubs generated by pyo3-stub-gen)
- [ ] 2.4.5: Add runnable examples to key class docstrings (Dataset, GBDTModel, GBLinearModel)
- [x] 2.4.6: Verify type hints display correctly (sphinx-autodoc-typehints)
- [x] 2.4.7: Add intersphinx links to numpy, sklearn, pandas
- [ ] 2.4.8: Add search functionality verification (Sphinx built-in search works)

**Definition of Done**:

- All public API documented
- Type hints visible in generated docs
- Cross-references to external docs work
- Stubs from pyo3-stub-gen integrated

### Story 2.5: Rust API Reference [S]

Link to Rustdoc.

**Tasks**:

- [x] 2.5.1: Create `docs/api/rust.rst` with link to `/rustdoc/boosters/`
- [x] 2.5.2: Add brief overview of Rust API structure
- [ ] 2.5.3: Verify rustdoc builds and is accessible at correct URL

**Definition of Done**:

- Clear path from Sphinx to Rustdoc
- Rustdoc deployed alongside Sphinx

### Story 2.6: Review and Demo (Epic 2) [S]

**Tasks**:

- [ ] 2.6.1: Review API coverage (all public symbols documented)
- [ ] 2.6.2: Verify quickstart code examples work
- [ ] 2.6.3: Demo: navigate from getting started to API reference

**Definition of Done**:

- Getting started → API reference flow is smooth
- No undocumented public symbols

---

## Epic 3: Explanations and Theory

Create conceptual documentation explaining how boosters works.

### Story 3.1: Gradient Boosting Explanation [L]

Explain gradient boosting fundamentals.

**Tasks**:

- [x] 3.1.1: Create `docs/explanations/index.rst` with overview
- [x] 3.1.2: Create `docs/explanations/gradient-boosting.rst`
- [x] 3.1.3: Include or embed content from `docs/research/gradient-boosting.md`
- [x] 3.1.4: Include additive model formulation:
  $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
  where $h_m$ minimizes $\sum_i L(y_i, F_{m-1}(x_i) + h_m(x_i))$
- [x] 3.1.5: Explain Newton-Raphson context: second-order Taylor expansion leads to optimal leaf weights $w^* = -\frac{\sum g_i}{\sum H_i + \lambda}$
- [x] 3.1.6: Add ensemble visualization diagram showing: Tree 1 prediction + Tree 2 prediction + ... = Final prediction
- [x] 3.1.7: Add "Further Reading" with academic references (Friedman 2001, Chen & Guestrin 2016)

**Definition of Done**:

- Additive model formulation clearly explained
- Second-order optimization context covered
- Math renders correctly
- Links to source papers

### Story 3.2: GBDT and GBLinear Explanations [M]

Explain the two booster types.

**Tasks**:

- [x] 3.2.1: Create `docs/explanations/gbdt.rst` explaining tree-based boosting
- [x] 3.2.2: Create `docs/explanations/gblinear.rst` explaining linear boosting
- [x] 3.2.3: Include when-to-use-which guidance
- [x] 3.2.4: Link to research docs for implementation details

**Definition of Done**:

- Both model types explained
- Clear guidance on when to use each

### Story 3.3: Hyperparameters Guide [L]

Comprehensive hyperparameter documentation.

**Tasks**:

- [x] 3.3.1: Create `docs/explanations/hyperparameters.rst`
- [x] 3.3.2: Document tree structure parameters (max_depth, max_leaves, min_child_weight, min_split_loss)
- [x] 3.3.3: Document regularization parameters with mathematical context:
  - $\ell_2$ (reg_lambda): Penalizes $\lambda \|w\|_2^2$, prevents coefficient explosion, shrinks toward zero
  - $\ell_1$ (reg_alpha): Penalizes $\alpha \|w\|_1$, promotes sparsity via subdifferential at zero
- [x] 3.3.4: Document learning parameters (learning_rate, n_estimators, early_stopping)
- [x] 3.3.5: Document subsampling parameters (subsample, colsample_*)
- [x] 3.3.6: Document GBLinear parameters (feature_selector, alpha, lambda)
- [x] 3.3.7: For each parameter: What, How, Why, Trade-offs
- [x] 3.3.8: Include interaction effects (e.g., depth vs n_estimators, learning_rate vs n_estimators)
- [x] 3.3.9: Add bias-variance perspective: how depth increases variance, regularization reduces it
- [x] 3.3.10: Create hyperparameter reference table summarizing all parameters with defaults, ranges, and when to tune

**Definition of Done**:

- All hyperparameters documented with mathematical context
- Trade-offs clearly explained
- Follows What/How/Why/Trade-offs format
- Regularization theory explained
- Quick reference table included

### Story 3.4: Objectives and Metrics [M]

Document available objectives and metrics.

**Tasks**:

- [x] 3.4.1: Create `docs/explanations/objectives-metrics.rst`
- [x] 3.4.2: Document each objective with formula, gradient ($g$), and Hessian ($H$):
  - **Squared Error**: $L = \frac{1}{2}(y - \hat{y})^2$, $g = \hat{y} - y$, $H = 1$
  - **Logistic**: $L = -y\log(\sigma(\hat{y})) - (1-y)\log(1-\sigma(\hat{y}))$, $g = \sigma(\hat{y}) - y$, $H = \sigma(\hat{y})(1-\sigma(\hat{y}))$
  - **Softmax**: For class $k$: $g_k = p_k - \mathbf{1}_{y=k}$, $H_k = p_k(1-p_k)$
  - **Huber**: Piecewise linear-quadratic with $\delta$ threshold
  - **Quantile**: $L = (y - \hat{y})(\alpha - \mathbf{1}_{y < \hat{y}})$
- [x] 3.4.3: Explain why Hessian matters: determines leaf weight via $w^* = -G/(H + \lambda)$
- [x] 3.4.4: Document each metric with formula
- [x] 3.4.5: Include usage examples for custom objectives

**Definition of Done**:

- All objectives and metrics documented with formulas
- Gradient/Hessian shown for each objective
- Connection to optimal leaf weights explained

### Story 3.5: Benchmarks Explanation [M]

Document benchmarking and boosters-eval usage.

**Tasks**:

- [x] 3.5.1: Create `docs/explanations/benchmarks.rst`
- [x] 3.5.2: Link to latest benchmark results from `docs/benchmarks/`
- [x] 3.5.3: Document boosters-eval CLI usage
- [x] 3.5.4: Explain how to interpret benchmark results
- [ ] 3.5.5: Include performance comparison tables

**Definition of Done**:

- Benchmarks accessible from docs
- boosters-eval usage documented

### Story 3.6: Review and Demo (Epic 3) [S]

**Tasks**:

- [ ] 3.6.1: Verify all math renders correctly
- [ ] 3.6.2: Review hyperparameter coverage against XGBoost/LightGBM docs
- [ ] 3.6.3: Demo: navigate explanation sections

**Definition of Done**:

- Theory documentation complete
- Math rendering verified

---

## Epic 4: Tutorials

Create 9 Jupyter notebook tutorials.

### Story 4.1: Basic Training Tutorial [M]

Tutorial 01: Python Basic GBDT Training.

**Tasks**:

- [x] 4.1.1: Create `docs/tutorials/index.rst` with tutorial overview, difficulty legend, and download links
- [x] 4.1.2: Create `docs/tutorials/01-basic-training.ipynb`
- [x] 4.1.3: Cover: load data, create Dataset, configure model, train, predict, evaluate
- [x] 4.1.4: Set random seed for reproducibility
- [x] 4.1.5: Include learning curve plot
- [x] 4.1.6: Add "Download notebook" link via nbsphinx configuration
- [x] 4.1.7: Verify notebook executes cleanly with nbsphinx

**Definition of Done**:

- Tutorial executes without error
- Output is deterministic
- Download link available
- Difficulty badge: 🟢 Beginner

### Story 4.2: sklearn Integration Tutorial [M]

Tutorial 02: sklearn Integration.

**Tasks**:

- [x] 4.2.1: Create `docs/tutorials/02-sklearn-integration.ipynb`
- [x] 4.2.2: Cover: GBDTRegressor, GBDTClassifier
- [x] 4.2.3: Demo: Pipeline with StandardScaler
- [x] 4.2.4: Demo: cross_val_score
- [x] 4.2.5: Demo: GridSearchCV
- [x] 4.2.6: Set random seeds throughout

**Definition of Done**:

- sklearn workflows demonstrated
- Difficulty badge: 🟢 Beginner

### Story 4.3: Classification Tutorials [M]

Tutorials 03-04: Binary and Multiclass Classification.

**Tasks**:

- [x] 4.3.1: Create `docs/tutorials/03-classification.ipynb` (binary)
- [x] 4.3.2: Cover: Objective.logistic(), predict_proba, AUC, confusion matrix
- [x] 4.3.3: Create `docs/tutorials/04-multiclass.ipynb`
- [x] 4.3.4: Cover: Objective.softmax(n_classes), multiclass metrics
- [x] 4.3.5: Include ROC curves and confusion matrices

**Definition of Done**:

- Both classification types covered
- Visualizations included

### Story 4.4: Early Stopping Tutorial [M]

Tutorial 05: Early Stopping and Validation.

**Tasks**:

- [x] 4.4.1: Create `docs/tutorials/05-early-stopping.ipynb`
- [x] 4.4.2: Cover: validation set usage, early_stopping_rounds
- [x] 4.4.3: Demo: monitoring training vs validation loss
- [x] 4.4.4: Show overfitting detection
- [x] 4.4.5: Include train/val learning curves

**Definition of Done**:

- Early stopping mechanism explained
- Overfitting visualized

### Story 4.5: GBLinear and Sparse Data Tutorial [M]

Tutorial 06: GBLinear and Sparse Data.

**Tasks**:

- [x] 4.5.1: Create `docs/tutorials/06-gblinear-sparse.ipynb`
- [x] 4.5.2: Cover: GBLinearModel, GBLinearConfig
- [x] 4.5.3: Demo: scipy.sparse input
- [x] 4.5.4: Compare GBDT vs GBLinear on linear vs non-linear data
- [x] 4.5.5: Explain when linear models excel

**Definition of Done**:

- GBLinear workflow demonstrated
- Sparse data handling shown

### Story 4.6: Hyperparameter Tuning Tutorial [M]

Tutorial 07: Hyperparameter Tuning.

**Tasks**:

- [x] 4.6.1: Create `docs/tutorials/07-hyperparameter-tuning.ipynb`
- [x] 4.6.2: Demo: effect of max_depth on performance
- [x] 4.6.3: Demo: learning_rate vs n_estimators trade-off
- [x] 4.6.4: Demo: regularization (reg_lambda, reg_alpha)
- [x] 4.6.5: Include parameter sweep visualizations

**Definition of Done**:

- Key hyperparameter effects visualized
- Tuning strategies explained

### Story 4.7: Explainability Tutorial [M]

Tutorial 08: Explainability.

**Tasks**:

- [x] 4.7.1: Create `docs/tutorials/08-explainability.ipynb`
- [x] 4.7.2: Cover: feature_importance (Split, Gain)
- [x] 4.7.3: Cover: SHAP values computation
- [x] 4.7.4: Demo: SHAP summary plot (using matplotlib or shap library)
- [x] 4.7.5: Show feature contribution interpretation

**Definition of Done**:

- Feature importance and SHAP demonstrated
- Interpretation guidance included

### Story 4.8: Model Serialization Tutorial [M]

Tutorial 09: Model Serialization.

**Tasks**:

- [x] 4.8.1: Create `docs/tutorials/09-model-serialization.ipynb`
- [x] 4.8.2: Cover: native binary format (to_bytes/from_bytes)
- [x] 4.8.3: Cover: JSON format (to_json_bytes/from_json_bytes)
- [x] 4.8.4: Cover: pickle serialization (pickle.dump/load)
- [x] 4.8.5: Demo: converting XGBoost/LightGBM models
- [x] 4.8.6: Compare file sizes between formats

**Definition of Done**:

- All serialization methods demonstrated
- Format trade-offs explained

### Story 4.9: Review and Demo (Epic 4) [S]

**Tasks**:

- [ ] 4.9.1: Execute all tutorials in clean environment
- [ ] 4.9.2: Verify deterministic output (set PYTHONHASHSEED=0)
- [ ] 4.9.3: Verify idempotency (run each notebook twice, compare outputs)
- [ ] 4.9.4: Verify difficulty progression: 01-02 beginner, 03-09 intermediate
- [ ] 4.9.5: Clear notebook outputs before committing (prevents merge conflicts)
- [ ] 4.9.6: Demo: tutorial navigation from index

**Definition of Done**:

- All 9 tutorials execute without error
- Outputs are deterministic and idempotent
- Difficulty badges accurate

---

## Epic 5: How-To Guides

Create task-oriented guides and recipes.

### Story 5.1: Missing Values Guide [S]

**Tasks**:

- [x] 5.1.1: Create `docs/howto/index.rst` with guide overview
- [x] 5.1.2: Create `docs/howto/missing-values.rst`
- [x] 5.1.3: Explain how boosters handles NaN values
- [x] 5.1.4: Include configuration options

**Definition of Done**:

- Missing value handling documented

### Story 5.2: Categorical Features Guide [M]

**Tasks**:

- [x] 5.2.1: Create `docs/howto/categorical-features.rst`
- [x] 5.2.2: Explain declaring categorical features
- [x] 5.2.3: Compare native categoricals vs one-hot encoding
- [x] 5.2.4: Include performance comparison

**Definition of Done**:

- Categorical feature usage documented

### Story 5.3: Custom Objectives Guide [M]

**Tasks**:

- [x] 5.3.1: Create `docs/howto/custom-objectives.rst`
- [x] 5.3.2: Explain custom objective interface
- [x] 5.3.3: Include example implementation
- [x] 5.3.4: Document gradient/Hessian requirements

**Definition of Done**:

- Custom objective creation documented

### Story 5.4: Debugging Performance Guide [M]

**Tasks**:

- [x] 5.4.1: Create `docs/howto/debugging-performance.rst`
- [x] 5.4.2: Cover: diagnosing underfitting (increase depth, more trees)
- [x] 5.4.3: Cover: diagnosing overfitting (regularization, early stopping)
- [x] 5.4.4: Include common mistakes checklist

**Definition of Done**:

- Troubleshooting guide complete

### Story 5.5: Production Deployment Guide [M]

**Tasks**:

- [x] 5.5.1: Create `docs/howto/production-deployment.rst`
- [x] 5.5.2: Cover: model serving options
- [x] 5.5.3: Cover: latency optimization tips (batch size selection, row batching vs single-row)
- [x] 5.5.4: Cover: threading configuration (`n_jobs` parameter, thread pool behavior)
- [x] 5.5.5: Document memory usage patterns and optimization

**Definition of Done**:

- Production deployment guidance provided
- Threading and latency optimization documented

### Story 5.6: Recipes Page [M]

**Tasks**:

- [x] 5.6.1: Create `docs/howto/recipes.rst`
- [x] 5.6.2: Add recipe: Cross-validation setup (complete, copy-paste ready)
  - Format: **Title** → Description → Code block
- [x] 5.6.3: Add recipe: Save/load model (native and pickle)
- [x] 5.6.4: Add recipe: Get feature importance
- [x] 5.6.5: Add recipe: Early stopping pattern
- [x] 5.6.6: Add recipe: Multiclass with sklearn
- [x] 5.6.7: Ensure all recipes are self-contained (include imports, sample data)
- [x] 5.6.8: Use consistent format: title, 1-2 sentence description, then code block

**Definition of Done**:

- Common patterns documented as copy-paste recipes
- Each recipe is a complete, runnable code block

### Story 5.7: FAQ Page [S]

**Tasks**:

- [x] 5.7.1: Create `docs/howto/faq.rst`
- [x] 5.7.2: Add common questions:
  - "How do I handle missing values?"
  - "How do I choose between GBDT and GBLinear?"
  - "How do I convert an XGBoost model?"
  - "Why is my model slow?"
  - "How do I get feature importances?"
- [x] 5.7.3: Link FAQ answers to detailed pages

**Definition of Done**:

- FAQ covers top 5+ common questions
- Links to detailed documentation

### Story 5.9: Glossary Page [S]

**Tasks**:

- [x] 5.9.1: Create `docs/howto/glossary.rst`
- [x] 5.9.2: Define key terms: GBDT, GBLinear, GOSS, SHAP, Feature Importance, Gradient, Hessian, Boosting Round, Tree Depth, Leaf Weight
- [x] 5.9.3: Use Sphinx glossary directive for cross-referencing
- [x] 5.9.4: Add links to relevant explanation pages

**Definition of Done**:

- Glossary has 10+ essential terms
- Terms are cross-referenceable from other pages

### Story 5.10: Review and Demo (Epic 5) [S]

**Tasks**:

- [ ] 5.10.1: Review how-to guides for completeness
- [ ] 5.10.2: Verify code examples execute
- [ ] 5.10.3: Demo: navigate how-to section

**Definition of Done**:

- All how-to guides complete

---

## Epic 6: READMEs and Contributing

Update package READMEs and create contributing guide.

### Story 6.1: Main README Update [M]

**Tasks**:

- [x] 6.1.1: Update root `README.md` with documentation links
- [x] 6.1.2: Add documentation badges (docs online, PyPI, etc.)
- [x] 6.1.3: Ensure quickstart code matches Sphinx quickstart exactly
- [x] 6.1.4: Add "For Rust Users" section linking to rustdoc

**Definition of Done**:

- README links to documentation
- Code examples consistent

### Story 6.2: Package READMEs [S]

**Tasks**:

- [x] 6.2.1: Update `packages/boosters-python/README.md` with docs link
- [x] 6.2.2: Update `packages/boosters-eval/README.md` with docs link
- [x] 6.2.3: Ensure package READMEs are concise with "See full docs" links

**Definition of Done**:

- All package READMEs link to main docs

### Story 6.3: Contributing Guide [M]

**Tasks**:

- [x] 6.3.1: Create `docs/contributing/index.rst`
- [x] 6.3.2: Create `docs/contributing/development.rst` with setup instructions
- [x] 6.3.3: Create `docs/contributing/architecture.rst` with codebase overview
- [x] 6.3.4: Link to existing `.github/instructions/CONTRIBUTING.instructions.md`

**Definition of Done**:

- Contributing guide complete
- Development setup documented

### Story 6.4: CHANGELOG Setup [S]

**Tasks**:

- [x] 6.4.1: Create `CHANGELOG.md` at repo root (if not exists)
- [x] 6.4.2: Follow Keep a Changelog format
- [x] 6.4.3: Link to CHANGELOG from docs navigation

**Definition of Done**:

- CHANGELOG exists and is linked from docs

### Story 6.5: Final Review and Launch [M]

**Tasks**:

- [ ] 6.5.1: Run full docs build with warnings as errors (`uv run poe docs:build`)
- [ ] 6.5.2: Run link checker (`uv run poe docs:linkcheck`)
- [ ] 6.5.3: Run coverage check (`uv run poe docs:coverage`) to identify undocumented items
- [ ] 6.5.4: Verify all RFC acceptance criteria met:
  - [ ] Documentation builds without warnings
  - [ ] All 9 tutorials execute without error
  - [ ] Zero broken internal links
  - [ ] API coverage: every public symbol documented
  - [ ] Sphinx search returns relevant results
  - [ ] Local development workflow functional
- [ ] 6.5.5: Deploy to GitHub Pages
- [ ] 6.5.6: Update RFC-0017 status to "Implemented"

**Definition of Done**:

- Documentation live on GitHub Pages
- All acceptance criteria verified and checked off
- RFC status updated

---

## Dependencies

| Story | Depends On | Notes |
|-------|------------|-------|
| 1.2 | 1.1 | Local dev requires Sphinx setup |
| 1.3 | 1.1 | CI requires Sphinx setup |
| 1.4 | 1.1 | RFC embedding requires Sphinx |
| 2.1-2.5 | 1.1 | All getting started content |
| 3.1-3.5 | 1.1 | All explanations content |
| 4.1 | 1.1 | Tutorials require Sphinx+nbsphinx |
| 4.2-4.8 | 4.1 | Later tutorials follow index pattern |
| 5.1-5.6 | 1.1 | How-to guides require Sphinx |
| 6.1-6.2 | 2.1, 2.2 | READMEs need consistent quickstarts |
| 6.3 | 1.1 | Contributing guide in Sphinx |
| 6.5 | All | Final review after all epics |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| nbsphinx notebook execution failures | Medium | High | Set explicit random seeds, use cached datasets |
| External link rot | Medium | Low | Weekly linkcheck in CI |
| API changes break docs | Medium | Medium | Run docs build in main CI, fail on warnings |
| pydata-sphinx-theme breaking changes | Low | Medium | Pin version in pyproject.toml |
| Large docs build times | Medium | Low | Add caching, consider notebook pre-execution |
| Documentation-code drift | Medium | High | Autodoc regeneration in CI, doctest examples |
| Intersphinx external service downtime | Low | Low | Cache intersphinx inventories locally |

---

## Changelog

- 2026-01-13: Initial backlog created from RFC-0017
- 2026-01-13: Round 1 (ML Researcher) - Added additive model formula, Newton-Raphson context, regularization theory, expanded objective formulas
- 2026-01-13: Round 2 (Performance Engineer) - Added caching details, watch target, notebook pre-execution strategy, build time target
- 2026-01-13: Round 3 (End User Engineer) - Added notebook downloads, FAQ page, copy-paste recipe requirements
- 2026-01-13: Round 4 (Architect) - Added intersphinx config, version notice, milestone effort estimates, documentation-code drift risk
- 2026-01-13: Round 5 (QA Engineer) - Added idempotency verification, coverage check, explicit acceptance checklist, doctest integration
- 2026-01-13: Round 6 (Product Owner) - Removed Makefile (use poe tasks only), added hyperparameter reference table, workspace integration task, aligned with RFC
- 2026-01-13: Round 7 (Technical Writer) - Added code formatting requirements, glossary page, ensemble diagram, recipe structure format, notebook output clearing
- 2026-01-13: Implementation started - Marked completed tasks in Epic 1 (Stories 1.1-1.4), Epic 2 (Stories 2.1-2.5), Epic 3 (Stories 3.1-3.5), Epic 4 (Stories 4.1-4.8), Epic 5 (Stories 5.1-5.7, 5.9), Epic 6 (Story 6.3)
- 2026-01-14: Fixed notebook errors (Objective.squared() usage, GBDTModel.train signature), improved explainability tutorial to use California Housing dataset with named features, verified all tutorials execute successfully
