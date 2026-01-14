# RFC-0017: Documentation and READMEs

**Status**: Accepted  
**Created**: 2026-01-13  
**Updated**: 2026-01-13  
**Author**: Team

## Summary

Comprehensive documentation infrastructure for boosters:

- **Sphinx docs** (Python-centric): Main documentation with tutorials, getting started, API reference, explanations, and embedded research/RFCs
- **Rustdoc** (Rust-centric): Rust API reference documentation  
- **READMEs**: Package-level quick starts linking to main docs
- **GitHub Actions**: Automated builds and GitHub Pages deployment

## Motivation

| Need | Problem Without Docs |
|------|---------------------|
| Adoption | Users can't discover or learn the library |
| Onboarding | Contributors have no guide to architecture |
| Discoverability | Advanced features remain hidden |
| Trust | Lack of documentation signals immaturity |

Current state: READMEs exist but no unified documentation site. RFCs and research exist but are developer-facing only.

## Non-Goals

- Maintaining separate docs for Rust library users (Rustdoc serves this)
- Translating docs to multiple languages
- Building a custom documentation framework
- Video tutorials (future enhancement)

## Design

### Documentation Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           GitHub Pages                   │
                    │         (boosters.github.io)             │
                    └─────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┴───────────────────────┐
            │                                                   │
            ▼                                                   ▼
    ┌───────────────┐                               ┌───────────────┐
    │  Sphinx Docs  │                               │    Rustdoc    │
    │   (Python)    │                               │    (Rust)     │
    │   /           │                               │    /rustdoc/  │
    └───────────────┘                               └───────────────┘
            │
    ┌───────┴────────────────────────────────────────────────┐
    │                                                        │
    ├── Getting Started                                      │
    ├── Tutorials (Jupyter via nbsphinx)                     │
    ├── How-To Guides                                        │
    ├── Explanations (Theory + Research)                     │
    ├── API Reference (autodoc)                              │
    ├── Design Docs (embedded RFCs)                          │
    ├── Benchmarks                                           │
    └── Contributing                                         │
```

### Directory Structure

```
docs/
├── conf.py                     # Sphinx configuration (root level)
├── index.rst                   # Landing page
├── Makefile                    # Build shortcuts
├── Makefile                    # Build shortcuts
├── _static/                    # Static assets (CSS, images, logo)
├── _templates/                 # Custom templates
├── scripts/                    # Build utilities
│   └── validate_links.py       # RFC link validation
│
├── getting-started/            # Onboarding section
│   ├── index.rst
│   ├── installation.rst
│   ├── quickstart-python.rst
│   └── quickstart-rust.rst
│
├── tutorials/                  # Step-by-step tutorials (notebooks)
│   ├── index.rst
│   ├── 01-basic-training.ipynb
│   ├── 02-sklearn-integration.ipynb
│   ├── 03-classification.ipynb
│   ├── 04-multiclass.ipynb
│   ├── 05-early-stopping.ipynb
│   ├── 06-gblinear-sparse.ipynb
│   ├── 07-hyperparameter-tuning.ipynb
│   ├── 08-explainability.ipynb
│   └── 09-model-serialization.ipynb
│
├── howto/                      # Task-oriented guides
│   ├── index.rst
│   ├── missing-values.rst
│   ├── categorical-features.rst
│   ├── custom-objectives.rst
│   ├── debugging-performance.rst
│   ├── production-deployment.rst
│   └── recipes.rst             # Common patterns cheatsheet
│
├── explanations/               # Conceptual documentation
│   ├── index.rst
│   ├── gradient-boosting.rst   # Embeds research/gradient-boosting.md
│   ├── gbdt.rst
│   ├── gblinear.rst
│   ├── hyperparameters.rst     # Comprehensive parameter guide
│   ├── objectives-metrics.rst
│   └── benchmarks.rst
│
├── api/                        # API reference
│   ├── index.rst
│   ├── python/                 # Python API (autodoc)
│   └── rust.rst                # Link to Rustdoc
│
├── design/                     # Design documentation
│   ├── index.rst
│   └── rfcs/                   # Rendered RFCs (includes from rfcs/)
│
├── contributing/               # Contributor guide
│   ├── index.rst
│   ├── development.rst
│   └── architecture.rst
│
├── rfcs/                       # RFC source files (existing)
│   ├── 0001-dataset.md
│   ├── ...
│   └── 0017-documentation.md
│
├── research/                   # Research documents (existing)
│   ├── gradient-boosting.md
│   └── ...
│
└── benchmarks/                 # Benchmark reports (existing)
```

**Note**: Sphinx source files are at `docs/` root level. RFCs, research, and benchmarks remain in their original locations and are embedded/linked into the Sphinx structure.

### Package Structure

```
packages/
├── boosters-python/
│   └── README.md               # Quick start → links to main docs
├── boosters-eval/
│   └── README.md               # Quick start → links to main docs
└── boosters-docs/              # NEW: Documentation package
    ├── pyproject.toml          # Sphinx + dependencies
    └── src/boosters_docs/      # Optional: doc generation utilities
```

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Theme | pydata-sphinx-theme | Professional, used by pandas/numpy/scipy |
| Notebooks | nbsphinx | Execute and render Jupyter notebooks |
| Markdown | myst-parser | Use .md files alongside .rst |
| API docs | sphinx-autodoc | Auto-generate from docstrings |
| Type hints | sphinx-autodoc-typehints | Show type annotations |
| Build | GitHub Actions | CI/CD integration |
| Hosting | GitHub Pages | Free, reliable |

### Sphinx Configuration

```python
# conf.py
project = "boosters"
copyright = "2026, boosters contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcheck",         # Verify external links
    "sphinx.ext.coverage",          # API documentation coverage
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
    "sphinx_design",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/your-org/booste-rs",
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/boosters/", "icon": "fas fa-box"},
    ],
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "search_bar_text": "Search documentation...",
}

# nbsphinx configuration
nbsphinx_execute = "auto"  # Execute notebooks during build
nbsphinx_allow_errors = False
nbsphinx_timeout = 300     # 5 minute timeout per notebook

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

# Math rendering with amsmath support
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "packages": {"[+]": ["ams"]},
    },
}

# Link checking
linkcheck_ignore = [
    r"http://localhost:\d+/",  # Local dev servers
    r"https://pypi\.org/project/boosters/",  # Before first publish
]
linkcheck_allowed_redirects = {
    r"https://github\.com/.*": r"https://github\.com/.*",  # GitHub redirects
}

# Coverage settings
coverage_show_missing_items = True

# Intersphinx: link to external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
```

### Landing Page Design

Inspired by OpenSTEF, the landing page features:

1. **Hero section**: Project name, tagline, badges
2. **Feature cards**: Key capabilities with icons
3. **Quick navigation**: Getting Started, Tutorials, API Reference
4. **Installation snippet**: `pip install boosters`
5. **Links**: GitHub, PyPI, Documentation sections

```rst
.. landing page index.rst structure
Welcome to boosters!
====================

Fast gradient boosting for Python and Rust.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: getting-started/index
      :link-type: doc

      Install and run your first model in 5 minutes.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Learn through hands-on Jupyter notebooks.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Complete Python and Rust API documentation.
```

### Tutorial Plan

| # | Title | Level | Content | Skills Taught |
|---|-------|-------|---------|---------------|
| 01 | Python: Basic GBDT Training | 🟢 Beginner | Train GBDT regressor, predict, evaluate | Core workflow |
| 02 | sklearn Integration | 🟢 Beginner | Pipelines, cross-validation, grid search | sklearn compat |
| 03 | Binary Classification | 🟢 Beginner | Logistic objective, AUC, predict_proba | Classification |
| 04 | Multiclass Classification | 🟡 Intermediate | Softmax, multiple outputs | Multi-output |
| 05 | Early Stopping & Validation | 🟡 Intermediate | Validation sets, early stopping, monitoring | Overfitting prevention |
| 06 | GBLinear & Sparse Data | 🟡 Intermediate | Linear boosting, scipy.sparse, high-dim data | Model types, sparse |
| 07 | Hyperparameter Tuning | 🟡 Intermediate | Depth, learning rate, regularization | Tuning |
| 08 | Explainability | 🟡 Intermediate | Feature importance, SHAP values | Interpretability |
| 09 | Model Serialization | 🟡 Intermediate | Save/load, pickle, format conversion | Persistence |

Each tutorial is self-contained (no cross-notebook dependencies) to ensure reliable execution and testing.

**Determinism requirements**: All tutorials must:
- Set explicit random seeds (`np.random.seed(42)`, `random_state=42`)
- Not rely on system time or environment-specific values
- Produce identical outputs across runs (enforced in CI)
- Use cached/local datasets where possible (avoid network dependencies)
- Include learning curve plots (train vs validation loss) where applicable

**Serialization tutorial (09) covers**:
- Native binary format (`.bstr`) and JSON format (`.bstr.json`)
- Pickle serialization (`pickle.dump()` / `pickle.load()`)
- Format conversion utilities
- Loading XGBoost/LightGBM models

### How-To Guides

| Guide | Content |
|-------|---------|
| Missing Values | How boosters handles NaN, configuration options |
| Categorical Features | Declaring categoricals, native vs one-hot encoding |
| Custom Objectives | Implementing custom loss functions |
| Debugging Performance | Diagnosing underfitting/overfitting, common mistakes |
| Production Deployment | Model serving, latency optimization |
| Recipes | Copy-paste patterns for common tasks |

The **Recipes** page provides quick solutions:
- Cross-validation setup
- Save/load model (native format and pickle)
- Get feature importance
- Early stopping pattern
- Multiclass with sklearn

### Hyperparameter Documentation

The `explanations/hyperparameters.rst` covers each parameter with:

1. **What**: Parameter name and valid values
2. **How**: How it affects the model
3. **Why**: When to adjust it
4. **Trade-offs**: Speed vs accuracy vs overfitting

Example structure:

```rst
Tree Depth (``max_depth``)
--------------------------

**What**: Maximum depth of each tree (default: 6).

**How**: Controls model complexity. Deeper trees capture more interactions
but require more computation and risk overfitting.

**Why tune it**:
- Increase for complex data with many interactions
- Decrease to prevent overfitting on small datasets

**Trade-offs**:
- Depth 3-4: Fast, low variance, may underfit
- Depth 6-8: Balanced (default recommendation)
- Depth 10+: Slow, high variance, risk overfitting

**Interaction**: Lower depth works well with more trees.
```

### Hyperparameter Documentation Coverage

The `explanations/hyperparameters.rst` comprehensively documents:

| Category | Parameters |
|----------|------------|
| **Tree Structure** | `max_depth`, `max_leaves`, `min_child_weight`, `min_split_loss` (gamma) |
| **Regularization** | `reg_lambda` (L2), `reg_alpha` (L1), `min_child_weight` |
| **Learning** | `learning_rate` (eta), `n_estimators`, `early_stopping_rounds` |
| **Subsampling** | `subsample`, `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` |
| **GOSS** | `top_rate`, `other_rate` |
| **Tree Growth** | `grow_policy` (depthwise vs leafwise) |
| **GBLinear** | `alpha`, `lambda_`, `feature_selector`, `top_k` |

Each parameter includes: default value, valid range, interaction effects, and when to tune.

**Feature selector guide** (GBLinear):
| Selector | When to Use |
|----------|-------------|
| `cyclic` | Default, deterministic, good for reproducibility |
| `shuffle` | Better for correlated features, introduces randomness |
| `greedy` | High-dimensional sparse data, selects best feature each round |
| `thrifty` | Like greedy but with approximate selection, faster |

### Objectives and Metrics Documentation

The `explanations/objectives-metrics.rst` includes for each objective:

| Objective | Formula | Gradient | Hessian | Use Case |
|-----------|---------|----------|---------|----------|
| Squared Error | $(y - \hat{y})^2$ | $2(\hat{y} - y)$ | $2$ | Regression |
| Logistic | $-y\log(\sigma) - (1-y)\log(1-\sigma)$ | $\sigma - y$ | $\sigma(1-\sigma)$ | Binary classification |
| Softmax | $-\sum_k y_k \log(p_k)$ | $p_k - y_k$ | $p_k(1-p_k)$ | Multiclass |

This helps users understand model behavior and debug custom objectives.

### Further Reading (Explanations Section)

The `explanations/index.rst` includes a "Further Reading" section linking to foundational papers:

- Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

These references provide theoretical background for users who want to understand the algorithms deeply.

### GitHub Actions Workflow

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly linkcheck

concurrency:
  group: docs-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build-sphinx:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        
      - name: Cache uv packages
        uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: ${{ runner.os }}-uv-docs-${{ hashFiles('docs/pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-uv-docs-

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-docs-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: |
            ${{ runner.os }}-cargo-docs-
        
      - name: Install dependencies
        run: uv sync --package boosters-docs
        
      - name: Build Python package (for autodoc)
        run: uv run maturin develop -m packages/boosters-python/Cargo.toml --release
        
      - name: Validate RFC links
        run: uv run python docs/scripts/validate_links.py
        
      - name: Build Sphinx docs
        run: uv run sphinx-build -W -j auto docs docs/_build/html
        env:
          PYTHONHASHSEED: "0"  # Deterministic notebook execution
        
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/_build/html

  build-rustdoc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-rustdoc-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: |
            ${{ runner.os }}-cargo-rustdoc-
        
      - name: Build Rustdoc
        run: cargo doc --no-deps --package boosters
        
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: rustdoc
          path: target/doc

  deploy:
    needs: [build-sphinx, build-rustdoc]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    permissions:
      pages: write
      id-token: write
      
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
      
    steps:
      - name: Download Sphinx artifact
        uses: actions/download-artifact@v4
        with:
          name: github-pages
          path: public
          
      - name: Download Rustdoc artifact
        uses: actions/download-artifact@v4
        with:
          name: rustdoc
          path: public/rustdoc
          
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### boosters-docs Package

The documentation package lives in `docs/` and is registered as a workspace member:

```toml
# In root pyproject.toml, add to workspace members:
[tool.uv.workspace]
members = ["packages/*", "docs"]
```

```toml
# docs/pyproject.toml
[project]
name = "boosters-docs"
version = "0.1.0"
description = "Documentation for boosters"
requires-python = ">=3.12"

dependencies = [
    "boosters",                    # For autodoc
    "sphinx>=7.0",
    "pydata-sphinx-theme>=0.15",
    "myst-parser>=2.0",
    "nbsphinx>=0.9",
    "sphinx-autodoc-typehints>=2.0",
    "sphinx-design>=0.5",
    "ipykernel>=6.0",             # For notebook execution
    "matplotlib>=3.8",            # For tutorial plots
    "pandas>=2.0",                # For tutorial examples
    "seaborn>=0.13",              # For tutorial visualizations
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = []  # No Python source, just docs
```

Note: This package contains no Python source code—it exists only to manage documentation dependencies within the uv workspace.

### Embedding RFCs and Research

RFCs and research documents are included via MyST's `{include}` directive:

```rst
.. in design/rfcs/index.rst

RFC-0008: GBDT Training
-----------------------

.. include:: ../../../rfcs/0008-gbdt-training.md
   :parser: myst_parser.sphinx_
```

**Link preservation**: A pre-build script validates that cross-references in included files resolve correctly. Relative links in RFCs (e.g., `../research/gradient-boosting.md`) are rewritten during the build process.

```python
# docs/scripts/validate_links.py
"""Validate RFC cross-references before Sphinx build."""
import re
from pathlib import Path

def validate_rfc_links(docs_dir: Path) -> list[str]:
    """Check that all relative links in RFCs resolve to existing files."""
    errors = []
    for rfc in (docs_dir / "rfcs").glob("*.md"):
        content = rfc.read_text()
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
            link_text, link_target = match.groups()
            if link_target.startswith(("http://", "https://", "#")):
                continue  # Skip external and anchor links
            resolved = (rfc.parent / link_target).resolve()
            if not resolved.exists():
                errors.append(f"{rfc.name}: broken link to {link_target}")
    return errors

if __name__ == "__main__":
    errors = validate_rfc_links(Path(__file__).parent.parent)
    if errors:
        print("Broken links found:")
        for e in errors:
            print(f"  - {e}")
        exit(1)
    print("All RFC links valid.")
```

### URL Structure

Deployed documentation will be available at:

- **Main docs**: `https://<org>.github.io/booste-rs/` (Sphinx)
- **Rust API**: `https://<org>.github.io/booste-rs/rustdoc/boosters/` (Rustdoc)

Internal links use relative paths to ensure portability between local builds and deployed versions.

### Local Development

```makefile
# docs/Makefile
.PHONY: html serve clean linkcheck coverage

html:
	uv run sphinx-build -W -j auto . _build/html

serve: html
	python -m http.server -d _build/html 8000

clean:
	rm -rf _build

linkcheck:
	uv run sphinx-build -b linkcheck . _build/linkcheck

coverage:
	uv run sphinx-build -b coverage . _build/coverage
	cat _build/coverage/python.txt
```

Add a `.gitignore` in `docs/`:
```gitignore
# docs/.gitignore
_build/
*.pyc
__pycache__/
.ipynb_checkpoints/
```

Developers can build and preview locally:
```bash
cd docs
make serve
# Open http://localhost:8000
```

**Note**: Add `docs/_build/` to `.gitignore` to exclude build artifacts.

### Installation Page Structure

The `getting-started/installation.rst` covers:

1. **Quick install** (pip):
   ```bash
   pip install boosters
   ```

2. **Building from source** (for development):
   ```bash
   git clone https://github.com/org/booste-rs
   cd booste-rs
   uv sync
   uv run maturin develop -m packages/boosters-python/Cargo.toml --release
   ```

3. **Troubleshooting**:
   - GLIBC version requirements (Linux)
   - Rust toolchain issues
   - Missing build dependencies

4. **Verifying installation**:
   ```python
   import boosters
   print(boosters.__version__)
   ```

### PR Preview Strategy

For pull requests, documentation artifacts are uploaded and available for download:

1. The `build-sphinx` job uploads the artifact
2. Contributors can download and inspect locally
3. Maintainers can view rendered docs before merge

Future enhancement: Deploy PR previews to a staging URL using Netlify or Vercel.

### Changelog and Release Notes

Release notes are maintained in `CHANGELOG.md` at the repository root following [Keep a Changelog](https://keepachangelog.com/) format. The documentation links to this file from the main navigation.

```rst
.. in contributing/index.rst or top-level toctree

Release Notes
-------------
See `CHANGELOG.md <https://github.com/org/booste-rs/blob/main/CHANGELOG.md>`_ for version history.
```

### README Updates

#### Main README

```markdown
# 🚀 boosters

Fast gradient boosting for Python and Rust.

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://your-org.github.io/booste-rs/)
[![PyPI](https://img.shields.io/pypi/v/boosters)](https://pypi.org/project/boosters/)

## Quick Start

\`\`\`bash
pip install boosters
\`\`\`

\`\`\`python
from boosters.sklearn import GBDTRegressor

model = GBDTRegressor(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
\`\`\`

## Documentation

📚 **[Full Documentation](https://your-org.github.io/booste-rs/)** — Tutorials, API reference, and more

- [Getting Started](https://your-org.github.io/booste-rs/getting-started/)
- [Tutorials](https://your-org.github.io/booste-rs/tutorials/)
- [API Reference](https://your-org.github.io/booste-rs/api/)
- [Rust API (rustdoc)](https://your-org.github.io/booste-rs/rustdoc/)

## For Rust Users

\`\`\`rust
use boosters::{GBDTModel, GBDTConfig};
// See rustdoc for complete API
\`\`\`

## License

MIT
```

#### Package READMEs

Package READMEs (boosters-python, boosters-eval) should:
1. Short description
2. Installation
3. 5-line quick start
4. Link to full documentation

## API

### Documentation CLI (poe tasks)

```toml
# Added to root pyproject.toml
[tool.poe.tasks."docs:build"]
help = "Build Sphinx documentation"
cmd = "make -C docs html"

[tool.poe.tasks."docs:serve"]
help = "Serve documentation locally"
cmd = "make -C docs serve"

[tool.poe.tasks."docs:clean"]
help = "Clean documentation build"
cmd = "make -C docs clean"

[tool.poe.tasks."docs:linkcheck"]
help = "Check documentation links"
cmd = "make -C docs linkcheck"
```

### Local Development

```bash
# Build and serve locally
uv run poe docs:serve
# Open http://localhost:8000

# Build only
uv run poe docs:build

# Clean rebuild
uv run poe docs:clean && uv run poe docs:build
```

## Testing

| What | How |
|------|-----|
| Sphinx builds | `sphinx-build` exits 0, no warnings |
| Links work | `sphinx-build -W` (warnings as errors) + linkcheck |
| Notebooks execute | nbsphinx runs all notebooks |
| API docs generate | autodoc finds all public symbols |
| Rustdoc builds | `cargo doc` exits 0 |
| External links | `sphinx-build -b linkcheck` validates URLs |

### Acceptance Criteria

Documentation is considered complete when:

1. **Coverage**: All public Python classes, methods, and functions have docstrings
2. **Tutorials**: All 10 tutorials execute without error on a fresh environment
3. **Links**: Zero broken internal links (`sphinx-build -W` passes)
4. **External links**: `linkcheck` reports no dead URLs (excluding allowlist)
5. **API completeness**: Every symbol exported in `boosters.__all__` appears in API docs
6. **Search**: Documentation search returns relevant results for "GBDT", "sklearn", "SHAP"

### CI Checks

```yaml
# In docs workflow
- name: Build with strict warnings
  run: uv run sphinx-build -W docs/sphinx docs/_build/html
```

## Alternatives

### Read the Docs

**Pros**: Free hosting, version dropdowns, search  
**Cons**: Less control over deployment, slower builds, ads on free tier

**Decision**: GitHub Pages for full control and no ads.

### MkDocs

**Pros**: Fast, Markdown-native  
**Cons**: Less mature notebook support, smaller ecosystem

**Decision**: Sphinx for nbsphinx and autodoc ecosystem.

### Docusaurus

**Pros**: Modern React-based  
**Cons**: Not Python-native, no autodoc

**Decision**: Sphinx is standard for Python scientific libraries.

## Design Decisions

**DD-1: Sphinx as primary.** Python is primary audience; Sphinx has best Python ecosystem integration.

**DD-2: pydata-sphinx-theme.** Used by pandas, numpy, scipy — familiar to target audience.

**DD-3: Notebooks for tutorials.** Executable documentation ensures examples work. nbsphinx renders them beautifully.

**DD-4: Rustdoc as subdirectory.** Rust users get native rustdoc at `/rustdoc/`, not a second-class experience.

**DD-5: Docs as uv package.** Dependencies managed via pyproject.toml, not requirements.txt. Consistent with workspace pattern.

**DD-6: Embed RFCs, don't duplicate.** Use myst-parser to include RFC markdown files directly. Single source of truth.

**DD-7: GitHub Pages over RTD.** Full control, no ads, simpler CI integration with existing workflows.

**DD-8: Self-contained tutorials.** Each notebook is independent with no cross-notebook state dependencies. This ensures reliable CI execution and allows users to run any tutorial in isolation.

## Future Work

- **Version dropdowns**: When v1.0 releases, add pydata-sphinx-theme version switcher to maintain v0.x docs
- **API coverage tooling**: Integrate `interrogate` or similar to enforce docstring coverage
- **Internationalization**: If community demand exists, add i18n support via sphinx-intl
- **Video tutorials**: Embed short screencasts for complex workflows

## Changelog

- 2026-01-13: Initial RFC created and accepted after 5 review rounds
  - Round 1: Added acceptance criteria, link checking, ranking tutorial, RFC embedding mechanism
  - Round 2: Restructured docs directory, added how-to guides, sphinx.ext.coverage
  - Round 3: Added intersphinx, caching, difficulty badges, hyperparameter coverage table
  - Round 4: Added Makefile, lighthouse config, PR preview strategy, installation details
  - Round 5: Final polish — concurrency, CHANGELOG reference, academic citations
