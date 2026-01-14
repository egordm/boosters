# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add boosters-python to path for autodoc
sys.path.insert(0, os.path.abspath("../packages/boosters-python/python"))

# -- Project information -----------------------------------------------------
project = "boosters"
copyright = "2026, boosters contributors"
author = "boosters contributors"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
    "sphinx_design",
]

# Generate autosummary stubs
autosummary_generate = True

# Avoid duplicate object warnings by not importing members from parent docs
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# MyST configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# nbsphinx configuration
# Execute notebooks to show output (plots, numbers)
# Caching: nbsphinx with "auto" only re-executes notebooks that have changed.
# The executed notebooks are cached in _build/.doctrees. Avoid `rm -rf _build`
# for faster incremental builds - use `make clean` or only delete _build/html.
nbsphinx_execute = "auto"  # Execute notebooks that don't have output
nbsphinx_allow_errors = True  # Continue build even if notebook execution fails
nbsphinx_timeout = 600  # 10 minute timeout per notebook
nbsphinx_kernel_name = "python3"
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

.. note::

   This tutorial is available as a Jupyter notebook.
   `Download notebook <https://github.com/egordm/booste-rs/blob/main/docs/{{ docname }}>`_
"""

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# Exclude patterns
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # Benchmarks - template
    "benchmarks/TEMPLATE.md",
    # Backlogs - internal tracking, not user-facing
    "backlogs/*",
    # Research README - replaced by research/index.rst
    "research/README.md",
    "research/gbdt/training/linear-trees.md",  # Planned content
    # Old structure directories (replaced by user-guide)
    "getting-started/*",
    "howto/*",
    "explanations/*",
    "design/index.rst",  # Using design/rfcs.rst directly
    # Standalone docs that are not part of user-facing toctree
    "README.md",
    "ROADMAP.md",
    # RFC template and README (not rendered directly)
    "rfcs/TEMPLATE.md",
    "rfcs/README.md",
]

# Templates path
templates_path = ["_templates"]

# Static files path
html_static_path = ["_static"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/egordm/booste-rs",
    "logo": {
        "text": "boosters",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/boosters/",
            "icon": "fas fa-box",
        },
    ],
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "announcement": (
        "⚠️ <b>Pre-release:</b> boosters is under active development. "
        "API may change before v1.0."
    ),
    "navbar_align": "left",
    # Simplified navbar - remove duplicate search
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    # Primary sidebar navigation
    "primary_sidebar_end": [],
}

html_context = {
    "github_user": "egordm",
    "github_repo": "booste-rs",
    "github_version": "main",
    "doc_path": "docs",
}

html_title = "boosters"
html_short_title = "boosters"

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Suppress duplicate object warnings from autosummary + autodoc interaction
# and minor docutils warnings from auto-generated docstrings
# The "duplicate object description" warnings are expected: autosummary generates
# stubs that document objects also documented by autodoc - both are needed for
# cross-references to work, but the warning is harmless.
suppress_warnings = [
    "autodoc.duplicate_object",
    "autodoc",  # Suppress all autodoc warnings (duplicate descriptions)
    "misc.highlighting_failure",  # JSON with comments, minor
    "docutils",  # Suppress docutils warnings from PyO3-generated docstrings
    "ref.python",  # Suppress python ref warnings
    "duplicate.object",  # Another form of duplicate object warning
    "py.duplicate",  # Python domain duplicate object warnings (py:class, py:attr, etc.)
    "py",  # Suppress all Python domain warnings including duplicate descriptions
]

# -- Math rendering with MathJax ---------------------------------------------
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "packages": {"[+]": ["ams"]},
    },
}

# -- Link checking configuration ---------------------------------------------
linkcheck_ignore = [
    r"http://localhost:\d+/",  # Local dev servers
    r"https://pypi\.org/project/boosters/",  # Before first publish
]
linkcheck_allowed_redirects = {
    r"https://github\.com/.*": r"https://github\.com/.*",
}
linkcheck_timeout = 30
linkcheck_retries = 3

# -- Coverage configuration --------------------------------------------------
coverage_show_missing_items = True


# -- Linkcode configuration (link to source on GitHub) ----------------------
def linkcode_resolve(domain, info):
    """Resolve GitHub source links for API documentation."""
    if domain != "py":
        return None
    if not info["module"]:
        return None

    # boosters is a compiled extension, link to the Rust source
    # This is a placeholder - adjust based on actual source structure
    return None
