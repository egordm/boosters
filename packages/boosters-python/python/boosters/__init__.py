"""Boosters - Fast gradient boosting library with native Rust core.

This package provides gradient boosted decision trees (GBDT) and linear models
with a clean Pythonic API and scikit-learn compatibility.

Example:
    >>> import boosters as bst
    >>> import numpy as np
    >>> X = np.random.rand(100, 10).astype(np.float32)
    >>> y = np.random.rand(100).astype(np.float32)
    >>> # More examples after model implementation

See Also:
    - RFC-0014 for API design rationale
    - https://github.com/your-org/boosters for documentation
"""

from boosters._boosters_rs import __version__

# Type aliases (populated as types are implemented)
# from boosters.config import ...
# from boosters.objectives import ...
# from boosters.metrics import ...
# from boosters.data import ...
# from boosters.model import ...

__all__ = [
    "__version__",
    # Config types (Epic 2)
    # Objective types (Epic 2)
    # Metric types (Epic 2)
    # Data types (Epic 3)
    # Model types (Epic 4)
]
