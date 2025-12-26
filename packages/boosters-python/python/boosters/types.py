"""Common type aliases for boosters.

This module defines type aliases used throughout the boosters package.
"""

from typing import Literal

# =============================================================================
# Tree Building Types
# =============================================================================

GrowthStrategy = Literal["leafwise", "depthwise"]
"""Tree growth strategy.

- ``"leafwise"``: Grow leaf-wise (best-first). Expands the leaf with the highest
  gain, which typically produces deeper trees and better accuracy but may overfit.
  This is the LightGBM default behavior.
  
- ``"depthwise"``: Grow depth-wise (level-by-level). Expands all nodes at the
  current depth before moving deeper. More regularized than leaf-wise.
  This is the XGBoost default behavior.
"""
