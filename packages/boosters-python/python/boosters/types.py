"""Common type aliases for boosters.

This module defines type aliases used throughout the boosters package.
"""

# =============================================================================
# Tree Building Types
# =============================================================================

# GrowthStrategy is now a proper enum exported from the Rust core.
# Import it from config module for backwards compatibility.
from boosters._boosters_rs import GrowthStrategy

__all__ = ["GrowthStrategy"]
