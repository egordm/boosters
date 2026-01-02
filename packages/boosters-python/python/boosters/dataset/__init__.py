"""Dataset-related types.

Public surface:
- `Feature`: single feature column (dense or sparse)
- `DatasetBuilder`: incremental dataset construction
- `Dataset`: constructed dataset
"""

from __future__ import annotations

from .builder import DatasetBuilder
from .dataset import Dataset
from .feature import Feature

__all__ = [
    "Dataset",
    "DatasetBuilder",
    "Feature",
]
