"""Persistence schema types for boosters models.

This module provides pydantic v2 models that mirror the Rust schema,
enabling native JSON parsing of boosters model files in Python.

Requires the [schema] extra to be installed:
    pip install boosters[schema]

Example:
-------
>>> from boosters.persist import JsonEnvelope, GBDTModelSchema
>>>
>>> with open("model.bstr.json") as f:
...     envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(f.read())
>>> print(envelope.model_type)  # "gbdt"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "CategoriesSchema",
    "FeatureType",
    "ForestSchema",
    "GBDTModelSchema",
    "GBLinearModelSchema",
    "JsonEnvelope",
    "LeafValuesSchema",
    "LinearCoefficientsSchema",
    "LinearWeightsSchema",
    "ModelMetaSchema",
    "TreeSchema",
]


def __getattr__(name: str):  # noqa: ANN202
    """Lazy import schema types, raising helpful error if pydantic is missing."""
    if name in __all__:
        try:
            from boosters.persist import schema as _schema

            return getattr(_schema, name)
        except ImportError as e:
            msg = (
                f"Cannot import '{name}' because pydantic is not installed. Install with: pip install boosters[schema]"
            )
            raise ImportError(msg) from e
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


if TYPE_CHECKING:
    from boosters.persist.schema import (
        CategoriesSchema,
        FeatureType,
        ForestSchema,
        GBDTModelSchema,
        GBLinearModelSchema,
        JsonEnvelope,
        LeafValuesSchema,
        LinearCoefficientsSchema,
        LinearWeightsSchema,
        ModelMetaSchema,
        TreeSchema,
    )
