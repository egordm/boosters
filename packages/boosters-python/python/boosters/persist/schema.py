"""Pydantic v2 models mirroring the Rust boosters schema.

These models enable native JSON parsing of `.bstr.json` model files.
Field names and types match the Rust serialization format exactly.

Example:
-------
>>> import json
>>> from boosters.persist.schema import JsonEnvelope, GBDTModelSchema
>>>
>>> # Parse a GBDT model file
>>> with open("model.bstr.json") as f:
...     envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(f.read())
>>> print(envelope.model_type)  # "gbdt"
>>> print(len(envelope.model.forest.trees))  # number of trees
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# -----------------------------------------------------------------------------
# JSON Envelope
# -----------------------------------------------------------------------------


class JsonEnvelope[T](BaseModel):
    """Top-level JSON envelope wrapping any model schema.

    The envelope provides version and type information for schema evolution.

    Attributes:
    ----------
    bstr_version
        Schema version number (currently 1).
    model_type
        Model type identifier ("gbdt" or "gblinear").
    model
        The model schema payload.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    bstr_version: int
    model_type: str
    model: T


# -----------------------------------------------------------------------------
# Simple Type Aliases
# -----------------------------------------------------------------------------

OutputTransform = Literal["identity", "sigmoid", "softmax"]

FeatureType = Literal["numeric", "categorical"]


# -----------------------------------------------------------------------------
# Model Metadata
# -----------------------------------------------------------------------------


class ModelMetaSchema(BaseModel):
    """Model metadata.

    Mirrors Rust `ModelMetaSchema`.

    Attributes:
    ----------
    num_features
        Number of input features.
    feature_names
        Optional list of feature names.
    feature_types
        Optional list of feature types (numeric or categorical).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    num_features: int
    num_groups: int = 1
    feature_names: list[str] | None = None
    feature_types: list[FeatureType] | None = None
    objective_name: str | None = None


# -----------------------------------------------------------------------------
# Leaf Values (discriminated union)
# -----------------------------------------------------------------------------


class ScalarLeafValues(BaseModel):
    """Scalar leaf values (one f64 per leaf)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["scalar"]
    values: list[float]


class VectorLeafValues(BaseModel):
    """Vector leaf values (multiple f64 per leaf)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["vector"]
    values: list[list[float]]


LeafValuesSchema = Annotated[
    ScalarLeafValues | VectorLeafValues,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Categories and Linear Coefficients
# -----------------------------------------------------------------------------


class CategoriesSchema(BaseModel):
    """Category mapping for categorical splits.

    Attributes:
    ----------
    node_indices
        Indices of nodes that have category sets.
    category_sets
        Category sets (one per node in node_indices).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    node_indices: list[int] = Field(default_factory=list)
    category_sets: list[list[int]] = Field(default_factory=list)


class LinearCoefficientsSchema(BaseModel):
    """Linear coefficients for linear-in-leaves models.

    Attributes:
    ----------
    node_indices
        Indices of nodes that have linear coefficients.
    coefficients
        Coefficient arrays (one per node in node_indices).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    node_indices: list[int] = Field(default_factory=list)
    coefficients: list[list[float]] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Tree Schema
# -----------------------------------------------------------------------------


class TreeSchema(BaseModel):
    """Single tree in Structure-of-Arrays layout.

    Attributes:
    ----------
    num_nodes
        Total number of nodes (internal + leaves).
    split_indices
        Split feature index for each internal node.
    thresholds
        Split threshold for each internal node.
    children_left
        Left child index for each internal node (0 = missing indicator).
    children_right
        Right child index for each internal node (0 = missing indicator).
    default_left
        Default direction for missing values (True = go left).
    leaf_values
        Leaf values (scalar or vector).
    categories
        Optional category mappings for categorical splits.
    linear_coefficients
        Optional linear coefficients for linear-in-leaves.
    gains
        Optional split gains for each internal node.
    covers
        Optional sample covers for each node.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    num_nodes: int
    split_indices: list[int]
    thresholds: list[float]
    children_left: list[int]
    children_right: list[int]
    default_left: list[bool]
    leaf_values: LeafValuesSchema
    categories: CategoriesSchema = Field(default_factory=CategoriesSchema)
    linear_coefficients: LinearCoefficientsSchema = Field(default_factory=LinearCoefficientsSchema)
    gains: list[float] | None = None
    covers: list[float] | None = None


# -----------------------------------------------------------------------------
# Forest Schema
# -----------------------------------------------------------------------------


class ForestSchema(BaseModel):
    """Collection of trees forming the ensemble.

    Attributes:
    ----------
    trees
        Trees in iteration order.
    tree_groups
        Tree group boundaries (for multi-output models).
    n_groups
        Number of output groups.
    base_score
        Base score(s) for the ensemble.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    trees: list[TreeSchema]
    tree_groups: list[int] | None = None
    n_groups: int
    base_score: list[float]


# -----------------------------------------------------------------------------
# Linear Weights
# -----------------------------------------------------------------------------


class LinearWeightsSchema(BaseModel):
    """GBLinear weight storage.

    Attributes:
    ----------
    values
        Weight values in [group x feature] order (row-major).
    num_features
        Number of features.
    num_groups
        Number of output groups.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    values: list[float]
    num_features: int
    num_groups: int


# -----------------------------------------------------------------------------
# Full Model Schemas
# -----------------------------------------------------------------------------


class GBDTModelSchema(BaseModel):
    """Complete GBDT model schema.

    Mirrors Rust `GBDTModelSchema`.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    meta: ModelMetaSchema
    forest: ForestSchema
    output_transform: OutputTransform

    MODEL_TYPE: str = "gbdt"


class GBLinearModelSchema(BaseModel):
    """Complete GBLinear model schema.

    Mirrors Rust `GBLinearModelSchema`.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    meta: ModelMetaSchema
    weights: LinearWeightsSchema
    base_score: list[float]
    output_transform: OutputTransform

    MODEL_TYPE: str = "gblinear"
