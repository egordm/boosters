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

from typing import Annotated, Any, Literal

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

    model_config = ConfigDict(strict=True)

    bstr_version: int
    model_type: str
    model: T


# -----------------------------------------------------------------------------
# Simple Type Aliases
# -----------------------------------------------------------------------------

TaskKind = Literal["regression", "binary_classification", "multiclass_classification", "ranking"]

FeatureType = Literal["numeric", "categorical"]

Verbosity = Literal["silent", "warning", "info", "debug"]

UpdateStrategy = Literal["shotgun", "sequential"]


# -----------------------------------------------------------------------------
# Model Metadata
# -----------------------------------------------------------------------------


class ModelMetaSchema(BaseModel):
    """Model metadata including task type and feature information.

    Attributes:
    ----------
    task
        The learning task type.
    num_features
        Number of input features.
    num_classes
        Number of classes (for multiclass tasks), or None.
    feature_names
        Optional list of feature names.
    feature_types
        Optional list of feature types (numeric or categorical).
    """

    model_config = ConfigDict(strict=True)

    task: TaskKind
    num_features: int
    num_classes: int | None = None
    feature_names: list[str] | None = None
    feature_types: list[FeatureType] | None = None


# -----------------------------------------------------------------------------
# Leaf Values (discriminated union)
# -----------------------------------------------------------------------------


class ScalarLeafValues(BaseModel):
    """Scalar leaf values (one f64 per leaf)."""

    model_config = ConfigDict(strict=True)

    type: Literal["scalar"]
    values: list[float]


class VectorLeafValues(BaseModel):
    """Vector leaf values (multiple f64 per leaf)."""

    model_config = ConfigDict(strict=True)

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

    model_config = ConfigDict(strict=True)

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

    model_config = ConfigDict(strict=True)

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

    model_config = ConfigDict(strict=True)

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

    model_config = ConfigDict(strict=True)

    trees: list[TreeSchema]
    tree_groups: list[int] | None = None
    n_groups: int
    base_score: list[float]


# -----------------------------------------------------------------------------
# Objective Schema (discriminated union)
# -----------------------------------------------------------------------------


class SquaredLossObjective(BaseModel):
    """Squared error loss."""

    model_config = ConfigDict(strict=True)
    type: Literal["squared_loss"]


class AbsoluteLossObjective(BaseModel):
    """Absolute error loss."""

    model_config = ConfigDict(strict=True)
    type: Literal["absolute_loss"]


class LogisticLossObjective(BaseModel):
    """Logistic loss for binary classification."""

    model_config = ConfigDict(strict=True)
    type: Literal["logistic_loss"]


class HingeLossObjective(BaseModel):
    """Hinge loss for classification."""

    model_config = ConfigDict(strict=True)
    type: Literal["hinge_loss"]


class SoftmaxLossObjective(BaseModel):
    """Softmax loss for multiclass classification."""

    model_config = ConfigDict(strict=True)
    type: Literal["softmax_loss"]
    n_classes: int


class PinballLossObjective(BaseModel):
    """Pinball loss for quantile regression."""

    model_config = ConfigDict(strict=True)
    type: Literal["pinball_loss"]
    alphas: list[float]


class PseudoHuberLossObjective(BaseModel):
    """Pseudo-Huber loss for robust regression."""

    model_config = ConfigDict(strict=True)
    type: Literal["pseudo_huber_loss"]
    delta: float


class PoissonLossObjective(BaseModel):
    """Poisson loss for count data."""

    model_config = ConfigDict(strict=True)
    type: Literal["poisson_loss"]


class CustomObjective(BaseModel):
    """Custom user-defined objective."""

    model_config = ConfigDict(strict=True)
    type: Literal["custom"]
    name: str


ObjectiveSchema = Annotated[
    SquaredLossObjective
    | AbsoluteLossObjective
    | LogisticLossObjective
    | HingeLossObjective
    | SoftmaxLossObjective
    | PinballLossObjective
    | PseudoHuberLossObjective
    | PoissonLossObjective
    | CustomObjective,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Metric Schema (discriminated union)
# -----------------------------------------------------------------------------


class NoneMetric(BaseModel):
    """No evaluation metric."""

    model_config = ConfigDict(strict=True)
    type: Literal["none"]


class RmseMetric(BaseModel):
    """Root Mean Squared Error."""

    model_config = ConfigDict(strict=True)
    type: Literal["rmse"]


class MaeMetric(BaseModel):
    """Mean Absolute Error."""

    model_config = ConfigDict(strict=True)
    type: Literal["mae"]


class MapeMetric(BaseModel):
    """Mean Absolute Percentage Error."""

    model_config = ConfigDict(strict=True)
    type: Literal["mape"]


class LogLossMetric(BaseModel):
    """Log Loss (binary cross-entropy)."""

    model_config = ConfigDict(strict=True)
    type: Literal["log_loss"]


class AccuracyMetric(BaseModel):
    """Classification accuracy with threshold."""

    model_config = ConfigDict(strict=True)
    type: Literal["accuracy"]
    threshold: float


class MarginAccuracyMetric(BaseModel):
    """Margin-based accuracy."""

    model_config = ConfigDict(strict=True)
    type: Literal["margin_accuracy"]


class AucMetric(BaseModel):
    """Area Under ROC Curve."""

    model_config = ConfigDict(strict=True)
    type: Literal["auc"]


class MulticlassLogLossMetric(BaseModel):
    """Multiclass Log Loss."""

    model_config = ConfigDict(strict=True)
    type: Literal["multiclass_log_loss"]


class MulticlassAccuracyMetric(BaseModel):
    """Multiclass Accuracy."""

    model_config = ConfigDict(strict=True)
    type: Literal["multiclass_accuracy"]


class QuantileMetric(BaseModel):
    """Quantile loss metric."""

    model_config = ConfigDict(strict=True)
    type: Literal["quantile"]
    alphas: list[float]


class HuberMetric(BaseModel):
    """Huber loss metric."""

    model_config = ConfigDict(strict=True)
    type: Literal["huber"]
    delta: float


class PoissonDevianceMetric(BaseModel):
    """Poisson deviance metric."""

    model_config = ConfigDict(strict=True)
    type: Literal["poisson_deviance"]


class CustomMetric(BaseModel):
    """Custom user-defined metric."""

    model_config = ConfigDict(strict=True)
    type: Literal["custom"]
    name: str


MetricSchema = Annotated[
    NoneMetric
    | RmseMetric
    | MaeMetric
    | MapeMetric
    | LogLossMetric
    | AccuracyMetric
    | MarginAccuracyMetric
    | AucMetric
    | MulticlassLogLossMetric
    | MulticlassAccuracyMetric
    | QuantileMetric
    | HuberMetric
    | PoissonDevianceMetric
    | CustomMetric,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Growth Strategy (discriminated union)
# -----------------------------------------------------------------------------


class DepthWiseGrowth(BaseModel):
    """Depth-wise tree growth."""

    model_config = ConfigDict(strict=True)
    type: Literal["depth_wise"]
    max_depth: int


class LeafWiseGrowth(BaseModel):
    """Leaf-wise (best-first) tree growth."""

    model_config = ConfigDict(strict=True)
    type: Literal["leaf_wise"]
    max_leaves: int


GrowthStrategySchema = Annotated[
    DepthWiseGrowth | LeafWiseGrowth,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Feature Selector (discriminated union)
# -----------------------------------------------------------------------------


class CyclicSelector(BaseModel):
    """Cyclic feature selection."""

    model_config = ConfigDict(strict=True)
    type: Literal["cyclic"]


class ShuffleSelector(BaseModel):
    """Shuffled feature selection."""

    model_config = ConfigDict(strict=True)
    type: Literal["shuffle"]


class RandomSelector(BaseModel):
    """Random feature selection."""

    model_config = ConfigDict(strict=True)
    type: Literal["random"]


class GreedySelector(BaseModel):
    """Greedy top-k feature selection."""

    model_config = ConfigDict(strict=True)
    type: Literal["greedy"]
    top_k: int


class ThriftySelector(BaseModel):
    """Thrifty top-k feature selection."""

    model_config = ConfigDict(strict=True)
    type: Literal["thrifty"]
    top_k: int


FeatureSelectorSchema = Annotated[
    CyclicSelector | ShuffleSelector | RandomSelector | GreedySelector | ThriftySelector,
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Binning Config
# -----------------------------------------------------------------------------


class BinningConfigSchema(BaseModel):
    """Histogram binning configuration.

    Attributes:
    ----------
    max_bins
        Maximum number of bins for histograms.
    sparsity_threshold
        Threshold for treating features as sparse.
    enable_bundling
        Whether to enable exclusive feature bundling.
    max_categorical_cardinality
        Maximum cardinality for categorical features.
    sample_cnt
        Number of samples for bin boundary estimation.
    """

    model_config = ConfigDict(strict=True)

    max_bins: int
    sparsity_threshold: float
    enable_bundling: bool
    max_categorical_cardinality: int
    sample_cnt: int


# -----------------------------------------------------------------------------
# Linear Leaf Config
# -----------------------------------------------------------------------------


class LinearLeafConfigSchema(BaseModel):
    """Linear-in-leaves configuration.

    Attributes:
    ----------
    lambda_
        L2 regularization for linear coefficients.
    alpha
        L1 regularization for linear coefficients.
    max_iterations
        Maximum iterations for coefficient fitting.
    tolerance
        Convergence tolerance.
    min_samples
        Minimum samples to fit linear model in a leaf.
    coefficient_threshold
        Threshold below which coefficients are zeroed.
    max_features
        Maximum features to use in linear model.
    """

    model_config = ConfigDict(strict=True)

    lambda_: float = Field(alias="lambda")
    alpha: float
    max_iterations: int
    tolerance: float
    min_samples: int
    coefficient_threshold: float
    max_features: int


# -----------------------------------------------------------------------------
# GBDT Config
# -----------------------------------------------------------------------------


class GBDTConfigSchema(BaseModel):
    """GBDT training configuration.

    Attributes:
    ----------
    objective
        Loss function to optimize.
    metric
        Evaluation metric (None means no metric).
    n_trees
        Number of boosting rounds.
    learning_rate
        Step size shrinkage.
    growth_strategy
        Tree growth strategy (depth-wise or leaf-wise).
    max_onehot_cats
        Maximum categories for one-hot encoding.
    lambda_
        L2 regularization on leaf weights.
    alpha
        L1 regularization on leaf weights.
    min_child_weight
        Minimum sum of hessians in a child.
    min_gain
        Minimum gain to make a split.
    min_samples_leaf
        Minimum samples in a leaf.
    subsample
        Row subsampling ratio.
    colsample_bytree
        Column subsampling ratio per tree.
    colsample_bylevel
        Column subsampling ratio per level.
    binning
        Histogram binning configuration.
    linear_leaves
        Optional linear-in-leaves configuration.
    early_stopping_rounds
        Early stopping patience (None = disabled).
    cache_size
        Histogram cache size.
    seed
        Random seed.
    verbosity
        Logging verbosity level.
    extra
        Additional parameters.
    """

    model_config = ConfigDict(strict=True, populate_by_name=True)

    objective: ObjectiveSchema
    metric: MetricSchema | None = None
    n_trees: int
    learning_rate: float
    growth_strategy: GrowthStrategySchema
    max_onehot_cats: int
    lambda_: float = Field(alias="lambda")
    alpha: float
    min_child_weight: float
    min_gain: float
    min_samples_leaf: int
    subsample: float
    colsample_bytree: float
    colsample_bylevel: float
    binning: BinningConfigSchema
    linear_leaves: LinearLeafConfigSchema | None = None
    early_stopping_rounds: int | None = None
    cache_size: int
    seed: int
    verbosity: Verbosity
    extra: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Full GBDT Model Schema
# -----------------------------------------------------------------------------


class GBDTModelSchema(BaseModel):
    """Complete GBDT model schema.

    Attributes:
    ----------
    meta
        Model metadata (task type, features, etc.).
    forest
        Tree ensemble.
    config
        Training configuration.
    """

    model_config = ConfigDict(strict=True)

    meta: ModelMetaSchema
    forest: ForestSchema
    config: GBDTConfigSchema

    MODEL_TYPE: str = "gbdt"


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

    model_config = ConfigDict(strict=True)

    values: list[float]
    num_features: int
    num_groups: int


# -----------------------------------------------------------------------------
# GBLinear Config
# -----------------------------------------------------------------------------


class GBLinearConfigSchema(BaseModel):
    """GBLinear training configuration.

    Attributes:
    ----------
    objective
        Loss function to optimize.
    metric
        Evaluation metric (None = no metric).
    n_rounds
        Number of boosting rounds.
    learning_rate
        Step size shrinkage.
    alpha
        L1 regularization.
    lambda_
        L2 regularization.
    update_strategy
        Coordinate descent update strategy.
    feature_selector
        Feature selection strategy.
    max_delta_step
        Maximum per-coordinate Newton step (0 = disabled).
    early_stopping_rounds
        Early stopping patience (None = disabled).
    seed
        Random seed.
    verbosity
        Logging verbosity level.
    extra
        Additional parameters.
    """

    model_config = ConfigDict(strict=True, populate_by_name=True)

    objective: ObjectiveSchema
    metric: MetricSchema | None = None
    n_rounds: int
    learning_rate: float
    alpha: float
    lambda_: float = Field(alias="lambda")
    update_strategy: UpdateStrategy
    feature_selector: FeatureSelectorSchema
    max_delta_step: float
    early_stopping_rounds: int | None = None
    seed: int
    verbosity: Verbosity
    extra: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Full GBLinear Model Schema
# -----------------------------------------------------------------------------


class GBLinearModelSchema(BaseModel):
    """Complete GBLinear model schema.

    Attributes:
    ----------
    meta
        Model metadata (task type, features, etc.).
    weights
        Linear model weights.
    base_score
        Base score(s).
    config
        Training configuration.
    """

    model_config = ConfigDict(strict=True)

    meta: ModelMetaSchema
    weights: LinearWeightsSchema
    base_score: list[float]
    config: GBLinearConfigSchema

    MODEL_TYPE: str = "gblinear"
