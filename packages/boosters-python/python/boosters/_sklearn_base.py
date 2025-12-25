"""Base classes and utilities for sklearn integration.

This module provides shared logic for sklearn-compatible estimators,
including kwargs→config conversion and common estimator patterns.
"""

from boosters import (
    CategoricalConfig,
    EFBConfig,
    GBDTConfig,
    GBLinearConfig,
    LinearLeavesConfig,
    RegularizationConfig,
    SamplingConfig,
    TreeConfig,
)


def build_gbdt_config(
    *,
    objective,
    metric=None,
    # Top-level
    n_estimators: int = 100,
    learning_rate: float = 0.3,
    early_stopping_rounds: int | None = None,
    seed: int = 42,
    # Tree
    max_depth: int = -1,
    n_leaves: int = 31,
    min_samples_leaf: int = 1,
    min_gain_to_split: float = 0.0,
    growth_strategy: str = "depthwise",
    # Regularization
    l1: float = 0.0,
    l2: float = 1.0,
    min_hessian: float = 1.0,
    # Sampling (sklearn-friendly names)
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    goss_top_rate: float = 0.0,
    goss_other_rate: float = 0.0,
    # Categorical (sklearn-friendly names)
    min_samples_cat: int = 10,
    max_cat_threshold: int = 256,
    # EFB (sklearn-friendly names)
    enable_efb: bool = True,
    max_conflict_rate: float = 0.0,
    # Linear leaves (sklearn-friendly names)
    enable_linear_leaves: bool = False,
    linear_leaves_l2: float = 0.01,
) -> GBDTConfig:
    """Build a GBDTConfig from flat kwargs.

    This function takes sklearn-style flat keyword arguments and constructs
    the nested config structure expected by the core API.

    Returns:
        GBDTConfig with all nested configs populated.
    """
    tree = TreeConfig(
        max_depth=max_depth,
        n_leaves=n_leaves,
        min_samples_leaf=min_samples_leaf,
        min_gain_to_split=min_gain_to_split,
        growth_strategy=growth_strategy,
    )

    regularization = RegularizationConfig(
        l1=l1,
        l2=l2,
        min_hessian=min_hessian,
    )

    # Map sklearn-style params to core API params
    sampling = SamplingConfig(
        subsample=subsample,
        colsample=colsample_bytree,  # sklearn uses colsample_bytree
        goss_alpha=goss_top_rate,  # sklearn uses goss_top_rate
        goss_beta=goss_other_rate,  # sklearn uses goss_other_rate
    )

    categorical = CategoricalConfig(
        max_categories=max_cat_threshold,  # sklearn uses max_cat_threshold
        min_category_count=min_samples_cat,  # sklearn uses min_samples_cat
    )

    efb = EFBConfig(
        enable=enable_efb,  # core uses enable, not enable_efb
        max_conflict_rate=max_conflict_rate,
    )

    linear_leaves = LinearLeavesConfig(
        enable=enable_linear_leaves,  # core uses enable
        l2=linear_leaves_l2,
    )

    return GBDTConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        objective=objective,
        metric=metric,
        tree=tree,
        regularization=regularization,
        sampling=sampling,
        categorical=categorical,
        efb=efb,
        linear_leaves=linear_leaves,
    )


def build_gblinear_config(
    *,
    objective,
    metric=None,
    n_estimators: int = 100,
    learning_rate: float = 0.5,
    l1: float = 0.0,
    l2: float = 1.0,
    early_stopping_rounds: int | None = None,
    seed: int = 42,
) -> GBLinearConfig:
    """Build a GBLinearConfig from flat kwargs.

    Returns:
        GBLinearConfig with all parameters populated.
    """
    return GBLinearConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        l1=l1,
        l2=l2,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        objective=objective,
        metric=metric,
    )
