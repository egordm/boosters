"""Conversion utilities for XGBoost and LightGBM models.

This module provides functions to convert models trained with XGBoost or LightGBM
to the boosters native .bstr format. The conversion uses the respective library's
introspection APIs to extract tree structure, avoiding format parsing.

Example:
    >>> import xgboost as xgb
    >>> from boosters.convert import xgboost_to_bstr
    >>>
    >>> # Train with XGBoost
    >>> xgb_model = xgb.train(params, dtrain)
    >>>
    >>> # Convert to boosters format (one-time)
    >>> xgboost_to_bstr(xgb_model, "model.bstr")
    >>>
    >>> # Load with boosters (fast, for inference)
    >>> from boosters import GBDTBooster
    >>> model = GBDTBooster.load("model.bstr")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightgbm as lgb
    import xgboost as xgb


def xgboost_to_bstr(
    model: xgb.Booster | xgb.XGBModel,
    path: str | Path,
    *,
    include_feature_names: bool = True,
) -> None:
    """Convert XGBoost model to boosters .bstr format.

    Uses XGBoost's introspection APIs to extract model structure:
    - model.trees_to_dataframe() for tree structure
    - model.get_config() or attributes for objective, base_score

    This avoids parsing XGBoost's serialization format directly.

    Args:
        model: Trained XGBoost Booster or XGBModel (XGBRegressor, XGBClassifier, etc.)
        path: Output .bstr file path
        include_feature_names: Include feature names in metadata

    Raises:
        ImportError: If xgboost is not installed
        ValueError: If the model type is not supported (e.g., gblinear)

    Example:
        >>> import xgboost as xgb
        >>> from boosters.convert import xgboost_to_bstr
        >>>
        >>> # Train XGBoost model
        >>> xgb_model = xgb.XGBRegressor().fit(X, y)
        >>>
        >>> # Convert to boosters format
        >>> xgboost_to_bstr(xgb_model, "model.bstr")
        >>>
        >>> # Load with boosters
        >>> from boosters import GBDTBooster
        >>> model = GBDTBooster.load("model.bstr")
    """
    import importlib.util

    if importlib.util.find_spec("xgboost") is None:
        raise ImportError(
            "xgboost is required for xgboost_to_bstr. "
            "Install with: pip install boosters[xgboost]"
        )

    from boosters import GBDTBooster

    # Handle sklearn-style wrapper
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    else:
        booster = model

    # Check booster type - we only support gbtree (and dart which uses same structure)
    config = booster.save_config()
    import json

    config_dict = json.loads(config)
    booster_type = (
        config_dict.get("learner", {})
        .get("gradient_booster", {})
        .get("name", "gbtree")
    )

    if booster_type == "gblinear":
        raise ValueError(
            "gblinear models are not yet supported by xgboost_to_bstr. "
            "Use GBLinearBooster for linear models."
        )

    # Get tree dataframe
    trees_df = booster.trees_to_dataframe()

    # Get number of features
    num_feature = int(
        config_dict.get("learner", {})
        .get("learner_model_param", {})
        .get("num_feature", 0)
    )

    # Get base score
    base_score_str = (
        config_dict.get("learner", {})
        .get("learner_model_param", {})
        .get("base_score", "0.5")
    )
    # Handle both string and float formats
    if isinstance(base_score_str, str):
        # XGBoost sometimes stores as "[0.5]" or "0.5"
        base_score_str = base_score_str.strip("[]")
        base_score = float(base_score_str)
    else:
        base_score = float(base_score_str)

    # Get number of classes for multiclass
    num_class = int(
        config_dict.get("learner", {})
        .get("learner_model_param", {})
        .get("num_class", 1)
    )
    n_groups = max(1, num_class)

    # Get feature names
    feature_names = None
    if include_feature_names:
        try:
            feature_names = booster.feature_names
            if feature_names is not None:
                feature_names = list(feature_names)
        except AttributeError:
            pass

    # Extract trees from dataframe
    trees = []
    tree_groups = []

    for tree_id in trees_df["Tree"].unique():
        tree_df = trees_df[trees_df["Tree"] == tree_id].copy()

        # Sort by node ID to ensure correct order
        tree_df = tree_df.sort_values("Node").reset_index(drop=True)

        len(tree_df)

        # Initialize arrays
        split_indices = []
        thresholds = []
        left_children = []
        right_children = []
        default_left = []
        is_leaf = []
        leaf_values = []

        # Build node ID to index mapping
        node_id_to_idx = {row["ID"]: idx for idx, row in tree_df.iterrows()}

        for _, row in tree_df.iterrows():
            is_leaf_node = row["Feature"] == "Leaf"
            is_leaf.append(is_leaf_node)

            if is_leaf_node:
                # Leaf node
                split_indices.append(0)
                thresholds.append(0.0)
                left_children.append(0)
                right_children.append(0)
                default_left.append(False)
                leaf_values.append(float(row["Gain"]))  # Gain column contains leaf value
            else:
                # Internal node
                # Parse feature index from "f0", "f1", etc.
                feature = row["Feature"]
                if feature.startswith("f"):
                    feat_idx = int(feature[1:])
                else:
                    # Named feature - find index
                    if feature_names and feature in feature_names:
                        feat_idx = feature_names.index(feature)
                    else:
                        feat_idx = int(feature.replace("f", ""))

                split_indices.append(feat_idx)
                thresholds.append(float(row["Split"]))

                # Get child indices
                left_id = row["Yes"]
                right_id = row["No"]
                left_children.append(node_id_to_idx.get(left_id, 0))
                right_children.append(node_id_to_idx.get(right_id, 0))

                # Default direction
                missing_id = row["Missing"]
                default_left.append(missing_id == left_id)

                leaf_values.append(0.0)  # Non-leaf nodes have no value

        tree_dict = {
            "split_indices": split_indices,
            "thresholds": thresholds,
            "left_children": left_children,
            "right_children": right_children,
            "default_left": default_left,
            "is_leaf": is_leaf,
            "leaf_values": leaf_values,
        }
        trees.append(tree_dict)

        # For multiclass, trees cycle through classes
        tree_groups.append(tree_id % n_groups)

    # Build base scores array
    base_scores = [base_score] * n_groups

    # Create boosters model from trees
    booster_model = GBDTBooster.from_trees(
        trees=trees,
        tree_groups=tree_groups,
        n_features=num_feature,
        base_scores=base_scores,
        feature_names=feature_names,
    )

    # Save to .bstr format
    booster_model.save(str(path))


def lightgbm_to_bstr(
    model: lgb.Booster | lgb.LGBMModel,
    path: str | Path,
    *,
    include_feature_names: bool = True,
) -> None:
    """Convert LightGBM model to boosters .bstr format.

    Uses LightGBM's introspection APIs:
    - model.trees_to_dataframe() for standard tree structure
    - model.dump_model() for configuration and metadata

    This avoids parsing LightGBM's text format directly.

    Args:
        model: Trained LightGBM Booster or LGBMModel (LGBMRegressor, etc.)
        path: Output .bstr file path
        include_feature_names: Include feature names in metadata

    Raises:
        ImportError: If lightgbm is not installed
        ValueError: If the model uses unsupported features (e.g., linear trees)

    Example:
        >>> import lightgbm as lgb
        >>> from boosters.convert import lightgbm_to_bstr
        >>>
        >>> # Train LightGBM model
        >>> lgb_model = lgb.LGBMRegressor().fit(X, y)
        >>>
        >>> # Convert to boosters format
        >>> lightgbm_to_bstr(lgb_model, "model.bstr")
        >>>
        >>> # Load with boosters
        >>> from boosters import GBDTBooster
        >>> model = GBDTBooster.load("model.bstr")
    """
    import importlib.util

    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError(
            "lightgbm is required for lightgbm_to_bstr. "
            "Install with: pip install boosters[lightgbm]"
        )

    from boosters import GBDTBooster

    # Handle sklearn-style wrapper
    if hasattr(model, "booster_"):
        booster = model.booster_
    else:
        booster = model

    # Get model dump for metadata
    model_dump = booster.dump_model()

    # Check for linear trees (not yet supported)
    if model_dump.get("parameters", {}).get("linear_tree", False):
        raise ValueError(
            "Linear trees are not yet supported by lightgbm_to_bstr. "
            "Use models trained without linear_tree=True."
        )

    # Get number of features
    num_feature = model_dump.get("max_feature_idx", 0) + 1

    # Get number of classes
    num_class = model_dump.get("num_class", 1)
    n_groups = max(1, num_class)

    # Get feature names
    feature_names = None
    if include_feature_names:
        feature_names = model_dump.get("feature_names")
        if feature_names is not None:
            feature_names = list(feature_names)

    # Get tree dataframe
    trees_df = booster.trees_to_dataframe()

    # Extract trees from dataframe
    trees = []
    tree_groups = []

    for tree_idx in trees_df["tree_index"].unique():
        tree_df = trees_df[trees_df["tree_index"] == tree_idx].copy()

        # Build node name to row mapping
        node_data = {row["node_index"]: row for _, row in tree_df.iterrows()}

        # Find root node (the one with no parent, or parent_index is None)
        root_nodes = [row["node_index"] for _, row in tree_df.iterrows()
                     if row.get("parent_index") is None or
                     (isinstance(row.get("parent_index"), float) and
                      row.get("parent_index") != row.get("parent_index"))]

        if not root_nodes:
            # Fallback: find by node_depth == 1
            root_nodes = [row["node_index"] for _, row in tree_df.iterrows()
                         if row["node_depth"] == 1]

        if not root_nodes:
            raise ValueError(f"Could not find root node for tree {tree_idx}")

        root_node_name = root_nodes[0]

        # BFS to build arrays in correct order
        # Node arrays indexed by our new indices
        split_indices = []
        thresholds = []
        left_children = []
        right_children = []
        default_left = []
        is_leaf = []
        leaf_values = []

        # Map from LightGBM node name to our array index
        node_name_to_idx: dict[str, int] = {}

        # BFS queue
        from collections import deque
        queue: deque[str] = deque([root_node_name])

        while queue:
            node_name = queue.popleft()
            row = node_data[node_name]

            # Assign new index for this node
            idx = len(split_indices)
            node_name_to_idx[node_name] = idx

            is_leaf_node = row["left_child"] is None or (
                isinstance(row["left_child"], float) and row["left_child"] != row["left_child"]
            )  # NaN check
            is_leaf.append(is_leaf_node)

            if is_leaf_node:
                # Leaf node
                split_indices.append(0)
                thresholds.append(0.0)
                left_children.append(0)  # Will not be used
                right_children.append(0)  # Will not be used
                default_left.append(False)
                leaf_values.append(float(row["value"]))
            else:
                # Internal node
                # Get feature index
                split_feature = row["split_feature"]
                if isinstance(split_feature, str):
                    # Named feature - find index
                    if feature_names and split_feature in feature_names:
                        feat_idx = feature_names.index(split_feature)
                    else:
                        # Try parsing as "Column_N"
                        feat_idx = int(split_feature.replace("Column_", ""))
                else:
                    feat_idx = int(split_feature)

                split_indices.append(feat_idx)

                # Threshold
                threshold = row["threshold"]
                thresholds.append(float(threshold) if threshold is not None else 0.0)

                # Placeholders for children - will be filled later
                left_children.append(0)  # Will be updated after we know child indices
                right_children.append(0)

                # Queue children for processing
                left_child = row["left_child"]
                right_child = row["right_child"]
                queue.append(left_child)
                queue.append(right_child)

                # Default direction - LightGBM uses decision_type and missing_direction
                missing_dir = row.get("missing_direction", "None")
                default_left.append(str(missing_dir).lower() == "left")

                leaf_values.append(0.0)  # Non-leaf nodes have no value

        # Second pass: fill in child indices now that we have the mapping
        for _, row in tree_df.iterrows():
            node_name = row["node_index"]
            if node_name not in node_name_to_idx:
                continue
            idx = node_name_to_idx[node_name]

            if not is_leaf[idx]:
                left_child = row["left_child"]
                right_child = row["right_child"]
                left_children[idx] = node_name_to_idx.get(left_child, 0)
                right_children[idx] = node_name_to_idx.get(right_child, 0)

        tree_dict = {
            "split_indices": split_indices,
            "thresholds": thresholds,
            "left_children": left_children,
            "right_children": right_children,
            "default_left": default_left,
            "is_leaf": is_leaf,
            "leaf_values": leaf_values,
        }
        trees.append(tree_dict)

        # For multiclass, trees cycle through classes
        tree_groups.append(tree_idx % n_groups)

    # Build base scores array (LightGBM uses 0 as default)
    base_scores = [0.0] * n_groups

    # Create boosters model from trees
    booster_model = GBDTBooster.from_trees(
        trees=trees,
        tree_groups=tree_groups,
        n_features=num_feature,
        base_scores=base_scores,
        feature_names=feature_names,
    )

    # Save to .bstr format
    booster_model.save(str(path))


__all__ = [
    "lightgbm_to_bstr",
    "xgboost_to_bstr",
]
