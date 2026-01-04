"""XGBoost and LightGBM model conversion utilities.

Convert external gradient boosting models to boosters' native `.bstr.json` format.

This module provides pure-Python conversion that produces JSON-serializable schema
objects without requiring the boosters native runtime.

Example:
-------
>>> from boosters.convert import xgboost_to_json_bytes
>>>
>>> # Convert from XGBoost model file
>>> json_bytes = xgboost_to_json_bytes("model.json")
>>> with open("model.bstr.json", "wb") as f:
...     f.write(json_bytes)
>>>
>>> # Or convert from XGBoost Booster object
>>> import xgboost as xgb
>>> booster = xgb.Booster(model_file="model.json")
>>> json_bytes = xgboost_to_json_bytes(booster)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from boosters.persist.schema import (
    CategoriesSchema,
    FeatureType,
    ForestSchema,
    GBDTModelSchema,
    GBLinearModelSchema,
    JsonEnvelope,
    LinearCoefficientsSchema,
    LinearWeightsSchema,
    ModelMetaSchema,
    OutputTransform,
    ScalarLeafValues,
    TreeSchema,
)

if TYPE_CHECKING:
    import lightgbm as lgb
    import xgboost as xgb


# =============================================================================
# XGBoost Conversion
# =============================================================================


def _objective_to_output_transform(objective_name: str, n_groups: int) -> OutputTransform:
    """Infer the output transform from objective + number of groups."""
    obj = objective_name.lower().split()[0]

    if n_groups > 1:
        return "softmax"

    if obj in ("multi:softprob", "multi:softmax", "multiclass", "multiclassova", "multiclass_ova"):
        return "softmax"

    if obj in ("binary:logistic", "reg:logistic", "binary", "binary_logloss", "cross_entropy"):
        return "sigmoid"

    return "identity"


def _load_xgboost_json(path_or_booster: str | Path | xgb.Booster) -> dict[str, Any]:
    """Load XGBoost JSON model from path or Booster object."""
    if isinstance(path_or_booster, (str, Path)):
        with Path(path_or_booster).open() as f:
            data: Any = json.load(f)
            if not isinstance(data, dict):
                msg = "XGBoost JSON root must be an object"
                raise TypeError(msg)
            return cast(dict[str, Any], data)
    else:
        # xgb.Booster - save to JSON bytes and parse
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            path_or_booster.save_model(tmp.name)
            with Path(tmp.name).open() as f:
                data: Any = json.load(f)
                if not isinstance(data, dict):
                    msg = "XGBoost JSON root must be an object"
                    raise TypeError(msg)
                return cast(dict[str, Any], data)


def _parse_number_field(value: Any) -> float:
    """Parse a potentially stringified number field from XGBoost JSON."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Handle bracketed arrays like "[1.5E0]"
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            # Check if it contains commas (multi-value array)
            if "," in inner:
                return float(inner.split(",")[0])
            return float(inner)
        return float(s)
    if isinstance(value, list):
        return float(value[0])
    msg = f"Cannot parse number from: {value}"
    raise ValueError(msg)


def _parse_base_scores(value: Any, n_groups: int, objective: str) -> list[float]:
    """Parse base_score(s) from XGBoost JSON and convert to margin space.

    XGBoost multiclass can have per-class base_scores as a stringified array.
    """
    scores: list[float] = []

    if isinstance(value, (int, float)):
        scores = [float(value)] * n_groups
    elif isinstance(value, list):
        scores = [float(v) for v in value]
    elif isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            scores = [float(x.strip()) for x in inner.split(",")] if "," in inner else [float(inner)] * n_groups
        else:
            scores = [float(s)] * n_groups
    else:
        msg = f"Cannot parse base_score from: {value}"
        raise TypeError(msg)

    # Extend or truncate to match n_groups
    if len(scores) < n_groups:
        scores = scores + [scores[0]] * (n_groups - len(scores))
    elif len(scores) > n_groups:
        scores = scores[:n_groups]

    # Convert to margin space
    return [_prob_to_margin(s, objective) for s in scores]


def _prob_to_margin(base_score: float, objective: str) -> float:
    """Convert base_score from probability space to margin space.

    XGBoost stores base_score in probability/original space in JSON,
    but the predictor uses margin space.
    """
    if objective in ("binary:logistic", "reg:logistic"):
        # logit transform: log(p / (1 - p))
        p = max(1e-7, min(1.0 - 1e-7, base_score))
        return math.log(p / (1.0 - p))
    if objective in ("reg:gamma", "reg:tweedie"):
        return math.log(max(1e-7, base_score))
    return base_score


def _xgboost_feature_type_to_boosters(ft: str) -> FeatureType:
    """Map XGBoost feature type to boosters feature type."""
    if ft in ("c", "categorical"):
        return "categorical"
    return "numeric"


def _convert_xgb_tree(tree_data: dict[str, Any]) -> TreeSchema:
    """Convert a single XGBoost tree to TreeSchema.

    The output format matches boosters' Rust schema where ALL nodes (internal + leaves)
    appear in every array. Leaf nodes have children_left=0, children_right=0 as sentinels.
    """
    num_nodes = int(tree_data["tree_param"]["num_nodes"])
    left_children = tree_data["left_children"]
    right_children = tree_data["right_children"]
    split_indices_raw = tree_data["split_indices"]
    split_conditions = tree_data["split_conditions"]
    default_left = tree_data["default_left"]
    base_weights = tree_data["base_weights"]

    # XGBoost split_type: 0 = numeric, 1 = categorical
    split_types = tree_data.get("split_type", [0] * num_nodes)

    # Build categorical map
    cat_nodes = tree_data.get("categories_nodes", [])
    cat_segments = tree_data.get("categories_segments", [])
    cat_sizes = tree_data.get("categories_sizes", [])
    categories_raw = tree_data.get("categories", [])

    cat_map: dict[int, list[int]] = {}
    for i, node_idx in enumerate(cat_nodes):
        start = cat_segments[i]
        size = cat_sizes[i]
        cat_map[node_idx] = [int(c) for c in categories_raw[start : start + size]]

    # XGBoost marks leaves with left_child == -1
    is_leaf = [lc == -1 for lc in left_children]

    # Output arrays - same length as num_nodes
    out_split_indices: list[int] = []
    out_thresholds: list[float] = []
    out_children_left: list[int] = []
    out_children_right: list[int] = []
    out_default_left: list[bool] = []
    out_leaf_values: list[float] = []

    # Categories (only for categorical split nodes)
    cat_node_indices: list[int] = []
    cat_sets: list[list[int]] = []

    for node_id in range(num_nodes):
        out_split_indices.append(int(split_indices_raw[node_id]))
        out_thresholds.append(float(split_conditions[node_id]))
        out_leaf_values.append(float(base_weights[node_id]))
        out_default_left.append(default_left[node_id] != 0)

        if is_leaf[node_id]:
            # Leaf node: use 0 as sentinel for "no child"
            out_children_left.append(0)
            out_children_right.append(0)
        else:
            # Internal node: convert -1 sentinel to actual child indices
            lc = left_children[node_id]
            rc = right_children[node_id]
            out_children_left.append(lc if lc != -1 else 0)
            out_children_right.append(rc if rc != -1 else 0)

            # Track categorical nodes
            if split_types[node_id] == 1:  # categorical
                cat_node_indices.append(node_id)
                cat_sets.append(cat_map.get(node_id, []))

    return TreeSchema(
        num_nodes=num_nodes,
        split_indices=out_split_indices,
        thresholds=out_thresholds,
        children_left=out_children_left,
        children_right=out_children_right,
        default_left=out_default_left,
        leaf_values=ScalarLeafValues(type="scalar", values=out_leaf_values),
        categories=CategoriesSchema(node_indices=cat_node_indices, category_sets=cat_sets),
    )


def _convert_xgb_gbtree(
    xgb_json: dict[str, Any],
) -> tuple[GBDTModelSchema, str]:
    """Convert XGBoost gbtree/dart model to GBDTModelSchema."""
    learner = xgb_json["learner"]
    learner_params = learner["learner_model_param"]
    objective_data = learner["objective"]
    gradient_booster = learner["gradient_booster"]

    # Get model type (gbtree or dart)
    booster_name = gradient_booster["name"]
    model_trees = gradient_booster["gbtree"]["model"] if booster_name == "dart" else gradient_booster["model"]

    # Parse learner params
    num_features = int(_parse_number_field(learner_params["num_feature"]))
    num_class = int(_parse_number_field(learner_params.get("num_class", 1)))

    # Get objective name
    objective_name = objective_data["name"]

    # Number of groups
    n_groups = max(1, num_class) if num_class > 2 else 1

    # Parse and convert base scores
    base_scores = _parse_base_scores(
        learner_params["base_score"],
        n_groups,
        objective_name,
    )

    # Convert trees
    trees = [_convert_xgb_tree(t) for t in model_trees["trees"]]

    # Tree groups (which group each tree belongs to)
    tree_info = model_trees.get("tree_info", list(range(len(trees))))
    tree_groups = [int(g) for g in tree_info]

    # Feature names and types
    feature_names = learner.get("feature_names", None)
    feature_types_raw = learner.get("feature_types", None)
    feature_types: list[FeatureType] | None = None
    if feature_types_raw:
        feature_types = [_xgboost_feature_type_to_boosters(ft) for ft in feature_types_raw]

    # Build metadata
    meta = ModelMetaSchema(
        num_features=num_features,
        num_groups=n_groups,
        feature_names=feature_names if feature_names else None,
        feature_types=feature_types,
        objective_name=objective_name,
    )

    # Build forest
    forest = ForestSchema(
        trees=trees,
        tree_groups=tree_groups,
        n_groups=n_groups,
        base_score=base_scores,
    )

    model = GBDTModelSchema(
        meta=meta,
        forest=forest,
        output_transform=_objective_to_output_transform(objective_name, n_groups),
    )
    return model, booster_name


def _convert_xgb_gblinear(xgb_json: dict[str, Any]) -> GBLinearModelSchema:
    """Convert XGBoost gblinear model to GBLinearModelSchema."""
    learner = xgb_json["learner"]
    learner_params = learner["learner_model_param"]
    objective_data = learner["objective"]
    gradient_booster = learner["gradient_booster"]

    # Parse learner params
    num_features = int(_parse_number_field(learner_params["num_feature"]))
    num_class = int(_parse_number_field(learner_params.get("num_class", 1)))

    # Get objective name
    objective_name = objective_data["name"]

    # Number of groups
    n_groups = max(1, num_class) if num_class > 2 else 1

    # Parse and convert base scores
    base_scores = _parse_base_scores(
        learner_params["base_score"],
        n_groups,
        objective_name,
    )

    # Get weights from gblinear model
    weights = gradient_booster["model"]["weights"]

    # XGBoost stores weights as [n_features + 1, n_groups] row-major
    # The last row is bias. We need to bake base_score into bias.
    weight_list = [float(w) for w in weights]

    # Bake base_scores into bias (last row). Our GBLinear predictor currently
    # only uses the weight matrix (including bias), so base_score must be folded
    # into the bias to match XGBoost behavior.
    for g in range(n_groups):
        bias_idx = num_features * n_groups + g
        weight_list[bias_idx] += base_scores[g]

    # Feature names and types
    feature_names = learner.get("feature_names", None)
    feature_types_raw = learner.get("feature_types", None)
    feature_types: list[FeatureType] | None = None
    if feature_types_raw:
        feature_types = [_xgboost_feature_type_to_boosters(ft) for ft in feature_types_raw]

    # Build metadata
    meta = ModelMetaSchema(
        num_features=num_features,
        num_groups=n_groups,
        feature_names=feature_names if feature_names else None,
        feature_types=feature_types,
        objective_name=objective_name,
    )

    # Build weights schema
    weights_schema = LinearWeightsSchema(
        values=weight_list,
        num_features=num_features,  # NOT including bias row
        num_groups=n_groups,
    )

    return GBLinearModelSchema(
        meta=meta,
        weights=weights_schema,
        base_score=[0.0] * n_groups,  # baked into weights
        output_transform=_objective_to_output_transform(objective_name, n_groups),
    )


def xgboost_to_schema(path_or_booster: str | Path | xgb.Booster) -> GBDTModelSchema | GBLinearModelSchema:
    """Convert XGBoost model to a boosters schema object.

    Parameters
    ----------
    path_or_booster
        Path to XGBoost JSON model file, or an xgb.Booster object.

    Returns:
    -------
    GBDTModelSchema | GBLinearModelSchema
        The converted model schema.

    Example:
    -------
    >>> from boosters.convert import xgboost_to_schema
    >>> schema = xgboost_to_schema("model.json")
    >>> print(schema.meta.objective_name)
    """
    xgb_json = _load_xgboost_json(path_or_booster)
    gradient_booster = xgb_json["learner"]["gradient_booster"]
    booster_name = gradient_booster["name"]

    if booster_name == "gblinear":
        return _convert_xgb_gblinear(xgb_json)
    return _convert_xgb_gbtree(xgb_json)[0]


def xgboost_to_json_bytes(path_or_booster: str | Path | xgb.Booster, *, pretty: bool = True) -> bytes:
    """Convert XGBoost model to `.bstr.json` bytes.

    Parameters
    ----------
    path_or_booster
        Path to XGBoost JSON model file, or an xgb.Booster object.
    pretty
        Whether to pretty-print the JSON output.

    Returns:
    -------
    bytes
        UTF-8 encoded JSON bytes in `.bstr.json` format.

    Example:
    -------
    >>> from boosters.convert import xgboost_to_json_bytes
    >>> json_bytes = xgboost_to_json_bytes("xgb_model.json")
    >>> with open("model.bstr.json", "wb") as f:
    ...     f.write(json_bytes)
    """
    schema = xgboost_to_schema(path_or_booster)

    if isinstance(schema, GBLinearModelSchema):
        envelope = JsonEnvelope[GBLinearModelSchema](
            bstr_version=2,
            model_type="gblinear",
            model=schema,
        )
    else:
        envelope = JsonEnvelope[GBDTModelSchema](
            bstr_version=2,
            model_type="gbdt",
            model=schema,
        )

    indent = 2 if pretty else None
    return envelope.model_dump_json(by_alias=True, indent=indent).encode("utf-8")


# =============================================================================
# LightGBM Conversion
# =============================================================================


def _load_lightgbm_json(path_or_booster: str | Path | lgb.Booster) -> dict[str, Any]:
    """Load LightGBM model as JSON dict."""
    if isinstance(path_or_booster, (str, Path)):
        path = Path(path_or_booster)
        # If .json file, load directly
        if path.suffix == ".json":
            with path.open() as f:
                data: Any = json.load(f)
                if not isinstance(data, dict):
                    msg = "LightGBM JSON root must be an object"
                    raise TypeError(msg)
                return cast(dict[str, Any], data)
        # Otherwise use lightgbm to load text model
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(path_or_booster))
        dumped: Any = model.dump_model()
        if not isinstance(dumped, dict):
            msg = "LightGBM dump_model() must return a dict"
            raise TypeError(msg)
        return cast(dict[str, Any], dumped)
    dumped: Any = path_or_booster.dump_model()
    if not isinstance(dumped, dict):
        msg = "LightGBM dump_model() must return a dict"
        raise TypeError(msg)
    return cast(dict[str, Any], dumped)


def _parse_lgb_tree_structure(
    tree_struct: dict[str, Any],
    nodes_out: list[dict[str, Any]],
    parent_idx: int,
    *,
    is_left: bool,
) -> int:
    """Recursively parse LightGBM tree structure into flat node list.

    Returns the index of this node in nodes_out.
    """
    node_idx = len(nodes_out)
    node: dict[str, Any] = {
        "parent": parent_idx,
        "is_left": is_left,
    }
    nodes_out.append(node)

    if "leaf_value" in tree_struct:
        # Leaf node
        node["is_leaf"] = True
        node["leaf_value"] = tree_struct["leaf_value"]

        # LightGBM linear-tree leaves may include a per-leaf linear model.
        # - leaf_const: intercept term
        # - leaf_features: feature indices used in the linear model
        # - leaf_coeff: coefficients aligned with leaf_features
        if "leaf_const" in tree_struct:
            node["leaf_const"] = tree_struct["leaf_const"]
        if "leaf_features" in tree_struct:
            node["leaf_features"] = tree_struct["leaf_features"]
        if "leaf_coeff" in tree_struct:
            node["leaf_coeff"] = tree_struct["leaf_coeff"]
    else:
        # Internal node
        node["is_leaf"] = False
        node["split_feature"] = tree_struct["split_feature"]
        node["threshold"] = tree_struct.get("threshold", 0.0)
        node["decision_type"] = tree_struct.get("decision_type", "<=")
        node["default_left"] = tree_struct.get("default_left", True)

        # Recursively add children
        left_idx = _parse_lgb_tree_structure(tree_struct["left_child"], nodes_out, node_idx, is_left=True)
        right_idx = _parse_lgb_tree_structure(tree_struct["right_child"], nodes_out, node_idx, is_left=False)
        node["left_child"] = left_idx
        node["right_child"] = right_idx

    return node_idx


def _convert_lgb_tree(tree_data: dict[str, Any]) -> TreeSchema:
    """Convert a single LightGBM tree to TreeSchema.

    The output format matches boosters' Rust schema where ALL nodes (internal + leaves)
    appear in every array. Leaf nodes have children_left=0, children_right=0 as sentinels.
    """
    # Parse tree structure recursively to get flat node list
    nodes: list[dict[str, Any]] = []
    _parse_lgb_tree_structure(tree_data["tree_structure"], nodes, parent_idx=-1, is_left=True)

    num_nodes = len(nodes)

    # Output arrays - same length as num_nodes
    out_split_indices: list[int] = []
    out_thresholds: list[float] = []
    out_children_left: list[int] = []
    out_children_right: list[int] = []
    out_default_left: list[bool] = []
    out_leaf_values: list[float] = []

    # Linear leaf support (LightGBM linear trees): packed coefficients per node.
    linear_node_indices: list[int] = []
    linear_coefficients: list[list[float]] = []

    for node_id, node in enumerate(nodes):
        if node["is_leaf"]:
            # Leaf node: placeholder values for split info, actual leaf value
            out_split_indices.append(0)
            out_thresholds.append(0.0)
            out_children_left.append(0)  # Sentinel for "no child"
            out_children_right.append(0)
            out_default_left.append(False)
            out_leaf_values.append(float(node["leaf_value"]))

            leaf_features = node.get("leaf_features", [])
            leaf_coeff = node.get("leaf_coeff", [])
            if leaf_features or leaf_coeff:
                if len(leaf_features) != len(leaf_coeff):
                    msg = (
                        "LightGBM leaf_features and leaf_coeff must have the same length "
                        f"(got {len(leaf_features)} and {len(leaf_coeff)})"
                    )
                    raise ValueError(msg)

                intercept = float(node.get("leaf_const", node["leaf_value"]))
                packed = [intercept] + [float(f) for f in leaf_features] + [float(c) for c in leaf_coeff]
                linear_node_indices.append(node_id)
                linear_coefficients.append(packed)
        else:
            # Internal node
            out_split_indices.append(int(node["split_feature"]))
            out_thresholds.append(float(node["threshold"]))
            out_children_left.append(int(node["left_child"]))
            out_children_right.append(int(node["right_child"]))
            out_default_left.append(bool(node["default_left"]))
            out_leaf_values.append(0.0)  # Placeholder for internal nodes

    return TreeSchema(
        num_nodes=num_nodes,
        split_indices=out_split_indices,
        thresholds=out_thresholds,
        children_left=out_children_left,
        children_right=out_children_right,
        default_left=out_default_left,
        leaf_values=ScalarLeafValues(type="scalar", values=out_leaf_values),
        categories=CategoriesSchema(),
        linear_coefficients=LinearCoefficientsSchema(
            node_indices=linear_node_indices,
            coefficients=linear_coefficients,
        ),
    )


def lightgbm_to_schema(path_or_booster: str | Path | lgb.Booster) -> GBDTModelSchema:
    """Convert LightGBM model to a boosters GBDTModelSchema.

    Parameters
    ----------
    path_or_booster
        Path to LightGBM model file (text or JSON), or an lgb.Booster object.

    Returns:
    -------
    GBDTModelSchema
        The converted model schema.

    Example:
    -------
    >>> from boosters.convert import lightgbm_to_schema
    >>> schema = lightgbm_to_schema("model.txt")
    >>> print(schema.meta.objective_name)
    """
    lgb_json = _load_lightgbm_json(path_or_booster)

    # Extract model info
    num_features = lgb_json.get("max_feature_idx", 0) + 1
    objective = lgb_json.get("objective", "regression")

    # Parse average_output which indicates number of groups
    # LightGBM uses num_tree_per_iteration for groups
    num_tree_per_iter = lgb_json.get("num_tree_per_iteration", 1)
    n_groups = num_tree_per_iter

    # Feature names
    feature_names = lgb_json.get("feature_names", None)

    # Convert trees
    tree_infos = lgb_json.get("tree_info", [])
    trees = [_convert_lgb_tree(t) for t in tree_infos]

    # Tree groups - LightGBM stores trees round-robin across groups
    tree_groups = [i % n_groups for i in range(len(trees))]

    # Build metadata
    meta = ModelMetaSchema(
        num_features=num_features,
        num_groups=n_groups,
        feature_names=feature_names,
        feature_types=None,
        objective_name=objective,
    )

    # Build forest
    forest = ForestSchema(
        trees=trees,
        tree_groups=tree_groups,
        n_groups=n_groups,
        base_score=[0.0] * n_groups,  # LightGBM handles base score differently
    )

    return GBDTModelSchema(
        meta=meta,
        forest=forest,
        output_transform=_objective_to_output_transform(objective, n_groups),
    )


def _tree_max_depth(tree: TreeSchema) -> int:
    """Compute max depth (root depth = 0) from a TreeSchema.

    Note: our TreeSchema uses `0` as the sentinel for "no child".
    Root node id is also `0`, but internal nodes never reference `0` as a child,
    so we treat a node as leaf when both children are `0`.
    """
    if tree.num_nodes <= 0:
        return 0

    max_depth = 0
    stack: list[tuple[int, int]] = [(0, 0)]  # (node_id, depth)

    while stack:
        node_id, depth = stack.pop()
        max_depth = max(max_depth, depth)

        if node_id < 0 or node_id >= tree.num_nodes:
            continue

        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])

        is_leaf = left == 0 and right == 0
        if is_leaf:
            continue

        if left != 0:
            stack.append((left, depth + 1))
        if right != 0:
            stack.append((right, depth + 1))

    return max_depth


def _forest_max_depth(trees: list[TreeSchema]) -> int:
    if not trees:
        return 0
    return max(_tree_max_depth(t) for t in trees)


def lightgbm_to_json_bytes(path_or_booster: str | Path | lgb.Booster, *, pretty: bool = True) -> bytes:
    """Convert LightGBM model to `.bstr.json` bytes.

    Parameters
    ----------
    path_or_booster
        Path to LightGBM model file (text or JSON), or an lgb.Booster object.
    pretty
        Whether to pretty-print the JSON output.

    Returns:
    -------
    bytes
        UTF-8 encoded JSON bytes in `.bstr.json` format.

    Example:
    -------
    >>> from boosters.convert import lightgbm_to_json_bytes
    >>> json_bytes = lightgbm_to_json_bytes("lgb_model.txt")
    >>> with open("model.bstr.json", "wb") as f:
    ...     f.write(json_bytes)
    """
    schema = lightgbm_to_schema(path_or_booster)

    envelope = JsonEnvelope[GBDTModelSchema](
        bstr_version=2,
        model_type="gbdt",
        model=schema,
    )

    indent = 2 if pretty else None
    return envelope.model_dump_json(by_alias=True, indent=indent).encode("utf-8")
