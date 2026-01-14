"""Tree visualization utilities for boosters models."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from boosters import GBDTModel

__all__ = ["plot_tree", "tree_to_dataframe"]


def _get_tree_data(model: GBDTModel, tree_index: int) -> dict[str, Any]:
    """Extract tree data from model JSON.

    Parameters
    ----------
    model : GBDTModel
        Trained GBDT model.
    tree_index : int
        Index of tree to extract (0-indexed).

    Returns:
    -------
    dict
        Tree data with nodes, splits, thresholds, etc.
    """
    json_bytes = model.to_json_bytes()
    schema = json.loads(json_bytes)
    trees = schema["model"]["forest"]["trees"]

    if tree_index < 0 or tree_index >= len(trees):
        msg = f"tree_index {tree_index} out of range [0, {len(trees)})"
        raise ValueError(msg)

    return trees[tree_index]


def tree_to_dataframe(
    model: GBDTModel,
    tree_index: int = 0,
    *,
    feature_names: list[str] | None = None,
) -> Any:
    """Convert a tree to a pandas DataFrame showing node structure.

    Parameters
    ----------
    model : GBDTModel
        Trained GBDT model.
    tree_index : int, default=0
        Index of tree to display (0-indexed).
    feature_names : list of str, optional
        Feature names for readable output. If None, uses names from the model
        (if available) or falls back to "f0", "f1", etc.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns: node_id, is_leaf, feature, threshold,
        left_child, right_child, value, gain, cover.

    Examples:
    --------
    >>> import boosters
    >>> from boosters.plotting import tree_to_dataframe
    >>> model = boosters.GBDTModel.train(train_data, config=config)
    >>> df = tree_to_dataframe(model, tree_index=0, feature_names=["age", "income"])
    >>> print(df)
    """
    try:
        import pandas as pd
    except ImportError as e:
        msg = "pandas is required for tree_to_dataframe. Install with: pip install pandas"
        raise ImportError(msg) from e

    tree = _get_tree_data(model, tree_index)

    num_nodes = tree["num_nodes"]
    split_indices = tree["split_indices"]
    thresholds = tree["thresholds"]
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    leaf_values = tree["leaf_values"]["values"]
    gains = tree["gains"]
    covers = tree["covers"]

    # Determine feature names: explicit > model > fallback
    if feature_names is None:
        feature_names = model.feature_names
    if feature_names is None:
        n_features = model.n_features
        feature_names = [f"f{i}" for i in range(n_features)]

    rows = []
    for node_id in range(num_nodes):
        left = children_left[node_id]
        right = children_right[node_id]
        is_leaf = left == 0 and right == 0

        if is_leaf:
            rows.append({
                "node_id": node_id,
                "is_leaf": True,
                "feature": None,
                "threshold": None,
                "left_child": None,
                "right_child": None,
                "value": leaf_values[node_id],
                "gain": None,
                "cover": covers[node_id],
            })
        else:
            feat_idx = split_indices[node_id]
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
            rows.append({
                "node_id": node_id,
                "is_leaf": False,
                "feature": feat_name,
                "threshold": thresholds[node_id],
                "left_child": left,
                "right_child": right,
                "value": None,
                "gain": gains[node_id],
                "cover": covers[node_id],
            })

    return pd.DataFrame(rows)


def plot_tree(
    model: GBDTModel,
    tree_index: int = 0,
    *,
    feature_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    precision: int = 3,
    filled: bool = True,
    ax: Any = None,
) -> Any:
    """Plot a single tree from a GBDT model.

    Creates a visual representation of the tree structure showing
    split decisions, thresholds, leaf values, and node statistics.

    Parameters
    ----------
    model : GBDTModel
        Trained GBDT model.
    tree_index : int, default=0
        Index of tree to plot (0-indexed).
    feature_names : list of str, optional
        Feature names for readable labels. If None, uses names from the model
        (if available) or falls back to "f0", "f1", etc.
    figsize : tuple of float, optional
        Figure size as (width, height). Default is auto-calculated.
    precision : int, default=3
        Number of decimal places for numeric values.
    filled : bool, default=True
        Whether to fill nodes with color based on value.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns:
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Examples:
    --------
    >>> import boosters
    >>> from boosters.plotting import plot_tree
    >>> model = boosters.GBDTModel.train(train_data, config=config)
    >>> ax = plot_tree(model, tree_index=0, feature_names=["age", "income", "score"])
    >>> plt.show()

    Notes:
    -----
    Similar to LightGBM's `plot_tree` and XGBoost's `plot_tree` functions.
    Shows split conditions, leaf values, gain, and sample coverage.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as e:
        msg = "matplotlib is required for plot_tree. Install with: pip install matplotlib"
        raise ImportError(msg) from e

    tree = _get_tree_data(model, tree_index)

    num_nodes = tree["num_nodes"]
    split_indices = tree["split_indices"]
    thresholds = tree["thresholds"]
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    leaf_values = tree["leaf_values"]["values"]
    gains = tree["gains"]
    covers = tree["covers"]

    # Determine feature names: explicit > model > fallback
    if feature_names is None:
        feature_names = model.feature_names
    if feature_names is None:
        n_features = model.n_features
        feature_names = [f"f{i}" for i in range(n_features)]

    # Build tree structure for layout
    def is_leaf(node_id: int) -> bool:
        return children_left[node_id] == 0 and children_right[node_id] == 0

    # Compute depth and position for each node
    def compute_layout(node_id: int, depth: int, left: float, right: float) -> dict[int, tuple[float, float]]:
        """Compute x,y positions for each node."""
        positions = {}
        x = (left + right) / 2
        y = -depth
        positions[node_id] = (x, y)

        if not is_leaf(node_id):
            mid = (left + right) / 2
            left_child = children_left[node_id]
            right_child = children_right[node_id]
            positions.update(compute_layout(left_child, depth + 1, left, mid))
            positions.update(compute_layout(right_child, depth + 1, mid, right))

        return positions

    positions = compute_layout(0, 0, 0, 1)

    # Compute tree depth for figsize
    max_depth = -min(y for x, y in positions.values())

    if figsize is None:
        width = max(12, num_nodes * 0.8)
        height = max(8, (max_depth + 1) * 2.5)
        figsize = (width, height)

    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)
    else:
        _fig = ax.figure

    # Find value range for coloring
    all_values = [leaf_values[i] for i in range(num_nodes) if is_leaf(i)]
    if all_values:
        min_val, max_val = min(all_values), max(all_values)
        val_range = max_val - min_val if max_val != min_val else 1
    else:
        min_val, max_val, val_range = 0, 1, 1

    # Draw edges
    for node_id in range(num_nodes):
        if not is_leaf(node_id):
            x, y = positions[node_id]
            left_child = children_left[node_id]
            right_child = children_right[node_id]

            lx, ly = positions[left_child]
            rx, ry = positions[right_child]

            ax.plot([x, lx], [y, ly], "k-", lw=1, zorder=1)
            ax.plot([x, rx], [y, ry], "k-", lw=1, zorder=1)

            # Add edge labels
            ax.text(
                (x + lx) / 2 - 0.02,
                (y + ly) / 2,
                "≤",
                fontsize=8,
                ha="right",
                va="center",
            )
            ax.text(
                (x + rx) / 2 + 0.02,
                (y + ry) / 2,
                ">",
                fontsize=8,
                ha="left",
                va="center",
            )

    # Draw nodes - calculate box width based on text
    box_width = 0.18
    box_height = 0.40

    for node_id in range(num_nodes):
        x, y = positions[node_id]

        if is_leaf(node_id):
            # Leaf node
            value = leaf_values[node_id]
            cover = covers[node_id]

            # Color based on value
            if filled:
                # Normalize value to [0, 1], blue for negative, red for positive
                norm_val = (value - min_val) / val_range if val_range != 0 else 0.5
                # Blue → White → Red colormap
                if value < 0:
                    color = (0.4 + 0.6 * (1 - abs(norm_val)), 0.6 + 0.4 * (1 - abs(norm_val)), 1.0)
                else:
                    color = (1.0, 0.6 + 0.4 * (1 - norm_val), 0.4 + 0.6 * (1 - norm_val))
            else:
                color = "white"

            # Create rounded rectangle
            rect = FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(rect)

            # Node text
            text = f"leaf\nvalue={value:.{precision}f}\ncover={cover:.0f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=7, zorder=3)

        else:
            # Split node
            feat_idx = split_indices[node_id]
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
            threshold = thresholds[node_id]
            gain = gains[node_id]
            cover = covers[node_id]

            color = "#E8E8E8" if filled else "white"

            rect = FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                zorder=2,
            )
            ax.add_patch(rect)

            text = f"{feat_name} ≤ {threshold:.{precision}f}\ngain={gain:.1f}\ncover={cover:.0f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=7, zorder=3)

    # Set axis properties
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-max_depth - 0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Tree {tree_index}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return ax
