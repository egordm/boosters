# RFC-0023: Training Constraints

- **Status**: Delayed
- **Created**: 2024-12-01
- **Updated**: 2024-12-01
- **Depends on**: RFC-0011 (GBTree Core), RFC-0014 (Split Finding)
- **Scope**: Monotonic and interaction constraints during training

> **Note**: This feature has been delayed for proper implementation in a future epic.
> Initial implementation was removed to simplify the codebase and ensure a clean API.
> See backlog for future work on constraints.

## Summary

This RFC defines constraint mechanisms for gradient boosting training:

1. **Monotonic Constraints** — Force predictions to increase/decrease with feature values
2. **Interaction Constraints** — Limit which features can appear together in trees

These constraints enable domain knowledge integration and improve model interpretability.

## Motivation

### Domain Knowledge

Many relationships have known directions:

| Domain | Feature | Expected Relationship |
|--------|---------|----------------------|
| Credit | Income | Higher income → lower default risk |
| Insurance | Age (driving) | Teenager → higher risk |
| Pricing | Quantity | More quantity → lower unit price |

Unconstrained models may learn spurious correlations from noise.

### Regulatory Requirements

Financial models often require monotonicity for fairness/explainability:

- FICO guidelines require monotonic credit scores
- Insurance regulations may mandate interpretable pricing factors

## Design

### Monotonic Constraints

For each feature, specify: `None`, `Increasing`, or `Decreasing`.

**Enforcement during split finding**:

1. After computing left/right leaf weights, check ordering
2. If `Increasing` and `weight_left > weight_right`, clamp both to midpoint
3. Propagate bounds to children: left child bounded above, right child bounded below

**Bounds tracking**: Each node carries `(lower_bound, upper_bound)` inherited from parent.
Leaf values are clamped to these bounds.

### Interaction Constraints

Specify groups of features that can interact:

```text
Groups: [[0, 1, 2], [3, 4, 5]]
```

Features in different groups cannot both appear on the path from root to leaf.

**Enforcement**:

1. Track which features have been used on path to current node
2. When finding splits, only consider features that share a group with all path features
3. First split can use any feature; subsequent splits are restricted

### Integration Points

| Component | Change |
|-----------|--------|
| `GBTreeParams` | Add `monotonic_constraints: Vec<MonotonicConstraint>` |
| `GBTreeParams` | Add `interaction_constraints: Option<Vec<Vec<usize>>>` |
| `BuildingNode` | Add `bounds: (f32, f32)` and `path_features: Vec<usize>` |
| `GreedySplitFinder` | Filter features by interaction, validate monotonicity |

## Design Decisions

### DD-1: Midpoint Clamping

**Decision**: When monotonicity is violated, set both children to midpoint.

**Rationale**: Preserves split structure while ensuring monotonicity.
XGBoost uses similar approach.

### DD-2: Path-based Interaction Checking

**Decision**: Track features in path, check new features against path.

**Rationale**: O(depth × groups) per feature check. Depth is typically small.

### DD-3: Feature in Multiple Groups

**Decision**: Allow features to be in multiple interaction groups.

**Rationale**: More flexible — feature X can interact with both Y and Z if desired.
Matches XGBoost semantics.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monotone_constraints` | `[]` | Per-feature: -1 (decreasing), 0 (none), 1 (increasing) |
| `interaction_constraints` | `None` | Groups of features that can interact |

## Design Decisions (Continued)

### DD-4: Hard Constraints Only

**Decision**: Implement hard constraints; defer soft constraints to future work.

**Rationale**: Hard constraints are simpler, deterministic, and match XGBoost/LightGBM
behavior. Soft constraints (penalty terms) add complexity and tuning burden.
If needed later, can be added as a separate feature.

### DD-5: No Piecewise Monotonicity

**Decision**: Do not support piecewise monotonicity (changing direction at thresholds).

**Rationale**: Adds significant complexity for niche use case. If a relationship
changes direction (e.g., U-shaped), user should model with separate features or
use unconstrained training. Neither XGBoost nor LightGBM support this.

## Testing Strategy

### Unit Tests

- Monotonicity violation detection is correct
- Midpoint clamping produces valid bounds
- Bounds propagation to children is correct
- Interaction constraint filtering works correctly
- Path feature tracking is accurate

### Integration Tests

- Trained model with monotonic constraints is actually monotonic
- Trained model with interaction constraints respects groupings
- Constraints work with both tree growth strategies (depthwise, leaf-wise)

### Validation Tests

- Compare constrained model predictions against XGBoost with same constraints
- Tolerance: predictions within 1e-2 for same hyperparameters
- Verify constraint enforcement matches XGBoost (sample test cases)
- If behavior differs, investigate XGBoost source code

### Qualitative Tests

- Train monotonic model on credit data, verify higher income → lower risk
- Train interaction-constrained model, verify expected feature separation
- Set constraint compliance expectations; investigate any violations

### Property Tests

- Monotonicity: For increasing constraint on feature F, increasing F should
  never decrease prediction (scan along feature axis)
- Interaction: Features from disjoint groups should never appear in same leaf path

## References

- [XGBoost monotonic constraints](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html)
- [XGBoost interaction constraints](https://xgboost.readthedocs.io/en/latest/tutorials/feature_interaction_constraint.html)
