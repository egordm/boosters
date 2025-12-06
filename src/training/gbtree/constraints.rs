//! Training constraints implementation.
//!
//! This module implements RFC-0023: constraint mechanisms for gradient boosting training:
//!
//! 1. **Monotonic Constraints** — Force predictions to increase/decrease with feature values
//! 2. **Interaction Constraints** — Limit which features can appear together in trees
//!
//! # Monotonic Constraints
//!
//! For each feature, specify: `None`, `Increasing`, or `Decreasing`.
//!
//! - `Increasing`: Higher feature values must lead to higher predictions
//! - `Decreasing`: Higher feature values must lead to lower predictions
//!
//! Enforcement happens during tree building:
//! 1. Track bounds (lower, upper) for each node inherited from parent
//! 2. When a split is found, check if child weights satisfy monotonicity
//! 3. If violated, clamp both children to midpoint
//! 4. Propagate tighter bounds to children
//!
//! # Interaction Constraints
//!
//! Specify groups of features that can interact:
//!
//! ```text
//! Groups: [[0, 1, 2], [3, 4, 5]]
//! ```
//!
//! Features in different groups cannot both appear on the path from root to leaf.
//!
//! Enforcement:
//! 1. Track which features have been used on path to current node
//! 2. When finding splits, only consider features that share a group with all path features
//! 3. First split can use any feature; subsequent splits are restricted
//!
//! See RFC-0023 for design rationale.

use std::collections::HashSet;

// ============================================================================
// MonotonicConstraint
// ============================================================================

/// Monotonic constraint type for a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MonotonicConstraint {
    /// No constraint (default)
    #[default]
    None,
    /// Predictions must increase with feature value
    Increasing,
    /// Predictions must decrease with feature value
    Decreasing,
}

impl MonotonicConstraint {
    /// Create from integer representation (XGBoost style).
    ///
    /// - `0`: No constraint
    /// - `1`: Increasing
    /// - `-1`: Decreasing
    pub fn from_int(value: i32) -> Self {
        match value {
            1 => Self::Increasing,
            -1 => Self::Decreasing,
            _ => Self::None,
        }
    }

    /// Convert to integer representation.
    pub fn to_int(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Increasing => 1,
            Self::Decreasing => -1,
        }
    }

    /// Check if this constraint is active.
    #[inline]
    pub fn is_constrained(self) -> bool {
        !matches!(self, Self::None)
    }
}

// ============================================================================
// MonotonicBounds
// ============================================================================

/// Monotonicity bounds for a node.
///
/// Tracks the allowed range of leaf weights based on monotonic constraints
/// inherited from ancestors.
#[derive(Debug, Clone, Copy)]
pub struct MonotonicBounds {
    /// Lower bound on leaf weight (NEG_INFINITY if unbounded)
    pub lower: f32,
    /// Upper bound on leaf weight (INFINITY if unbounded)
    pub upper: f32,
}

impl Default for MonotonicBounds {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl MonotonicBounds {
    /// Create unbounded (no constraint).
    pub fn unbounded() -> Self {
        Self {
            lower: f32::NEG_INFINITY,
            upper: f32::INFINITY,
        }
    }

    /// Check if bounds are unbounded (no constraint).
    #[inline]
    pub fn is_unbounded(&self) -> bool {
        self.lower == f32::NEG_INFINITY && self.upper == f32::INFINITY
    }

    /// Clamp a weight to these bounds.
    #[inline]
    pub fn clamp(&self, weight: f32) -> f32 {
        weight.max(self.lower).min(self.upper)
    }

    /// Check if a weight satisfies these bounds.
    #[inline]
    pub fn contains(&self, weight: f32) -> bool {
        weight >= self.lower && weight <= self.upper
    }

    /// Compute child bounds after a split on a feature with monotonic constraint.
    ///
    /// When splitting on feature F with constraint C:
    /// - For `Increasing`: left child (lower values) has upper bound = parent weight
    ///                    right child (higher values) has lower bound = parent weight
    /// - For `Decreasing`: left child has lower bound = parent weight
    ///                    right child has upper bound = parent weight
    ///
    /// The `split_weight` is the parent node's weight (or midpoint if clamped).
    pub fn child_bounds(
        &self,
        constraint: MonotonicConstraint,
        split_weight: f32,
    ) -> (MonotonicBounds, MonotonicBounds) {
        match constraint {
            MonotonicConstraint::None => (*self, *self),
            MonotonicConstraint::Increasing => {
                // Left child (lower feature values) -> lower predictions
                // Right child (higher feature values) -> higher predictions
                let left = MonotonicBounds {
                    lower: self.lower,
                    upper: split_weight.min(self.upper),
                };
                let right = MonotonicBounds {
                    lower: split_weight.max(self.lower),
                    upper: self.upper,
                };
                (left, right)
            }
            MonotonicConstraint::Decreasing => {
                // Left child (lower feature values) -> higher predictions
                // Right child (higher feature values) -> lower predictions
                let left = MonotonicBounds {
                    lower: split_weight.max(self.lower),
                    upper: self.upper,
                };
                let right = MonotonicBounds {
                    lower: self.lower,
                    upper: split_weight.min(self.upper),
                };
                (left, right)
            }
        }
    }
}

// ============================================================================
// MonotonicChecker
// ============================================================================

/// Helper for checking and enforcing monotonic constraints during split finding.
pub struct MonotonicChecker {
    /// Constraints per feature (indexed by feature ID)
    constraints: Vec<MonotonicConstraint>,
    /// Whether any feature has a constraint
    has_constraints: bool,
}

impl MonotonicChecker {
    /// Create a checker from a list of constraints.
    ///
    /// # Arguments
    ///
    /// * `constraints` - Per-feature constraints (can be empty for no constraints)
    /// * `num_features` - Total number of features (constraints extended with None if shorter)
    pub fn new(constraints: &[MonotonicConstraint], num_features: usize) -> Self {
        let mut full_constraints = constraints.to_vec();
        full_constraints.resize(num_features, MonotonicConstraint::None);

        let has_constraints = full_constraints.iter().any(|c| c.is_constrained());

        Self {
            constraints: full_constraints,
            has_constraints,
        }
    }

    /// Create from integer constraints (XGBoost style: -1, 0, 1).
    pub fn from_ints(constraints: &[i32], num_features: usize) -> Self {
        let typed: Vec<_> = constraints
            .iter()
            .map(|&c| MonotonicConstraint::from_int(c))
            .collect();
        Self::new(&typed, num_features)
    }

    /// Check if any constraints are active.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.has_constraints
    }

    /// Get the constraint for a feature.
    #[inline]
    pub fn get(&self, feature: u32) -> MonotonicConstraint {
        self.constraints
            .get(feature as usize)
            .copied()
            .unwrap_or(MonotonicConstraint::None)
    }

    /// Check if a split satisfies monotonic constraints and optionally fix it.
    ///
    /// For a split with constraint:
    /// - `Increasing`: weight_left must be <= weight_right
    /// - `Decreasing`: weight_left must be >= weight_right
    ///
    /// If violated, returns adjusted weights clamped to midpoint.
    ///
    /// # Arguments
    ///
    /// * `feature` - Feature being split on
    /// * `weight_left` - Computed left child weight
    /// * `weight_right` - Computed right child weight
    /// * `bounds` - Current node's monotonic bounds
    ///
    /// # Returns
    ///
    /// `(adjusted_left, adjusted_right, is_valid)` - Adjusted weights and whether
    /// the original split was already valid.
    pub fn check_and_fix(
        &self,
        feature: u32,
        weight_left: f32,
        weight_right: f32,
        bounds: &MonotonicBounds,
    ) -> (f32, f32, bool) {
        let constraint = self.get(feature);

        let (is_valid, adjusted_left, adjusted_right) = match constraint {
            MonotonicConstraint::None => (true, weight_left, weight_right),
            MonotonicConstraint::Increasing => {
                // Left (lower values) should have lower or equal weight
                if weight_left <= weight_right {
                    (true, weight_left, weight_right)
                } else {
                    // Violation: clamp to midpoint
                    let mid = (weight_left + weight_right) / 2.0;
                    (false, mid, mid)
                }
            }
            MonotonicConstraint::Decreasing => {
                // Left (lower values) should have higher or equal weight
                if weight_left >= weight_right {
                    (true, weight_left, weight_right)
                } else {
                    // Violation: clamp to midpoint
                    let mid = (weight_left + weight_right) / 2.0;
                    (false, mid, mid)
                }
            }
        };

        // Also clamp to parent bounds
        let final_left = bounds.clamp(adjusted_left);
        let final_right = bounds.clamp(adjusted_right);

        (final_left, final_right, is_valid)
    }
}

// ============================================================================
// InteractionConstraints
// ============================================================================

/// Interaction constraints manager.
///
/// Tracks which features can interact based on user-specified groups.
#[derive(Debug, Clone)]
pub struct InteractionConstraints {
    /// Groups of features that can interact (feature indices)
    groups: Vec<Vec<u32>>,
    /// Mapping from feature to its groups (a feature can be in multiple groups)
    feature_to_groups: Vec<HashSet<usize>>,
    /// Total number of features
    num_features: u32,
    /// Whether constraints are active
    has_constraints: bool,
}

impl InteractionConstraints {
    /// Create interaction constraints from group definitions.
    ///
    /// # Arguments
    ///
    /// * `groups` - Groups of features that can interact (e.g., `[[0, 1, 2], [3, 4]]`)
    /// * `num_features` - Total number of features
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Features 0,1,2 can interact with each other
    /// // Features 3,4,5 can interact with each other
    /// // But 0 cannot interact with 3
    /// let constraints = InteractionConstraints::new(&[[0, 1, 2], [3, 4, 5]], 6);
    /// ```
    pub fn new(groups: &[Vec<u32>], num_features: u32) -> Self {
        let has_constraints = !groups.is_empty();

        // Build feature -> groups mapping
        let mut feature_to_groups: Vec<HashSet<usize>> =
            vec![HashSet::new(); num_features as usize];

        for (group_idx, group) in groups.iter().enumerate() {
            for &feat in group {
                if (feat as usize) < feature_to_groups.len() {
                    feature_to_groups[feat as usize].insert(group_idx);
                }
            }
        }

        Self {
            groups: groups.to_vec(),
            feature_to_groups,
            num_features,
            has_constraints,
        }
    }

    /// Create with no constraints.
    pub fn none() -> Self {
        Self {
            groups: Vec::new(),
            feature_to_groups: Vec::new(),
            num_features: 0,
            has_constraints: false,
        }
    }

    /// Check if any constraints are active.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.has_constraints
    }

    /// Get allowed features for a node given the features used in its path.
    ///
    /// A feature is allowed if it shares at least one group with every feature
    /// in the path, or if the path is empty.
    ///
    /// # Arguments
    ///
    /// * `path_features` - Features used on the path from root to this node
    ///
    /// # Returns
    ///
    /// Vector of allowed feature indices
    pub fn allowed_features(&self, path_features: &[u32]) -> Vec<u32> {
        if !self.has_constraints || path_features.is_empty() {
            // No constraints or empty path: all features allowed
            return (0..self.num_features).collect();
        }

        // Find the intersection of groups for all path features
        let mut allowed_groups: Option<HashSet<usize>> = None;

        for &feat in path_features {
            if let Some(groups) = self.feature_to_groups.get(feat as usize) {
                match &mut allowed_groups {
                    None => allowed_groups = Some(groups.clone()),
                    Some(current) => {
                        *current = current.intersection(groups).cloned().collect();
                    }
                }
            }
        }

        let allowed_groups = match allowed_groups {
            Some(groups) => groups,
            None => return (0..self.num_features).collect(), // No valid path features
        };

        // Collect all features that belong to at least one allowed group
        let mut allowed = Vec::new();
        for feat in 0..self.num_features {
            let feat_groups = &self.feature_to_groups[feat as usize];
            if feat_groups.iter().any(|g| allowed_groups.contains(g)) {
                allowed.push(feat);
            }
        }

        allowed
    }

    /// Check if a feature can be used given the path features.
    #[inline]
    pub fn is_feature_allowed(&self, feature: u32, path_features: &[u32]) -> bool {
        if !self.has_constraints || path_features.is_empty() {
            return true;
        }

        let feat_groups = match self.feature_to_groups.get(feature as usize) {
            Some(g) => g,
            None => return false,
        };

        // Check that this feature shares a group with every path feature
        for &path_feat in path_features {
            if let Some(path_groups) = self.feature_to_groups.get(path_feat as usize) {
                if feat_groups.is_disjoint(path_groups) {
                    return false;
                }
            }
        }

        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MonotonicConstraint tests ----

    #[test]
    fn test_monotonic_constraint_from_int() {
        assert_eq!(MonotonicConstraint::from_int(0), MonotonicConstraint::None);
        assert_eq!(MonotonicConstraint::from_int(1), MonotonicConstraint::Increasing);
        assert_eq!(MonotonicConstraint::from_int(-1), MonotonicConstraint::Decreasing);
        assert_eq!(MonotonicConstraint::from_int(99), MonotonicConstraint::None);
    }

    #[test]
    fn test_monotonic_constraint_to_int() {
        assert_eq!(MonotonicConstraint::None.to_int(), 0);
        assert_eq!(MonotonicConstraint::Increasing.to_int(), 1);
        assert_eq!(MonotonicConstraint::Decreasing.to_int(), -1);
    }

    #[test]
    fn test_monotonic_constraint_is_constrained() {
        assert!(!MonotonicConstraint::None.is_constrained());
        assert!(MonotonicConstraint::Increasing.is_constrained());
        assert!(MonotonicConstraint::Decreasing.is_constrained());
    }

    // ---- MonotonicBounds tests ----

    #[test]
    fn test_bounds_unbounded() {
        let bounds = MonotonicBounds::unbounded();
        assert!(bounds.is_unbounded());
        assert!(bounds.contains(0.0));
        assert!(bounds.contains(-1000.0));
        assert!(bounds.contains(1000.0));
    }

    #[test]
    fn test_bounds_clamp() {
        let bounds = MonotonicBounds {
            lower: -1.0,
            upper: 1.0,
        };
        assert!(!bounds.is_unbounded());
        assert_eq!(bounds.clamp(0.5), 0.5);
        assert_eq!(bounds.clamp(-2.0), -1.0);
        assert_eq!(bounds.clamp(2.0), 1.0);
    }

    #[test]
    fn test_bounds_contains() {
        let bounds = MonotonicBounds {
            lower: -1.0,
            upper: 1.0,
        };
        assert!(bounds.contains(0.0));
        assert!(bounds.contains(-1.0));
        assert!(bounds.contains(1.0));
        assert!(!bounds.contains(-2.0));
        assert!(!bounds.contains(2.0));
    }

    #[test]
    fn test_child_bounds_increasing() {
        let parent = MonotonicBounds::unbounded();
        let split_weight = 0.5;

        let (left, right) = parent.child_bounds(MonotonicConstraint::Increasing, split_weight);

        // Left child: upper bounded by split weight
        assert_eq!(left.lower, f32::NEG_INFINITY);
        assert_eq!(left.upper, 0.5);

        // Right child: lower bounded by split weight
        assert_eq!(right.lower, 0.5);
        assert_eq!(right.upper, f32::INFINITY);
    }

    #[test]
    fn test_child_bounds_decreasing() {
        let parent = MonotonicBounds::unbounded();
        let split_weight = 0.5;

        let (left, right) = parent.child_bounds(MonotonicConstraint::Decreasing, split_weight);

        // Left child: lower bounded by split weight (higher values for lower features)
        assert_eq!(left.lower, 0.5);
        assert_eq!(left.upper, f32::INFINITY);

        // Right child: upper bounded by split weight
        assert_eq!(right.lower, f32::NEG_INFINITY);
        assert_eq!(right.upper, 0.5);
    }

    #[test]
    fn test_child_bounds_none() {
        let parent = MonotonicBounds {
            lower: -1.0,
            upper: 1.0,
        };
        let split_weight = 0.0;

        let (left, right) = parent.child_bounds(MonotonicConstraint::None, split_weight);

        // Both children inherit parent bounds unchanged
        assert_eq!(left.lower, parent.lower);
        assert_eq!(left.upper, parent.upper);
        assert_eq!(right.lower, parent.lower);
        assert_eq!(right.upper, parent.upper);
    }

    #[test]
    fn test_child_bounds_with_existing_bounds() {
        // Parent already has bounds from previous constraint
        let parent = MonotonicBounds {
            lower: 0.0,
            upper: 2.0,
        };
        let split_weight = 1.0;

        let (left, right) = parent.child_bounds(MonotonicConstraint::Increasing, split_weight);

        // Left: upper = min(split, parent.upper) = min(1.0, 2.0) = 1.0
        assert_eq!(left.lower, 0.0); // Inherited
        assert_eq!(left.upper, 1.0);

        // Right: lower = max(split, parent.lower) = max(1.0, 0.0) = 1.0
        assert_eq!(right.lower, 1.0);
        assert_eq!(right.upper, 2.0); // Inherited
    }

    // ---- MonotonicChecker tests ----

    #[test]
    fn test_checker_no_constraints() {
        let checker = MonotonicChecker::new(&[], 5);
        assert!(!checker.is_enabled());
        assert_eq!(checker.get(0), MonotonicConstraint::None);
    }

    #[test]
    fn test_checker_with_constraints() {
        let constraints = vec![
            MonotonicConstraint::Increasing,
            MonotonicConstraint::None,
            MonotonicConstraint::Decreasing,
        ];
        let checker = MonotonicChecker::new(&constraints, 5);

        assert!(checker.is_enabled());
        assert_eq!(checker.get(0), MonotonicConstraint::Increasing);
        assert_eq!(checker.get(1), MonotonicConstraint::None);
        assert_eq!(checker.get(2), MonotonicConstraint::Decreasing);
        assert_eq!(checker.get(3), MonotonicConstraint::None); // Extended
        assert_eq!(checker.get(4), MonotonicConstraint::None);
    }

    #[test]
    fn test_checker_from_ints() {
        let checker = MonotonicChecker::from_ints(&[1, 0, -1], 3);
        assert_eq!(checker.get(0), MonotonicConstraint::Increasing);
        assert_eq!(checker.get(1), MonotonicConstraint::None);
        assert_eq!(checker.get(2), MonotonicConstraint::Decreasing);
    }

    #[test]
    fn test_check_and_fix_no_constraint() {
        let checker = MonotonicChecker::new(&[], 5);
        let bounds = MonotonicBounds::unbounded();

        let (left, right, valid) = checker.check_and_fix(0, -1.0, 1.0, &bounds);
        assert!(valid);
        assert_eq!(left, -1.0);
        assert_eq!(right, 1.0);
    }

    #[test]
    fn test_check_and_fix_increasing_valid() {
        let constraints = vec![MonotonicConstraint::Increasing];
        let checker = MonotonicChecker::new(&constraints, 1);
        let bounds = MonotonicBounds::unbounded();

        // Left <= Right: valid for increasing
        let (left, right, valid) = checker.check_and_fix(0, -1.0, 1.0, &bounds);
        assert!(valid);
        assert_eq!(left, -1.0);
        assert_eq!(right, 1.0);
    }

    #[test]
    fn test_check_and_fix_increasing_violated() {
        let constraints = vec![MonotonicConstraint::Increasing];
        let checker = MonotonicChecker::new(&constraints, 1);
        let bounds = MonotonicBounds::unbounded();

        // Left > Right: violated for increasing, should clamp to midpoint
        let (left, right, valid) = checker.check_and_fix(0, 1.0, -1.0, &bounds);
        assert!(!valid);
        assert_eq!(left, 0.0); // Midpoint
        assert_eq!(right, 0.0);
    }

    #[test]
    fn test_check_and_fix_decreasing_valid() {
        let constraints = vec![MonotonicConstraint::Decreasing];
        let checker = MonotonicChecker::new(&constraints, 1);
        let bounds = MonotonicBounds::unbounded();

        // Left >= Right: valid for decreasing
        let (left, right, valid) = checker.check_and_fix(0, 1.0, -1.0, &bounds);
        assert!(valid);
        assert_eq!(left, 1.0);
        assert_eq!(right, -1.0);
    }

    #[test]
    fn test_check_and_fix_decreasing_violated() {
        let constraints = vec![MonotonicConstraint::Decreasing];
        let checker = MonotonicChecker::new(&constraints, 1);
        let bounds = MonotonicBounds::unbounded();

        // Left < Right: violated for decreasing
        let (left, right, valid) = checker.check_and_fix(0, -1.0, 1.0, &bounds);
        assert!(!valid);
        assert_eq!(left, 0.0);
        assert_eq!(right, 0.0);
    }

    #[test]
    fn test_check_and_fix_with_bounds_clamping() {
        let constraints = vec![MonotonicConstraint::None];
        let checker = MonotonicChecker::new(&constraints, 1);
        let bounds = MonotonicBounds {
            lower: -0.5,
            upper: 0.5,
        };

        // Weights outside bounds should be clamped
        let (left, right, valid) = checker.check_and_fix(0, -2.0, 2.0, &bounds);
        assert!(valid);
        assert_eq!(left, -0.5);
        assert_eq!(right, 0.5);
    }

    // ---- InteractionConstraints tests ----

    #[test]
    fn test_interaction_no_constraints() {
        let constraints = InteractionConstraints::none();
        assert!(!constraints.is_enabled());

        let allowed = constraints.allowed_features(&[]);
        assert!(allowed.is_empty()); // num_features = 0
    }

    #[test]
    fn test_interaction_empty_path() {
        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let constraints = InteractionConstraints::new(&groups, 6);

        assert!(constraints.is_enabled());

        // Empty path: all features allowed
        let allowed = constraints.allowed_features(&[]);
        assert_eq!(allowed, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_interaction_single_path_feature() {
        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let constraints = InteractionConstraints::new(&groups, 6);

        // Path contains feature 0 (group 0)
        // Only features in group 0 are allowed
        let allowed = constraints.allowed_features(&[0]);
        assert_eq!(allowed, vec![0, 1, 2]);

        // Path contains feature 3 (group 1)
        let allowed = constraints.allowed_features(&[3]);
        assert_eq!(allowed, vec![3, 4, 5]);
    }

    #[test]
    fn test_interaction_multiple_path_features() {
        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let constraints = InteractionConstraints::new(&groups, 6);

        // Path contains features from same group
        let allowed = constraints.allowed_features(&[0, 1]);
        assert_eq!(allowed, vec![0, 1, 2]);
    }

    #[test]
    fn test_interaction_feature_in_multiple_groups() {
        // Feature 2 is in both groups
        let groups = vec![vec![0, 1, 2], vec![2, 3, 4]];
        let constraints = InteractionConstraints::new(&groups, 5);

        // Path contains feature 0 (only in group 0)
        // Allowed: features in group 0
        let allowed = constraints.allowed_features(&[0]);
        assert_eq!(allowed, vec![0, 1, 2]);

        // Path contains feature 3 (only in group 1)
        let allowed = constraints.allowed_features(&[3]);
        assert_eq!(allowed, vec![2, 3, 4]);

        // Path contains feature 2 (in both groups)
        // All features are allowed since 2 bridges both groups
        let allowed = constraints.allowed_features(&[2]);
        assert_eq!(allowed, vec![0, 1, 2, 3, 4]);

        // Path contains features 0 and 2
        // 0 is only in group 0, so only group 0 features allowed
        let allowed = constraints.allowed_features(&[0, 2]);
        assert_eq!(allowed, vec![0, 1, 2]);
    }

    #[test]
    fn test_is_feature_allowed() {
        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let constraints = InteractionConstraints::new(&groups, 6);

        // Empty path: all allowed
        assert!(constraints.is_feature_allowed(0, &[]));
        assert!(constraints.is_feature_allowed(3, &[]));

        // Path with feature 0: only group 0 allowed
        assert!(constraints.is_feature_allowed(0, &[0]));
        assert!(constraints.is_feature_allowed(1, &[0]));
        assert!(constraints.is_feature_allowed(2, &[0]));
        assert!(!constraints.is_feature_allowed(3, &[0]));
        assert!(!constraints.is_feature_allowed(4, &[0]));
        assert!(!constraints.is_feature_allowed(5, &[0]));
    }

    #[test]
    fn test_interaction_disjoint_path_features() {
        let groups = vec![vec![0, 1], vec![2, 3]];
        let constraints = InteractionConstraints::new(&groups, 4);

        // Path contains features from different groups (shouldn't happen in valid tree)
        // This would mean no common groups exist
        let allowed = constraints.allowed_features(&[0, 2]);
        // Features 0 (group 0) and 2 (group 1) have no common group
        // Intersection of groups is empty, so no features allowed
        assert!(allowed.is_empty());
    }
}
