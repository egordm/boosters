//! Tree node types.

use super::leaf::LeafValue;

/// Split condition for a decision node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitCondition {
    /// Feature index to split on
    pub feature_index: u32,
    /// Threshold value (go left if feature < threshold)
    pub threshold: f32,
    /// Direction for missing values (true = left, false = right)
    pub default_left: bool,
}

impl SplitCondition {
    pub fn new(feature_index: u32, threshold: f32, default_left: bool) -> Self {
        Self {
            feature_index,
            threshold,
            default_left,
        }
    }

    /// Evaluate which direction to go for a feature value.
    /// Returns true for left, false for right.
    #[inline]
    pub fn go_left(&self, feature_value: f32) -> bool {
        if feature_value.is_nan() {
            self.default_left
        } else {
            feature_value < self.threshold
        }
    }
}

/// A node in a decision tree.
#[derive(Debug, Clone)]
pub enum Node<L: LeafValue> {
    /// Internal split node
    Split {
        condition: SplitCondition,
        left: u32,
        right: u32,
    },
    /// Leaf node with a value
    Leaf(L),
}

impl<L: LeafValue> Node<L> {
    /// Create a new split node.
    pub fn split(condition: SplitCondition, left: u32, right: u32) -> Self {
        Self::Split {
            condition,
            left,
            right,
        }
    }

    /// Create a new leaf node.
    pub fn leaf(value: L) -> Self {
        Self::Leaf(value)
    }

    /// Returns true if this is a leaf node.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    /// Get the leaf value, if this is a leaf.
    #[inline]
    pub fn leaf_value(&self) -> Option<&L> {
        match self {
            Self::Leaf(v) => Some(v),
            Self::Split { .. } => None,
        }
    }

    /// Get the split condition, if this is a split node.
    #[inline]
    pub fn split_condition(&self) -> Option<&SplitCondition> {
        match self {
            Self::Split { condition, .. } => Some(condition),
            Self::Leaf(_) => None,
        }
    }

    /// Get child indices, if this is a split node.
    #[inline]
    pub fn children(&self) -> Option<(u32, u32)> {
        match self {
            Self::Split { left, right, .. } => Some((*left, *right)),
            Self::Leaf(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trees::leaf::ScalarLeaf;

    #[test]
    fn split_condition_numeric() {
        let cond = SplitCondition::new(0, 0.5, true);

        assert!(cond.go_left(0.3)); // < threshold
        assert!(!cond.go_left(0.7)); // >= threshold
        assert!(!cond.go_left(0.5)); // == threshold goes right
    }

    #[test]
    fn split_condition_missing_default_left() {
        let cond = SplitCondition::new(0, 0.5, true);
        assert!(cond.go_left(f32::NAN));
    }

    #[test]
    fn split_condition_missing_default_right() {
        let cond = SplitCondition::new(0, 0.5, false);
        assert!(!cond.go_left(f32::NAN));
    }

    #[test]
    fn node_leaf() {
        let node: Node<ScalarLeaf> = Node::leaf(ScalarLeaf(1.5));

        assert!(node.is_leaf());
        assert_eq!(node.leaf_value(), Some(&ScalarLeaf(1.5)));
        assert_eq!(node.split_condition(), None);
        assert_eq!(node.children(), None);
    }

    #[test]
    fn node_split() {
        let cond = SplitCondition::new(2, 0.5, false);
        let node: Node<ScalarLeaf> = Node::split(cond, 1, 2);

        assert!(!node.is_leaf());
        assert_eq!(node.leaf_value(), None);
        assert_eq!(node.split_condition(), Some(&cond));
        assert_eq!(node.children(), Some((1, 2)));
    }
}
