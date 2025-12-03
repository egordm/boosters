//! Leaf value types for tree nodes.

/// Trait for values stored in leaf nodes.
pub trait LeafValue: Clone + Default + Send + Sync {
    /// Accumulate another leaf value (for prediction summation)
    fn accumulate(&mut self, other: &Self);
}

/// Scalar leaf value (single f32).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ScalarLeaf(pub f32);

impl LeafValue for ScalarLeaf {
    #[inline]
    fn accumulate(&mut self, other: &Self) {
        self.0 += other.0;
    }
}

impl From<f32> for ScalarLeaf {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<ScalarLeaf> for f32 {
    fn from(leaf: ScalarLeaf) -> Self {
        leaf.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_leaf_accumulates() {
        let mut acc = ScalarLeaf(0.0);
        acc.accumulate(&ScalarLeaf(1.5));
        acc.accumulate(&ScalarLeaf(2.5));
        assert_eq!(acc.0, 4.0);
    }

    #[test]
    fn scalar_leaf_default_is_zero() {
        let leaf = ScalarLeaf::default();
        assert_eq!(leaf.0, 0.0);
    }

    #[test]
    fn scalar_leaf_from_f32() {
        let leaf: ScalarLeaf = 2.5.into();
        assert_eq!(leaf.0, 2.5);
    }
}
