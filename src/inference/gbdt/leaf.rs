//! Leaf value types for tree nodes.

/// Trait for values stored in leaf nodes.
pub trait LeafValue: Clone + Default + Send + Sync {
    /// Accumulate another leaf value (for prediction summation)
    fn accumulate(&mut self, other: &Self);

    /// Scale the leaf value by a factor (for learning rate application)
    fn scale(&mut self, factor: f32);
}

/// Scalar leaf value (single f32).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ScalarLeaf(pub f32);

impl LeafValue for ScalarLeaf {
    #[inline]
    fn accumulate(&mut self, other: &Self) {
        self.0 += other.0;
    }

    #[inline]
    fn scale(&mut self, factor: f32) {
        self.0 *= factor;
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

/// Vector leaf value for multi-output trees (K f32 values).
///
/// Used when a single tree produces K outputs simultaneously,
/// rather than the one-output-per-tree strategy where K separate
/// trees each produce scalar outputs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VectorLeaf {
    /// K output values
    pub values: Vec<f32>,
}

impl VectorLeaf {
    /// Create a new vector leaf with the given values.
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Number of outputs (K).
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the leaf has no outputs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get output value at index k.
    #[inline]
    pub fn get(&self, k: usize) -> f32 {
        self.values[k]
    }
}

impl LeafValue for VectorLeaf {
    #[inline]
    fn accumulate(&mut self, other: &Self) {
        debug_assert_eq!(self.values.len(), other.values.len());
        for (a, b) in self.values.iter_mut().zip(other.values.iter()) {
            *a += *b;
        }
    }

    #[inline]
    fn scale(&mut self, factor: f32) {
        for v in &mut self.values {
            *v *= factor;
        }
    }
}

impl From<Vec<f32>> for VectorLeaf {
    fn from(values: Vec<f32>) -> Self {
        Self { values }
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

    #[test]
    fn vector_leaf_accumulates() {
        let mut acc = VectorLeaf::new(vec![0.0, 0.0, 0.0]);
        acc.accumulate(&VectorLeaf::new(vec![1.0, 2.0, 3.0]));
        acc.accumulate(&VectorLeaf::new(vec![0.5, 0.5, 0.5]));
        assert_eq!(acc.values, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn vector_leaf_default_is_empty() {
        let leaf = VectorLeaf::default();
        assert!(leaf.is_empty());
    }

    #[test]
    fn vector_leaf_from_vec() {
        let leaf: VectorLeaf = vec![1.0, 2.0, 3.0].into();
        assert_eq!(leaf.len(), 3);
        assert_eq!(leaf.get(1), 2.0);
    }
}
