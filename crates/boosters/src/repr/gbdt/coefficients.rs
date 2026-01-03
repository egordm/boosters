//! Linear leaf coefficient storage for trees.
//!
//! Stores feature indices and coefficients for linear leaves in a packed format.
//! Follows the same pattern as `CategoriesStorage` for efficient inference.
//!
//! See RFC-0010 for design rationale.

use super::NodeId;

/// Storage for linear leaf coefficients in a tree.
///
/// Stores feature indices and coefficients in flat arrays with per-node segments.
/// Only leaves with non-zero coefficients have entries.
///
/// # Format
///
/// - `feature_indices`: flat array of feature indices for all linear leaves
/// - `coefficients`: flat array of coefficients (same length as feature_indices)
/// - `segments`: per-node `(start, len)` into the arrays. Nodes without
///   linear coefficients have `(0, 0)`.
/// - `intercepts`: per-node intercept adjustment. Zero for non-linear leaves.
///
/// # Prediction
///
/// For a leaf node `n` with features `x`:
/// ```ignore
/// let base = leaf_value(n);
/// if let Some((feat_idx, coefs)) = coefficients.leaf_terms(n) {
///     // Check for NaN in linear features
///     for &f in feat_idx {
///         if x[f].is_nan() {
///             return base; // Fall back to constant
///         }
///     }
///     let intercept = coefficients.intercept(n);
///     let linear: f32 = feat_idx.iter().zip(coefs).map(|(&f, &c)| c * x[f]).sum();
///     base + intercept + linear
/// } else {
///     base
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct LeafCoefficients {
    /// Flat array of feature indices for all linear leaves.
    feature_indices: Box<[u32]>,
    /// Flat array of coefficients (parallel to feature_indices).
    coefficients: Box<[f32]>,
    /// Per-node segment: `(start, len)` into the arrays.
    /// Indexed by node_idx. Nodes without linear coefficients have `(0, 0)`.
    segments: Box<[(u32, u16)]>,
    /// Per-node intercept adjustment. Zero for non-linear leaves.
    /// Indexed by node_idx.
    intercepts: Box<[f32]>,
}

use std::sync::LazyLock;

/// Static empty LeafCoefficients for returning references.
static EMPTY_COEFFICIENTS: LazyLock<LeafCoefficients> = LazyLock::new(LeafCoefficients::empty);

impl LeafCoefficients {
    /// Get a static reference to empty coefficients.
    #[inline]
    pub fn empty_ref() -> &'static Self {
        &EMPTY_COEFFICIENTS
    }

    /// Create empty coefficients storage.
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create coefficients storage from raw data.
    ///
    /// # Arguments
    ///
    /// * `feature_indices` - Flat array of feature indices
    /// * `coefficients` - Flat array of coefficients (same length as feature_indices)
    /// * `segments` - Per-node `(start, len)` into arrays
    /// * `intercepts` - Per-node intercept values
    pub fn new(
        feature_indices: Vec<u32>,
        coefficients: Vec<f32>,
        segments: Vec<(u32, u16)>,
        intercepts: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(
            feature_indices.len(),
            coefficients.len(),
            "feature_indices and coefficients must have same length"
        );
        Self {
            feature_indices: feature_indices.into_boxed_slice(),
            coefficients: coefficients.into_boxed_slice(),
            segments: segments.into_boxed_slice(),
            intercepts: intercepts.into_boxed_slice(),
        }
    }

    /// Get the linear terms for a leaf node.
    ///
    /// Returns `Some((feature_indices, coefficients))` if the node has
    /// linear coefficients or an intercept, `None` otherwise.
    /// May return `Some(([], []))` for intercept-only linear leaves.
    #[inline]
    pub fn leaf_terms(&self, node: NodeId) -> Option<(&[u32], &[f32])> {
        if self.segments.is_empty() {
            return None;
        }
        let (start, len) = self.segments[node as usize];
        // Check if this node has linear data (either coefficients or intercept)
        // If len == 0 but there's a non-zero intercept, return empty slices
        if len == 0 {
            // Check for non-zero intercept
            if self
                .intercepts
                .get(node as usize)
                .copied()
                .unwrap_or(0.0)
                .abs()
                < 1e-10
            {
                return None;
            }
            // Return empty slices - caller should use leaf_intercept
            return Some((&[], &[]));
        }
        let start = start as usize;
        let end = start + len as usize;
        Some((
            &self.feature_indices[start..end],
            &self.coefficients[start..end],
        ))
    }

    /// Get the intercept for a leaf node.
    ///
    /// Returns 0.0 for nodes without linear coefficients.
    #[inline]
    pub fn intercept(&self, node: NodeId) -> f32 {
        if self.intercepts.is_empty() {
            0.0
        } else {
            self.intercepts[node as usize]
        }
    }

    /// Whether this storage has any linear coefficients.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.feature_indices.is_empty()
    }

    /// Get the segments array (for serialization).
    #[inline]
    pub fn segments(&self) -> &[(u32, u16)] {
        &self.segments
    }

    /// Get the feature indices array (for serialization).
    #[inline]
    pub fn feature_indices(&self) -> &[u32] {
        &self.feature_indices
    }

    /// Get the coefficients array (for serialization).
    #[inline]
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Get the intercepts array (for serialization).
    #[inline]
    pub fn intercepts(&self) -> &[f32] {
        &self.intercepts
    }

    /// Check if a node has linear coefficients.
    #[inline]
    pub fn has_linear(&self, node: NodeId) -> bool {
        if self.segments.is_empty() {
            return false;
        }
        self.segments[node as usize].1 > 0
    }
}

/// Builder for creating LeafCoefficients during freeze().
///
/// Accumulates linear leaf data and packs it into the final format.
#[derive(Debug, Default)]
pub struct LeafCoefficientsBuilder {
    feature_indices: Vec<u32>,
    coefficients: Vec<f32>,
    /// Temporary storage: (node_idx, start, len)
    entries: Vec<(u32, u32, u16)>,
    intercepts: Vec<(u32, f32)>,
}

impl LeafCoefficientsBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a linear leaf entry.
    ///
    /// # Arguments
    /// * `node` - Node ID
    /// * `features` - Feature indices used
    /// * `intercept` - Intercept adjustment
    /// * `coefs` - Coefficients (parallel to features)
    pub fn add(&mut self, node: NodeId, features: &[u32], intercept: f32, coefs: &[f32]) {
        debug_assert_eq!(features.len(), coefs.len());
        if features.is_empty() && intercept.abs() < 1e-10 {
            return; // Nothing to store
        }

        let start = self.feature_indices.len() as u32;
        let len = features.len() as u16;

        self.feature_indices.extend_from_slice(features);
        self.coefficients.extend_from_slice(coefs);
        self.entries.push((node, start, len));
        self.intercepts.push((node, intercept));
    }

    /// Check if any entries have been added.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build the final LeafCoefficients storage.
    ///
    /// # Arguments
    /// * `num_nodes` - Total number of nodes in the tree
    pub fn build(self, num_nodes: usize) -> LeafCoefficients {
        if self.entries.is_empty() {
            return LeafCoefficients::empty();
        }

        // Build segments array indexed by node
        let mut segments = vec![(0u32, 0u16); num_nodes];
        for (node, start, len) in &self.entries {
            segments[*node as usize] = (*start, *len);
        }

        // Build intercepts array indexed by node
        let mut intercepts = vec![0.0f32; num_nodes];
        for (node, intercept) in &self.intercepts {
            intercepts[*node as usize] = *intercept;
        }

        LeafCoefficients::new(
            self.feature_indices,
            self.coefficients,
            segments,
            intercepts,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_coefficients() {
        let coefs = LeafCoefficients::empty();
        assert!(coefs.is_empty());
        assert!(coefs.leaf_terms(0).is_none());
        assert_eq!(coefs.intercept(0), 0.0);
    }

    #[test]
    fn leaf_terms_basic() {
        let coefs = LeafCoefficients::new(
            vec![0, 2],           // feature indices
            vec![0.5, -0.3],      // coefficients
            vec![(0, 0), (0, 2)], // node 0: none, node 1: 2 terms at offset 0
            vec![0.0, 1.5],       // intercepts
        );

        assert!(!coefs.has_linear(0));
        assert!(coefs.has_linear(1));

        assert!(coefs.leaf_terms(0).is_none());

        let (feats, coef) = coefs.leaf_terms(1).unwrap();
        assert_eq!(feats, &[0, 2]);
        assert_eq!(coef, &[0.5, -0.3]);
        assert_eq!(coefs.intercept(1), 1.5);
    }

    #[test]
    fn multiple_linear_leaves() {
        // Tree: 3 nodes, nodes 1 and 2 are linear leaves
        let coefs = LeafCoefficients::new(
            vec![0, 1, 2], // 3 terms total
            vec![1.0, 2.0, 3.0],
            vec![(0, 0), (0, 2), (2, 1)], // node 0: none, node 1: 2 terms, node 2: 1 term
            vec![0.0, 0.5, 0.1],
        );

        let (f1, c1) = coefs.leaf_terms(1).unwrap();
        assert_eq!(f1, &[0, 1]);
        assert_eq!(c1, &[1.0, 2.0]);

        let (f2, c2) = coefs.leaf_terms(2).unwrap();
        assert_eq!(f2, &[2]);
        assert_eq!(c2, &[3.0]);
    }

    #[test]
    fn builder_basic() {
        let mut builder = LeafCoefficientsBuilder::new();
        builder.add(1, &[0, 2], 1.5, &[0.5, -0.3]);
        builder.add(2, &[1], 0.0, &[1.0]);

        let coefs = builder.build(3);

        assert!(!coefs.has_linear(0));
        assert!(coefs.has_linear(1));
        assert!(coefs.has_linear(2));

        let (f, c) = coefs.leaf_terms(1).unwrap();
        assert_eq!(f, &[0, 2]);
        assert_eq!(c, &[0.5, -0.3]);
        assert_eq!(coefs.intercept(1), 1.5);
    }

    #[test]
    fn builder_empty_skipped() {
        let mut builder = LeafCoefficientsBuilder::new();
        builder.add(1, &[], 0.0, &[]); // Empty, should be skipped

        assert!(builder.is_empty());
    }

    #[test]
    fn builder_intercept_only() {
        let mut builder = LeafCoefficientsBuilder::new();
        builder.add(1, &[], 2.5, &[]); // No coefficients, but has intercept

        let coefs = builder.build(2);

        // Has intercept stored even though no coefficients
        assert_eq!(coefs.intercept(1), 2.5);
        // leaf_terms returns Some with empty slices for intercept-only linear leaves
        assert_eq!(coefs.leaf_terms(1), Some((&[][..], &[][..])));
    }
}
