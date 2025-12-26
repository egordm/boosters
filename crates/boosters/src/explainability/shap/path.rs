//! PathState for TreeSHAP algorithm.
//!
//! Implements the path tracking structure from Lundberg & Lee (2017):
//! "A Unified Approach to Interpreting Model Predictions"
//!
//! The path tracks which features have been used in the current tree path
//! and maintains weights for computing SHAP contributions.

/// Path tracking state for TreeSHAP algorithm.
///
/// This struct tracks the features encountered along a decision path
/// and maintains the weights needed to compute SHAP values.
///
/// Based on Algorithm 2 from Lundberg et al. (2020):
/// "From local explanations to global understanding with explainable AI for trees"
#[derive(Clone, Debug)]
pub struct PathState {
    /// Feature indices along the path (-1 = no feature / root)
    features: Vec<i32>,
    /// Fraction of samples going left when feature not in coalition
    zero_fractions: Vec<f64>,
    /// Fraction of samples going left when feature in coalition
    one_fractions: Vec<f64>,
    /// Path weights for contribution computation
    weights: Vec<f64>,
    /// Current depth in the path
    depth: usize,
}

impl PathState {
    /// Create a new path state with capacity for up to `max_depth` nodes.
    pub fn new(max_depth: usize) -> Self {
        let capacity = max_depth + 2; // Extra space for safety
        Self {
            features: vec![-1; capacity],
            zero_fractions: vec![0.0; capacity],
            one_fractions: vec![0.0; capacity],
            weights: vec![0.0; capacity],
            depth: 0,
        }
    }

    /// Reset the path state for reuse.
    #[inline]
    pub fn reset(&mut self) {
        self.depth = 0;
        // Initialize root weight
        self.weights[0] = 1.0;
    }

    /// Current depth in the path.
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Extend the path with a new feature.
    ///
    /// # Arguments
    /// * `feature` - Feature index (or -1 for no feature)
    /// * `zero_fraction` - Fraction going this way when feature not in coalition
    /// * `one_fraction` - Fraction going this way when feature is in coalition
    pub fn extend(&mut self, feature: i32, zero_fraction: f64, one_fraction: f64) {
        let d = self.depth;
        self.depth += 1;

        // Store the new path element
        self.features[d] = feature;
        self.zero_fractions[d] = zero_fraction;
        self.one_fractions[d] = one_fraction;

        // Initialize weight for new depth
        self.weights[d] = if d == 0 { 1.0 } else { 0.0 };

        // Update weights using the SHAP path formula
        // This is the core recursion from Algorithm 2
        for i in (0..d).rev() {
            self.weights[i + 1] += one_fraction * self.weights[i] * (i + 1) as f64 / (d + 1) as f64;
            self.weights[i] *= zero_fraction * (d - i) as f64 / (d + 1) as f64;
        }
    }

    /// Unwind the path by one step (inverse of extend).
    ///
    /// This is used when backtracking in the tree traversal.
    pub fn unwind(&mut self) {
        if self.depth == 0 {
            return;
        }

        let d = self.depth - 1;
        let one_fraction = self.one_fractions[d];
        let zero_fraction = self.zero_fractions[d];

        // Inverse of the extend operation
        for i in 0..=d {
            let numer = (d + 1) as f64;
            if one_fraction != 0.0 {
                let denom = one_fraction * (i + 1) as f64;
                let scale = numer / denom;
                let term = if i > 0 { self.weights[i - 1] } else { 0.0 };
                self.weights[i] = (self.weights[i] - term) * scale;
            } else {
                // When one_fraction is 0, use the zero_fraction inverse
                let denom = zero_fraction * (d - i + 1) as f64;
                if denom != 0.0 {
                    self.weights[i] *= numer / denom;
                }
            }
        }

        self.depth = d;
    }

    /// Compute the unwound sum for a target feature index.
    ///
    /// This computes the contribution for a specific feature by summing
    /// weighted path contributions where that feature appears.
    ///
    /// # Arguments
    /// * `target_idx` - Index in the path (not feature index)
    ///
    /// # Returns
    /// The weighted sum contribution for this path position
    pub fn unwound_sum(&self, target_idx: usize) -> f64 {
        let d = self.depth;
        if target_idx >= d {
            return 0.0;
        }

        let one_fraction = self.one_fractions[target_idx];
        let zero_fraction = self.zero_fractions[target_idx];

        let mut total = 0.0;
        let mut next_one_portion = if d > 0 { self.weights[d - 1] } else { 1.0 };

        for i in (0..d).rev() {
            if one_fraction != 0.0 {
                let w = next_one_portion * (d as f64) / ((i + 1) as f64 * one_fraction);
                total += w;
                let prev_weight = if i > 0 { self.weights[i - 1] } else { 0.0 };
                next_one_portion =
                    prev_weight - self.weights[i] * zero_fraction * ((d - i) as f64) / (d as f64);
                if i == 0 {
                    next_one_portion = 0.0;
                }
            } else if zero_fraction != 0.0 {
                total += self.weights[i] / (zero_fraction * (d - i) as f64 / d as f64);
            }
        }

        total
    }

    /// Get the feature at a given path index.
    #[inline]
    pub fn feature(&self, idx: usize) -> i32 {
        self.features[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_path() {
        let path = PathState::new(10);
        assert_eq!(path.depth(), 0);
    }

    #[test]
    fn test_reset() {
        let mut path = PathState::new(10);
        path.reset();
        assert_eq!(path.depth(), 0);
        assert_eq!(path.weights[0], 1.0);
    }

    #[test]
    fn test_extend_single() {
        let mut path = PathState::new(10);
        path.reset();

        // Extend with a feature (50% go each way regardless of coalition)
        path.extend(0, 0.5, 0.5);

        assert_eq!(path.depth(), 1);
        assert_eq!(path.feature(0), 0);
    }

    #[test]
    fn test_extend_unwind_depth() {
        let mut path = PathState::new(10);
        path.reset();

        path.extend(0, 0.5, 0.5);
        assert_eq!(path.depth(), 1);

        path.unwind();
        assert_eq!(path.depth(), 0);
    }

    #[test]
    fn test_multiple_extend_unwind() {
        let mut path = PathState::new(10);
        path.reset();

        // Build a path
        path.extend(0, 0.6, 1.0); // Feature 0 goes left always when in coalition
        path.extend(1, 0.4, 0.0); // Feature 1 goes right always when in coalition
        path.extend(2, 0.5, 0.5); // Feature 2 is 50/50

        assert_eq!(path.depth(), 3);

        // Unwind back
        path.unwind();
        assert_eq!(path.depth(), 2);

        path.unwind();
        assert_eq!(path.depth(), 1);

        path.unwind();
        assert_eq!(path.depth(), 0);
    }

    #[test]
    fn test_path_weights_sum() {
        let mut path = PathState::new(10);
        path.reset();

        // After reset, weight should be 1
        assert_eq!(path.weights[0], 1.0);

        // Extend and check weights are being computed
        path.extend(0, 0.5, 0.5);

        // Weights should be non-negative
        assert!(path.weights[0] >= 0.0);
        assert!(path.weights[1] >= 0.0);
    }
}
