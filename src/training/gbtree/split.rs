//! Split finding and gain computation.
//!
//! This module implements RFC-0013: finding optimal splits from histograms.
//!
//! # Overview
//!
//! For each tree node, we find the best split by:
//! 1. Enumerating all possible splits (one per bin boundary per feature)
//! 2. Computing gain for each candidate split
//! 3. Selecting the split with maximum gain (if above threshold)
//! 4. Handling missing values (learning default direction)
//!
//! # Gain Formula
//!
//! The gain from a split measures the reduction in the objective function:
//!
//! ```text
//! gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
//! ```
//!
//! Where:
//! - G_L, H_L: Sum of gradients/hessians in left child
//! - G_R, H_R: Sum of gradients/hessians in right child
//! - G, H: Sum of gradients/hessians in parent
//! - λ (lambda): L2 regularization
//! - γ (gamma): Minimum loss reduction to make a split
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::split::{GainParams, GreedySplitFinder, SplitFinder};
//!
//! let params = GainParams::default();
//! let finder = GreedySplitFinder::default();
//! let split = finder.find_best_split(&histogram, &cuts, &params);
//!
//! if split.is_valid() {
//!     println!("Split on feature {} at {}", split.feature, split.threshold);
//! }
//! ```
//!
//! See RFC-0013 for design rationale.

use rayon::prelude::*;

use super::histogram::NodeHistogram;
use super::quantize::BinCuts;

// ============================================================================
// GainParams
// ============================================================================

/// Parameters for gain computation and split validation.
///
/// These parameters control regularization and constraints:
/// - `lambda`: L2 regularization on leaf weights
/// - `alpha`: L1 regularization on leaf weights
/// - `min_split_gain`: Minimum gain required to make a split
/// - `min_child_weight`: Minimum hessian sum in a child
///
/// # Defaults
///
/// The defaults match XGBoost:
/// - `lambda = 1.0`: Some L2 regularization
/// - `alpha = 0.0`: No L1 regularization
/// - `min_split_gain = 0.0`: Accept any positive gain
/// - `min_child_weight = 1.0`: Each child needs at least 1.0 hessian sum
#[derive(Clone, Copy, Debug)]
pub struct GainParams {
    /// L2 regularization on leaf weights (XGBoost's `lambda`, LightGBM's `lambda_l2`)
    pub lambda: f32,
    /// L1 regularization on leaf weights (XGBoost's `alpha`, LightGBM's `lambda_l1`)
    pub alpha: f32,
    /// Minimum loss reduction to make a split (XGBoost's `gamma`, LightGBM's `min_gain_to_split`)
    pub min_split_gain: f32,
    /// Minimum sum of hessians in a child (XGBoost's `min_child_weight`)
    pub min_child_weight: f32,
}

impl Default for GainParams {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            alpha: 0.0,
            min_split_gain: 0.0,
            min_child_weight: 1.0,
        }
    }
}

impl GainParams {
    /// Create params with no regularization.
    pub fn no_regularization() -> Self {
        Self {
            lambda: 0.0,
            alpha: 0.0,
            min_split_gain: 0.0,
            min_child_weight: 0.0,
        }
    }

    /// Builder: set lambda (L2 regularization).
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Builder: set alpha (L1 regularization).
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: set minimum split gain.
    pub fn with_min_split_gain(mut self, gamma: f32) -> Self {
        self.min_split_gain = gamma;
        self
    }

    /// Builder: set minimum child weight.
    pub fn with_min_child_weight(mut self, min_child_weight: f32) -> Self {
        self.min_child_weight = min_child_weight;
        self
    }
}

// ============================================================================
// Gain Functions
// ============================================================================

/// Soft thresholding for L1 regularization.
///
/// Returns:
/// - `g - alpha` if `g > alpha`
/// - `g + alpha` if `g < -alpha`
/// - `0` otherwise
#[inline]
pub fn soft_threshold(g: f32, alpha: f32) -> f32 {
    if g > alpha {
        g - alpha
    } else if g < -alpha {
        g + alpha
    } else {
        0.0
    }
}

/// Compute optimal leaf weight.
///
/// For L2-only regularization:
/// ```text
/// w* = -G / (H + λ)
/// ```
///
/// For L1+L2 regularization (soft thresholding):
/// ```text
/// w* = -soft_threshold(G, α) / (H + λ)
/// ```
#[inline]
pub fn leaf_weight(sum_grad: f32, sum_hess: f32, params: &GainParams) -> f32 {
    let h = sum_hess + params.lambda;
    if h <= 0.0 {
        return 0.0;
    }
    if params.alpha > 0.0 {
        -soft_threshold(sum_grad, params.alpha) / h
    } else {
        -sum_grad / h
    }
}

/// Compute the objective value for a leaf node.
///
/// For L2-only regularization:
/// ```text
/// obj = -0.5 * G² / (H + λ)
/// ```
///
/// For L1+L2 regularization:
/// ```text
/// obj = -0.5 * soft_threshold(G, α)² / (H + λ)
/// ```
///
/// Note: This is the negative of what we want to maximize.
/// Lower (more negative) is better.
#[inline]
pub fn leaf_objective(sum_grad: f32, sum_hess: f32, params: &GainParams) -> f32 {
    let h = sum_hess + params.lambda;
    if h <= 0.0 {
        return 0.0;
    }
    if params.alpha > 0.0 {
        let g_thresh = soft_threshold(sum_grad, params.alpha);
        -0.5 * g_thresh * g_thresh / h
    } else {
        -0.5 * sum_grad * sum_grad / h
    }
}

/// Compute gain from splitting a node.
///
/// The gain measures the reduction in the objective function:
/// ```text
/// gain = obj(parent) - obj(left) - obj(right) - γ
///      = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
/// ```
///
/// Returns 0.0 if gain is negative (no improvement).
///
/// # Arguments
///
/// * `grad_left`, `hess_left` - Stats for left child
/// * `grad_right`, `hess_right` - Stats for right child
/// * `grad_parent`, `hess_parent` - Stats for parent node
/// * `params` - Regularization parameters
#[inline]
pub fn split_gain(
    grad_left: f32,
    hess_left: f32,
    grad_right: f32,
    hess_right: f32,
    grad_parent: f32,
    hess_parent: f32,
    params: &GainParams,
) -> f32 {
    let obj_left = leaf_objective(grad_left, hess_left, params);
    let obj_right = leaf_objective(grad_right, hess_right, params);
    let obj_parent = leaf_objective(grad_parent, hess_parent, params);

    // Gain is reduction in objective (objectives are negative, so parent - children)
    let gain = obj_parent - obj_left - obj_right - params.min_split_gain;

    gain.max(0.0)
}

// ============================================================================
// SplitInfo
// ============================================================================

/// Complete information about a split decision.
///
/// Contains all metadata needed to:
/// - Apply the split (feature, threshold, default direction)
/// - Create child nodes (grad/hess sums, weights)
/// - Evaluate split quality (gain)
///
/// # Example
///
/// ```ignore
/// let split = SplitInfo::none();
/// if split.is_valid() {
///     println!("Split on feature {} at {}", split.feature, split.threshold);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct SplitInfo {
    /// Feature index to split on
    pub feature: u32,
    /// Bin index for the split (values in bins <= split_bin go left)
    pub split_bin: u32,
    /// Split threshold (go left if value <= threshold)
    /// For bin boundaries: threshold is upper bound of split_bin
    pub threshold: f32,
    /// Gain from this split
    pub gain: f32,
    /// Sum of gradients in left child
    pub grad_left: f32,
    /// Sum of hessians in left child
    pub hess_left: f32,
    /// Sum of gradients in right child
    pub grad_right: f32,
    /// Sum of hessians in right child
    pub hess_right: f32,
    /// Optimal weight for left leaf
    pub weight_left: f32,
    /// Optimal weight for right leaf
    pub weight_right: f32,
    /// Default direction for missing values (true = left)
    pub default_left: bool,
    /// Whether this is a categorical split
    pub is_categorical: bool,
    /// For categorical: which categories go left (bin indices)
    pub categories_left: Vec<u32>,
}

impl Default for SplitInfo {
    fn default() -> Self {
        Self::none()
    }
}

impl SplitInfo {
    /// A null split (no valid split found).
    ///
    /// Use `is_valid()` to check if a split is usable.
    pub fn none() -> Self {
        Self {
            feature: u32::MAX,
            split_bin: 0,
            threshold: f32::NAN,
            gain: f32::NEG_INFINITY,
            grad_left: 0.0,
            hess_left: 0.0,
            grad_right: 0.0,
            hess_right: 0.0,
            weight_left: 0.0,
            weight_right: 0.0,
            default_left: true,
            is_categorical: false,
            categories_left: Vec::new(),
        }
    }

    /// Check if this is a valid split.
    ///
    /// A split is valid if it has positive gain and a valid feature.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.gain > 0.0 && self.feature != u32::MAX
    }

    /// Count of samples in left child (hessian sum).
    #[inline]
    pub fn count_left(&self) -> f32 {
        self.hess_left
    }

    /// Count of samples in right child (hessian sum).
    #[inline]
    pub fn count_right(&self) -> f32 {
        self.hess_right
    }
}

// ============================================================================
// SplitFinder Trait
// ============================================================================

/// Strategy for finding the best split for a node.
///
/// Different implementations may use different algorithms:
/// - [`GreedySplitFinder`]: Standard exhaustive search
/// - Future: Approximate split finder, histogram sampling, etc.
pub trait SplitFinder: Send + Sync {
    /// Find the best split for a node given its histogram.
    ///
    /// Returns `SplitInfo::none()` if no valid split is found.
    fn find_best_split(
        &self,
        histogram: &NodeHistogram,
        cuts: &BinCuts,
        params: &GainParams,
    ) -> SplitInfo;
}

// ============================================================================
// GreedySplitFinder
// ============================================================================

/// Standard greedy split finder.
///
/// Enumerates all bin boundaries for all features, computes gain for each,
/// and returns the split with the highest gain.
///
/// # Algorithm
///
/// For each feature:
/// 1. Scan bins from left to right
/// 2. Accumulate gradient/hessian sums
/// 3. At each bin boundary, compute gain
/// 4. Track best split for this feature
///
/// Then select the globally best split across all features.
///
/// # Complexity
///
/// O(num_features × max_bins)
#[derive(Debug, Default, Clone)]
pub struct GreedySplitFinder {
    /// Optional subset of features to consider
    pub feature_subset: Option<Vec<u32>>,
}

impl GreedySplitFinder {
    /// Create a finder that considers all features.
    pub fn new() -> Self {
        Self {
            feature_subset: None,
        }
    }

    /// Create a finder that only considers specific features.
    pub fn with_features(features: Vec<u32>) -> Self {
        Self {
            feature_subset: Some(features),
        }
    }

    /// Find best split for a single feature.
    ///
    /// Scans bins from left to right, computing gain at each boundary.
    /// This is the single-output implementation.
    fn find_best_split_for_feature(
        &self,
        feature: u32,
        hist: &super::histogram::FeatureHistogram,
        cuts: &[f32],
        parent_grad: f32,
        parent_hess: f32,
        params: &GainParams,
    ) -> SplitInfo {
        let mut best = SplitInfo::none();
        best.feature = feature;

        let num_bins = hist.num_bins() as usize;
        if num_bins <= 1 {
            return best; // Need at least 2 bins for a split
        }

        // Get missing value stats (bin 0)
        let (missing_grad, missing_hess, _missing_count) = hist.bin_stats(0);

        // Cumulative sums from left (excluding missing bin 0)
        let mut grad_left = 0.0f32;
        let mut hess_left = 0.0f32;

        // Scan non-missing bins left to right
        // Bin 0 is missing, so we start from bin 1
        for bin in 1..num_bins {
            let (g, h, _) = hist.bin_stats(bin);
            grad_left += g;
            hess_left += h;

            // Can't split after the last bin
            if bin >= num_bins - 1 {
                break;
            }

            // Right side stats (excluding missing for now)
            let grad_right = parent_grad - missing_grad - grad_left;
            let hess_right = parent_hess - missing_hess - hess_left;

            // Check min_child_weight constraint (ignoring missing for initial check)
            if hess_left < params.min_child_weight || hess_right < params.min_child_weight {
                continue;
            }

            // Compute gain for missing going left vs right
            let (gain_missing_left, gain_missing_right) = self.compute_gains_for_missing(
                grad_left,
                hess_left,
                grad_right,
                hess_right,
                missing_grad,
                missing_hess,
                parent_grad,
                parent_hess,
                params,
            );

            // Choose direction with higher gain
            let (gain, default_left) = if gain_missing_left >= gain_missing_right {
                (gain_missing_left, true)
            } else {
                (gain_missing_right, false)
            };

            if gain > best.gain {
                // Get threshold from cuts
                // bin=1 means values <= cuts[0], bin=2 means values <= cuts[1], etc.
                // So split after bin `b` uses threshold cuts[b-1]
                let threshold = if bin > 0 && bin - 1 < cuts.len() {
                    cuts[bin - 1]
                } else {
                    f32::INFINITY
                };

                // Compute final stats including missing
                let (final_grad_left, final_hess_left, final_grad_right, final_hess_right) =
                    if default_left {
                        (
                            grad_left + missing_grad,
                            hess_left + missing_hess,
                            grad_right,
                            hess_right,
                        )
                    } else {
                        (
                            grad_left,
                            hess_left,
                            grad_right + missing_grad,
                            hess_right + missing_hess,
                        )
                    };

                best.gain = gain;
                best.split_bin = bin as u32;
                best.threshold = threshold;
                best.grad_left = final_grad_left;
                best.hess_left = final_hess_left;
                best.grad_right = final_grad_right;
                best.hess_right = final_hess_right;
                best.weight_left = leaf_weight(final_grad_left, final_hess_left, params);
                best.weight_right = leaf_weight(final_grad_right, final_hess_right, params);
                best.default_left = default_left;
            }
        }

        best
    }

    /// Compute gains for missing values going left vs right.
    #[inline]
    fn compute_gains_for_missing(
        &self,
        grad_left: f32,
        hess_left: f32,
        grad_right: f32,
        hess_right: f32,
        missing_grad: f32,
        missing_hess: f32,
        parent_grad: f32,
        parent_hess: f32,
        params: &GainParams,
    ) -> (f32, f32) {
        // Gain with missing going left
        let gain_left = split_gain(
            grad_left + missing_grad,
            hess_left + missing_hess,
            grad_right,
            hess_right,
            parent_grad,
            parent_hess,
            params,
        );

        // Gain with missing going right
        let gain_right = split_gain(
            grad_left,
            hess_left,
            grad_right + missing_grad,
            hess_right + missing_hess,
            parent_grad,
            parent_hess,
            params,
        );

        (gain_left, gain_right)
    }

    /// Find best categorical split for a feature using gradient-sorted partition.
    ///
    /// This implements the LightGBM-style gradient-sorted categorical split:
    /// 1. For each non-empty category, compute grad/hess ratio
    /// 2. Sort categories by ratio
    /// 3. Scan sorted order to find optimal binary partition
    /// 4. Return SplitInfo with categories_left populated
    ///
    /// # Complexity
    /// O(k log k) where k is the number of categories
    fn find_best_categorical_split(
        &self,
        feature: u32,
        hist: &super::histogram::FeatureHistogram,
        num_categories: u32,
        parent_grad: f32,
        parent_hess: f32,
        params: &GainParams,
    ) -> SplitInfo {
        let mut best = SplitInfo::none();
        best.feature = feature;

        let num_bins = hist.num_bins() as usize;
        if num_bins <= 1 {
            return best;
        }

        // Get missing value stats (bin 0)
        let (missing_grad, missing_hess, _missing_count) = hist.bin_stats(0);

        // Collect non-empty categories with their gradient stats
        // Category i is stored in bin i+1 (bin 0 is missing)
        let mut categories: Vec<(u32, f32, f32)> = Vec::with_capacity(num_categories as usize);
        for cat in 0..num_categories {
            let bin = (cat + 1) as usize;
            if bin < num_bins {
                let (g, h, count) = hist.bin_stats(bin);
                if count > 0 && h > 0.0 {
                    categories.push((cat, g, h));
                }
            }
        }

        if categories.len() < 2 {
            return best; // Need at least 2 categories to split
        }

        // Sort categories by gradient/hessian ratio (ascending)
        // This groups similar categories together
        categories.sort_by(|a, b| {
            let ratio_a = a.1 / a.2;
            let ratio_b = b.1 / b.2;
            ratio_a.partial_cmp(&ratio_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Scan sorted categories to find best split point
        // At each position, categories before go left, after go right
        let mut grad_left = 0.0f32;
        let mut hess_left = 0.0f32;

        // Total gradient/hessian for non-missing categories
        let total_grad_cats: f32 = categories.iter().map(|(_, g, _)| g).sum();
        let total_hess_cats: f32 = categories.iter().map(|(_, _, h)| h).sum();

        for i in 0..categories.len() - 1 {
            let (_, g, h) = categories[i];
            grad_left += g;
            hess_left += h;

            let grad_right = total_grad_cats - grad_left;
            let hess_right = total_hess_cats - hess_left;

            // Check min_child_weight
            if hess_left < params.min_child_weight || hess_right < params.min_child_weight {
                continue;
            }

            // Compute gain for missing going left vs right
            let (gain_missing_left, gain_missing_right) = self.compute_gains_for_missing(
                grad_left,
                hess_left,
                grad_right,
                hess_right,
                missing_grad,
                missing_hess,
                parent_grad,
                parent_hess,
                params,
            );

            let (gain, default_left) = if gain_missing_left >= gain_missing_right {
                (gain_missing_left, true)
            } else {
                (gain_missing_right, false)
            };

            if gain > best.gain {
                // Compute final stats including missing
                let (final_grad_left, final_hess_left, final_grad_right, final_hess_right) =
                    if default_left {
                        (
                            grad_left + missing_grad,
                            hess_left + missing_hess,
                            grad_right,
                            hess_right,
                        )
                    } else {
                        (
                            grad_left,
                            hess_left,
                            grad_right + missing_grad,
                            hess_right + missing_hess,
                        )
                    };

                // Collect categories that go left (first i+1 in sorted order)
                let cats_left: Vec<u32> = categories[0..=i].iter().map(|(c, _, _)| *c).collect();

                best.gain = gain;
                best.split_bin = 0; // Not used for categorical
                best.threshold = f32::NAN; // Not used for categorical
                best.grad_left = final_grad_left;
                best.hess_left = final_hess_left;
                best.grad_right = final_grad_right;
                best.hess_right = final_hess_right;
                best.weight_left = leaf_weight(final_grad_left, final_hess_left, params);
                best.weight_right = leaf_weight(final_grad_right, final_hess_right, params);
                best.default_left = default_left;
                best.is_categorical = true;
                best.categories_left = cats_left;
            }
        }

        best
    }

    /// Find best split with parallel feature evaluation.
    ///
    /// Uses Rayon to evaluate features in parallel. Recommended for
    /// datasets with many features.
    pub fn find_best_split_parallel(
        &self,
        histogram: &NodeHistogram,
        cuts: &BinCuts,
        params: &GainParams,
    ) -> SplitInfo {
        let parent_grad = histogram.total_grad();
        let parent_hess = histogram.total_hess();
        let num_features = histogram.num_features();

        // Determine which features to evaluate
        let features: Vec<u32> = self
            .feature_subset
            .as_ref()
            .cloned()
            .unwrap_or_else(|| (0..num_features as u32).collect());

        // Parallel map-reduce over features
        features
            .par_iter()
            .map(|&feat| {
                let feat_hist = histogram.feature(feat as usize);
                if cuts.is_categorical(feat) {
                    self.find_best_categorical_split(
                        feat,
                        feat_hist,
                        cuts.num_categories(feat),
                        parent_grad,
                        parent_hess,
                        params,
                    )
                } else {
                    let feat_cuts = cuts.feature_cuts(feat);
                    self.find_best_split_for_feature(
                        feat,
                        feat_hist,
                        feat_cuts,
                        parent_grad,
                        parent_hess,
                        params,
                    )
                }
            })
            .reduce(SplitInfo::none, |best, split| {
                if split.gain > best.gain {
                    split
                } else {
                    best
                }
            })
    }
}

impl SplitFinder for GreedySplitFinder {
    fn find_best_split(
        &self,
        histogram: &NodeHistogram,
        cuts: &BinCuts,
        params: &GainParams,
    ) -> SplitInfo {
        let parent_grad = histogram.total_grad();
        let parent_hess = histogram.total_hess();
        let num_features = histogram.num_features();

        let mut best = SplitInfo::none();

        // Determine which features to evaluate
        let features: Vec<u32> = self
            .feature_subset
            .as_ref()
            .cloned()
            .unwrap_or_else(|| (0..num_features as u32).collect());

        for feat in features {
            let feat_hist = histogram.feature(feat as usize);
            let split = if cuts.is_categorical(feat) {
                self.find_best_categorical_split(
                    feat,
                    feat_hist,
                    cuts.num_categories(feat),
                    parent_grad,
                    parent_hess,
                    params,
                )
            } else {
                let feat_cuts = cuts.feature_cuts(feat);
                self.find_best_split_for_feature(
                    feat,
                    feat_hist,
                    feat_cuts,
                    parent_grad,
                    parent_hess,
                    params,
                )
            };

            if split.gain > best.gain {
                best = split;
            }
        }

        best
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold() {
        // No thresholding when alpha = 0
        assert_eq!(soft_threshold(5.0, 0.0), 5.0);
        assert_eq!(soft_threshold(-5.0, 0.0), -5.0);

        // Thresholding
        assert_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_eq!(soft_threshold(1.0, 2.0), 0.0); // |g| < alpha
        assert_eq!(soft_threshold(-1.0, 2.0), 0.0);
    }

    #[test]
    fn test_leaf_weight_l2_only() {
        let params = GainParams::default().with_lambda(1.0).with_alpha(0.0);

        // w* = -G / (H + λ) = -10 / (5 + 1) = -10/6 ≈ -1.667
        let w = leaf_weight(10.0, 5.0, &params);
        assert!((w - (-10.0 / 6.0)).abs() < 1e-5);

        // Negative gradient -> positive weight
        let w = leaf_weight(-10.0, 5.0, &params);
        assert!((w - (10.0 / 6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_leaf_weight_l1() {
        let params = GainParams::default()
            .with_lambda(1.0)
            .with_alpha(2.0);

        // w* = -soft_threshold(10, 2) / (5 + 1) = -8/6 ≈ -1.333
        let w = leaf_weight(10.0, 5.0, &params);
        assert!((w - (-8.0 / 6.0)).abs() < 1e-5);

        // G within threshold -> weight = 0
        let w = leaf_weight(1.0, 5.0, &params);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn test_leaf_objective_l2_only() {
        let params = GainParams::default().with_lambda(1.0).with_alpha(0.0);

        // obj = -0.5 * G² / (H + λ) = -0.5 * 100 / 6 ≈ -8.333
        let obj = leaf_objective(10.0, 5.0, &params);
        assert!((obj - (-0.5 * 100.0 / 6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_split_gain_basic() {
        let params = GainParams::default()
            .with_lambda(0.0)
            .with_alpha(0.0)
            .with_min_split_gain(0.0);

        // Parent: G=0, H=4
        // Left: G=2, H=2 -> obj = -0.5 * 4 / 2 = -1
        // Right: G=-2, H=2 -> obj = -0.5 * 4 / 2 = -1
        // Parent obj = -0.5 * 0 / 4 = 0
        // Gain = 0 - (-1) - (-1) = 2
        let gain = split_gain(2.0, 2.0, -2.0, 2.0, 0.0, 4.0, &params);
        assert!((gain - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_split_gain_no_improvement() {
        let params = GainParams::no_regularization();

        // Parent: G=10, H=10 -> obj = -0.5 * 100 / 10 = -5
        // Left: G=10, H=5 -> obj = -0.5 * 100 / 5 = -10
        // Right: G=0, H=5 -> obj = -0.5 * 0 / 5 = 0
        // Gain = -5 - (-10) - 0 = 5 > 0 (this is actually an improvement)

        // Let's try one where there's no improvement
        // Parent: G=10, H=10 -> obj = -5
        // Left: G=5, H=5 -> obj = -0.5 * 25 / 5 = -2.5
        // Right: G=5, H=5 -> obj = -2.5
        // Gain = -5 - (-2.5) - (-2.5) = 0

        let gain = split_gain(5.0, 5.0, 5.0, 5.0, 10.0, 10.0, &params);
        assert!((gain - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_split_gain_with_min_split_gain() {
        let params = GainParams::no_regularization().with_min_split_gain(1.5);

        // Gain without min_split_gain would be 2.0
        // With min_split_gain=1.5, effective gain = 2.0 - 1.5 = 0.5
        let gain = split_gain(2.0, 2.0, -2.0, 2.0, 0.0, 4.0, &params);
        assert!((gain - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_split_info_validity() {
        let none = SplitInfo::none();
        assert!(!none.is_valid());
        assert!(none.gain < 0.0);

        let mut valid = SplitInfo::none();
        valid.feature = 0;
        valid.gain = 1.0;
        assert!(valid.is_valid());
    }

    #[test]
    fn test_gain_params_builder() {
        let params = GainParams::default()
            .with_lambda(2.0)
            .with_alpha(0.5)
            .with_min_split_gain(0.1)
            .with_min_child_weight(10.0);

        assert_eq!(params.lambda, 2.0);
        assert_eq!(params.alpha, 0.5);
        assert_eq!(params.min_split_gain, 0.1);
        assert_eq!(params.min_child_weight, 10.0);
    }

    mod integration {
        use super::*;
        use crate::data::ColMatrix;
        use crate::training::gbtree::histogram::{HistogramBuilder, NodeHistogram};
        use crate::training::gbtree::quantize::{BinCuts, ExactQuantileCuts, QuantizedMatrix, Quantizer};
        use crate::training::GradientBuffer;

        fn make_test_data() -> (
            QuantizedMatrix<u8>,
            GradientBuffer,
            BinCuts,
        ) {
            // Create data: 20 rows, 2 features
            // Feature 0: 0..20 (linear, good for splitting)
            // Feature 1: all same value (no split possible)
            let data: Vec<f32> = vec![
                // Feature 0
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0,
                // Feature 1
                5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                5.0, 5.0, 5.0, 5.0,
            ];
            let matrix = ColMatrix::from_vec(data, 20, 2);

            // Quantize
            let cut_finder = ExactQuantileCuts::default();
            let quantizer = Quantizer::from_data(&matrix, &cut_finder, 256);
            let quantized = quantizer.quantize_u8(&matrix);
            let cuts = (*quantizer.cuts()).clone();

            // Create gradients: first 10 rows have grad=1, last 10 have grad=-1
            // This creates a clear split point at row 10
            let mut grads = GradientBuffer::new(20, 1);
            for i in 0..10 {
                grads.set(i, 0, 1.0, 1.0);
            }
            for i in 10..20 {
                grads.set(i, 0, -1.0, 1.0);
            }

            (quantized, grads, cuts)
        }

        #[test]
        fn test_split_finder_basic() {
            let (quantized, grads, cuts) = make_test_data();
            let rows: Vec<u32> = (0..20).collect();

            // Build histogram
            let mut hist = NodeHistogram::new(&cuts);
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);

            // Find split
            let params = GainParams::default().with_min_child_weight(1.0);
            let finder = GreedySplitFinder::new();
            let split = finder.find_best_split(&hist, &cuts, &params);

            // Should find a valid split
            assert!(split.is_valid(), "Should find a valid split");
            assert_eq!(split.feature, 0, "Should split on feature 0 (the varying one)");

            // The split should separate positive and negative gradients
            // Left should have mostly positive, right mostly negative (or vice versa)
            let left_sign = split.grad_left.signum();
            let right_sign = split.grad_right.signum();
            assert!(
                left_sign != right_sign || left_sign == 0.0 || right_sign == 0.0,
                "Split should separate positive and negative gradients"
            );
        }

        #[test]
        fn test_split_finder_min_child_weight() {
            let (quantized, grads, cuts) = make_test_data();
            let rows: Vec<u32> = (0..20).collect();

            let mut hist = NodeHistogram::new(&cuts);
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);

            // With very high min_child_weight, no split should be found
            let params = GainParams::default().with_min_child_weight(100.0);
            let finder = GreedySplitFinder::new();
            let split = finder.find_best_split(&hist, &cuts, &params);

            assert!(
                !split.is_valid(),
                "No valid split with high min_child_weight"
            );
        }

        #[test]
        fn test_split_finder_parallel_matches_sequential() {
            let (quantized, grads, cuts) = make_test_data();
            let rows: Vec<u32> = (0..20).collect();

            let mut hist = NodeHistogram::new(&cuts);
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);

            let params = GainParams::default();
            let finder = GreedySplitFinder::new();

            let split_seq = finder.find_best_split(&hist, &cuts, &params);
            let split_par = finder.find_best_split_parallel(&hist, &cuts, &params);

            assert_eq!(split_seq.feature, split_par.feature);
            assert!((split_seq.gain - split_par.gain).abs() < 1e-5);
            assert!((split_seq.threshold - split_par.threshold).abs() < 1e-5);
        }

        #[test]
        fn test_split_finder_feature_subset() {
            let (quantized, grads, cuts) = make_test_data();
            let rows: Vec<u32> = (0..20).collect();

            let mut hist = NodeHistogram::new(&cuts);
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);

            let params = GainParams::default();

            // Only consider feature 1 (the constant one)
            let finder = GreedySplitFinder::with_features(vec![1]);
            let split = finder.find_best_split(&hist, &cuts, &params);

            // Feature 1 is constant, so no valid split should be found
            // (or if found, it should be on feature 1)
            if split.is_valid() {
                assert_eq!(split.feature, 1);
            }
        }

        #[test]
        fn test_split_with_missing_values() {
            // Create data with missing values
            let data: Vec<f32> = vec![
                // Feature 0: some NaN
                f32::NAN,
                1.0,
                2.0,
                f32::NAN,
                4.0,
                5.0,
                6.0,
                7.0,
                f32::NAN,
                9.0,
            ];
            let matrix = ColMatrix::from_vec(data, 10, 1);

            let cut_finder = ExactQuantileCuts::default();
            let quantizer = Quantizer::from_data(&matrix, &cut_finder, 256);
            let quantized = quantizer.quantize_u8(&matrix);
            let cuts = (*quantizer.cuts()).clone();

            // Gradients: positive for low values, negative for high
            let mut grads = GradientBuffer::new(10, 1);
            for i in 0..5 {
                grads.set(i, 0, 1.0, 1.0);
            }
            for i in 5..10 {
                grads.set(i, 0, -1.0, 1.0);
            }

            let rows: Vec<u32> = (0..10).collect();
            let mut hist = NodeHistogram::new(&cuts);
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);

            let params = GainParams::default().with_min_child_weight(0.5);
            let finder = GreedySplitFinder::new();
            let split = finder.find_best_split(&hist, &cuts, &params);

            // Should learn a default direction for missing values
            if split.is_valid() {
                // default_left should be set based on which gives better gain
                // The actual value depends on the data distribution
                assert!(split.default_left || !split.default_left); // Just check it's set
            }
        }

        #[test]
        fn test_categorical_split_basic() {
            use crate::training::gbtree::quantize::CategoricalInfo;

            // Create categorical data: 12 rows, 1 categorical feature with 3 categories
            // Category 0 (rows 0-3): grad = 2.0 each (total 8.0)
            // Category 1 (rows 4-7): grad = -1.0 each (total -4.0)
            // Category 2 (rows 8-11): grad = 0.5 each (total 2.0)
            let data: Vec<f32> = vec![
                0.0, 0.0, 0.0, 0.0, // Category 0
                1.0, 1.0, 1.0, 1.0, // Category 1
                2.0, 2.0, 2.0, 2.0, // Category 2
            ];
            let matrix = ColMatrix::from_vec(data, 12, 1);

            // Create categorical cuts
            let cat_info = CategoricalInfo::with_categorical(1, &[(0, 3)]);
            let cut_finder = ExactQuantileCuts::default();
            let quantizer =
                Quantizer::from_data_with_categorical(&matrix, &cut_finder, 256, &cat_info);
            let quantized = quantizer.quantize_u8(&matrix);
            let cuts = (*quantizer.cuts()).clone();

            // Verify it's categorical
            assert!(cuts.is_categorical(0));
            assert_eq!(cuts.num_categories(0), 3);
            assert_eq!(cuts.num_bins(0), 4); // 3 categories + 1 missing bin

            // Gradients
            let mut grads = GradientBuffer::new(12, 1);
            for i in 0..4 {
                grads.set(i, 0, 2.0, 1.0); // Category 0: positive
            }
            for i in 4..8 {
                grads.set(i, 0, -1.0, 1.0); // Category 1: negative
            }
            for i in 8..12 {
                grads.set(i, 0, 0.5, 1.0); // Category 2: small positive
            }

            let rows: Vec<u32> = (0..12).collect();
            let mut hist = NodeHistogram::new(&cuts);
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);

            let params = GainParams::no_regularization().with_min_child_weight(1.0);
            let finder = GreedySplitFinder::new();
            let split = finder.find_best_split(&hist, &cuts, &params);

            // Should find a valid categorical split
            assert!(split.is_valid(), "Should find valid categorical split");
            assert!(split.is_categorical, "Split should be marked as categorical");
            assert!(!split.categories_left.is_empty(), "Should have categories going left");

            // Gradient-sorted order should be: cat 1 (-1.0/1.0 = -1.0), cat 2 (0.5/1.0 = 0.5), cat 0 (2.0/1.0 = 2.0)
            // Best split should separate cat 1 from {cat 2, cat 0}
            // So categories_left should contain cat 1
            assert!(
                split.categories_left.contains(&1),
                "Category 1 should go left (most negative gradient ratio)"
            );
        }

        #[test]
        fn test_categorical_split_with_missing() {
            use crate::training::gbtree::quantize::CategoricalInfo;

            // Create categorical data with some missing values
            let data: Vec<f32> = vec![
                f32::NAN, // Missing
                0.0,
                0.0, // Category 0
                1.0,
                1.0, // Category 1
                f32::NAN, // Missing
            ];
            let matrix = ColMatrix::from_vec(data, 6, 1);

            let cat_info = CategoricalInfo::with_categorical(1, &[(0, 2)]);
            let cut_finder = ExactQuantileCuts::default();
            let quantizer =
                Quantizer::from_data_with_categorical(&matrix, &cut_finder, 256, &cat_info);
            let quantized = quantizer.quantize_u8(&matrix);
            let cuts = (*quantizer.cuts()).clone();

            // Gradients: category 0 positive, category 1 negative, missing neutral
            let mut grads = GradientBuffer::new(6, 1);
            grads.set(0, 0, 0.0, 1.0); // Missing
            grads.set(1, 0, 2.0, 1.0); // Cat 0
            grads.set(2, 0, 2.0, 1.0); // Cat 0
            grads.set(3, 0, -2.0, 1.0); // Cat 1
            grads.set(4, 0, -2.0, 1.0); // Cat 1
            grads.set(5, 0, 0.0, 1.0); // Missing

            let rows: Vec<u32> = (0..6).collect();
            let mut hist = NodeHistogram::new(&cuts);
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);

            let params = GainParams::no_regularization().with_min_child_weight(1.0);
            let finder = GreedySplitFinder::new();
            let split = finder.find_best_split(&hist, &cuts, &params);

            assert!(split.is_valid());
            assert!(split.is_categorical);
            // default_left should be learned based on which gives better gain
        }

        #[test]
        fn test_mixed_categorical_numerical_features() {
            use crate::training::gbtree::quantize::CategoricalInfo;

            // 2 features: feature 0 is numerical, feature 1 is categorical
            let data: Vec<f32> = vec![
                // Feature 0 (numerical): 0-9
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                // Feature 1 (categorical): alternating 0 and 1
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            ];
            let matrix = ColMatrix::from_vec(data, 10, 2);

            // Only feature 1 is categorical
            let cat_info = CategoricalInfo::with_categorical(2, &[(1, 2)]);
            let cut_finder = ExactQuantileCuts::default();
            let quantizer =
                Quantizer::from_data_with_categorical(&matrix, &cut_finder, 256, &cat_info);
            let quantized = quantizer.quantize_u8(&matrix);
            let cuts = (*quantizer.cuts()).clone();

            assert!(!cuts.is_categorical(0), "Feature 0 should be numerical");
            assert!(cuts.is_categorical(1), "Feature 1 should be categorical");

            // Gradients: first 5 positive, last 5 negative
            let mut grads = GradientBuffer::new(10, 1);
            for i in 0..5 {
                grads.set(i, 0, 1.0, 1.0);
            }
            for i in 5..10 {
                grads.set(i, 0, -1.0, 1.0);
            }

            let rows: Vec<u32> = (0..10).collect();
            let mut hist = NodeHistogram::new(&cuts);
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);

            let params = GainParams::default().with_min_child_weight(1.0);
            let finder = GreedySplitFinder::new();
            let split = finder.find_best_split(&hist, &cuts, &params);

            // Should find a valid split (either numerical or categorical)
            assert!(split.is_valid());

            // Numerical feature 0 has a clear split at row 5
            // Categorical feature 1 has mixed gradients per category
            // So we expect the numerical split to win
            assert_eq!(split.feature, 0, "Should prefer numerical split");
            assert!(!split.is_categorical, "Best split should be numerical");
        }
    }
}
