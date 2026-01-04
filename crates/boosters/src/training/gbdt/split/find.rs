//! Greedy split finder implementation.

use super::super::categorical::CatBitset;
use super::super::histograms::{HistogramBin, HistogramSlot};
use super::gain::{GainParams, NodeGainContext};
use super::types::SplitInfo;
use crate::utils::Parallelism;

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// Maximum categories supported in sorted partition scratch buffer.
const MAX_SCRATCH_CATS: usize = 256;

/// Default maximum categories for one-hot split strategy.
pub const DEFAULT_MAX_ONEHOT_CATS: u32 = 4;

/// Minimum features to consider parallel split finding.
const MIN_FEATURES_PARALLEL: usize = 16;

// =============================================================================
// Greedy Splitter
// =============================================================================

/// Greedy split finder that exhaustively scans all bins.
///
/// This is the standard split finding algorithm used by XGBoost and LightGBM.
/// For each feature, it scans all possible split points and returns the one
/// with highest gain.
///
/// # Configuration
///
/// The splitter owns its `GainParams` since these are static for training.
/// Parallelism is configured at construction and self-corrects based on workload.
#[derive(Clone, Debug)]
pub struct GreedySplitter {
    /// Gain computation and constraint parameters.
    gain_params: GainParams,
    /// Maximum categories to use one-hot strategy (above uses sorted partition).
    max_onehot_cats: u32,
    /// Parallelism hint (self-corrects if workload is too small).
    parallelism: Parallelism,
    /// Scratch buffer for categorical split (avoids allocation).
    cat_scratch: Vec<(u32, f64)>,
}

impl Default for GreedySplitter {
    fn default() -> Self {
        Self::new(GainParams::default())
    }
}

impl GreedySplitter {
    /// Create a new greedy splitter with default settings (sequential).
    pub fn new(gain_params: GainParams) -> Self {
        Self {
            gain_params,
            max_onehot_cats: DEFAULT_MAX_ONEHOT_CATS,
            parallelism: Parallelism::Sequential,
            cat_scratch: Vec::with_capacity(MAX_SCRATCH_CATS),
        }
    }

    /// Create a greedy splitter with custom configuration.
    pub fn with_config(
        gain_params: GainParams,
        max_onehot_cats: u32,
        parallelism: Parallelism,
    ) -> Self {
        Self {
            gain_params,
            max_onehot_cats,
            parallelism,
            cat_scratch: Vec::with_capacity(MAX_SCRATCH_CATS),
        }
    }

    /// Get the gain parameters.
    #[inline]
    pub fn gain_params(&self) -> &GainParams {
        &self.gain_params
    }

    /// Compute leaf weight using the splitter's regularization parameters.
    ///
    /// This delegates to `GainParams::compute_leaf_weight`.
    #[inline]
    pub fn compute_leaf_weight(&self, grad_sum: f64, hess_sum: f64) -> f32 {
        self.gain_params.compute_leaf_weight(grad_sum, hess_sum)
    }

    // =========================================================================
    // Main Entry Point
    // =========================================================================

    /// Find the best split for a node.
    ///
    /// Automatically selects between sequential and parallel search based on
    /// the configured strategy and number of features.
    ///
    /// # Arguments
    /// * `histogram` - Histogram with (grad, hess) bins per feature
    /// * `parent_grad` - Total gradient sum in the node
    /// * `parent_hess` - Total hessian sum in the node
    /// * `parent_count` - Number of samples in the node
    /// * `feature_types` - Whether each feature is categorical (true = categorical)
    /// * `feature_has_missing` - Whether each feature has missing values
    /// * `feature_is_bundle` - Whether each feature is a bundle column (bin 0 reserved)
    /// * `features` - Slice of feature indices to consider for split finding
    ///
    /// # Returns
    /// The best split found, or an invalid split if none meets constraints.
    #[allow(clippy::too_many_arguments)]
    pub fn find_split(
        &mut self,
        histogram: &HistogramSlot<'_>,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        feature_types: &[bool],
        feature_has_missing: &[bool],
        feature_is_bundle: &[bool],
        features: &[u32],
    ) -> SplitInfo {
        // Use sequential for small workloads
        let parallelism = if features.len() < MIN_FEATURES_PARALLEL {
            Parallelism::Sequential
        } else {
            self.parallelism
        };

        if parallelism.is_parallel() {
            self.find_split_parallel(
                histogram,
                parent_grad,
                parent_hess,
                parent_count,
                feature_types,
                feature_has_missing,
                feature_is_bundle,
                features,
            )
        } else {
            self.find_split_sequential(
                histogram,
                parent_grad,
                parent_hess,
                parent_count,
                feature_types,
                feature_has_missing,
                feature_is_bundle,
                features,
            )
        }
    }

    /// Find the best split using sequential search over features.
    #[allow(clippy::too_many_arguments)]
    pub fn find_split_sequential(
        &mut self,
        histogram: &HistogramSlot<'_>,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        feature_types: &[bool],
        feature_has_missing: &[bool],
        feature_is_bundle: &[bool],
        features: &[u32],
    ) -> SplitInfo {
        let mut best = SplitInfo::invalid();

        // Pre-compute parent score context once for all features
        let ctx = NodeGainContext::new(parent_grad, parent_hess, &self.gain_params);

        // Iterate over provided features
        for &feature in features {
            let feature = feature as usize;
            let is_categorical = feature_types.get(feature).copied().unwrap_or(false);
            let has_missing = feature_has_missing.get(feature).copied().unwrap_or(false);
            let is_bundle = feature_is_bundle.get(feature).copied().unwrap_or(false);
            let bins = histogram.feature_bins(feature);

            // Find best split for this feature (type-specific)
            let split = if is_categorical {
                self.find_categorical_split(
                    bins,
                    feature as u32,
                    parent_grad,
                    parent_hess,
                    parent_count,
                    &ctx,
                )
            } else {
                self.find_numerical_split(
                    bins,
                    feature as u32,
                    parent_grad,
                    parent_hess,
                    parent_count,
                    has_missing,
                    is_bundle,
                    &ctx,
                )
            };

            // Update best if this feature is better
            if split.gain > best.gain {
                best = split;
            }
        }

        best
    }

    /// Find the best split using parallel search over features.
    ///
    /// Uses rayon to evaluate features concurrently. Suitable for datasets
    /// with many features where parallel overhead is amortized.
    #[allow(clippy::too_many_arguments)]
    pub fn find_split_parallel(
        &self,
        histogram: &HistogramSlot<'_>,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        feature_types: &[bool],
        feature_has_missing: &[bool],
        feature_is_bundle: &[bool],
        features: &[u32],
    ) -> SplitInfo {
        // Pre-compute parent score context once for all features
        let ctx = NodeGainContext::new(parent_grad, parent_hess, &self.gain_params);

        // Parallel search: each thread evaluates different features
        // Note: categorical sorted splits allocate per-thread scratch space
        features
            .par_iter()
            .map(|&feature| {
                let feature = feature as usize;
                let is_categorical = feature_types.get(feature).copied().unwrap_or(false);
                let has_missing = feature_has_missing.get(feature).copied().unwrap_or(false);
                let is_bundle = feature_is_bundle.get(feature).copied().unwrap_or(false);
                let bins = histogram.feature_bins(feature);

                if is_categorical {
                    self.find_categorical_split_parallel(
                        bins,
                        feature as u32,
                        parent_grad,
                        parent_hess,
                        parent_count,
                        &ctx,
                    )
                } else {
                    self.find_numerical_split(
                        bins,
                        feature as u32,
                        parent_grad,
                        parent_hess,
                        parent_count,
                        has_missing,
                        is_bundle,
                        &ctx,
                    )
                }
            })
            .reduce(
                SplitInfo::invalid,
                |a, b| if a.gain > b.gain { a } else { b },
            )
    }

    // =========================================================================
    // Numerical Split Finding
    // =========================================================================

    /// Find the best numerical split for a feature.
    ///
    /// Performs bidirectional scanning:
    /// 1. **Forward scan**: Accumulate left stats, missing goes right
    /// 2. **Backward scan**: Accumulate right stats, missing goes left (only if `has_missing`)
    /// 3. Return best split with optimal missing value handling
    ///
    /// For bundle columns (`is_bundle = true`), bin 0 is reserved for "all features at default"
    /// and cannot be used as a split point since it doesn't map to a representable threshold.
    #[allow(clippy::too_many_arguments)]
    fn find_numerical_split(
        &self,
        bins: &[HistogramBin],
        feature: u32,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        has_missing: bool,
        is_bundle: bool,
        ctx: &NodeGainContext,
    ) -> SplitInfo {
        let n_bins = bins.len();
        if n_bins < 2 {
            return SplitInfo::invalid();
        }

        let mut best = SplitInfo::invalid();

        // For bundles, skip bin 0 (reserved for "all defaults").
        // This bin cannot be represented as a single-feature threshold split.
        let start_bin = if is_bundle { 1 } else { 0 };

        // Forward scan: missing values go right.
        //
        // Convention:
        // - Bundles: bin 0 is the default bin ("all defaults"), treated as missing/default.
        // - Standalone numeric with MissingType::NaN: NaN values are encoded as the LAST bin.
        //
        // The grower currently only sets `has_missing` for those two cases, so we can
        // derive the missing bin position from `(has_missing, is_bundle)`.
        let missing_bin = if has_missing {
            Some(if is_bundle { 0 } else { n_bins - 1 })
        } else {
            None
        };

        let mut left_grad = 0.0;
        let mut left_hess = 0.0;
        let mut left_count = 0u32;

        // Pre-accumulate bins before start_bin (they all go left but can't be split points)
        for &(bin_grad, bin_hess) in bins.iter().take(start_bin) {
            left_grad += bin_grad;
            left_hess += bin_hess;
            left_count += 1;
        }

        for (bin, &(bin_grad, bin_hess)) in bins
            .iter()
            .enumerate()
            .skip(start_bin)
            .take(n_bins - 1 - start_bin)
        {
            // Accumulate current bin into left child
            left_grad += bin_grad;
            left_hess += bin_hess;
            left_count += 1;

            // Right child gets the remainder
            let right_grad = parent_grad - left_grad;
            let right_hess = parent_hess - left_hess;
            let right_count = parent_count.saturating_sub(left_count);

            if self
                .gain_params
                .is_valid_split(left_hess, right_hess, left_count, right_count)
            {
                let gain = ctx.compute_gain(left_grad, left_hess, right_grad, right_hess);
                if gain > best.gain {
                    // For bundles, bin 0 was pre-accumulated into LEFT, so we must set
                    // default_left=true to match. The partitioner uses default_left to
                    // determine where bin 0 (the "all defaults" bin) goes.
                    // For non-bundles, bin 0 is a regular bin, and missing values should
                    // go right (default_left=false) per the forward scan semantics.
                    let default_left = is_bundle;
                    best = SplitInfo::numerical(feature, bin as u16, gain, default_left);
                }
            }
        }

        // Backward scan: missing values go left (only if has_missing)
        if has_missing {
            let mut right_grad = 0.0;
            let mut right_hess = 0.0;
            let mut right_count = 0u32;

            // For bundles, don't allow splitting at bin 0 (so stop backward scan at bin 1)
            let min_split_bin = if is_bundle { 1 } else { 0 };

            // Exclude the missing bin from the RIGHT accumulation so that it stays in LEFT.
            let max_bin_exclusive = match missing_bin {
                Some(mb) if mb + 1 == n_bins => n_bins - 1, // last bin is missing
                _ => n_bins,
            };

            for bin in (min_split_bin + 1..max_bin_exclusive).rev() {
                // Accumulate current bin into right child
                let (bin_grad, bin_hess) = bins[bin];
                right_grad += bin_grad;
                right_hess += bin_hess;
                right_count += 1;

                // Left child gets the remainder (including missing)
                let left_grad = parent_grad - right_grad;
                let left_hess = parent_hess - right_hess;
                let left_count = parent_count.saturating_sub(right_count);

                if self
                    .gain_params
                    .is_valid_split(left_hess, right_hess, left_count, right_count)
                {
                    let gain = ctx.compute_gain(left_grad, left_hess, right_grad, right_hess);
                    if gain > best.gain {
                        // Split at bin-1 since bin goes right
                        best = SplitInfo::numerical(feature, (bin - 1) as u16, gain, true);
                    }
                }
            }
        }

        best
    }

    // =========================================================================
    // Categorical Split Finding
    // =========================================================================

    /// Find the best categorical split for a feature.
    ///
    /// Strategy selection based on cardinality:
    /// - **One-hot** (≤ max_onehot_cats): O(k) simple scan
    /// - **Sorted partition** (> max_onehot_cats): O(k log k) sort + O(k) scan
    fn find_categorical_split(
        &mut self,
        bins: &[HistogramBin],
        feature: u32,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        ctx: &NodeGainContext,
    ) -> SplitInfo {
        let n_cats = bins.len() as u32;
        if n_cats < 2 {
            return SplitInfo::invalid();
        }

        if n_cats <= self.max_onehot_cats {
            self.find_onehot_split(bins, feature, parent_grad, parent_hess, parent_count, ctx)
        } else {
            self.find_sorted_split(bins, feature, parent_grad, parent_hess, parent_count, ctx)
        }
    }

    /// Parallel-safe categorical split (uses local allocation).
    fn find_categorical_split_parallel(
        &self,
        bins: &[HistogramBin],
        feature: u32,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        ctx: &NodeGainContext,
    ) -> SplitInfo {
        let n_cats = bins.len() as u32;
        if n_cats < 2 {
            return SplitInfo::invalid();
        }

        if n_cats <= self.max_onehot_cats {
            self.find_onehot_split(bins, feature, parent_grad, parent_hess, parent_count, ctx)
        } else {
            self.find_sorted_split_with_scratch(
                bins,
                feature,
                parent_grad,
                parent_hess,
                parent_count,
                ctx,
                &mut Vec::with_capacity(bins.len().min(MAX_SCRATCH_CATS)),
            )
        }
    }

    /// One-hot categorical split: try each category as a singleton left partition.
    ///
    /// Simple O(k) scan where k = number of categories.
    fn find_onehot_split(
        &self,
        bins: &[HistogramBin],
        feature: u32,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        ctx: &NodeGainContext,
    ) -> SplitInfo {
        let mut best = SplitInfo::invalid();

        // Try each category as singleton left partition
        for (cat, &(left_grad, left_hess)) in bins.iter().enumerate() {
            // This category alone goes left
            let left_count = 1u32;
            let right_grad = parent_grad - left_grad;
            let right_hess = parent_hess - left_hess;
            let right_count = parent_count.saturating_sub(left_count);

            if self
                .gain_params
                .is_valid_split(left_hess, right_hess, left_count, right_count)
            {
                let gain = ctx.compute_gain(left_grad, left_hess, right_grad, right_hess);
                if gain > best.gain {
                    best = SplitInfo::categorical(
                        feature,
                        CatBitset::singleton(cat as u32),
                        gain,
                        false,
                    );
                }
            }
        }

        best
    }

    /// Sorted categorical split using the splitter's scratch buffer.
    fn find_sorted_split(
        &mut self,
        bins: &[HistogramBin],
        feature: u32,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        ctx: &NodeGainContext,
    ) -> SplitInfo {
        // Step 1: Collect (category, grad/hess ratio) for valid categories
        self.cat_scratch.clear();
        for (cat, &(grad, hess)) in bins.iter().enumerate() {
            if hess > 1e-10 {
                // Negative ratio for ascending sort (most negative gradient first)
                self.cat_scratch.push((cat as u32, -grad / (hess + 1e-6)));
            }
        }

        if self.cat_scratch.len() < 2 {
            return SplitInfo::invalid();
        }

        // Step 2: Sort by gradient ratio (ascending)
        self.cat_scratch
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Step 3: Scan for optimal partition
        let mut best = SplitInfo::invalid();
        let mut left_cats = CatBitset::empty();
        let mut left_grad = 0.0;
        let mut left_hess = 0.0;
        let mut left_count = 0u32;

        // Don't try last category on left (always goes right)
        for i in 0..(self.cat_scratch.len() - 1) {
            let (cat, _) = self.cat_scratch[i];

            // Add this category to left partition
            let cat_idx = cat as usize;
            left_cats.insert(cat);
            let (bin_grad, bin_hess) = bins[cat_idx];
            left_grad += bin_grad;
            left_hess += bin_hess;
            left_count += 1;

            // Right gets remainder
            let right_grad = parent_grad - left_grad;
            let right_hess = parent_hess - left_hess;
            let right_count = parent_count.saturating_sub(left_count);

            if self
                .gain_params
                .is_valid_split(left_hess, right_hess, left_count, right_count)
            {
                let gain = ctx.compute_gain(left_grad, left_hess, right_grad, right_hess);
                if gain > best.gain {
                    best = SplitInfo::categorical(feature, left_cats.clone(), gain, false);
                }
            }
        }

        best
    }

    /// Sorted categorical split: sort by gradient ratio, scan for optimal partition.
    ///
    /// Algorithm (LightGBM-style):
    /// 1. Compute grad/hess ratio for each category with sufficient hessian
    /// 2. Sort categories by ratio (optimal ordering for binary partition)
    /// 3. Scan sorted order, accumulating left partition
    /// 4. Return best partition found
    #[allow(clippy::too_many_arguments)]
    fn find_sorted_split_with_scratch(
        &self,
        bins: &[HistogramBin],
        feature: u32,
        parent_grad: f64,
        parent_hess: f64,
        parent_count: u32,
        ctx: &NodeGainContext,
        scratch: &mut Vec<(u32, f64)>,
    ) -> SplitInfo {
        // Step 1: Collect (category, grad/hess ratio) for valid categories
        scratch.clear();
        for (cat, &(grad, hess)) in bins.iter().enumerate() {
            if hess > 1e-10 {
                // Negative ratio for ascending sort (most negative gradient first)
                scratch.push((cat as u32, -grad / (hess + 1e-6)));
            }
        }

        if scratch.len() < 2 {
            return SplitInfo::invalid();
        }

        // Step 2: Sort by gradient ratio (ascending)
        scratch.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Step 3: Scan for optimal partition
        let mut best = SplitInfo::invalid();
        let mut left_cats = CatBitset::empty();
        let mut left_grad = 0.0;
        let mut left_hess = 0.0;
        let mut left_count = 0u32;

        // Don't try last category on left (always goes right)
        for (cat, _) in scratch.iter().take(scratch.len() - 1) {
            // Add this category to left partition
            let cat_idx = *cat as usize;
            left_cats.insert(*cat);
            let (bin_grad, bin_hess) = bins[cat_idx];
            left_grad += bin_grad;
            left_hess += bin_hess;
            left_count += 1;

            // Right gets remainder
            let right_grad = parent_grad - left_grad;
            let right_hess = parent_hess - left_hess;
            let right_count = parent_count.saturating_sub(left_count);

            if self
                .gain_params
                .is_valid_split(left_hess, right_hess, left_count, right_count)
            {
                let gain = ctx.compute_gain(left_grad, left_hess, right_grad, right_hess);
                if gain > best.gain {
                    best = SplitInfo::categorical(feature, left_cats.clone(), gain, false);
                }
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::SplitType;
    use super::*;
    use crate::training::gbdt::histograms::{HistogramLayout, HistogramPool};

    fn make_pool(bin_counts: &[u32]) -> HistogramPool {
        let mut offset = 0;
        let features: Vec<HistogramLayout> = bin_counts
            .iter()
            .map(|&n_bins| {
                let meta = HistogramLayout { offset, n_bins };
                offset += n_bins;
                meta
            })
            .collect();
        HistogramPool::new(features, 4, 4)
    }

    #[test]
    fn test_numerical_split_simple() {
        let splitter = GreedySplitter::default();

        // 4 bins, gradients suggest split at bin 1
        let bins: [(f64, f64); 4] = [(10.0, 5.0), (5.0, 5.0), (-5.0, 5.0), (-10.0, 5.0)];

        let parent_grad = 0.0;
        let parent_hess = 20.0;
        let ctx = NodeGainContext::new(parent_grad, parent_hess, &splitter.gain_params);
        let split = splitter.find_numerical_split(
            &bins,
            0,
            parent_grad,
            parent_hess,
            20,
            false, // no missing values in test
            false, // not a bundle
            &ctx,
        );

        assert!(split.is_valid());
        assert_eq!(split.feature, 0);
    }

    #[test]
    fn test_categorical_onehot() {
        let mut splitter = GreedySplitter::with_config(
            GainParams::default(),
            4, // max_onehot_cats
            Parallelism::Sequential,
        );

        // 3 categories, middle one has distinct gradient
        let bins: [(f64, f64); 3] = [(5.0, 5.0), (-10.0, 5.0), (5.0, 5.0)];

        let parent_grad = 0.0;
        let parent_hess = 15.0;
        let ctx = NodeGainContext::new(parent_grad, parent_hess, &splitter.gain_params);
        let split = splitter.find_categorical_split(&bins, 0, parent_grad, parent_hess, 15, &ctx);

        assert!(split.is_valid());
        if let SplitType::Categorical { ref left_cats } = split.split_type {
            assert!(left_cats.contains(1));
        } else {
            panic!("Expected categorical split");
        }
    }

    #[test]
    fn test_categorical_sorted() {
        let mut splitter = GreedySplitter::with_config(
            GainParams::default(),
            2, // Force sorted strategy (< 5 categories)
            Parallelism::Sequential,
        );

        // 5 categories with gradient pattern
        let bins: [(f64, f64); 5] = [(5.0, 3.0), (3.0, 3.0), (0.0, 3.0), (-3.0, 3.0), (-5.0, 3.0)];

        let parent_grad = 0.0;
        let parent_hess = 15.0;
        let ctx = NodeGainContext::new(parent_grad, parent_hess, &splitter.gain_params);
        let split = splitter.find_categorical_split(&bins, 0, parent_grad, parent_hess, 15, &ctx);

        assert!(split.is_valid());
        assert!(matches!(split.split_type, SplitType::Categorical { .. }));
    }

    #[test]
    fn test_find_split_multiple_features() {
        let mut pool = make_pool(&[3, 3, 3]);
        let slot = pool.acquire(0).slot();

        {
            let view = pool.slot_mut(slot);
            // Feature 0: weak split
            view.bins[0] = (1.0, 5.0);
            view.bins[1] = (0.0, 5.0);
            view.bins[2] = (-1.0, 5.0);
            // Feature 1: strong split (should be chosen)
            view.bins[3] = (10.0, 5.0);
            view.bins[4] = (0.0, 5.0);
            view.bins[5] = (-10.0, 5.0);
            // Feature 2: medium split
            view.bins[6] = (3.0, 5.0);
            view.bins[7] = (0.0, 5.0);
            view.bins[8] = (-3.0, 5.0);
        }

        let histogram = pool.slot(slot);
        let feature_types = [false, false, false];
        let feature_has_missing = [false, false, false];
        let feature_is_bundle = [false, false, false];
        let features: Vec<u32> = (0..3).collect();
        let mut splitter = GreedySplitter::default();

        let split = splitter.find_split(
            &histogram,
            0.0,
            45.0,
            45,
            &feature_types,
            &feature_has_missing,
            &feature_is_bundle,
            &features,
        );
        assert!(split.is_valid());
        assert_eq!(split.feature, 1);
    }

    #[test]
    fn test_invalid_single_bin() {
        let splitter = GreedySplitter::default();
        let bins: [(f64, f64); 1] = [(5.0, 5.0)];
        let parent_grad = 5.0;
        let parent_hess = 5.0;
        let ctx = NodeGainContext::new(parent_grad, parent_hess, &splitter.gain_params);
        let split = splitter.find_numerical_split(
            &bins,
            0,
            parent_grad,
            parent_hess,
            5,
            false,
            false,
            &ctx,
        );
        assert!(!split.is_valid());
    }

    #[test]
    fn test_split_with_feature_filtering() {
        // Test that column sampling correctly excludes features
        let mut pool = make_pool(&[3, 3, 3]);
        let slot = pool.acquire(0).slot();

        {
            let view = pool.slot_mut(slot);
            // Feature 0: weak split
            view.bins[0] = (1.0, 5.0);
            view.bins[1] = (0.0, 5.0);
            view.bins[2] = (-1.0, 5.0);
            // Feature 1: strong split (would be chosen without filtering)
            view.bins[3] = (10.0, 5.0);
            view.bins[4] = (0.0, 5.0);
            view.bins[5] = (-10.0, 5.0);
            // Feature 2: medium split
            view.bins[6] = (3.0, 5.0);
            view.bins[7] = (0.0, 5.0);
            view.bins[8] = (-3.0, 5.0);
        }

        let histogram = pool.slot(slot);
        let feature_types = [false, false, false];
        let feature_has_missing = [false, false, false];
        let feature_is_bundle = [false, false, false];
        let mut splitter = GreedySplitter::default();

        // All features: should pick feature 1 (strongest)
        let all_features = [0u32, 1, 2];
        let split_all = splitter.find_split(
            &histogram,
            0.0,
            45.0,
            45,
            &feature_types,
            &feature_has_missing,
            &feature_is_bundle,
            &all_features,
        );
        assert_eq!(split_all.feature, 1);

        // Exclude feature 1, should pick feature 2 (medium)
        let filtered = [0u32, 2];
        let split_filtered = splitter.find_split(
            &histogram,
            0.0,
            45.0,
            45,
            &feature_types,
            &feature_has_missing,
            &feature_is_bundle,
            &filtered,
        );
        assert!(split_filtered.is_valid());
        assert_eq!(split_filtered.feature, 2);

        // Single feature: should only pick that feature
        let single = [0u32];
        let split_single = splitter.find_split(
            &histogram,
            0.0,
            45.0,
            45,
            &feature_types,
            &feature_has_missing,
            &feature_is_bundle,
            &single,
        );
        assert!(split_single.is_valid());
        assert_eq!(split_single.feature, 0);
    }
}
