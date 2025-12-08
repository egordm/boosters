//! Per-node gradient histogram collection.

use super::feature::FeatureHistogram;
use crate::training::gbtree::quantize::BinCuts;

/// Histograms for all features at a single tree node.
///
/// Contains one [`FeatureHistogram`] per feature, plus cached totals.
///
/// # Memory Usage
///
/// For 100 features × 256 bins:
/// - 100 × 3 KB = 300 KB per node
///
/// # Example
///
/// ```ignore
/// let mut hist = NodeHistogram::new(&cuts);
/// HistogramBuilder.build(&mut hist, &quantized, &gradients, &rows);
/// ```
#[derive(Debug, Clone)]
pub struct NodeHistogram {
    /// Per-feature histograms
    features: Box<[FeatureHistogram]>,
    /// Cached total gradient (sum across all features, all bins)
    /// Updated after build()
    total_grad: f32,
    /// Cached total hessian
    total_hess: f32,
    /// Cached total count
    total_count: u32,
}

impl NodeHistogram {
    /// Create histograms for all features based on bin cuts.
    ///
    /// Each feature gets a histogram with the appropriate number of bins.
    pub fn new(cuts: &BinCuts) -> Self {
        let features: Box<[FeatureHistogram]> = (0..cuts.num_features())
            .map(|f| {
                let num_bins = cuts.num_bins(f as u32);
                FeatureHistogram::new(num_bins as u16)
            })
            .collect();

        Self {
            features,
            total_grad: 0.0,
            total_hess: 0.0,
            total_count: 0,
        }
    }

    /// Create from explicit bin counts per feature.
    ///
    /// Useful for testing or when cuts aren't available.
    pub fn from_bin_counts(bin_counts: &[u16]) -> Self {
        let features: Box<[FeatureHistogram]> =
            bin_counts.iter().map(|&n| FeatureHistogram::new(n)).collect();

        Self {
            features,
            total_grad: 0.0,
            total_hess: 0.0,
            total_count: 0,
        }
    }

    /// Number of features.
    #[inline]
    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    /// Get histogram for a specific feature.
    #[inline]
    pub fn feature(&self, feat: usize) -> &FeatureHistogram {
        &self.features[feat]
    }

    /// Get mutable histogram for a specific feature.
    #[inline]
    pub fn feature_mut(&mut self, feat: usize) -> &mut FeatureHistogram {
        &mut self.features[feat]
    }

    /// Get all feature histograms as a slice.
    #[inline]
    pub fn features(&self) -> &[FeatureHistogram] {
        &self.features
    }

    /// Get all feature histograms as a mutable slice.
    #[inline]
    pub fn features_mut(&mut self) -> &mut [FeatureHistogram] {
        &mut self.features
    }

    /// Reset all histograms to zero.
    pub fn reset(&mut self) {
        for hist in self.features.iter_mut() {
            hist.reset();
        }
        self.total_grad = 0.0;
        self.total_hess = 0.0;
        self.total_count = 0;
    }

    /// Update cached totals from histogram contents.
    ///
    /// Should be called after building histograms.
    /// Totals are computed from feature 0 (all features should have same totals).
    pub fn update_totals(&mut self) {
        if self.features.is_empty() {
            return;
        }

        let first = &self.features[0];
        self.total_grad = first.total_grad();
        self.total_hess = first.total_hess();
        self.total_count = first.total_count();
    }

    /// Set totals directly.
    pub fn set_totals(&mut self, grad: f32, hess: f32, count: u32) {
        self.total_grad = grad;
        self.total_hess = hess;
        self.total_count = count;
    }

    /// Get cached total gradient.
    #[inline]
    pub fn total_grad(&self) -> f32 {
        self.total_grad
    }

    /// Get cached total hessian.
    #[inline]
    pub fn total_hess(&self) -> f32 {
        self.total_hess
    }

    /// Get cached total count.
    #[inline]
    pub fn total_count(&self) -> u32 {
        self.total_count
    }

    /// Compute sibling histograms via subtraction.
    ///
    /// After calling, `self` contains `parent - self`.
    pub fn subtract_from(&mut self, parent: &Self) {
        debug_assert_eq!(self.features.len(), parent.features.len());

        for (child, par) in self.features.iter_mut().zip(parent.features.iter()) {
            child.subtract_from(par);
        }
        self.total_grad = parent.total_grad - self.total_grad;
        self.total_hess = parent.total_hess - self.total_hess;
        self.total_count = parent.total_count - self.total_count;
    }

    /// Create a new histogram by subtracting another from self.
    ///
    /// Returns `self - other`. Useful for histogram subtraction optimization.
    pub fn subtract(&self, other: &Self) -> Self {
        debug_assert_eq!(self.features.len(), other.features.len());

        let features: Box<[FeatureHistogram]> = self
            .features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| a.subtract(b))
            .collect();

        Self {
            features,
            total_grad: self.total_grad - other.total_grad,
            total_hess: self.total_hess - other.total_hess,
            total_count: self.total_count - other.total_count,
        }
    }

    /// Copy from another histogram.
    pub fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.features.len(), src.features.len());

        for (dst, src) in self.features.iter_mut().zip(src.features.iter()) {
            dst.copy_from(src);
        }
        self.total_grad = src.total_grad;
        self.total_hess = src.total_hess;
        self.total_count = src.total_count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_histogram_creation() {
        // Create cuts with different bin counts per feature
        let cut_values = vec![0.5, 1.5, 2.5, 10.0]; // Feature 0: 3 cuts, Feature 1: 1 cut
        let cut_ptrs = vec![0, 3, 4];
        let cuts = BinCuts::new(cut_values, cut_ptrs);

        let hist = NodeHistogram::new(&cuts);

        assert_eq!(hist.num_features(), 2);
        // 3 cuts creates 4 regions + 1 missing bin = 5 bins
        assert_eq!(hist.feature(0).num_bins(), 5);
        // 1 cut creates 2 regions + 1 missing bin = 3 bins
        assert_eq!(hist.feature(1).num_bins(), 3);
    }

    #[test]
    fn test_node_histogram_subtraction() {
        let mut parent = NodeHistogram::from_bin_counts(&[4, 4]);
        let mut child = NodeHistogram::from_bin_counts(&[4, 4]);

        // Parent: feature 0, bin 0 = (10, 5, 2); feature 1, bin 1 = (8, 4, 2)
        parent.feature_mut(0).add(0, 5.0, 2.5);
        parent.feature_mut(0).add(0, 5.0, 2.5);
        parent.feature_mut(1).add(1, 4.0, 2.0);
        parent.feature_mut(1).add(1, 4.0, 2.0);
        parent.set_totals(18.0, 9.0, 4);

        // Child: feature 0, bin 0 = (5, 2.5, 1); feature 1, bin 1 = (4, 2, 1)
        child.feature_mut(0).add(0, 5.0, 2.5);
        child.feature_mut(1).add(1, 4.0, 2.0);
        child.set_totals(9.0, 4.5, 2);

        // Subtract
        child.subtract_from(&parent);

        // Check sibling
        assert_eq!(child.feature(0).bin_stats(0), (5.0, 2.5, 1));
        assert_eq!(child.feature(1).bin_stats(1), (4.0, 2.0, 1));
        assert_eq!(child.total_grad(), 9.0);
        assert_eq!(child.total_hess(), 4.5);
        assert_eq!(child.total_count(), 2);
    }
}
