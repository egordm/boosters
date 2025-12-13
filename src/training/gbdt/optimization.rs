//! Optimization profiles for automatic strategy selection.
//!
//! This module provides automatic selection of split finding strategies based on
//! data characteristics. Histogram building has its own auto-selection in the
//! histograms module.
//!
//! The main entry point is [`OptimizationProfile`], which resolves to a [`SplitStrategy`]
//! based on data shape (rows, features).

use super::split::SplitStrategy;

/// Minimum features to enable parallel split finding.
const MIN_FEATURES_PARALLEL: usize = 16;

// =============================================================================
// Optimization Profile
// =============================================================================

/// High-level optimization profile for split finding strategy.
///
/// Users select a profile based on their data characteristics or use `Auto`
/// for automatic detection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptimizationProfile {
    /// Auto-detect based on data characteristics.
    #[default]
    Auto,
    /// Optimized for small datasets (<100k rows). Sequential split finding.
    SmallData,
    /// Optimized for medium/large datasets (>=100k rows). Parallel split finding.
    LargeData,
}

impl OptimizationProfile {
    /// Resolve profile to split finding strategy.
    ///
    /// # Arguments
    /// * `n_rows` - Number of samples
    /// * `n_features` - Number of features
    pub fn resolve(&self, n_rows: usize, n_features: usize) -> SplitStrategy {
        match self {
            Self::Auto => Self::auto_detect(n_rows, n_features),
            Self::SmallData => SplitStrategy::Sequential,
            Self::LargeData => {
                if n_features >= MIN_FEATURES_PARALLEL {
                    SplitStrategy::Parallel
                } else {
                    SplitStrategy::Sequential
                }
            }
        }
    }

    /// Auto-detect optimal split strategy from data characteristics.
    fn auto_detect(n_rows: usize, n_features: usize) -> SplitStrategy {
        // Use parallel split finding for larger datasets with enough features
        if n_rows >= 100_000 && n_features >= MIN_FEATURES_PARALLEL {
            SplitStrategy::Parallel
        } else {
            SplitStrategy::Sequential
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_small() {
        let strategy = OptimizationProfile::Auto.resolve(10_000, 50);
        assert_eq!(strategy, SplitStrategy::Sequential);
    }

    #[test]
    fn test_auto_detect_medium() {
        let strategy = OptimizationProfile::Auto.resolve(500_000, 100);
        assert_eq!(strategy, SplitStrategy::Parallel);
    }

    #[test]
    fn test_auto_detect_few_features() {
        // Even large dataset should use sequential if few features
        let strategy = OptimizationProfile::Auto.resolve(1_000_000, 5);
        assert_eq!(strategy, SplitStrategy::Sequential);
    }

    #[test]
    fn test_large_data_profile() {
        let strategy = OptimizationProfile::LargeData.resolve(10_000, 100);
        assert_eq!(strategy, SplitStrategy::Parallel);
    }

    #[test]
    fn test_small_data_profile() {
        let strategy = OptimizationProfile::SmallData.resolve(1_000_000, 100);
        assert_eq!(strategy, SplitStrategy::Sequential);
    }
}

