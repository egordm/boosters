//! Bin boundaries, index types, and categorical feature information.

// ============================================================================
// BinIndex trait
// ============================================================================

/// Trait for bin index types.
///
/// Bin indices can be u8 (256 bins), u16 (65536 bins), or u32 (4B bins).
/// Most use cases need only u8, but the generic parameter allows flexibility.
pub trait BinIndex: Copy + Send + Sync + Default + 'static {
    /// Maximum number of bins this type can represent.
    const MAX_BINS: usize;

    /// Convert from usize, saturating at MAX_BINS - 1.
    fn from_usize(v: usize) -> Self;

    /// Convert to usize.
    fn to_usize(self) -> usize;
}

impl BinIndex for u8 {
    const MAX_BINS: usize = 256;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v.min(255) as u8
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl BinIndex for u16 {
    const MAX_BINS: usize = 65536;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v.min(65535) as u16
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl BinIndex for u32 {
    const MAX_BINS: usize = u32::MAX as usize;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v.min(u32::MAX as usize) as u32
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

// ============================================================================
// CategoricalInfo
// ============================================================================

/// Information about which features are categorical and their cardinality.
#[derive(Debug, Clone, Default)]
pub struct CategoricalInfo {
    /// For each feature: Some(num_categories) if categorical, None if numerical.
    pub feature_types: Vec<Option<u32>>,
}

impl CategoricalInfo {
    /// Create with no categorical features.
    pub fn all_numerical(num_features: usize) -> Self {
        Self {
            feature_types: vec![None; num_features],
        }
    }

    /// Create from a list specifying categorical features.
    ///
    /// # Arguments
    /// * `num_features` - Total number of features
    /// * `categorical` - List of (feature_index, num_categories) pairs
    pub fn with_categorical(num_features: usize, categorical: &[(usize, u32)]) -> Self {
        let mut feature_types = vec![None; num_features];
        for &(feat, num_cats) in categorical {
            if feat < num_features {
                feature_types[feat] = Some(num_cats);
            }
        }
        Self { feature_types }
    }

    /// Check if a feature is categorical.
    pub fn is_categorical(&self, feature: usize) -> bool {
        self.feature_types
            .get(feature)
            .map(|t| t.is_some())
            .unwrap_or(false)
    }

    /// Get the number of categories for a feature, or None if numerical.
    pub fn num_categories(&self, feature: usize) -> Option<u32> {
        self.feature_types.get(feature).copied().flatten()
    }
}

// ============================================================================
// BinCuts
// ============================================================================

/// Bin boundaries for all features.
///
/// Stores cut points (thresholds) for each feature in a CSR-like format:
/// - `cut_values`: All cut values concatenated
/// - `cut_ptrs`: Offsets into `cut_values` for each feature
///
/// A value `v` for feature `f` maps to bin `b` where `cuts[b] <= v < cuts[b+1]`.
/// Bin 0 is reserved for missing values (NaN).
///
/// # Categorical Features
///
/// For categorical features, bins represent category indices directly:
/// - Bin 0: missing values (NaN)
/// - Bin 1: category 0
/// - Bin 2: category 1
/// - ...
///
/// Categorical features have no cut values (cuts are empty), and `is_categorical[f]`
/// is set to true. The number of bins equals the cardinality + 1 (for missing).
///
/// # Memory Layout
///
/// ```text
/// cut_ptrs:    [0, 5, 8, 12]  (offsets)
/// cut_values:  [0.1, 0.3, 0.5, 0.7, 0.9,   ← Feature 0: 5 cuts (6 bins)
///               1.0, 2.0, 3.0,              ← Feature 1: 3 cuts (4 bins)
///               0.0, 0.25, 0.5, 0.75]       ← Feature 2: 4 cuts (5 bins)
/// ```
#[derive(Debug, Clone)]
pub struct BinCuts {
    /// All cut values concatenated, sorted per feature.
    /// These are the upper bounds of each bin (exclusive).
    /// For categorical features, this is empty (no cuts needed).
    cut_values: Box<[f32]>,

    /// Offsets into cut_values: cut_ptrs[f] is start of feature f's cuts.
    /// Length: num_features + 1
    cut_ptrs: Box<[u32]>,

    /// Number of features.
    num_features: u32,

    /// Per-feature categorical flag.
    /// If `is_categorical[f]` is true, feature f is treated as categorical.
    is_categorical: Box<[bool]>,

    /// Per-feature number of categories (only meaningful for categorical features).
    /// For numerical features, this is 0.
    num_categories: Box<[u32]>,
}

impl BinCuts {
    /// Create new bin cuts from pre-computed values.
    ///
    /// # Arguments
    ///
    /// * `cut_values` - All cut values concatenated
    /// * `cut_ptrs` - Offsets into cut_values for each feature (length: num_features + 1)
    ///
    /// # Panics
    ///
    /// Panics if `cut_ptrs` is empty or last element doesn't match `cut_values.len()`.
    pub fn new(cut_values: Vec<f32>, cut_ptrs: Vec<u32>) -> Self {
        assert!(!cut_ptrs.is_empty(), "cut_ptrs must not be empty");
        assert_eq!(
            *cut_ptrs.last().unwrap() as usize,
            cut_values.len(),
            "Last cut_ptr must equal cut_values.len()"
        );

        let num_features = (cut_ptrs.len() - 1) as u32;

        Self {
            cut_values: cut_values.into_boxed_slice(),
            cut_ptrs: cut_ptrs.into_boxed_slice(),
            num_features,
            is_categorical: vec![false; num_features as usize].into_boxed_slice(),
            num_categories: vec![0; num_features as usize].into_boxed_slice(),
        }
    }

    /// Create bin cuts with categorical feature support.
    ///
    /// # Arguments
    ///
    /// * `cut_values` - All cut values concatenated (for numerical features)
    /// * `cut_ptrs` - Offsets into cut_values for each feature
    /// * `is_categorical` - Per-feature categorical flag
    /// * `num_categories` - Per-feature category count (0 for numerical)
    pub fn with_categorical(
        cut_values: Vec<f32>,
        cut_ptrs: Vec<u32>,
        is_categorical: Vec<bool>,
        num_categories: Vec<u32>,
    ) -> Self {
        assert!(!cut_ptrs.is_empty(), "cut_ptrs must not be empty");
        assert_eq!(
            *cut_ptrs.last().unwrap() as usize,
            cut_values.len(),
            "Last cut_ptr must equal cut_values.len()"
        );

        let num_features = (cut_ptrs.len() - 1) as u32;
        assert_eq!(
            is_categorical.len(),
            num_features as usize,
            "is_categorical length must match num_features"
        );
        assert_eq!(
            num_categories.len(),
            num_features as usize,
            "num_categories length must match num_features"
        );

        Self {
            cut_values: cut_values.into_boxed_slice(),
            cut_ptrs: cut_ptrs.into_boxed_slice(),
            num_features,
            is_categorical: is_categorical.into_boxed_slice(),
            num_categories: num_categories.into_boxed_slice(),
        }
    }

    /// Number of features.
    #[inline]
    pub fn num_features(&self) -> u32 {
        self.num_features
    }

    /// Check if a feature is categorical.
    #[inline]
    pub fn is_categorical(&self, feature: u32) -> bool {
        self.is_categorical[feature as usize]
    }

    /// Get the number of categories for a categorical feature.
    ///
    /// Returns 0 for numerical features.
    #[inline]
    pub fn num_categories(&self, feature: u32) -> u32 {
        self.num_categories[feature as usize]
    }

    /// Get bin boundaries for a specific feature.
    ///
    /// Returns a slice of cut values (bin upper bounds).
    /// For categorical features, this returns an empty slice.
    #[inline]
    pub fn feature_cuts(&self, feature: u32) -> &[f32] {
        let start = self.cut_ptrs[feature as usize] as usize;
        let end = self.cut_ptrs[feature as usize + 1] as usize;
        &self.cut_values[start..end]
    }

    /// Number of bins for a feature.
    ///
    /// For numerical features: number of cuts + 1 (for bin 0 which handles missing/below-min).
    /// For categorical features: num_categories + 1 (for bin 0 which handles missing).
    ///
    /// For numerical features with N cuts, there are N+2 bins:
    /// - Bin 0: missing values (NaN)
    /// - Bins 1..=N: regions (-∞, cut[0]], (cut[0], cut[1]], ..., (cut[N-2], cut[N-1]]
    /// - Bin N+1: region (cut[N-1], +∞)
    #[inline]
    pub fn num_bins(&self, feature: u32) -> usize {
        if self.is_categorical[feature as usize] {
            // Categorical: bins are 0 (missing) + categories
            self.num_categories[feature as usize] as usize + 1
        } else {
            // Numerical: bins are 0 (missing) + (N+1 regions for N cuts)
            let start = self.cut_ptrs[feature as usize];
            let end = self.cut_ptrs[feature as usize + 1];
            let num_cuts = (end - start) as usize;
            // N cuts create N+1 regions, plus bin 0 for missing = N+2 bins total
            num_cuts + 2
        }
    }

    /// Total bins across all features.
    ///
    /// Useful for pre-allocating histogram storage.
    pub fn total_bins(&self) -> usize {
        (0..self.num_features).map(|f| self.num_bins(f)).sum()
    }

    /// Map a single value to its bin index.
    ///
    /// For numerical features:
    /// - NaN values map to bin 0
    /// - Values below min cut map to bin 1
    /// - Values >= max cut map to max bin
    ///
    /// For categorical features:
    /// - NaN values map to bin 0
    /// - Category i maps to bin i + 1 (0-indexed categories)
    ///
    /// Uses binary search for numerical: O(log num_bins).
    /// Direct mapping for categorical: O(1).
    #[inline]
    pub fn bin_value(&self, feature: u32, value: f32) -> usize {
        // Missing values go to bin 0
        if value.is_nan() {
            return 0;
        }

        if self.is_categorical[feature as usize] {
            // Categorical: direct mapping, category i -> bin i+1
            // Value should be a non-negative integer
            let cat = value.round() as usize;
            let max_cat = self.num_categories[feature as usize] as usize;
            if cat >= max_cat {
                // Unknown category treated as missing (bin 0)
                0
            } else {
                cat + 1
            }
        } else {
            // Numerical: binary search for correct bin
            let cuts = self.feature_cuts(feature);
            if cuts.is_empty() {
                return 1; // Single bin for all non-missing values
            }

            // Binary search for the bin
            // We want the first cut that is > value, then bin = that index + 1
            // (because bin 0 is reserved for missing)
            //
            // Bin layout for cuts [c0, c1, c2]:
            // - bin 0: missing (NaN)
            // - bin 1: value <= c0
            // - bin 2: c0 < value <= c1
            // - bin 3: c1 < value <= c2
            // - bin 4: value > c2
            match cuts.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
                Ok(idx) => idx + 1,  // Exact match: value == cuts[idx], goes to bin idx+1
                Err(idx) => idx + 1, // Insert position: cuts[idx-1] < value < cuts[idx]
            }
        }
    }

    /// Map a value to bin index, returning a BinIndex type.
    #[inline]
    pub fn bin_value_as<B: BinIndex>(&self, feature: u32, value: f32) -> B {
        B::from_usize(self.bin_value(feature, value))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create BinCuts for testing
    fn make_cuts(features: &[&[f32]]) -> BinCuts {
        let mut cut_values = Vec::new();
        let mut cut_ptrs = vec![0u32];

        for cuts in features {
            cut_values.extend(*cuts);
            cut_ptrs.push(cut_values.len() as u32);
        }

        BinCuts::new(cut_values, cut_ptrs)
    }

    #[test]
    fn test_bin_index_trait() {
        // u8
        assert_eq!(u8::MAX_BINS, 256);
        assert_eq!(u8::from_usize(0), 0u8);
        assert_eq!(u8::from_usize(255), 255u8);
        assert_eq!(u8::from_usize(256), 255u8); // saturates
        assert_eq!(100u8.to_usize(), 100usize);

        // u16
        assert_eq!(u16::MAX_BINS, 65536);
        assert_eq!(u16::from_usize(0), 0u16);
        assert_eq!(u16::from_usize(65535), 65535u16);
        assert_eq!(u16::from_usize(65536), 65535u16); // saturates
    }

    #[test]
    fn test_categorical_info() {
        let info = CategoricalInfo::all_numerical(5);
        assert!(!info.is_categorical(0));
        assert!(!info.is_categorical(4));
        assert_eq!(info.num_categories(0), None);

        let info = CategoricalInfo::with_categorical(5, &[(1, 10), (3, 5)]);
        assert!(!info.is_categorical(0));
        assert!(info.is_categorical(1));
        assert!(!info.is_categorical(2));
        assert!(info.is_categorical(3));
        assert!(!info.is_categorical(4));
        assert_eq!(info.num_categories(1), Some(10));
        assert_eq!(info.num_categories(3), Some(5));
    }

    #[test]
    fn test_bin_cuts_basic() {
        // Feature 0: cuts at [0.5, 1.5, 2.5] -> bins 0 (missing), 1, 2, 3, 4
        // Feature 1: cuts at [10.0] -> bins 0 (missing), 1, 2
        let cuts = make_cuts(&[&[0.5, 1.5, 2.5], &[10.0]]);

        assert_eq!(cuts.num_features(), 2);
        assert_eq!(cuts.feature_cuts(0), &[0.5, 1.5, 2.5]);
        assert_eq!(cuts.feature_cuts(1), &[10.0]);
        assert_eq!(cuts.num_bins(0), 5); // 3 cuts -> 4 regions + 1 missing = 5 bins
        assert_eq!(cuts.num_bins(1), 3); // 1 cut -> 2 regions + 1 missing = 3 bins
    }

    #[test]
    fn test_bin_value_mapping() {
        let cuts = make_cuts(&[&[0.5, 1.5, 2.5]]);

        // NaN -> bin 0
        assert_eq!(cuts.bin_value(0, f32::NAN), 0);

        // Values map to correct bins
        assert_eq!(cuts.bin_value(0, 0.0), 1); // < 0.5 -> bin 1
        assert_eq!(cuts.bin_value(0, 0.5), 1); // == 0.5 -> bin 1
        assert_eq!(cuts.bin_value(0, 0.7), 2); // 0.5 < v < 1.5 -> bin 2
        assert_eq!(cuts.bin_value(0, 1.5), 2); // == 1.5 -> bin 2
        assert_eq!(cuts.bin_value(0, 2.0), 3); // 1.5 < v < 2.5 -> bin 3
        assert_eq!(cuts.bin_value(0, 2.5), 3); // == 2.5 -> bin 3
        assert_eq!(cuts.bin_value(0, 3.0), 4); // > 2.5 -> bin 4
        assert_eq!(cuts.bin_value(0, 100.0), 4); // way above -> bin 4
    }

    #[test]
    fn test_bin_value_edge_cases() {
        // Empty cuts (bin 0 for missing, bin 1 for all values)
        let cuts = make_cuts(&[&[]]);
        assert_eq!(cuts.num_bins(0), 2); // 0 cuts -> 1 region + 1 missing = 2 bins
        assert_eq!(cuts.bin_value(0, f32::NAN), 0);
        assert_eq!(cuts.bin_value(0, 0.0), 1);
        assert_eq!(cuts.bin_value(0, 100.0), 1);

        // Single cut
        let cuts = make_cuts(&[&[5.0]]);
        assert_eq!(cuts.num_bins(0), 3); // 1 cut -> 2 regions + 1 missing = 3 bins
        assert_eq!(cuts.bin_value(0, f32::NAN), 0);
        assert_eq!(cuts.bin_value(0, 4.0), 1);
        assert_eq!(cuts.bin_value(0, 5.0), 1);
        assert_eq!(cuts.bin_value(0, 6.0), 2);
    }
}
