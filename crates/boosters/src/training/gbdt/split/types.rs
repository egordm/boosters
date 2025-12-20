//! Split types and result structures.

use super::super::categorical::CatBitset;

// =============================================================================
// Split Types
// =============================================================================

/// Type of split (numerical threshold or categorical membership).
#[derive(Clone, Debug)]
pub enum SplitType {
    /// Numerical: value with bin index <= threshold goes left.
    Numerical {
        /// Bin index threshold (inclusive for left).
        bin: u16,
    },
    /// Categorical: categories in bitset go left.
    Categorical {
        /// Set of categories that go left.
        left_cats: CatBitset,
    },
}

impl Default for SplitType {
    fn default() -> Self {
        SplitType::Numerical { bin: 0 }
    }
}

/// Result of split finding for a node.
///
/// Contains all information needed to partition data and construct the tree node.
#[derive(Clone, Debug, Default)]
pub struct SplitInfo {
    /// Feature index.
    pub feature: u32,
    /// Split gain (higher is better).
    pub gain: f32,
    /// Direction for missing values (true = left).
    pub default_left: bool,
    /// Type of split.
    pub split_type: SplitType,
}

impl SplitInfo {
    /// Create an invalid split (no valid split found).
    #[inline]
    pub fn invalid() -> Self {
        Self {
            feature: 0,
            gain: f32::NEG_INFINITY,
            default_left: false,
            split_type: SplitType::default(),
        }
    }

    /// Check if this represents a valid split.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.gain > f32::NEG_INFINITY
    }

    /// Create a numerical split.
    #[inline]
    pub fn numerical(feature: u32, bin: u16, gain: f32, default_left: bool) -> Self {
        Self {
            feature,
            gain,
            default_left,
            split_type: SplitType::Numerical { bin },
        }
    }

    /// Create a categorical split.
    #[inline]
    pub fn categorical(feature: u32, left_cats: CatBitset, gain: f32, default_left: bool) -> Self {
        Self {
            feature,
            gain,
            default_left,
            split_type: SplitType::Categorical { left_cats },
        }
    }

    /// Get bin threshold for numerical split.
    #[inline]
    pub fn bin(&self) -> Option<u16> {
        match &self.split_type {
            SplitType::Numerical { bin } => Some(*bin),
            SplitType::Categorical { .. } => None,
        }
    }

    /// Get left categories for categorical split.
    #[inline]
    pub fn left_cats(&self) -> Option<&CatBitset> {
        match &self.split_type {
            SplitType::Numerical { .. } => None,
            SplitType::Categorical { left_cats } => Some(left_cats),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_info_invalid() {
        let split = SplitInfo::invalid();
        assert!(!split.is_valid());
        assert!(split.gain.is_infinite() && split.gain < 0.0);
    }

    #[test]
    fn test_split_info_numerical() {
        let split = SplitInfo::numerical(3, 10, 0.5, true);
        assert!(split.is_valid());
        assert_eq!(split.feature, 3);
        assert_eq!(split.bin(), Some(10));
        assert!(split.default_left);
        assert!(split.left_cats().is_none());
    }

    #[test]
    fn test_split_info_categorical() {
        let cats = CatBitset::singleton(5);
        let split = SplitInfo::categorical(2, cats, 0.3, false);
        assert!(split.is_valid());
        assert_eq!(split.feature, 2);
        assert!(split.bin().is_none());
        assert!(split.left_cats().unwrap().contains(5));
    }
}
