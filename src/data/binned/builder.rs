//! Builder for BinnedDataset.

use super::dataset::BinnedDataset;
use super::group::{FeatureGroup, FeatureMeta};
use super::storage::{BinStorage, BinType, GroupLayout};
use super::BinMapper;

/// Strategy for grouping features.
#[derive(Clone, Debug, Default)]
pub enum GroupStrategy {
    /// All features in one group with specified layout.
    SingleGroup {
        layout: GroupLayout,
    },

    /// Automatic grouping based on feature characteristics.
    /// - Dense numeric (<=256 bins): row-major u8 group
    /// - Wide numeric (>256 bins): column-major u16 group  
    /// - Categorical: column-major u8 group
    /// - Sparse (>90% zeros): column-major u8 group
    #[default]
    Auto,

    /// Custom grouping specified by user.
    /// Each inner Vec contains feature indices for one group.
    Custom(Vec<GroupSpec>),
}

/// Specification for a custom feature group.
#[derive(Clone, Debug)]
pub struct GroupSpec {
    /// Feature indices in this group.
    pub features: Vec<usize>,
    /// Storage layout.
    pub layout: GroupLayout,
}

impl GroupSpec {
    /// Create a new group specification.
    pub fn new(features: Vec<usize>, layout: GroupLayout) -> Self {
        Self { features, layout }
    }
}

/// Temporary feature data during building.
#[derive(Clone, Debug)]
pub(crate) struct FeatureData {
    /// Bin values (column-major: one per row).
    pub bins: Vec<u32>,
    /// Bin mapper.
    pub mapper: BinMapper,
    /// Optional name.
    pub name: Option<String>,
}

impl FeatureData {
    /// Get sparsity rate (fraction of zeros).
    pub fn sparsity(&self) -> f64 {
        if self.bins.is_empty() {
            return 0.0;
        }
        let zeros = self.bins.iter().filter(|&&b| b == 0).count();
        zeros as f64 / self.bins.len() as f64
    }

    /// Max bin value.
    #[allow(dead_code)] // Will be used for auto bin type selection
    pub fn max_bin(&self) -> u32 {
        self.bins.iter().copied().max().unwrap_or(0)
    }

    /// Is this a categorical feature?
    pub fn is_categorical(&self) -> bool {
        self.mapper.is_categorical()
    }

    /// Number of bins.
    pub fn n_bins(&self) -> u32 {
        self.mapper.n_bins()
    }
}

/// Builder for `BinnedDataset`.
///
/// # Example
///
/// Building from pre-binned data:
/// ```ignore
/// let dataset = BinnedDatasetBuilder::new()
///     .add_binned(bins, mapper)
///     .group_strategy(GroupStrategy::Auto)
///     .build()?;
/// ```
///
/// Building from a matrix with automatic binning:
/// ```ignore
/// let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build()?;
/// ```
#[derive(Debug, Default)]
pub struct BinnedDatasetBuilder {
    features: Vec<FeatureData>,
    n_rows: Option<usize>,
    group_strategy: GroupStrategy,
}

impl BinnedDatasetBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder from a column-major matrix with automatic binning.
    ///
    /// This is a convenience method that automatically bins all features
    /// using equal-width binning.
    ///
    /// # Arguments
    /// * `data` - Column-major matrix (each column is a feature)
    /// * `max_bins` - Maximum number of bins per feature (typically 256)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let col_matrix = ColMatrix::from_vec(features, n_samples, n_features);
    /// let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build()?;
    /// ```
    pub fn from_matrix<S: AsRef<[f32]>>(
        data: &crate::data::DenseMatrix<f32, crate::data::ColMajor, S>,
        max_bins: u32,
    ) -> Self {
        use super::MissingType;

        let n_rows = data.num_rows();
        let n_cols = data.num_cols();

        let mut builder = Self::new();

        for col_idx in 0..n_cols {
            // Collect non-NaN values for this column
            let mut values: Vec<f32> = Vec::with_capacity(n_rows);
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;

            for row_idx in 0..n_rows {
                let val = data.get(row_idx, col_idx).copied().unwrap_or(f32::NAN);
                if val.is_finite() {
                    values.push(val);
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }

            // Handle degenerate cases
            if values.is_empty() || min_val >= max_val {
                // All missing or constant - use single bin
                let bins: Vec<u32> = vec![0; n_rows];
                let mapper = BinMapper::numerical(
                    vec![f64::MAX],
                    MissingType::None,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                );
                builder = builder.add_binned(bins, mapper);
                continue;
            }

            // Compute equal-width bin boundaries
            let n_bins = max_bins.min(values.len() as u32);
            let width = (max_val - min_val) / n_bins as f32;

            let bounds: Vec<f64> = (1..=n_bins)
                .map(|i| {
                    if i == n_bins {
                        f64::MAX // Last bin catches everything
                    } else {
                        (min_val + width * i as f32) as f64
                    }
                })
                .collect();

            // Bin each value
            let bins: Vec<u32> = (0..n_rows)
                .map(|row_idx| {
                    let val = data.get(row_idx, col_idx).copied().unwrap_or(f32::NAN);
                    if !val.is_finite() {
                        0 // Map NaN to bin 0
                    } else {
                        // Find bin: linear search is fine for small n_bins
                        let bin = ((val - min_val) / width).floor() as u32;
                        bin.min(n_bins - 1)
                    }
                })
                .collect();

            let mapper = BinMapper::numerical(
                bounds,
                MissingType::None,
                0,
                0,
                0.0,
                min_val as f64,
                max_val as f64,
            );

            builder = builder.add_binned(bins, mapper);
        }

        builder
    }

    /// Add a pre-binned feature with existing mapper.
    ///
    /// # Arguments
    /// * `bins` - Bin indices (length = n_rows)
    /// * `mapper` - Bin mapper for this feature
    pub fn add_binned(mut self, bins: Vec<u32>, mapper: BinMapper) -> Self {
        self.validate_n_rows(bins.len());
        self.features.push(FeatureData {
            bins,
            mapper,
            name: None,
        });
        self
    }

    /// Add a pre-binned feature with name.
    pub fn add_binned_named(
        mut self,
        name: impl Into<String>,
        bins: Vec<u32>,
        mapper: BinMapper,
    ) -> Self {
        self.validate_n_rows(bins.len());
        self.features.push(FeatureData {
            bins,
            mapper,
            name: Some(name.into()),
        });
        self
    }

    /// Set the grouping strategy.
    pub fn group_strategy(mut self, strategy: GroupStrategy) -> Self {
        self.group_strategy = strategy;
        self
    }

    /// Build the dataset.
    pub fn build(self) -> Result<BinnedDataset, BuildError> {
        let n_rows = self.n_rows.unwrap_or(0);
        let n_features = self.features.len();

        if n_features == 0 {
            return Ok(BinnedDataset::empty());
        }

        // Determine grouping
        let group_specs = match &self.group_strategy {
            GroupStrategy::SingleGroup { layout } => {
                vec![GroupSpec::new((0..n_features).collect(), *layout)]
            }
            GroupStrategy::Auto => self.auto_group(),
            GroupStrategy::Custom(specs) => {
                self.validate_custom_groups(specs)?;
                specs.clone()
            }
        };

        // Build groups and feature metadata
        let (groups, features) = self.build_groups(n_rows, &group_specs)?;

        Ok(BinnedDataset::new(n_rows, features, groups))
    }

    /// Validate that a new feature has consistent row count.
    fn validate_n_rows(&mut self, rows: usize) {
        match self.n_rows {
            None => self.n_rows = Some(rows),
            Some(n) => assert_eq!(n, rows, "Feature row count mismatch: expected {}, got {}", n, rows),
        }
    }

    /// Auto-grouping based on feature characteristics.
    fn auto_group(&self) -> Vec<GroupSpec> {
        let mut dense_numeric = Vec::new();
        let mut wide_numeric = Vec::new();
        let mut categorical = Vec::new();
        let mut sparse = Vec::new();

        for (idx, f) in self.features.iter().enumerate() {
            if f.sparsity() > 0.9 {
                sparse.push(idx);
            } else if f.is_categorical() {
                categorical.push(idx);
            } else if f.n_bins() > 256 {
                wide_numeric.push(idx);
            } else {
                dense_numeric.push(idx);
            }
        }

        let mut specs = Vec::new();

        // Dense numeric: column-major for efficient histogram building (contiguous per-feature access)
        // Benchmark shows 13% speedup vs row-major on Apple Silicon (see layout_benchmark.rs)
        if !dense_numeric.is_empty() {
            specs.push(GroupSpec::new(dense_numeric, GroupLayout::ColumnMajor));
        }

        // Wide numeric: column-major (u16)
        if !wide_numeric.is_empty() {
            specs.push(GroupSpec::new(wide_numeric, GroupLayout::ColumnMajor));
        }

        // Categorical: column-major
        if !categorical.is_empty() {
            specs.push(GroupSpec::new(categorical, GroupLayout::ColumnMajor));
        }

        // Sparse: column-major (MFB optimization works better)
        if !sparse.is_empty() {
            specs.push(GroupSpec::new(sparse, GroupLayout::ColumnMajor));
        }

        // If nothing grouped, put everything in one row-major group
        if specs.is_empty() && !self.features.is_empty() {
            specs.push(GroupSpec::new(
                (0..self.features.len()).collect(),
                GroupLayout::RowMajor,
            ));
        }

        specs
    }

    /// Validate custom group specifications.
    fn validate_custom_groups(&self, specs: &[GroupSpec]) -> Result<(), BuildError> {
        let n_features = self.features.len();
        let mut seen = vec![false; n_features];

        for spec in specs {
            for &idx in &spec.features {
                if idx >= n_features {
                    return Err(BuildError::InvalidFeatureIndex(idx, n_features));
                }
                if seen[idx] {
                    return Err(BuildError::DuplicateFeature(idx));
                }
                seen[idx] = true;
            }
        }

        // Check all features are assigned
        for (idx, &assigned) in seen.iter().enumerate() {
            if !assigned {
                return Err(BuildError::UnassignedFeature(idx));
            }
        }

        Ok(())
    }

    /// Build feature groups and metadata from specifications.
    fn build_groups(
        &self,
        n_rows: usize,
        specs: &[GroupSpec],
    ) -> Result<(Vec<FeatureGroup>, Vec<FeatureMeta>), BuildError> {
        let n_features = self.features.len();

        // Pre-allocate feature metadata (will be filled in order)
        let mut feature_metas: Vec<Option<FeatureMeta>> = vec![None; n_features];
        let mut groups = Vec::with_capacity(specs.len());

        for (group_idx, spec) in specs.iter().enumerate() {
            let group_features: Vec<_> = spec.features.iter().map(|&i| &self.features[i]).collect();

            // Determine bin type for this group
            let max_bins = group_features.iter().map(|f| f.n_bins()).max().unwrap_or(256);
            let bin_type = BinType::for_max_bins(max_bins)
                .expect("max_bins should be > 0");

            // Build storage based on layout
            let storage = match spec.layout {
                GroupLayout::RowMajor => {
                    self.build_row_major_storage(n_rows, &spec.features, bin_type)
                }
                GroupLayout::ColumnMajor => {
                    self.build_column_major_storage(n_rows, &spec.features, bin_type)
                }
            };

            // Collect bin counts
            let bin_counts: Vec<u32> = spec.features.iter()
                .map(|&i| self.features[i].n_bins())
                .collect();

            // Create group
            let group = FeatureGroup::new(
                spec.features.iter().map(|&i| i as u32).collect(),
                spec.layout,
                n_rows,
                storage,
                bin_counts,
            );
            groups.push(group);

            // Create feature metadata
            for (idx_in_group, &feature_idx) in spec.features.iter().enumerate() {
                let f = &self.features[feature_idx];
                let mut meta = FeatureMeta::new(
                    f.mapper.clone(),
                    group_idx as u32,
                    idx_in_group as u32,
                );
                if let Some(name) = &f.name {
                    meta = meta.with_name(name.clone());
                }
                feature_metas[feature_idx] = Some(meta);
            }
        }

        // Unwrap all feature metadata (should all be Some now)
        let features: Vec<FeatureMeta> = feature_metas
            .into_iter()
            .enumerate()
            .map(|(i, opt)| opt.unwrap_or_else(|| panic!("Feature {} not assigned to any group", i)))
            .collect();

        Ok((groups, features))
    }

    /// Build row-major storage for a group.
    fn build_row_major_storage(
        &self,
        n_rows: usize,
        feature_indices: &[usize],
        bin_type: BinType,
    ) -> BinStorage {
        let n_features = feature_indices.len();
        let total_size = n_rows * n_features;

        match bin_type {
            BinType::U8 => {
                let mut data = Vec::with_capacity(total_size);
                for row in 0..n_rows {
                    for &fidx in feature_indices {
                        data.push(self.features[fidx].bins[row] as u8);
                    }
                }
                BinStorage::from_u8(data)
            }
            BinType::U16 => {
                let mut data = Vec::with_capacity(total_size);
                for row in 0..n_rows {
                    for &fidx in feature_indices {
                        data.push(self.features[fidx].bins[row] as u16);
                    }
                }
                BinStorage::from_u16(data)
            }
        }
    }

    /// Build column-major storage for a group.
    fn build_column_major_storage(
        &self,
        n_rows: usize,
        feature_indices: &[usize],
        bin_type: BinType,
    ) -> BinStorage {
        let n_features = feature_indices.len();
        let total_size = n_rows * n_features;

        match bin_type {
            BinType::U8 => {
                let mut data = Vec::with_capacity(total_size);
                for &fidx in feature_indices {
                    for row in 0..n_rows {
                        data.push(self.features[fidx].bins[row] as u8);
                    }
                }
                BinStorage::from_u8(data)
            }
            BinType::U16 => {
                let mut data = Vec::with_capacity(total_size);
                for &fidx in feature_indices {
                    for row in 0..n_rows {
                        data.push(self.features[fidx].bins[row] as u16);
                    }
                }
                BinStorage::from_u16(data)
            }
        }
    }
}

/// Errors during dataset building.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// Feature index out of bounds.
    InvalidFeatureIndex(usize, usize),
    /// Feature assigned to multiple groups.
    DuplicateFeature(usize),
    /// Feature not assigned to any group.
    UnassignedFeature(usize),
    /// Row count mismatch.
    RowCountMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFeatureIndex(idx, max) => {
                write!(f, "Feature index {} out of bounds (max {})", idx, max)
            }
            Self::DuplicateFeature(idx) => {
                write!(f, "Feature {} assigned to multiple groups", idx)
            }
            Self::UnassignedFeature(idx) => {
                write!(f, "Feature {} not assigned to any group", idx)
            }
            Self::RowCountMismatch { expected, got } => {
                write!(f, "Row count mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for BuildError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::binned::MissingType;

    fn make_simple_mapper(n_bins: u32) -> BinMapper {
        let bounds: Vec<f64> = (0..n_bins).map(|i| i as f64 + 0.5).collect();
        BinMapper::numerical(bounds, MissingType::None, 0, 0, 0.0, 0.0, (n_bins - 1) as f64)
    }

    #[test]
    fn test_builder_single_group_row_major() {
        let dataset = BinnedDatasetBuilder::new()
            .add_binned(vec![0, 1, 2, 3], make_simple_mapper(4))
            .add_binned(vec![1, 2, 3, 0], make_simple_mapper(4))
            .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::RowMajor })
            .build()
            .unwrap();

        assert_eq!(dataset.n_rows(), 4);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.n_groups(), 1);

        // Verify row-major layout
        let group = dataset.group(0);
        assert!(group.is_row_major());
        
        // Access bins via storage match
        match group.storage() {
            BinStorage::DenseU8(bins) => {
                // Row-major: row 0 has [feat0, feat1], row 1 has [feat0, feat1], etc.
                assert_eq!(bins[0], 0); // row 0, feat 0
                assert_eq!(bins[1], 1); // row 0, feat 1
                assert_eq!(bins[2], 1); // row 1, feat 0
                assert_eq!(bins[3], 2); // row 1, feat 1
            }
            _ => panic!("Expected DenseU8"),
        }
    }

    #[test]
    fn test_builder_single_group_column_major() {
        let dataset = BinnedDatasetBuilder::new()
            .add_binned(vec![0, 1, 2, 3], make_simple_mapper(4))
            .add_binned(vec![10, 11, 12, 13], make_simple_mapper(16))
            .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::ColumnMajor })
            .build()
            .unwrap();

        assert_eq!(dataset.n_rows(), 4);
        assert_eq!(dataset.n_groups(), 1);

        let group = dataset.group(0);
        assert!(group.is_column_major());
        
        // Access bins via storage match
        match group.storage() {
            BinStorage::DenseU8(bins) => {
                // Column-major: all feat0 values, then all feat1 values
                assert_eq!(bins[0], 0);  // feat 0, row 0
                assert_eq!(bins[3], 3);  // feat 0, row 3
                assert_eq!(bins[4], 10); // feat 1, row 0
                assert_eq!(bins[7], 13); // feat 1, row 3
            }
            _ => panic!("Expected DenseU8"),
        }
    }

    #[test]
    fn test_builder_auto_grouping() {
        // Feature 0: dense numeric (<=256 bins) -> column-major (optimized for histogram building)
        // Feature 1: wide numeric (>256 bins) -> column-major
        let dataset = BinnedDatasetBuilder::new()
            .add_binned(vec![0, 1, 2, 3], make_simple_mapper(100))  // dense
            .add_binned(vec![0, 100, 200, 300], make_simple_mapper(500))  // wide
            .group_strategy(GroupStrategy::Auto)
            .build()
            .unwrap();

        assert_eq!(dataset.n_groups(), 2);

        // Dense should be in column-major group (faster for training histograms)
        let group0 = dataset.group(0);
        assert!(group0.is_column_major());

        // Wide should be in column-major group
        let group1 = dataset.group(1);
        assert!(group1.is_column_major());
        assert_eq!(group1.bin_type(), BinType::U16);
    }

    #[test]
    fn test_builder_custom_groups() {
        let dataset = BinnedDatasetBuilder::new()
            .add_binned(vec![0, 1, 2], make_simple_mapper(4))
            .add_binned(vec![1, 2, 3], make_simple_mapper(4))
            .add_binned(vec![2, 3, 0], make_simple_mapper(4))
            .group_strategy(GroupStrategy::Custom(vec![
                GroupSpec::new(vec![0, 2], GroupLayout::RowMajor),
                GroupSpec::new(vec![1], GroupLayout::ColumnMajor),
            ]))
            .build()
            .unwrap();

        assert_eq!(dataset.n_groups(), 2);

        let group0 = dataset.group(0);
        assert_eq!(group0.n_features(), 2);
        assert_eq!(group0.feature_indices(), &[0, 2]);

        let group1 = dataset.group(1);
        assert_eq!(group1.n_features(), 1);
        assert_eq!(group1.feature_indices(), &[1]);
    }

    #[test]
    fn test_builder_empty() {
        let dataset = BinnedDatasetBuilder::new().build().unwrap();
        assert_eq!(dataset.n_rows(), 0);
        assert_eq!(dataset.n_features(), 0);
        assert_eq!(dataset.n_groups(), 0);
    }

    #[test]
    fn test_builder_named_features() {
        let dataset = BinnedDatasetBuilder::new()
            .add_binned_named("feature_x", vec![0, 1, 2], make_simple_mapper(4))
            .add_binned_named("feature_y", vec![1, 2, 3], make_simple_mapper(4))
            .build()
            .unwrap();

        assert_eq!(dataset.feature(0).name, Some("feature_x".to_string()));
        assert_eq!(dataset.feature(1).name, Some("feature_y".to_string()));
    }

    #[test]
    fn test_builder_error_duplicate_feature() {
        let result = BinnedDatasetBuilder::new()
            .add_binned(vec![0, 1, 2], make_simple_mapper(4))
            .add_binned(vec![1, 2, 3], make_simple_mapper(4))
            .group_strategy(GroupStrategy::Custom(vec![
                GroupSpec::new(vec![0, 1], GroupLayout::RowMajor),
                GroupSpec::new(vec![1], GroupLayout::ColumnMajor),  // duplicate!
            ]))
            .build();

        assert!(matches!(result, Err(BuildError::DuplicateFeature(1))));
    }

    #[test]
    fn test_builder_error_unassigned_feature() {
        let result = BinnedDatasetBuilder::new()
            .add_binned(vec![0, 1, 2], make_simple_mapper(4))
            .add_binned(vec![1, 2, 3], make_simple_mapper(4))
            .group_strategy(GroupStrategy::Custom(vec![
                GroupSpec::new(vec![0], GroupLayout::RowMajor),
                // feature 1 not assigned
            ]))
            .build();

        assert!(matches!(result, Err(BuildError::UnassignedFeature(1))));
    }

    #[test]
    fn test_feature_data_sparsity() {
        let f = FeatureData {
            bins: vec![0, 0, 0, 1, 0],
            mapper: make_simple_mapper(4),
            name: None,
        };
        assert!((f.sparsity() - 0.8).abs() < 0.001);
    }
}
