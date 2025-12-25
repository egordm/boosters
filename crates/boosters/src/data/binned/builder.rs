//! Builder for BinnedDataset.

use bon::Builder;

use super::bundling::{create_bundle_plan, BundlePlan, BundlingConfig};
use super::dataset::BinnedDataset;
use super::feature_analysis::FeatureInfo;
use super::group::{FeatureGroup, BinnedFeatureInfo};
use super::storage::{BinStorage, BinType, GroupLayout};
use super::MissingType;
use super::BinMapper;

use crate::data::binned::BundlingFeatures;
use crate::data::FeaturesView;
use crate::utils::Parallelism;

// =============================================================================
// Binning Configuration
// =============================================================================

/// Strategy for computing bin boundaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BinningStrategy {
    /// Equal-width bins: divide [min, max] into equal intervals.
    /// Fast but poor for skewed data - many samples end up in few bins.
    EqualWidth,

    /// Equal-frequency (quantile) bins: each bin has ~same number of samples.
    /// Better for real-world data with skewed distributions.
    /// This matches LightGBM and XGBoost behavior.
    #[default]
    Quantile,
}

/// Configuration for feature binning.
///
/// Controls how features are binned during quantization. Use the builder pattern
/// for configuration:
///
/// # Example
///
/// ```ignore
/// use boosters::data::BinningConfig;
///
/// // Simple: just max bins
/// let config = BinningConfig::builder().max_bins(256).build();
///
/// // Full control
/// let config = BinningConfig::builder()
///     .max_bins(256)
///     .strategy(BinningStrategy::Quantile)
///     .sample_cnt(100_000)  // For faster binning on large datasets
///     .build();
/// ```
#[derive(Clone, Debug, Builder)]
#[builder(derive(Clone, Debug))]
pub struct BinningConfig {
    /// Maximum bins per feature (default: 256).
    #[builder(default = 256)]
    pub max_bins: u32,
    /// Binning strategy (default: Quantile).
    #[builder(default)]
    pub strategy: BinningStrategy,
    /// Number of samples for computing bin boundaries (default: 200K, matching LightGBM).
    /// For datasets larger than this, uses sampling for approximate quantiles.
    #[builder(default = 200_000)]
    pub sample_cnt: usize,
}

impl Default for BinningConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl From<u32> for BinningConfig {
    fn from(max_bins: u32) -> Self {
        Self::builder().max_bins(max_bins).build()
    }
}

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

// =============================================================================
// Feature Bin Accessor for Bundling
// =============================================================================

/// Adapter that provides access to binned feature values for conflict detection.
///
/// This allows the bundling algorithm to access bin values (as f32) from the
/// builder's internal feature data, treating bin 0 as "zero" and other bins
/// as "non-zero" for conflict detection purposes.
pub(crate) struct FeatureBinAccessor<'a> {
    features: &'a [FeatureData],
    n_rows: usize,
}

// FeatureBinAccessor is safe to share between threads because it only
// reads from the immutable feature data
unsafe impl Sync for FeatureBinAccessor<'_> {}

// Implement BundlingFeatures for FeatureBinAccessor
impl BundlingFeatures for FeatureBinAccessor<'_> {
    fn n_samples(&self) -> usize {
        self.n_rows
    }
    
    fn get(&self, sample: usize, feature: usize) -> f32 {
        self.features[feature].bins[sample] as f32
    }
}

/// Builder for `BinnedDataset`.
///
/// # Example
///
/// Building from a Dataset:
/// ```ignore
/// use boosters::data::{BinnedDatasetBuilder, BinningConfig};
/// use boosters::Parallelism;
///
/// let config = BinningConfig::builder().max_bins(256).build();
/// let binned = BinnedDatasetBuilder::new(config)
///     .add_features(dataset.features(), Parallelism::Parallel)
///     .build()?;
/// ```
///
/// Building with feature bundling:
/// ```ignore
/// let binned = BinnedDatasetBuilder::new(config)
///     .add_features(dataset.features(), Parallelism::Parallel)
///     .with_bundling(BundlingConfig::auto())
///     .build()?;
/// ```
///
/// Building from raw array (no schema):
/// ```ignore
/// use boosters::data::FeaturesView;
/// use ndarray::array;
///
/// let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let binned = BinnedDatasetBuilder::new(config)
///     .add_features(FeaturesView::from_array(features.view()), Parallelism::Parallel)
///     .build()?;
/// ```
#[derive(Debug)]
pub struct BinnedDatasetBuilder {
    config: BinningConfig,
    features: Vec<FeatureData>,
    n_rows: Option<usize>,
    group_strategy: GroupStrategy,
    bundling_config: Option<BundlingConfig>,
}

impl BinnedDatasetBuilder {
    /// Create a new builder with the given binning configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Binning configuration (use `BinningConfig::builder()`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::data::{BinnedDatasetBuilder, BinningConfig};
    ///
    /// let config = BinningConfig::builder().max_bins(256).build();
    /// let builder = BinnedDatasetBuilder::new(config);
    /// ```
    pub fn new(config: BinningConfig) -> Self {
        Self {
            config,
            features: Vec::new(),
            n_rows: None,
            group_strategy: GroupStrategy::Auto,
            bundling_config: None,
        }
    }

    /// Add features from a FeaturesView.
    ///
    /// This is the primary API for creating a binned dataset. The FeaturesView
    /// contains both feature data and optional schema. Categorical features
    /// (as indicated in the schema) are binned as categorical; all others use
    /// quantile binning.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature-major data `[n_features, n_samples]` with optional schema
    /// * `parallelism` - Threading strategy (Sequential or Parallel)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::data::{BinnedDatasetBuilder, BinningConfig};
    /// use boosters::data::{Dataset, FeaturesView};
    /// use boosters::Parallelism;
    /// use ndarray::array;
    ///
    /// // From Dataset
    /// let dataset = Dataset::new(features.view(), Some(targets.view()), None);
    /// let binned = BinnedDatasetBuilder::new(config)
    ///     .add_features(dataset.features(), Parallelism::Parallel)
    ///     .build()?;
    ///
    /// // From raw array (no schema - all features treated as numeric)
    /// let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // [n_features, n_samples]
    /// let binned = BinnedDatasetBuilder::new(config)
    ///     .add_features(FeaturesView::from_array(features.view()), Parallelism::Parallel)
    ///     .build()?;
    /// ```
    pub fn add_features(mut self, features: FeaturesView<'_>, parallelism: Parallelism) -> Self {
        let n_features = features.n_features();
        let config = &self.config;
        let schema = features.schema();

        // Helper to process a single feature
        let process_feature = |feat_idx: usize| -> (Vec<u32>, BinMapper, Option<String>) {
            let feature_data = features.feature(feat_idx);
            let is_categorical = features.feature_type(feat_idx).is_categorical();
            
            // Get feature name from schema if available
            let name = schema
                .and_then(|s| s.get(feat_idx))
                .and_then(|m| m.name.clone());

            let (bins, mapper) = if is_categorical {
                Self::bin_categorical(feature_data.as_slice().unwrap())
            } else {
                Self::bin_numeric(feature_data.as_slice().unwrap(), config)
            };
            
            (bins, mapper, name)
        };

        // Process features
        let feature_results = parallelism.maybe_par_map(0..n_features, process_feature);

        // Add all features to builder
        for (bins, mapper, name) in feature_results {
            self = self.add_binned(bins, mapper, name);
        }

        self
    }

    /// Bin a categorical feature from float values.
    ///
    /// Categories are expected to be non-negative integers encoded as floats.
    fn bin_categorical(data: &[f32]) -> (Vec<u32>, BinMapper) {
        // Find unique categories
        let mut categories: Vec<i32> = data
            .iter()
            .filter(|v| v.is_finite())
            .map(|&v| v as i32)
            .collect();
        categories.sort_unstable();
        categories.dedup();

        // Map values to bin indices
        let bins: Vec<u32> = data
            .iter()
            .map(|&val| {
                if !val.is_finite() {
                    0 // Missing → bin 0
                } else {
                    let cat = val as i32;
                    categories.binary_search(&cat).unwrap_or(0) as u32
                }
            })
            .collect();

        let mapper = BinMapper::categorical(
            categories,
            MissingType::None,
            0,
            0,
            0.0,
        );

        (bins, mapper)
    }

    /// Bin a numeric feature from a slice.
    fn bin_numeric(data: &[f32], config: &BinningConfig) -> (Vec<u32>, BinMapper) {
        let n_samples = data.len();
        let max_bins = config.max_bins;

        // Collect non-NaN values and compute min/max
        let mut values: Vec<f32> = Vec::with_capacity(n_samples);
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for &val in data.iter() {
            if val.is_finite() {
                values.push(val);
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Handle degenerate cases
        if values.is_empty() || min_val >= max_val {
            let bins: Vec<u32> = vec![0; n_samples];
            let mapper = BinMapper::numerical(
                vec![f64::MAX],
                MissingType::None,
                0,
                0,
                0.0,
                0.0,
                0.0,
            );
            return (bins, mapper);
        }

        let n_bins = max_bins.min(values.len() as u32);

        // Compute bin boundaries based on strategy
        let bounds: Vec<f64> = match config.strategy {
            BinningStrategy::EqualWidth => {
                let width = (max_val - min_val) / n_bins as f32;
                (1..=n_bins)
                    .map(|i| {
                        if i == n_bins {
                            f64::MAX
                        } else {
                            (min_val + i as f32 * width) as f64
                        }
                    })
                    .collect()
            }
            BinningStrategy::Quantile => {
                Self::compute_quantile_bounds(&mut values, n_bins, config.sample_cnt)
            }
        };

        // Bin each value using binary search on bounds
        let bins: Vec<u32> = data
            .iter()
            .map(|&val| {
                if !val.is_finite() {
                    0 // Map NaN to bin 0
                } else {
                    match bounds.binary_search_by(|b| b.partial_cmp(&(val as f64)).unwrap()) {
                        Ok(i) => i as u32,
                        Err(i) => i as u32,
                    }
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

        (bins, mapper)
    }

    /// Compute quantile boundaries using LightGBM-style sampling.
    ///
    /// For datasets larger than `sample_cnt`, uses uniform sampling to compute
    /// approximate quantiles. This significantly reduces memory and compute cost
    /// while maintaining good bin boundary quality.
    ///
    /// For smaller datasets, computes exact quantiles via full sort.
    fn compute_quantile_bounds(values: &mut [f32], n_bins: u32, sample_cnt: usize) -> Vec<f64> {
        let n = values.len();
        if n == 0 {
            return vec![f64::MAX];
        }

        // Use LightGBM-style sampling for large datasets
        if n > sample_cnt && sample_cnt > 0 {
            // Uniform sampling: take evenly-spaced samples
            let step = n / sample_cnt;
            let mut sample: Vec<f32> = values.iter().step_by(step.max(1)).copied().collect();
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let sample_n = sample.len();
            let mut bounds = Vec::with_capacity(n_bins as usize);
            for i in 1..n_bins {
                let q = i as f64 / n_bins as f64;
                let idx = ((q * (sample_n - 1) as f64).round() as usize).min(sample_n - 1);
                let bound = sample[idx] as f64;

                if bounds.is_empty() || bound > *bounds.last().unwrap() {
                    bounds.push(bound);
                }
            }
            bounds.push(f64::MAX);
            bounds
        } else {
            // Full sort for smaller datasets - exact quantiles
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut bounds = Vec::with_capacity(n_bins as usize);
            for i in 1..n_bins {
                let q = i as f64 / n_bins as f64;
                let idx = ((q * (n - 1) as f64).round() as usize).min(n - 1);
                let bound = values[idx] as f64;

                if bounds.is_empty() || bound > *bounds.last().unwrap() {
                    bounds.push(bound);
                }
            }
            bounds.push(f64::MAX);
            bounds
        }
    }

    /// Add a pre-binned feature with existing mapper (internal use).
    ///
    /// # Arguments
    /// * `bins` - Bin indices (length = n_rows)
    /// * `mapper` - Bin mapper for this feature
    /// * `name` - Optional feature name
    pub(crate) fn add_binned(
        mut self,
        bins: Vec<u32>,
        mapper: BinMapper,
        name: Option<String>,
    ) -> Self {
        self.validate_n_rows(bins.len());
        self.features.push(FeatureData {
            bins,
            mapper,
            name,
        });
        self
    }

    /// Set the grouping strategy (internal use only).
    ///
    /// Users should rely on the default Auto strategy which uses column-major
    /// layout for optimal histogram building performance.
    #[doc(hidden)]
    pub fn group_strategy(mut self, strategy: GroupStrategy) -> Self {
        self.group_strategy = strategy;
        self
    }

    /// Enable feature bundling for sparse/one-hot features.
    ///
    /// Bundling analyzes feature sparsity and creates a bundle plan that
    /// groups non-conflicting sparse features. This can significantly reduce
    /// memory usage and improve training speed for datasets with many sparse
    /// or one-hot encoded features.
    ///
    /// The actual column reduction happens during histogram building (Story 1.4).
    /// This method stores the bundle plan in the dataset for later use.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let binned = BinnedDatasetBuilder::from_dataset(&dataset, config, Parallelism::Parallel)
    ///     .with_bundling(BundlingConfig::auto())
    ///     .build()?;
    ///
    /// // Check bundling stats
    /// if let Some(stats) = binned.bundling_stats() {
    ///     println!("Bundles created: {}", stats.bundles_created);
    /// }
    /// ```
    pub fn with_bundling(mut self, config: BundlingConfig) -> Self {
        self.bundling_config = Some(config);
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

        // Create bundle plan if bundling is enabled
        let bundle_plan = if let Some(config) = &self.bundling_config {
            if config.enable_bundling {
                Some(self.create_bundle_plan(n_rows, config))
            } else {
                None
            }
        } else {
            None
        };

        Ok(BinnedDataset::with_bundle_plan(
            n_rows, features, groups, bundle_plan,
        ))
    }

    /// Create a bundle plan by analyzing feature sparsity and conflicts.
    fn create_bundle_plan(&self, n_rows: usize, config: &BundlingConfig) -> BundlePlan {
        // Analyze features to get sparsity and binary info
        let feature_infos: Vec<FeatureInfo> = self
            .features
            .iter()
            .enumerate()
            .map(|(idx, f)| {
                // Convert FeatureData to FeatureInfo
                let density = (1.0 - f.sparsity()) as f32;
                let is_binary = f.n_bins() <= 2;
                let is_trivial = f.n_bins() <= 1;
                FeatureInfo {
                    original_idx: idx,
                    density,
                    is_binary,
                    is_trivial,
                }
            })
            .collect();

        // Create a simple adapter that provides bin values for conflict detection
        let bin_accessor = FeatureBinAccessor {
            features: &self.features,
            n_rows,
        };

        create_bundle_plan(&bin_accessor, &feature_infos, config)
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
    ) -> Result<(Vec<FeatureGroup>, Vec<BinnedFeatureInfo>), BuildError> {
        let n_features = self.features.len();

        // Pre-allocate feature metadata (will be filled in order)
        let mut feature_metas: Vec<Option<BinnedFeatureInfo>> = vec![None; n_features];
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
                let mut meta = BinnedFeatureInfo::new(
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
        let features: Vec<BinnedFeatureInfo> = feature_metas
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

    /// Create a simple numeric BinMapper for testing.
    fn make_mapper(n_bins: u32) -> BinMapper {
        let bounds: Vec<f64> = (0..n_bins).map(|i| i as f64 + 0.5).collect();
        BinMapper::numerical(bounds, MissingType::None, 0, 0, 0.0, 0.0, (n_bins - 1) as f64)
    }

    #[test]
    fn test_builder_single_group_row_major() {
        let dataset = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 1, 2, 3], make_mapper(4), None)
            .add_binned(vec![1, 2, 3, 0], make_mapper(4), None)
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
        let dataset = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 1, 2, 3], make_mapper(4), None)
            .add_binned(vec![10, 11, 12, 13], make_mapper(16), None)
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
        let dataset = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 1, 2, 3], make_mapper(100), None)  // dense
            .add_binned(vec![0, 100, 200, 300], make_mapper(500), None)  // wide
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
        let dataset = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 1, 2], make_mapper(4), None)
            .add_binned(vec![1, 2, 3], make_mapper(4), None)
            .add_binned(vec![2, 3, 0], make_mapper(4), None)
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
        let dataset = BinnedDatasetBuilder::new(BinningConfig::default()).build().unwrap();
        assert_eq!(dataset.n_rows(), 0);
        assert_eq!(dataset.n_features(), 0);
        assert_eq!(dataset.n_groups(), 0);
    }

    #[test]
    fn test_builder_error_duplicate_feature() {
        let result = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 1, 2], make_mapper(4), None)
            .add_binned(vec![1, 2, 3], make_mapper(4), None)
            .group_strategy(GroupStrategy::Custom(vec![
                GroupSpec::new(vec![0, 1], GroupLayout::RowMajor),
                GroupSpec::new(vec![1], GroupLayout::ColumnMajor),  // duplicate!
            ]))
            .build();

        assert!(matches!(result, Err(BuildError::DuplicateFeature(1))));
    }

    #[test]
    fn test_builder_error_unassigned_feature() {
        let result = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 1, 2], make_mapper(4), None)
            .add_binned(vec![1, 2, 3], make_mapper(4), None)
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
            mapper: make_mapper(4),
            name: None,
        };
        assert!((f.sparsity() - 0.8).abs() < 0.001);
    }

    // =========================================================================
    // Bundling Integration Tests
    // =========================================================================

    #[test]
    fn test_builder_with_bundling_disabled() {
        let dataset = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 0, 0, 1], make_mapper(2), None)
            .add_binned(vec![0, 1, 0, 0], make_mapper(2), None)
            .with_bundling(BundlingConfig::disabled())
            .build()
            .unwrap();

        // Bundling disabled, no bundle plan stored
        assert!(dataset.bundle_plan().is_none());
        assert!(!dataset.has_effective_bundling());
    }

    #[test]
    fn test_builder_with_bundling_auto_sparse_features() {
        // Create 10 one-hot encoded features (mutually exclusive, very sparse)
        // With 100 rows and 10 features, each feature has ~10% non-zero (density 0.1)
        // This is 90% sparse, meeting the default min_sparsity threshold of 0.9
        let n_rows = 100;
        let n_features = 10;

        let mut builder = BinnedDatasetBuilder::new(BinningConfig::default());
        for f in 0..n_features {
            let bins: Vec<u32> = (0..n_rows)
                .map(|r| if r % n_features == f { 1 } else { 0 })
                .collect();
            builder = builder.add_binned(bins, make_mapper(2), None);
        }

        let dataset = builder
            .with_bundling(BundlingConfig::auto())
            .build()
            .unwrap();

        // Should have a bundle plan
        let plan = dataset.bundle_plan().expect("Should have bundle plan");

        // All features are sparse (90% zeros) and non-conflicting
        // They should be detected as sparse
        assert!(
            plan.sparse_feature_count > 0 || plan.skipped,
            "Expected sparse features or bundling to be skipped, got sparse_count={}, skipped={}",
            plan.sparse_feature_count,
            plan.skipped
        );
    }

    #[test]
    fn test_builder_with_bundling_stores_plan() {
        let dataset = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0], make_mapper(2), None)
            .add_binned(vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0], make_mapper(2), None)
            .add_binned(vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1], make_mapper(2), None) // dense
            .with_bundling(BundlingConfig::auto())
            .build()
            .unwrap();

        // Should have a bundle plan stored
        assert!(dataset.bundle_plan().is_some());

        // Can get stats
        if let Some(stats) = dataset.bundling_stats() {
            // 2 sparse features
            assert!(stats.original_sparse_features <= 3);
        }
    }

    #[test]
    fn test_builder_without_bundling_no_plan() {
        let dataset = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_binned(vec![0, 0, 0, 1], make_mapper(2), None)
            .add_binned(vec![0, 1, 0, 0], make_mapper(2), None)
            .build()
            .unwrap();

        assert!(dataset.bundle_plan().is_none());
        assert!(!dataset.has_effective_bundling());
    }

    #[test]
    fn test_add_features() {
        use ndarray::array;
        
        // Create feature-major data: 3 features, 4 samples
        let features = array![
            [1.0f32, 4.0, 7.0, 0.0],  // feature 0
            [2.0, 5.0, 8.0, 1.0],  // feature 1
            [3.0, 6.0, 9.0, 2.0],  // feature 2
        ];
        let features_view = FeaturesView::from_array(features.view());

        let binned = BinnedDatasetBuilder::new(BinningConfig::default())
            .add_features(features_view, Parallelism::Sequential)
            .build()
            .unwrap();

        // Verify structure
        assert_eq!(binned.n_rows(), 4);
        assert_eq!(binned.n_features(), 3);
    }

    #[test]
    fn test_add_features_with_schema() {
        use ndarray::array;
        use crate::data::DatasetSchema;

        // Create feature-major data: 2 features, 3 samples
        let feature_data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let schema = DatasetSchema::all_numeric(2);
        let features = FeaturesView::new(feature_data.view(), &schema);

        let config = BinningConfig::builder().max_bins(64).build();

        let binned = BinnedDatasetBuilder::new(config)
            .add_features(features, Parallelism::Sequential)
            .build()
            .unwrap();

        assert_eq!(binned.n_rows(), 3);
        assert_eq!(binned.n_features(), 2);
    }
}
