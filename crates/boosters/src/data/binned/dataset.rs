//! Internal binned dataset representation.
//!
//! `BinnedDataset` is an internal type used for histogram building during training.
//! The only supported construction path is via [`BinnedDataset::from_dataset`].

// Allow dead code during migration - this will be used when we switch over in Epic 7
#![allow(dead_code)]

use thiserror::Error;

use super::BinData;
use super::bin_mapper::{BinMapper, MissingType};
use super::bundling::{BundlingConfig, apply_bundling, create_bundle_plan};
use super::feature_analysis::{
    BinningConfig, FeatureAnalysis, FeatureMetadata, GroupSpec, GroupType,
    analyze_features_dataset, compute_groups,
};
use super::group::FeatureGroup;
use super::storage::{
    BundleStorage, CategoricalStorage, FeatureStorage, NumericStorage, SparseCategoricalStorage,
    SparseNumericStorage,
};
use super::view::FeatureView;
use crate::data::Dataset;

/// Errors returned by binned dataset construction.
#[derive(Debug, Clone, Error)]
pub enum DatasetError {
    #[error("dataset is empty")]
    EmptyDataset,

    #[error("binning error: {0}")]
    BinningError(String),
}

/// Where a feature's data lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureLocation {
    /// Feature in a regular (Dense or Sparse) group.
    Direct { group_idx: u32, idx_in_group: u32 },
    /// Feature bundled into a Bundle group (EFB).
    /// Not yet implemented - reserved for future use.
    Bundled {
        bundle_group_idx: u32,
        position_in_bundle: u32,
    },
    /// Feature was skipped (trivial, constant value).
    Skipped,
}

impl FeatureLocation {
    /// Returns true if the feature is directly stored (not bundled or skipped).
    #[inline]
    pub fn is_direct(&self) -> bool {
        matches!(self, FeatureLocation::Direct { .. })
    }

    /// Returns true if the feature is bundled (EFB).
    #[inline]
    pub fn is_bundled(&self) -> bool {
        matches!(self, FeatureLocation::Bundled { .. })
    }

    /// Returns true if the feature was skipped.
    #[inline]
    pub fn is_skipped(&self) -> bool {
        matches!(self, FeatureLocation::Skipped)
    }
}

/// Metadata for a single feature.
#[derive(Debug, Clone)]
pub struct BinnedFeatureInfo {
    /// Optional feature name.
    pub name: Option<String>,
    /// The bin mapper for this feature (contains thresholds/categories).
    pub bin_mapper: BinMapper,
    /// Where this feature's data lives.
    pub location: FeatureLocation,
}

impl BinnedFeatureInfo {
    /// Create a new feature info.
    pub fn new(name: Option<String>, bin_mapper: BinMapper, location: FeatureLocation) -> Self {
        Self {
            name,
            bin_mapper,
            location,
        }
    }

    /// Returns true if this feature is categorical.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        self.bin_mapper.is_categorical()
    }

    /// Get the number of bins for this feature.
    #[inline]
    pub fn n_bins(&self) -> u32 {
        self.bin_mapper.n_bins()
    }
}

/// Where an effective feature column comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinnedFeatureColumnSource {
    /// An encoded bundle column (EFB). The value points to a bundle group.
    Bundle { group_idx: usize },
    /// A standalone (non-bundled) original feature.
    Standalone { feature_idx: usize },
}

/// Result of `feature_views()` - views in the effective layout.
///
/// Effective layout puts bundles first, then standalone features:
/// ```text
/// [bundle_0, bundle_1, ..., standalone_0, standalone_1, ...]
/// ```
#[derive(Debug)]
pub struct BinnedFeatureViews<'a> {
    /// Feature views in effective order.
    pub views: Vec<FeatureView<'a>>,
    /// Bin counts for each effective column.
    pub bin_counts: Vec<u32>,
    /// Whether each effective column is categorical.
    /// For bundles: always true (encoded bin space).
    pub is_categorical: Vec<bool>,
    /// Whether each effective column has missing values.
    /// For bundles: always true (bin 0 is "all defaults" / missing).
    pub has_missing: Vec<bool>,
    /// Source mapping for each effective column.
    pub sources: Vec<BinnedFeatureColumnSource>,
    /// Number of bundle columns (first `n_bundles` columns are bundles).
    pub n_bundles: usize,
}

impl<'a> BinnedFeatureViews<'a> {
    /// Number of effective columns (bundles + standalone features).
    #[inline]
    pub fn n_columns(&self) -> usize {
        self.views.len()
    }

    /// Whether the dataset contains any bundles.
    #[inline]
    pub fn has_bundles(&self) -> bool {
        self.n_bundles > 0
    }

    /// Returns true if the effective column is a bundle.
    #[inline]
    pub fn is_bundle(&self, effective_col: usize) -> bool {
        matches!(
            self.sources[effective_col],
            BinnedFeatureColumnSource::Bundle { .. }
        )
    }
}

/// The unified dataset type for training and inference.
///
/// Contains both binned data (for tree splits) and raw data (for linear regression).
/// This replaces the previous separate `Dataset` and `BinnedDataset` types.
#[derive(Debug, Clone)]
pub struct BinnedDataset {
    /// Number of samples.
    n_samples: usize,
    /// Per-feature metadata (name, bin mapper, location).
    features: Box<[BinnedFeatureInfo]>,
    /// Feature groups (actual storage).
    groups: Vec<FeatureGroup>,
    /// Global bin offsets for histogram allocation.
    /// `global_bin_offsets[i]` is the offset of feature i's bins in the global histogram.
    global_bin_offsets: Box<[u32]>,
    /// Optional labels (targets).
    labels: Option<Box<[f32]>>,
    /// Optional sample weights.
    weights: Option<Box<[f32]>>,
}

impl BinnedDataset {
    /// Create a BinnedDataset from a Dataset.
    ///
    /// This is the primary factory method for creating binned training data.
    /// It analyzes features, creates bin mappers, groups features, and applies
    /// Exclusive Feature Bundling (EFB) if enabled.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Source dataset with raw feature values
    /// * `config` - Binning configuration (max_bins, bundling settings, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::data::{Dataset, BinnedDataset, BinningConfig};
    ///
    /// let dataset = Dataset::from_array(features.view(), Some(targets.view()), None);
    /// let binned = BinnedDataset::from_dataset(&dataset, &BinningConfig::default())?;
    /// ```
    ///
    /// # Note
    ///
    /// Targets and weights are NOT copied to BinnedDataset. They should be
    /// accessed from the source Dataset during training.
    pub fn from_dataset(dataset: &Dataset, config: &BinningConfig) -> Result<Self, DatasetError> {
        let n_samples = dataset.n_samples();
        let n_features = dataset.n_features();
        if n_samples == 0 || n_features == 0 {
            return Err(DatasetError::EmptyDataset);
        }

        // Build feature metadata from Dataset schema.
        let mut names: Vec<String> = Vec::with_capacity(n_features);
        let mut categorical = Vec::new();
        for (idx, meta) in dataset.schema().iter_enumerated() {
            names.push(meta.name.clone().unwrap_or_default());
            if meta.feature_type.is_categorical() {
                categorical.push(idx);
            }
        }
        let metadata = FeatureMetadata::default()
            .names(names)
            .categorical(categorical);

        // Analyze features and compute groups.
        let analyses = analyze_features_dataset(dataset, config, Some(&metadata));
        let mut grouping = compute_groups(&analyses, config);

        // Optional Exclusive Feature Bundling (EFB).
        if config.enable_bundling {
            let bundling_config = BundlingConfig {
                min_sparsity: config.sparsity_threshold,
                ..Default::default()
            };
            let plan = create_bundle_plan(&analyses, dataset, &bundling_config);
            grouping = apply_bundling(grouping, &plan, &analyses);
        }

        // Create bin mappers and build feature groups.
        let bin_mappers = create_bin_mappers(&analyses, dataset, config, Some(&metadata));
        let mut groups = Vec::with_capacity(grouping.groups.len());
        for group_spec in &grouping.groups {
            groups.push(build_feature_group(
                group_spec,
                &analyses,
                &bin_mappers,
                dataset,
                n_samples,
            ));
        }

        // Build per-feature info and global bin offsets.
        let mut features = Vec::with_capacity(n_features);
        let mut global_bin_offsets = Vec::with_capacity(n_features + 1);
        let mut current_offset = 0u32;

        let mut is_trivial = vec![false; n_features];
        for &idx in &grouping.trivial_features {
            is_trivial[idx] = true;
        }

        let mut location_map: Vec<Option<FeatureLocation>> = vec![None; n_features];
        for (group_idx, group) in groups.iter().enumerate() {
            let is_bundle = group.is_bundle();
            for (idx_in_group, &global_idx) in group.feature_indices().iter().enumerate() {
                let location = if is_bundle {
                    FeatureLocation::Bundled {
                        bundle_group_idx: group_idx as u32,
                        position_in_bundle: idx_in_group as u32,
                    }
                } else {
                    FeatureLocation::Direct {
                        group_idx: group_idx as u32,
                        idx_in_group: idx_in_group as u32,
                    }
                };
                location_map[global_idx as usize] = Some(location);
            }
        }

        for feature_idx in 0..n_features {
            let bin_mapper = bin_mappers[feature_idx].clone();
            let n_bins = bin_mapper.n_bins();

            let location = if is_trivial[feature_idx] {
                FeatureLocation::Skipped
            } else {
                location_map[feature_idx].unwrap_or(FeatureLocation::Skipped)
            };

            let name = metadata.get_name(feature_idx).map(|s| s.to_owned());
            features.push(BinnedFeatureInfo::new(name, bin_mapper, location));

            global_bin_offsets.push(current_offset);
            current_offset += n_bins;
        }
        global_bin_offsets.push(current_offset);

        Ok(Self {
            n_samples,
            features: features.into_boxed_slice(),
            groups,
            global_bin_offsets: global_bin_offsets.into_boxed_slice(),
            labels: None,
            weights: None,
        })
    }

    // =========================================================================
    // Basic accessors
    // =========================================================================

    /// Get the number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Get the number of groups.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// Get the total number of bins across all features.
    #[inline]
    pub fn total_bins(&self) -> u32 {
        // Last element is the total
        self.global_bin_offsets.last().copied().unwrap_or(0)
    }

    /// Get feature info for a feature.
    #[inline]
    pub fn feature_info(&self, feature: usize) -> &BinnedFeatureInfo {
        &self.features[feature]
    }

    /// Get the location of a feature.
    #[inline]
    pub fn feature_location(&self, feature: usize) -> FeatureLocation {
        self.features[feature].location
    }

    /// Get the bin mapper for a feature.
    #[inline]
    pub fn bin_mapper(&self, feature: usize) -> &BinMapper {
        &self.features[feature].bin_mapper
    }

    /// Get the global bin offset for a feature.
    /// This is used for histogram indexing.
    #[inline]
    pub fn global_bin_offset(&self, feature: usize) -> u32 {
        self.global_bin_offsets[feature]
    }

    /// Check if a feature is categorical.
    #[inline]
    pub fn is_categorical(&self, feature: usize) -> bool {
        self.features[feature].is_categorical()
    }

    /// Get the number of bins for a feature.
    #[inline]
    pub fn n_bins(&self, feature: usize) -> u32 {
        self.features[feature].n_bins()
    }

    /// Check if the dataset has labels.
    #[inline]
    pub fn has_labels(&self) -> bool {
        self.labels.is_some()
    }

    /// Get the labels if present.
    #[inline]
    pub fn labels(&self) -> Option<&[f32]> {
        self.labels.as_deref()
    }

    /// Check if the dataset has weights.
    #[inline]
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Get the weights if present.
    #[inline]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Get a reference to the groups.
    #[inline]
    pub fn groups(&self) -> &[FeatureGroup] {
        &self.groups
    }

    /// Get a reference to a specific group.
    #[inline]
    pub fn group(&self, group_idx: usize) -> &FeatureGroup {
        &self.groups[group_idx]
    }

    // =========================================================================
    // Bin/Raw Access Methods
    // =========================================================================

    /// Get the bin value for a sample and feature.
    ///
    /// # Parameters
    /// - `sample`: Sample index (0..n_samples)
    /// - `feature`: Global feature index (0..n_features)
    ///
    /// # Panics
    ///
    /// Panics if the feature is skipped (trivial) or indices are out of bounds.
    #[inline]
    pub fn bin(&self, sample: usize, feature: usize) -> u32 {
        let location = self.features[feature].location;
        match location {
            FeatureLocation::Direct {
                group_idx,
                idx_in_group,
            } => self.groups[group_idx as usize].bin(sample, idx_in_group as usize),
            FeatureLocation::Bundled { .. } => {
                // TODO: Implement bundled feature access
                panic!("Bundled feature access not yet implemented")
            }
            FeatureLocation::Skipped => {
                panic!("Cannot access bin for skipped feature {feature}")
            }
        }
    }

    // =========================================================================
    // Histogram Building (Hot Path)
    // =========================================================================

    /// Get feature views for histogram building.
    ///
    /// Returns views in the effective layout used by the grower:
    /// - First: One view per bundle (the encoded bundle column)
    /// - Then: One view per standalone (non-bundled) feature
    ///
    /// # Effective Layout
    /// ```text
    /// Effective columns: [bundle_0, bundle_1, ..., standalone_0, standalone_1, ...]
    /// ```
    ///
    /// For datasets without bundling, this is simply all non-trivial features.
    pub fn feature_views(&self) -> BinnedFeatureViews<'_> {
        use crate::data::MissingType;

        let mut views = Vec::new();
        let mut bin_counts = Vec::new();
        let mut is_categorical = Vec::new();
        let mut has_missing = Vec::new();
        let mut sources = Vec::new();
        let mut n_bundles = 0usize;

        // Phase 1: Collect bundle views (bundle groups)
        for (group_idx, group) in self.groups.iter().enumerate() {
            if group.is_bundle() {
                // Bundle group: one view for the encoded bundle
                views.push(group.feature_view(0));
                if let Some(bundle) = group.as_bundle() {
                    bin_counts.push(bundle.total_bins());
                } else {
                    // Should not happen, but fallback to bin_counts[0]
                    bin_counts.push(group.bin_counts()[0]);
                }
                // Bundles are treated as categorical (encoded bin space)
                is_categorical.push(true);
                // Bundles always have "missing" (bin 0 = all defaults)
                has_missing.push(true);
                sources.push(BinnedFeatureColumnSource::Bundle { group_idx });
                n_bundles += 1;
            }
        }

        // Phase 2: Collect standalone feature views (non-bundle features)
        for feature_idx in 0..self.features.len() {
            let location = self.features[feature_idx].location;
            match location {
                FeatureLocation::Direct {
                    group_idx,
                    idx_in_group,
                } => {
                    let group = &self.groups[group_idx as usize];
                    views.push(group.feature_view(idx_in_group as usize));
                    bin_counts.push(group.bin_counts()[idx_in_group as usize]);
                    // Use original feature's categorical flag
                    is_categorical.push(self.features[feature_idx].is_categorical());
                    // Check if original feature has missing
                    has_missing.push(
                        self.features[feature_idx].bin_mapper.missing_type() != MissingType::None,
                    );
                    sources.push(BinnedFeatureColumnSource::Standalone { feature_idx });
                }
                FeatureLocation::Bundled { .. } | FeatureLocation::Skipped => {
                    // Bundled features are accessed via their bundle
                    // Skipped features are not included
                }
            }
        }

        BinnedFeatureViews {
            views,
            bin_counts,
            is_categorical,
            has_missing,
            sources,
            n_bundles,
        }
    }

    /// Decode a split on an effective column to the original feature and bin.
    ///
    /// When working with effective columns (from `feature_views()`), splits
    /// need to be decoded back to original feature indices for tree storage.
    ///
    /// # Parameters
    /// - `effective_views`: The BinnedFeatureViews from `feature_views()`
    /// - `effective_col`: The effective column index (0..n_effective_columns)
    /// - `bin`: The bin value for the split
    ///
    /// # Returns
    /// `(original_feature_idx, original_bin)` for use in tree split nodes.
    ///
    /// # Bundle Handling
    /// For bundle columns, decodes the encoded bin to find which original feature
    /// owns that bin. If bin is 0 (all defaults), returns the first feature in the
    /// bundle with bin 0.
    pub fn decode_split_to_original(
        &self,
        effective_views: &BinnedFeatureViews<'_>,
        effective_col: usize,
        bin: u32,
    ) -> (usize, u32) {
        match effective_views.sources[effective_col] {
            BinnedFeatureColumnSource::Bundle { group_idx } => {
                // Bundle column: decode using BundleStorage
                let group = &self.groups[group_idx];
                let Some(bundle) = group.as_bundle() else {
                    panic!("Bundle group {} has no BundleStorage", group_idx);
                };

                if let Some((pos_in_bundle, orig_bin)) = bundle.decode(bin as u16) {
                    let orig_feature = bundle.feature_indices()[pos_in_bundle] as usize;
                    (orig_feature, orig_bin)
                } else {
                    // bin 0 = all defaults. Pick the first feature in the bundle with bin 0.
                    let first_feature = bundle.feature_indices()[0] as usize;
                    (first_feature, 0)
                }
            }
            BinnedFeatureColumnSource::Standalone { feature_idx } => (feature_idx, bin),
        }
    }

    /// Check if the dataset contains any categorical features.
    ///
    /// Primarily useful for validating configs that disallow categoricals.
    pub fn has_categorical(&self) -> bool {
        self.features.iter().any(|f| {
            matches!(f.location, FeatureLocation::Direct { group_idx, .. } if {
                self.groups[group_idx as usize].is_categorical()
            })
        })
    }

    // NOTE: sample_blocks() method removed in RFC-0021.
    // SampleBlocks now works on Dataset, not BinnedDataset.
    // Use: SampleBlocks::new(&dataset, block_size).for_each_with(...)
}

// =============================================================================
// Internal helpers (inlined from the former builder module)
// =============================================================================

/// Create bin mappers for all features based on analysis results.
/// Data is in column-major format: [feature][sample].
fn create_bin_mappers(
    analyses: &[FeatureAnalysis],
    dataset: &Dataset,
    config: &BinningConfig,
    metadata: Option<&FeatureMetadata>,
) -> Vec<BinMapper> {
    fn sample_indices_for_binning(n_samples: usize, sample_cnt: usize) -> Option<Vec<u32>> {
        if n_samples <= sample_cnt {
            // Use all samples - no need to allocate indices
            return None;
        }

        // Deterministic pseudo-random sampling without extra dependencies.
        // This keeps binning stable across runs while avoiding a full scan of every feature.
        fn splitmix64_next(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        let k = sample_cnt.min(n_samples);
        let mut indices = Vec::with_capacity(k);
        let mut seen = std::collections::HashSet::with_capacity(k * 2);

        let mut state = 0xD1B54A32D192ED03u64 ^ (n_samples as u64);
        while indices.len() < k {
            let r = splitmix64_next(&mut state);
            let idx = (r % n_samples as u64) as u32;
            if seen.insert(idx) {
                indices.push(idx);
            }
        }

        indices.sort_unstable();
        Some(indices)
    }

    // None means use all samples, Some(indices) means sample
    let sample_indices = sample_indices_for_binning(dataset.n_samples(), config.sample_cnt);

    analyses
        .iter()
        .map(|analysis| {
            let feature_idx = analysis.feature_idx;
            let max_bins = metadata
                .and_then(|m| m.max_bins.get(&feature_idx).copied())
                .unwrap_or(config.max_bins);

            let has_nan = analysis.n_nan > 0;

            if analysis.is_numeric {
                // Optimize for common case: dense feature with all samples
                if sample_indices.is_none()
                    && let Some(slice) = dataset.feature(feature_idx).as_slice()
                {
                    // Fast path: direct slice access (no copy, no indirection)
                    return bin_numeric(slice, max_bins, has_nan);
                }

                // Fall back to gathering (sampled or sparse features)
                let indices = sample_indices.as_ref().map_or_else(
                    || (0..dataset.n_samples() as u32).collect::<Vec<_>>(),
                    |v| v.clone(),
                );
                let mut values = vec![0.0f32; indices.len()];
                dataset.gather_feature_values(feature_idx, &indices, &mut values);
                bin_numeric(&values, max_bins, has_nan)
            } else {
                // For categorical features, sampling can miss categories that exist in the dataset.
                // Build the category set directly without materializing the full column.
                let mut seen = std::collections::HashSet::new();
                dataset.for_each_feature_value_dense(feature_idx, |_idx, val| {
                    if val.is_finite() {
                        seen.insert(val as i32);
                    }
                });

                let mut categories: Vec<i32> = seen.into_iter().collect();
                categories.sort();
                if categories.is_empty() {
                    categories.push(0);
                }

                BinMapper::categorical(categories, MissingType::None, 0, 0, 0.0)
            }
        })
        .collect()
}

fn bin_numeric(data: &[f32], max_bins: u32, has_nan: bool) -> BinMapper {
    // Collect non-NaN values and compute min/max
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut values: Vec<f32> = Vec::with_capacity(data.len());

    for &val in data.iter() {
        if val.is_finite() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            values.push(val);
        }
    }

    let n_valid = values.len();
    if n_valid == 0 || min_val >= max_val {
        return BinMapper::numerical(vec![f64::MAX], MissingType::None, 0, 0, 0.0, 0.0, 0.0);
    }

    let n_bins = max_bins.min(n_valid as u32);

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup();
    let n_unique = values.len();

    let mut bounds = Vec::with_capacity(n_bins as usize);
    if n_unique <= n_bins as usize {
        for i in 0..n_unique.saturating_sub(1) {
            let bound = (values[i] as f64 + values[i + 1] as f64) / 2.0;
            bounds.push(bound);
        }
    } else {
        for i in 1..n_bins {
            let q = i as f64 / n_bins as f64;
            let idx = ((q * (n_unique - 1) as f64).floor() as usize).min(n_unique - 2);
            let bound = (values[idx] as f64 + values[idx + 1] as f64) / 2.0;
            if bounds.is_empty() || bound > *bounds.last().unwrap() {
                bounds.push(bound);
            }
        }
    }
    bounds.push(f64::MAX);

    // If this feature has NaNs, allocate a dedicated last bin for NaN values.
    // This enables LightGBM/XGBoost-style default-direction routing.
    if has_nan {
        bounds.push(f64::MAX);
    }

    let missing_type = if has_nan {
        MissingType::NaN
    } else {
        MissingType::None
    };

    let n_bins = bounds.len() as u32;
    let default_bin = if has_nan { n_bins - 1 } else { 0 };

    BinMapper::numerical(
        bounds,
        missing_type,
        default_bin,
        0,
        0.0,
        min_val as f64,
        max_val as f64,
    )
}

fn build_feature_group(
    spec: &GroupSpec,
    _analyses: &[FeatureAnalysis],
    bin_mappers: &[BinMapper],
    dataset: &Dataset,
    n_samples: usize,
) -> FeatureGroup {
    let storage = match spec.group_type {
        GroupType::NumericDense => build_numeric_dense(spec, bin_mappers, dataset, n_samples),
        GroupType::CategoricalDense => {
            build_categorical_dense(spec, bin_mappers, dataset, n_samples)
        }
        GroupType::SparseNumeric => build_sparse_numeric(spec, bin_mappers, dataset, n_samples),
        GroupType::SparseCategorical => {
            build_sparse_categorical(spec, bin_mappers, dataset, n_samples)
        }
        GroupType::Bundle => build_bundle(spec, bin_mappers, dataset, n_samples),
    };

    let bin_counts: Box<[u32]> = spec
        .feature_indices
        .iter()
        .map(|&idx| bin_mappers[idx].n_bins())
        .collect();

    let feature_indices: Box<[u32]> = spec.feature_indices.iter().map(|&idx| idx as u32).collect();

    FeatureGroup::new(feature_indices, n_samples, storage, bin_counts)
}

fn build_numeric_dense(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    dataset: &Dataset,
    n_samples: usize,
) -> FeatureStorage {
    let n_features = spec.n_features();

    let bins = if spec.needs_u16 {
        let mut bin_data = Vec::with_capacity(n_features * n_samples);
        for &feature_idx in &spec.feature_indices {
            let mapper = &bin_mappers[feature_idx];
            dataset.for_each_feature_value_dense(feature_idx, |_idx, value| {
                bin_data.push(mapper.value_to_bin(value as f64) as u16);
            });
        }
        BinData::U16(bin_data.into_boxed_slice())
    } else {
        let mut bin_data = Vec::with_capacity(n_features * n_samples);
        for &feature_idx in &spec.feature_indices {
            let mapper = &bin_mappers[feature_idx];
            dataset.for_each_feature_value_dense(feature_idx, |_idx, value| {
                bin_data.push(mapper.value_to_bin(value as f64) as u8);
            });
        }
        BinData::U8(bin_data.into_boxed_slice())
    };

    FeatureStorage::Numeric(NumericStorage::new(bins))
}

fn build_categorical_dense(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    dataset: &Dataset,
    n_samples: usize,
) -> FeatureStorage {
    let n_features = spec.n_features();

    let bins = if spec.needs_u16 {
        let mut bin_data = Vec::with_capacity(n_features * n_samples);
        for &feature_idx in &spec.feature_indices {
            let mapper = &bin_mappers[feature_idx];
            dataset.for_each_feature_value_dense(feature_idx, |_idx, value| {
                bin_data.push(mapper.value_to_bin(value as f64) as u16);
            });
        }
        BinData::U16(bin_data.into_boxed_slice())
    } else {
        let mut bin_data = Vec::with_capacity(n_features * n_samples);
        for &feature_idx in &spec.feature_indices {
            let mapper = &bin_mappers[feature_idx];
            dataset.for_each_feature_value_dense(feature_idx, |_idx, value| {
                bin_data.push(mapper.value_to_bin(value as f64) as u8);
            });
        }
        BinData::U8(bin_data.into_boxed_slice())
    };

    FeatureStorage::Categorical(CategoricalStorage::new(bins))
}

fn build_sparse_numeric(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    dataset: &Dataset,
    n_samples: usize,
) -> FeatureStorage {
    assert_eq!(
        spec.n_features(),
        1,
        "Sparse groups must have exactly one feature"
    );
    let feature_idx = spec.feature_indices[0];
    let mapper = &bin_mappers[feature_idx];

    let mut indices = Vec::new();
    let mut bins = Vec::new();

    dataset.for_each_feature_value_dense(feature_idx, |i, value| {
        if value != 0.0 && !value.is_nan() {
            indices.push(i as u32);
            bins.push(mapper.value_to_bin(value as f64) as u8);
        }
    });

    let bins_data = BinData::U8(bins.into_boxed_slice());
    FeatureStorage::SparseNumeric(SparseNumericStorage::new(
        indices.into_boxed_slice(),
        bins_data,
        n_samples,
    ))
}

fn build_sparse_categorical(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    dataset: &Dataset,
    n_samples: usize,
) -> FeatureStorage {
    assert_eq!(
        spec.n_features(),
        1,
        "Sparse groups must have exactly one feature"
    );
    let feature_idx = spec.feature_indices[0];
    let mapper = &bin_mappers[feature_idx];

    let mut indices = Vec::new();
    let mut bins = Vec::new();

    dataset.for_each_feature_value_dense(feature_idx, |i, value| {
        if value != 0.0 && !value.is_nan() {
            indices.push(i as u32);
            bins.push(mapper.value_to_bin(value as f64) as u8);
        }
    });

    let bins_data = BinData::U8(bins.into_boxed_slice());
    FeatureStorage::SparseCategorical(SparseCategoricalStorage::new(
        indices.into_boxed_slice(),
        bins_data,
        n_samples,
    ))
}

fn build_bundle(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    dataset: &Dataset,
    n_samples: usize,
) -> FeatureStorage {
    let n_features = spec.n_features();
    assert!(n_features >= 2, "Bundle must have at least 2 features");

    let mut bin_offsets = Vec::with_capacity(n_features);
    let mut feature_n_bins = Vec::with_capacity(n_features);
    let mut offset = 1u32;

    for &feature_idx in &spec.feature_indices {
        let n_bins = bin_mappers[feature_idx].n_bins();
        bin_offsets.push(offset);
        feature_n_bins.push(n_bins);
        offset += n_bins;
    }
    let total_bins = offset;

    let default_bins: Vec<u32> = spec
        .feature_indices
        .iter()
        .map(|&idx| bin_mappers[idx].value_to_bin(0.0))
        .collect();

    let mut encoded_bins = vec![0u16; n_samples];
    for (pos, &feature_idx) in spec.feature_indices.iter().enumerate() {
        let mapper = &bin_mappers[feature_idx];
        let offset = bin_offsets[pos];
        dataset.for_each_feature_value_dense(feature_idx, |sample_idx, value| {
            if encoded_bins[sample_idx] == 0 && value != 0.0 && !value.is_nan() {
                let bin = mapper.value_to_bin(value as f64);
                encoded_bins[sample_idx] = (offset + bin) as u16;
            }
        });
    }

    let feature_indices_u32: Box<[u32]> =
        spec.feature_indices.iter().map(|&idx| idx as u32).collect();

    FeatureStorage::Bundle(BundleStorage::new(
        encoded_bins.into_boxed_slice(),
        feature_indices_u32,
        bin_offsets.into_boxed_slice(),
        feature_n_bins.into_boxed_slice(),
        total_bins,
        default_bins.into_boxed_slice(),
        n_samples,
    ))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::binned::feature_analysis::BinningConfig;
    use ndarray::{Array1, array};

    #[test]
    fn test_from_dataset_single_feature() {
        // Use floats to ensure numeric detection
        let config = BinningConfig::default();

        let raw = Dataset::builder()
            .add_feature_unnamed(array![1.5, 2.5, 3.5, 4.5, 5.5])
            .build()
            .unwrap();
        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();

        assert_eq!(dataset.n_samples(), 5);
        assert_eq!(dataset.n_features(), 1);
        assert_eq!(dataset.n_groups(), 1);
    }

    #[test]
    fn test_from_dataset() {
        // Create a raw Dataset
        let f0 = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let f1 = array![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let targets = array![[0.0f32, 1.0, 0.0, 1.0, 0.0]]; // [1 output, 5 samples]
        let raw_dataset = Dataset::builder()
            .add_feature_unnamed(f0)
            .add_feature_unnamed(f1)
            .targets(targets.view())
            .build()
            .unwrap();

        // Convert to BinnedDataset
        let config = BinningConfig::default();
        let binned = BinnedDataset::from_dataset(&raw_dataset, &config).unwrap();

        // Verify structure matches
        assert_eq!(binned.n_samples(), 5);
        assert_eq!(binned.n_features(), 2);

        // Verify both features are in the dataset
        assert!(binned.feature_location(0).is_direct());
        assert!(binned.feature_location(1).is_direct());
    }

    #[test]
    fn test_from_dataset_empty_dataset() {
        // Empty dataset should fail (0 samples)
        let raw_dataset = Dataset::builder()
            .add_feature_unnamed(Array1::<f32>::zeros(0))
            .build()
            .unwrap();

        let config = BinningConfig::default();
        let result = BinnedDataset::from_dataset(&raw_dataset, &config);

        assert!(result.is_err());
    }

    #[test]
    fn test_feature_location() {
        let config = BinningConfig::default();

        let raw = Dataset::builder()
            .add_feature_unnamed(array![1.1, 3.3, 20.2])
            .add_feature_unnamed(array![2.2, 10.1, 30.3])
            .build()
            .unwrap();
        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();

        // Both features should be direct (not skipped or bundled)
        assert!(dataset.feature_location(0).is_direct());
        assert!(dataset.feature_location(1).is_direct());

        // Both should be in the same group (numeric dense)
        if let (
            FeatureLocation::Direct { group_idx: g0, .. },
            FeatureLocation::Direct { group_idx: g1, .. },
        ) = (dataset.feature_location(0), dataset.feature_location(1))
        {
            assert_eq!(g0, g1);
        }
    }

    #[test]
    fn test_global_bin_offsets() {
        let config = BinningConfig::builder().max_bins(10).build();

        let raw = Dataset::builder()
            .add_feature_unnamed(array![1.1, 2.2, 3.3, 4.4, 5.5])
            .build()
            .unwrap();
        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();

        // Feature 0 should start at offset 0
        assert_eq!(dataset.global_bin_offset(0), 0);
        // Total bins should equal the number of bins for feature 0
        assert_eq!(dataset.total_bins(), dataset.n_bins(0));
    }

    #[test]
    fn test_feature_is_categorical() {
        // Mix of numeric and categorical
        let config = BinningConfig::default();

        let raw = Dataset::builder()
            .add_feature_unnamed(array![1.1, 2.2, 3.3, 4.4, 5.5])
            .add_categorical_unnamed(array![0.0, 1.0, 2.0, 1.0, 0.0])
            .build()
            .unwrap();

        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();

        assert!(!dataset.is_categorical(0)); // Numeric
        assert!(dataset.is_categorical(1)); // Categorical
    }

    #[test]
    fn test_bin_access() {
        // Create a simple dataset with known values
        let config = BinningConfig::builder().max_bins(5).build();

        let raw = Dataset::builder()
            .add_feature_unnamed(array![1.1, 2.2, 3.3, 4.4, 5.5])
            .build()
            .unwrap();
        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();

        // Bin values should be 0..n_bins-1 for evenly spaced values
        // With 5 unique values and max_bins=5, each value should map to its own bin
        for sample in 0..5 {
            let bin = dataset.bin(sample, 0);
            assert!(bin < dataset.n_bins(0));
        }
    }

    #[test]
    fn test_feature_views_count() {
        let config = BinningConfig::default();

        let raw = Dataset::builder()
            .add_feature_unnamed(array![1.1, 3.3, 20.2])
            .add_feature_unnamed(array![2.2, 10.1, 30.3])
            .build()
            .unwrap();
        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();
        let views = dataset.feature_views();

        // Should have exactly 2 views (one per non-trivial feature)
        assert_eq!(views.views.len(), 2);
    }

    #[test]
    fn test_feature_views_dense() {
        let config = BinningConfig::default();

        let raw = Dataset::builder()
            .add_feature_unnamed(array![1.1, 2.2, 3.3, 4.4, 5.5])
            .build()
            .unwrap();
        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();
        let views = dataset.feature_views();

        assert_eq!(views.views.len(), 1);
        assert!(views.views[0].is_dense());
        assert_eq!(views.views[0].len(), 5); // 5 samples
    }

    #[test]
    fn test_mixed_feature_views() {
        let config = BinningConfig::default();

        let raw = Dataset::builder()
            .add_feature_unnamed(array![1.1, 2.2, 3.3, 4.4, 5.5])
            .add_categorical_unnamed(array![0.0, 1.0, 2.0, 1.0, 0.0])
            .build()
            .unwrap();

        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();
        let views = dataset.feature_views();

        // Should have 2 views (one numeric, one categorical)
        assert_eq!(views.views.len(), 2);

        // Both should be dense (not sparse)
        for view in &views.views {
            assert!(view.is_dense());
            assert_eq!(view.len(), 5);
        }
    }

    #[test]
    fn test_decode_split_to_original_with_bundle() {
        use crate::data::BinningConfig;

        // Create two sparse categorical features that will be bundled
        // Feature 0: non-zero only for rows 0-4
        // Feature 1: non-zero only for rows 50-54
        let n_samples = 100;

        let mut f0_values = Vec::with_capacity(n_samples);
        let mut f1_values = Vec::with_capacity(n_samples);
        for sample in 0..n_samples {
            let v0 = if sample < 5 {
                (sample % 3 + 1) as f32
            } else {
                0.0
            };
            let v1 = if (50..55).contains(&sample) {
                ((sample - 50) % 3 + 1) as f32
            } else {
                0.0
            };
            f0_values.push(v0);
            f1_values.push(v1);
        }

        let config = BinningConfig::builder()
            .enable_bundling(true)
            .sparsity_threshold(0.9)
            .build();

        let raw = Dataset::builder()
            .add_categorical_unnamed(Array1::from_vec(f0_values))
            .add_categorical_unnamed(Array1::from_vec(f1_values))
            .build()
            .unwrap();

        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();

        let effective = dataset.feature_views();

        // Verify bundling occurred
        assert_eq!(effective.n_bundles, 1, "Should have 1 bundle");
        assert_eq!(effective.n_columns(), 1, "Should have 1 effective column");

        // Test decode_split_to_original for the bundle column.
        let (feature, bin) = dataset.decode_split_to_original(&effective, 0, 0);
        assert_eq!(feature, 0, "Bin 0 should map to first feature");
        assert_eq!(bin, 0, "Bin 0 should decode to original bin 0");

        // A non-zero bin should decode to the original feature that owns it.
        let (feature1, _bin1) = dataset.decode_split_to_original(&effective, 0, 1);
        assert!(
            feature1 == 0 || feature1 == 1,
            "Bin 1 should map to feature 0 or 1, got {}",
            feature1
        );

        // For a higher bin (likely from second feature).
        let (feature5, _bin5) = dataset.decode_split_to_original(&effective, 0, 5);
        assert!(
            feature5 == 0 || feature5 == 1,
            "Bin 5 should map to a feature in the bundle"
        );
    }
}
