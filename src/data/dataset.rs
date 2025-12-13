//! User-facing dataset abstraction.
//!
//! This is the canonical entry point for training APIs.

use std::collections::BTreeMap;

use crate::data::{
    BinMapper, BinnedDataset, BinnedDatasetBuilder, BuildError, ColMajor, ColMatrix, GroupStrategy,
    MissingType,
};

/// A single feature column.
#[derive(Debug, Clone)]
pub enum FeatureColumn {
    /// Numeric feature values (one per row).
    Numeric {
        name: Option<String>,
        values: Vec<f32>,
    },
    /// Categorical feature values (one per row).
    ///
    /// Values are integer category IDs.
    Categorical {
        name: Option<String>,
        values: Vec<i32>,
    },
}

impl FeatureColumn {
    /// Feature name, if present.
    pub fn name(&self) -> Option<&str> {
        match self {
            FeatureColumn::Numeric { name, .. } => name.as_deref(),
            FeatureColumn::Categorical { name, .. } => name.as_deref(),
        }
    }

    /// Number of rows in this column.
    pub fn len(&self) -> usize {
        match self {
            FeatureColumn::Numeric { values, .. } => values.len(),
            FeatureColumn::Categorical { values, .. } => values.len(),
        }
    }

    /// Returns true if the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Dataset conversion/validation errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum DatasetError {
    #[error("inconsistent number of rows: feature {feature_idx} expected {expected}, got {got}")]
    InconsistentRows {
        feature_idx: usize,
        expected: usize,
        got: usize,
    },

    #[error("number of targets ({targets}) does not match number of rows ({rows})")]
    TargetLenMismatch { rows: usize, targets: usize },

    #[error("number of weights ({weights}) does not match number of rows ({rows})")]
    WeightLenMismatch { rows: usize, weights: usize },

    #[error("GBLinear does not support categorical features (feature index {feature_idx})")]
    CategoricalNotSupported { feature_idx: usize },
}

/// A user-facing dataset.
///
/// Targets are single-output for now (length = n_rows).
#[derive(Debug, Clone)]
pub struct Dataset {
    features: Vec<FeatureColumn>,
    targets: Vec<f32>,
    weights: Option<Vec<f32>>,
    n_rows: usize,
}

impl Dataset {
    /// Create a dataset from feature columns and a target vector.
    pub fn new(features: Vec<FeatureColumn>, targets: Vec<f32>) -> Result<Self, DatasetError> {
        let n_rows = targets.len();

        for (i, col) in features.iter().enumerate() {
            let got = col.len();
            if got != n_rows {
                return Err(DatasetError::InconsistentRows {
                    feature_idx: i,
                    expected: n_rows,
                    got,
                });
            }
        }

        Ok(Self {
            features,
            targets,
            weights: None,
            n_rows,
        })
    }

    /// Create a numeric-only dataset from an existing matrix.
    pub fn from_numeric<S: AsRef<[f32]>>(
        data: &ColMatrix<f32, S>,
        targets: Vec<f32>,
    ) -> Result<Self, DatasetError> {
        if targets.len() != data.num_rows() {
            return Err(DatasetError::TargetLenMismatch {
                rows: data.num_rows(),
                targets: targets.len(),
            });
        }

        let mut features = Vec::with_capacity(data.num_cols());
        for col in 0..data.num_cols() {
            features.push(FeatureColumn::Numeric {
                name: None,
                values: data.col_slice(col).to_vec(),
            });
        }

        Self::new(features, targets)
    }

    /// Attach per-row weights.
    pub fn with_weights(mut self, weights: Vec<f32>) -> Result<Self, DatasetError> {
        if weights.len() != self.n_rows {
            return Err(DatasetError::WeightLenMismatch {
                rows: self.n_rows,
                weights: weights.len(),
            });
        }
        self.weights = Some(weights);
        Ok(self)
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Targets (length = n_rows).
    pub fn targets(&self) -> &[f32] {
        &self.targets
    }

    /// Optional weights (length = n_rows).
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Feature columns.
    pub fn features(&self) -> &[FeatureColumn] {
        &self.features
    }

    /// Convert to a binned dataset for tree training.
    pub fn to_binned(&self, max_bins: u32) -> Result<BinnedDataset, BuildError> {
        let mut builder = BinnedDatasetBuilder::new().group_strategy(GroupStrategy::Auto);

        for col in &self.features {
            match col {
                FeatureColumn::Numeric { name, values } => {
                    let (bins, mapper) = bin_numeric(values, max_bins);
                    builder = match name {
                        None => builder.add_binned(bins, mapper),
                        Some(n) => builder.add_binned_named(n.clone(), bins, mapper),
                    };
                }
                FeatureColumn::Categorical { name, values } => {
                    let (bins, mapper) = bin_categorical(values);
                    builder = match name {
                        None => builder.add_binned(bins, mapper),
                        Some(n) => builder.add_binned_named(n.clone(), bins, mapper),
                    };
                }
            }
        }

        builder.build()
    }

    /// Convert into a column-major numeric matrix for GBLinear training.
    pub fn for_gblinear(&self) -> Result<ColMatrix<f32>, DatasetError> {
        let n_rows = self.n_rows;
        let n_features = self.features.len();

        let mut data = Vec::with_capacity(n_rows * n_features);

        for (feature_idx, col) in self.features.iter().enumerate() {
            match col {
                FeatureColumn::Numeric { values, .. } => {
                    debug_assert_eq!(values.len(), n_rows);
                    data.extend_from_slice(values);
                }
                FeatureColumn::Categorical { .. } => {
                    return Err(DatasetError::CategoricalNotSupported { feature_idx });
                }
            }
        }

        Ok(crate::data::DenseMatrix::<f32, ColMajor>::from_vec(
            data, n_rows, n_features,
        ))
    }
}

fn bin_numeric(values: &[f32], max_bins: u32) -> (Vec<u32>, BinMapper) {
    // Mirrors `BinnedDatasetBuilder::from_matrix` logic, but for a single column.
    let n_rows = values.len();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut n_finite = 0usize;

    for &v in values {
        if v.is_finite() {
            n_finite += 1;
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
    }

    if n_finite == 0 || min_val >= max_val {
        let bins = vec![0u32; n_rows];
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

    let n_bins = max_bins.min(n_finite as u32).max(1);
    let width = (max_val - min_val) / n_bins as f32;

    let bounds: Vec<f64> = (1..=n_bins)
        .map(|i| {
            if i == n_bins {
                f64::MAX
            } else {
                (min_val + width * i as f32) as f64
            }
        })
        .collect();

    let bins: Vec<u32> = values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                0
            } else {
                let bin = ((v - min_val) / width).floor() as u32;
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

    (bins, mapper)
}

fn bin_categorical(values: &[i32]) -> (Vec<u32>, BinMapper) {
    // Collect categories in stable order (sorted) and compute frequencies.
    let mut freqs: BTreeMap<i32, usize> = BTreeMap::new();
    for &v in values {
        *freqs.entry(v).or_default() += 1;
    }

    let categories: Vec<i32> = freqs.keys().copied().collect();

    // Find most frequent category bin index.
    let mut most_freq_category = 0i32;
    let mut most_freq_count = 0usize;
    for (&cat, &count) in freqs.iter() {
        if count > most_freq_count {
            most_freq_count = count;
            most_freq_category = cat;
        }
    }

    let sparse_rate = 0.0;
    let most_freq_bin = categories
        .iter()
        .position(|&c| c == most_freq_category)
        .unwrap_or(0) as u32;

    let mapper = BinMapper::categorical(categories, MissingType::None, 0, most_freq_bin, sparse_rate);

    let bins: Vec<u32> = values
        .iter()
        .map(|&v| mapper.value_to_bin(v as f64))
        .collect();

    (bins, mapper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_dataset_to_gblinear_matrix_is_col_major() {
        let features = vec![
            FeatureColumn::Numeric {
                name: Some("f0".into()),
                values: vec![1.0, 2.0, 3.0],
            },
            FeatureColumn::Numeric {
                name: Some("f1".into()),
                values: vec![10.0, 20.0, 30.0],
            },
        ];
        let ds = Dataset::new(features, vec![0.0, 1.0, 0.0]).unwrap();
        let m = ds.for_gblinear().unwrap();

        assert_eq!(m.num_rows(), 3);
        assert_eq!(m.num_cols(), 2);
        assert_eq!(m.col_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.col_slice(1), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn categorical_to_binned_roundtrips_bins() {
        let features = vec![FeatureColumn::Categorical {
            name: None,
            values: vec![1, 2, 1, 1, 3],
        }];
        let ds = Dataset::new(features, vec![0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let binned = ds.to_binned(16).unwrap();
        assert_eq!(binned.n_rows(), 5);
        assert_eq!(binned.n_features(), 1);
    }
}
