//! Semantic prediction wrappers.

use ndarray::Array2;

/// What do the prediction values represent?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionKind {
    /// Raw margins / logits / scores produced directly by the model.
    Margin,

    /// Regression-style value (identity transform) or mean parameter
    /// (e.g., exp for Poisson).
    Value,

    /// Probabilities in [0, 1] (binary) or rows that sum to 1 (multiclass).
    Probability,

    /// Predicted class index (0..K-1).
    ///
    /// Stored as f32 for compatibility with existing metric interfaces.
    ClassIndex,

    /// Ranking score (typically margin-like; objective decides).
    RankScore,
}

/// A prediction with explicit semantic meaning.
///
/// The output is stored as `Array2<f32>` with shape `(n_samples, n_groups)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Predictions {
    pub kind: PredictionKind,
    pub output: Array2<f32>,
}

impl Predictions {
    #[inline]
    pub fn raw_margin(output: Array2<f32>) -> Self {
        Self {
            kind: PredictionKind::Margin,
            output,
        }
    }

    #[inline]
    pub fn value(output: Array2<f32>) -> Self {
        Self {
            kind: PredictionKind::Value,
            output,
        }
    }

    #[inline]
    pub fn probability(output: Array2<f32>) -> Self {
        Self {
            kind: PredictionKind::Probability,
            output,
        }
    }

    #[inline]
    pub fn class_index(output: Array2<f32>) -> Self {
        Self {
            kind: PredictionKind::ClassIndex,
            output,
        }
    }

    #[inline]
    pub fn rank_score(output: Array2<f32>) -> Self {
        Self {
            kind: PredictionKind::RankScore,
            output,
        }
    }
}
