//! Semantic prediction wrappers.

use super::PredictionOutput;

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
#[derive(Debug, Clone, PartialEq)]
pub struct Predictions {
    pub kind: PredictionKind,
    pub output: PredictionOutput,
}

impl Predictions {
    #[inline]
    pub fn raw_margin(output: PredictionOutput) -> Self {
        Self {
            kind: PredictionKind::Margin,
            output,
        }
    }

    #[inline]
    pub fn value(output: PredictionOutput) -> Self {
        Self {
            kind: PredictionKind::Value,
            output,
        }
    }

    #[inline]
    pub fn probability(output: PredictionOutput) -> Self {
        Self {
            kind: PredictionKind::Probability,
            output,
        }
    }

    #[inline]
    pub fn class_index(output: PredictionOutput) -> Self {
        Self {
            kind: PredictionKind::ClassIndex,
            output,
        }
    }

    #[inline]
    pub fn rank_score(output: PredictionOutput) -> Self {
        Self {
            kind: PredictionKind::RankScore,
            output,
        }
    }
}
