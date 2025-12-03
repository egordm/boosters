//! Objective functions and output transformations.
//!
//! The [`Objective`] enum determines how raw model output is transformed
//! for final predictions. For example, binary classification uses sigmoid
//! to convert logits to probabilities.
//!
//! See RFC-0007 for design rationale.

use crate::predict::PredictionOutput;

/// Objective function for output transformation.
///
/// Determines how raw model output is transformed for final predictions.
/// For example, binary classification uses sigmoid to convert logits to probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Objective {
    // --- Regression ---
    /// Squared error loss (no transformation).
    #[default]
    SquaredError,

    /// Absolute error loss (no transformation).
    AbsoluteError,

    /// Tweedie regression.
    Tweedie { variance_power: f32 },

    /// Gamma regression (exp transform).
    Gamma,

    /// Poisson regression (exp transform).
    Poisson,

    // --- Binary Classification ---
    /// Binary logistic (sigmoid transform).
    BinaryLogistic,

    /// Binary logit raw (no transformation, return logits).
    BinaryLogitRaw,

    // --- Multiclass Classification ---
    /// Multiclass softmax (returns class index, not probabilities).
    MultiSoftmax { num_class: u32 },

    /// Multiclass softprob (softmax transform, returns probabilities).
    MultiSoftprob { num_class: u32 },

    // --- Ranking ---
    /// Pairwise ranking.
    RankPairwise,

    /// NDCG ranking.
    RankNdcg,

    /// MAP ranking.
    RankMap,

    // --- Survival ---
    /// Cox proportional hazards (exp transform).
    SurvivalCox,

    // --- Other ---
    /// Custom/unknown objective (no transformation).
    Custom,
}

impl Objective {
    /// Apply the objective transformation to predictions in-place.
    pub fn transform(&self, output: &mut PredictionOutput) {
        match self {
            // No transformation
            Objective::SquaredError
            | Objective::AbsoluteError
            | Objective::BinaryLogitRaw
            | Objective::RankPairwise
            | Objective::RankNdcg
            | Objective::RankMap
            | Objective::Custom => {}

            // Sigmoid for binary classification
            Objective::BinaryLogistic => {
                for val in output.as_mut_slice() {
                    *val = sigmoid(*val);
                }
            }

            // Exp transform
            Objective::Gamma | Objective::Poisson | Objective::SurvivalCox => {
                for val in output.as_mut_slice() {
                    *val = val.exp();
                }
            }

            // Tweedie: exp transform
            Objective::Tweedie { .. } => {
                for val in output.as_mut_slice() {
                    *val = val.exp();
                }
            }

            // Softmax for multiclass probabilities
            Objective::MultiSoftprob { .. } => {
                let num_groups = output.num_groups();
                for row_idx in 0..output.num_rows() {
                    let row = output.row_mut(row_idx);
                    softmax_inplace(row);
                    // Ensure we processed correct number of groups
                    debug_assert_eq!(row.len(), num_groups);
                }
            }

            // Softmax then argmax for class prediction
            Objective::MultiSoftmax { .. } => {
                for row_idx in 0..output.num_rows() {
                    let row = output.row_mut(row_idx);
                    let class_idx = argmax(row);
                    // Store class index in first position
                    row[0] = class_idx as f32;
                    // Zero out the rest (or could resize output, but simpler to keep shape)
                    for val in row.iter_mut().skip(1) {
                        *val = 0.0;
                    }
                }
            }
        }
    }

    /// Whether this objective produces probabilities.
    pub fn produces_probabilities(&self) -> bool {
        matches!(
            self,
            Objective::BinaryLogistic | Objective::MultiSoftprob { .. }
        )
    }

    /// Whether this objective produces class indices.
    pub fn produces_class_indices(&self) -> bool {
        matches!(self, Objective::MultiSoftmax { .. })
    }
}

// =============================================================================
// Transform functions
// =============================================================================

/// Sigmoid function: 1 / (1 + exp(-x))
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax in-place over a slice.
pub fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for val in values.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    // Normalize
    if sum > 0.0 {
        for val in values.iter_mut() {
            *val /= sum;
        }
    }
}

/// Argmax: index of maximum value.
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(2.0) - 0.8807971).abs() < 1e-5);
        assert!((sigmoid(-2.0) - 0.1192029).abs() < 1e-5);
    }

    #[test]
    fn softmax_function() {
        let mut values = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut values);

        // Should sum to 1
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Largest input should have largest probability
        assert!(values[2] > values[1]);
        assert!(values[1] > values[0]);
    }

    #[test]
    fn softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let mut values = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut values);

        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_empty() {
        let mut values: Vec<f32> = vec![];
        softmax_inplace(&mut values);
        assert!(values.is_empty());
    }

    #[test]
    fn argmax_function() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 3.0]), 2);
    }

    #[test]
    fn argmax_empty() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn objective_no_transform() {
        let mut output = PredictionOutput::new(vec![1.0, 2.0], 2, 1);
        Objective::SquaredError.transform(&mut output);
        assert_eq!(output.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn objective_sigmoid() {
        let mut output = PredictionOutput::new(vec![0.0, 2.0], 2, 1);
        Objective::BinaryLogistic.transform(&mut output);

        assert!((output.row(0)[0] - 0.5).abs() < 1e-6);
        assert!((output.row(1)[0] - 0.8807971).abs() < 1e-5);
    }

    #[test]
    fn objective_exp_transform() {
        let mut output = PredictionOutput::new(vec![0.0, 1.0], 2, 1);
        Objective::Poisson.transform(&mut output);

        assert!((output.row(0)[0] - 1.0).abs() < 1e-6); // exp(0) = 1
        assert!((output.row(1)[0] - std::f32::consts::E).abs() < 1e-5); // exp(1) = e
    }

    #[test]
    fn objective_softmax() {
        // 2 rows, 3 classes
        let mut output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], 2, 3);
        Objective::MultiSoftprob { num_class: 3 }.transform(&mut output);

        // First row: should be proper probabilities
        let row0 = output.row(0);
        let sum: f32 = row0.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Second row: uniform (all zeros → equal probs)
        let row1 = output.row(1);
        let sum: f32 = row1.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // All equal
        assert!((row1[0] - row1[1]).abs() < 1e-6);
        assert!((row1[1] - row1[2]).abs() < 1e-6);
    }

    #[test]
    fn objective_multiclass_argmax() {
        let mut output = PredictionOutput::new(vec![1.0, 3.0, 2.0], 1, 3);
        Objective::MultiSoftmax { num_class: 3 }.transform(&mut output);

        // Class 1 had highest score
        assert_eq!(output.row(0)[0], 1.0);
    }

    #[test]
    fn objective_produces_probabilities() {
        assert!(Objective::BinaryLogistic.produces_probabilities());
        assert!(Objective::MultiSoftprob { num_class: 3 }.produces_probabilities());
        assert!(!Objective::SquaredError.produces_probabilities());
        assert!(!Objective::MultiSoftmax { num_class: 3 }.produces_probabilities());
    }

    #[test]
    fn objective_produces_class_indices() {
        assert!(Objective::MultiSoftmax { num_class: 3 }.produces_class_indices());
        assert!(!Objective::MultiSoftprob { num_class: 3 }.produces_class_indices());
        assert!(!Objective::BinaryLogistic.produces_class_indices());
    }
}
