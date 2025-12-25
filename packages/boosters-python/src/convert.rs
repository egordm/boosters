//! Configuration conversion layer from Python to core Rust types.
//!
//! This module implements conversion from PyO3 config types to the core
//! `boosters` crate types. Conversion is lazy—it happens at fit-time,
//! not config construction. This keeps the Python layer simple and lets
//! Rust handle full validation.
//!
//! **Note**: Full GBDTConfig and GBLinearConfig conversion will be implemented
//! in Story 4.3 (GBDTModel.fit()) when the actual training API is built.
//! For now, this module provides the building blocks (objective/metric conversion).

use crate::metrics::PyMetric;
use crate::objectives::PyObjective;

// =============================================================================
// Objective Conversion
// =============================================================================

impl PyObjective {
    /// Convert to core `boosters::training::Objective`.
    pub fn to_core(&self) -> boosters::training::Objective {
        use boosters::training::Objective;

        match self {
            PyObjective::SquaredLoss(_) => Objective::squared(),
            PyObjective::AbsoluteLoss(_) => Objective::absolute(),
            PyObjective::PoissonLoss(_) => Objective::poisson(),
            PyObjective::LogisticLoss(_) => Objective::logistic(),
            PyObjective::HingeLoss(_) => Objective::hinge(),
            PyObjective::HuberLoss(h) => Objective::pseudo_huber(h.delta as f32),
            PyObjective::PinballLoss(p) => {
                if p.alpha.len() == 1 {
                    Objective::quantile(p.alpha[0] as f32)
                } else {
                    Objective::multi_quantile(p.alpha.iter().map(|&a| a as f32).collect())
                }
            }
            PyObjective::ArctanLoss(_) => {
                // Arctan loss maps to pseudo-huber with small delta
                // (approximates arctan behavior for small values)
                Objective::pseudo_huber(0.1)
            }
            PyObjective::SoftmaxLoss(s) => Objective::softmax(s.n_classes as usize),
            PyObjective::LambdaRankLoss(_) => {
                // LambdaRank not yet implemented in core - use logistic as placeholder
                // TODO: Add LambdaRank to core objective enum
                Objective::logistic()
            }
        }
    }
}

// =============================================================================
// Metric Conversion
// =============================================================================

impl PyMetric {
    /// Convert to core `boosters::training::Metric`.
    pub fn to_core(&self) -> boosters::training::Metric {
        use boosters::training::Metric;

        match self {
            PyMetric::Rmse(_) => Metric::rmse(),
            PyMetric::Mae(_) => Metric::mae(),
            PyMetric::Mape(_) => Metric::mape(),
            PyMetric::LogLoss(_) => Metric::logloss(),
            PyMetric::Auc(_) => Metric::auc(),
            PyMetric::Accuracy(_) => Metric::accuracy(),
            PyMetric::Ndcg(_) => {
                // NDCG not yet implemented in core - use rmse as placeholder
                // TODO: Add NDCG to core metric enum
                Metric::rmse()
            }
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
    fn test_objective_conversions() {
        // Test parameterless objectives
        assert!(matches!(
            PyObjective::SquaredLoss(crate::objectives::PySquaredLoss).to_core(),
            boosters::training::Objective::SquaredLoss(_)
        ));

        assert!(matches!(
            PyObjective::LogisticLoss(crate::objectives::PyLogisticLoss).to_core(),
            boosters::training::Objective::LogisticLoss(_)
        ));
    }

    #[test]
    fn test_metric_conversions() {
        assert!(matches!(
            PyMetric::Rmse(crate::metrics::PyRmse).to_core(),
            boosters::training::Metric::Rmse(_)
        ));

        assert!(matches!(
            PyMetric::Auc(crate::metrics::PyAuc).to_core(),
            boosters::training::Metric::Auc(_)
        ));
    }

    #[test]
    fn test_pinball_single_alpha() {
        let py_objective = PyObjective::PinballLoss(crate::objectives::PyPinballLoss {
            alpha: vec![0.5],
        });

        let core = py_objective.to_core();
        assert!(matches!(core, boosters::training::Objective::PinballLoss(_)));
    }

    #[test]
    fn test_pinball_multi_alpha() {
        let py_objective = PyObjective::PinballLoss(crate::objectives::PyPinballLoss {
            alpha: vec![0.1, 0.5, 0.9],
        });

        let core = py_objective.to_core();
        assert!(matches!(core, boosters::training::Objective::PinballLoss(_)));
    }

    #[test]
    fn test_softmax_conversion() {
        let py_objective = PyObjective::SoftmaxLoss(crate::objectives::PySoftmaxLoss {
            n_classes: 5,
        });

        let core = py_objective.to_core();
        assert!(matches!(core, boosters::training::Objective::SoftmaxLoss(_)));
    }

    #[test]
    fn test_huber_conversion() {
        let py_objective =
            PyObjective::HuberLoss(crate::objectives::PyHuberLoss { delta: 2.0 });

        let core = py_objective.to_core();
        assert!(matches!(
            core,
            boosters::training::Objective::PseudoHuberLoss(_)
        ));
    }
}
