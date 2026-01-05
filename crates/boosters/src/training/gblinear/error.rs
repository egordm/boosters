//! GBLinear training errors.

/// Errors that can occur when training a GBLinear model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GBLinearTrainError {
    /// GBLinear does not support categorical features.
    CategoricalFeaturesNotSupported,

    /// NaN was found in target values.
    NaNInTargets { dataset: &'static str },

    /// NaN was found in sample weights.
    NaNInWeights { dataset: &'static str },

    /// Validation set was provided but has no targets.
    ValidationSetMissingTargets,
}

impl std::fmt::Display for GBLinearTrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CategoricalFeaturesNotSupported => {
                write!(f, "GBLinear does not support categorical features")
            }
            Self::NaNInTargets { dataset } => write!(
                f,
                "GBLinear does not support NaNs in targets: found NaN in {dataset} targets"
            ),
            Self::NaNInWeights { dataset } => write!(
                f,
                "GBLinear does not support NaNs in weights: found NaN in {dataset} weights"
            ),
            Self::ValidationSetMissingTargets => {
                write!(f, "Validation set must have targets")
            }
        }
    }
}

impl std::error::Error for GBLinearTrainError {}
