//! XGBoost JSON model loader.
//!
//! Parses XGBoost >= 2.0 JSON format. These are "foreign types" used only for parsing;
//! conversion to native booste-rs types will be added later.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use serde_with::{DisplayFromStr, OneOrMany, serde_as};

// =============================================================================
// Custom deserializers for XGBoost-specific formats
// =============================================================================

fn deserialize_base_score<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error as SerdeError;

    let value = Value::deserialize(deserializer)?;
    // Normalize the value by unwrapping arrays and stringified arrays to a scalar
    let mut cur = value;
    loop {
        match cur {
            Value::Number(n) => {
                return n
                    .as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| SerdeError::custom("invalid number"));
            }
            Value::String(s) => {
                // direct parse
                if let Ok(f) = s.parse::<f32>() {
                    return Ok(f);
                }
                // bracketed like "[1.5E0]"
                let t = s.trim();
                if t.starts_with('[') && t.ends_with(']') {
                    let inner = &t[1..t.len() - 1];
                    if let Ok(f) = inner.parse::<f32>() {
                        return Ok(f);
                    }
                }
                // parse as JSON array string
                if let Ok(arr) = serde_json::from_str::<Vec<Value>>(&s) {
                    if arr.is_empty() {
                        return Err(SerdeError::custom("empty array"));
                    }
                    cur = arr.into_iter().next().unwrap();
                    continue;
                }
                return Err(SerdeError::custom(format!(
                    "cannot parse base_score from string: {}",
                    s
                )));
            }
            Value::Array(ref arr) => {
                if arr.is_empty() {
                    return Err(SerdeError::custom("empty array"));
                }
                cur = arr[0].clone();
                continue;
            }
            _ => {
                return Err(SerdeError::custom(
                    "base_score must be number, string, or array",
                ));
            }
        }
    }
}

fn deserialize_bool_any<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error as SerdeError;

    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Bool(b) => Ok(b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                return Ok(i != 0);
            }
            if let Some(f) = n.as_f64() {
                return Ok(f != 0.0);
            }
            Err(SerdeError::custom("invalid number for bool"))
        }
        Value::String(s) => {
            let t = s.trim();
            if t.eq_ignore_ascii_case("true") || t == "1" {
                return Ok(true);
            }
            if t.eq_ignore_ascii_case("false") || t == "0" {
                return Ok(false);
            }
            Err(SerdeError::custom(format!(
                "cannot parse bool from string: {}",
                s
            )))
        }
        _ => Err(SerdeError::custom("unsupported type for bool")),
    }
}

// =============================================================================
// Default value helpers for serde
// =============================================================================

fn default_scale_pos_weight() -> f32 {
    1.0
}
fn default_max_delta_step() -> f32 {
    0.7
}
fn default_tweedie_variance_power() -> f32 {
    1.5
}
fn default_num_pairsample() -> i64 {
    -1
}
fn default_fix_list_weight() -> f32 {
    0.0
}
fn default_lambdarank_num_pair_per_sample() -> i64 {
    -1
}
fn default_lambdarank_pair_method() -> String {
    "topk".to_string()
}
fn default_lambdarank_unbiased() -> bool {
    false
}
fn default_lambdarank_bias_norm() -> f64 {
    1.0
}
fn default_ndcg_exp_gain() -> bool {
    true
}
fn default_aft_loss_distribution() -> String {
    "normal".to_string()
}
fn default_aft_loss_distribution_scale() -> f32 {
    1.0
}
fn default_num_target() -> i64 {
    1
}
fn default_boost_from_average() -> bool {
    true
}
fn default_num_class() -> i64 {
    1
}

// =============================================================================
// Tree / model level definitions
// =============================================================================

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeParam {
    #[serde_as(as = "DisplayFromStr")]
    pub num_nodes: i64,
    #[serde_as(as = "DisplayFromStr")]
    pub size_leaf_vector: i64,
    #[serde_as(as = "DisplayFromStr")]
    pub num_feature: i64,
    #[serde_as(as = "DisplayFromStr")]
    pub num_deleted: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub tree_param: TreeParam,
    pub id: i32,
    pub loss_changes: Vec<f64>,
    pub sum_hessian: Vec<f64>,
    pub base_weights: Vec<f32>,
    pub left_children: Vec<i32>,
    pub right_children: Vec<i32>,
    pub parents: Vec<i32>,
    pub split_indices: Vec<i32>,
    pub split_conditions: Vec<f32>,
    pub split_type: Vec<i32>,
    pub default_left: Vec<i32>,
    pub categories: Vec<i32>,
    pub categories_nodes: Vec<i32>,
    pub categories_segments: Vec<i32>,
    pub categories_sizes: Vec<i32>,
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBTreeModelParam {
    #[serde_as(as = "DisplayFromStr")]
    pub num_trees: i64,
    #[serde_as(as = "DisplayFromStr")]
    pub num_parallel_tree: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrees {
    pub trees: Vec<Tree>,
    pub tree_info: Vec<i32>,
    pub gbtree_model_param: GBTreeModelParam,
}

// =============================================================================
// Gradient booster variants (gbtree | gblinear | dart)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbLinearModel {
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBTreeDefinition {
    pub name: String,
    pub model: ModelTrees,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "lowercase")]
pub enum GradientBooster {
    Gbtree {
        model: ModelTrees,
    },
    Gblinear {
        model: GbLinearModel,
    },
    Dart {
        gbtree: GBTreeDefinition,
        weight_drop: Vec<f32>,
    },
}

// =============================================================================
// Objective / learner-level definitions
// =============================================================================

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegLossParam {
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_scale_pos_weight")]
    pub scale_pos_weight: f32,
}

impl Default for RegLossParam {
    fn default() -> Self {
        Self {
            scale_pos_weight: 1.0,
        }
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonRegressionParam {
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_max_delta_step")]
    pub max_delta_step: f32,
}

impl Default for PoissonRegressionParam {
    fn default() -> Self {
        Self {
            max_delta_step: 0.7,
        }
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweedieRegressionParam {
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_tweedie_variance_power")]
    pub tweedie_variance_power: f32,
}

impl Default for TweedieRegressionParam {
    fn default() -> Self {
        Self {
            tweedie_variance_power: 1.5,
        }
    }
}

#[serde_as]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantileLossParam {
    #[serde_as(as = "OneOrMany<DisplayFromStr>")]
    #[serde(default)]
    pub quantle_alpha: Vec<f32>,
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxMulticlassParam {
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_num_class")]
    pub num_class: i64,
}

impl Default for SoftmaxMulticlassParam {
    fn default() -> Self {
        Self { num_class: 1 }
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaRankParam {
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_num_pairsample")]
    pub num_pairsample: i64,
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_fix_list_weight")]
    pub fix_list_weight: f32,
}

impl Default for LambdaRankParam {
    fn default() -> Self {
        Self {
            num_pairsample: -1,
            fix_list_weight: 0.0,
        }
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdarankParam {
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_lambdarank_num_pair_per_sample")]
    pub lambdarank_num_pair_per_sample: i64,
    #[serde(default = "default_lambdarank_pair_method")]
    pub lambdarank_pair_method: String,
    #[serde(deserialize_with = "deserialize_bool_any")]
    #[serde(default = "default_lambdarank_unbiased")]
    pub lambdarank_unbiased: bool,
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_lambdarank_bias_norm")]
    pub lambdarank_bias_norm: f64,
    #[serde(deserialize_with = "deserialize_bool_any")]
    #[serde(default = "default_ndcg_exp_gain")]
    pub ndcg_exp_gain: bool,
}

impl Default for LambdarankParam {
    fn default() -> Self {
        Self {
            lambdarank_num_pair_per_sample: -1,
            lambdarank_pair_method: "topk".to_string(),
            lambdarank_unbiased: false,
            lambdarank_bias_norm: 1.0,
            ndcg_exp_gain: true,
        }
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AftLossParam {
    #[serde(default = "default_aft_loss_distribution")]
    pub aft_loss_distribution: String,
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_aft_loss_distribution_scale")]
    pub aft_loss_distribution_scale: f32,
}

impl Default for AftLossParam {
    fn default() -> Self {
        Self {
            aft_loss_distribution: "normal".to_string(),
            aft_loss_distribution_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureType {
    #[serde(rename = "float", alias = "float32", alias = "f")]
    Float,
    #[serde(rename = "int", alias = "i")]
    Int,
    #[serde(rename = "indicator")]
    Indicator,
    #[serde(rename = "q", alias = "quantitative")]
    Quantitative,
    #[serde(rename = "c", alias = "categorical")]
    Categorical,
}

impl std::fmt::Display for FeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureType::Float => write!(f, "float"),
            FeatureType::Int => write!(f, "int"),
            FeatureType::Indicator => write!(f, "i"),
            FeatureType::Quantitative => write!(f, "q"),
            FeatureType::Categorical => write!(f, "c"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "name")]
pub enum Objective {
    #[serde(rename = "reg:squarederror")]
    RegSquaredError {
        #[serde(default)]
        reg_loss_param: RegLossParam,
    },
    #[serde(rename = "reg:pseudohubererror")]
    RegPseudohuberError {
        #[serde(default)]
        reg_loss_param: RegLossParam,
    },
    #[serde(rename = "reg:squaredlogerror")]
    RegSquaredLogError {
        #[serde(default)]
        reg_loss_param: RegLossParam,
    },
    #[serde(rename = "reg:linear")]
    RegLinear {
        #[serde(default)]
        reg_loss_param: RegLossParam,
    },
    #[serde(rename = "reg:logistic")]
    RegLogistic {
        #[serde(default)]
        reg_loss_param: RegLossParam,
    },
    #[serde(rename = "binary:logistic")]
    BinaryLogistic {
        #[serde(default)]
        reg_loss_param: RegLossParam,
    },
    #[serde(rename = "binary:logitraw")]
    BinaryLogitRaw {
        #[serde(default)]
        reg_loss_param: RegLossParam,
    },
    #[serde(rename = "count:poisson")]
    CountPoisson {
        #[serde(default)]
        poisson_regression_param: PoissonRegressionParam,
    },
    #[serde(rename = "reg:tweedie")]
    RegTweedie {
        #[serde(default)]
        tweedie_regression_param: TweedieRegressionParam,
    },
    #[serde(rename = "reg:absoluteerror")]
    RegAbsoluteError,
    #[serde(rename = "reg:quantileerror")]
    RegQuantileError {
        #[serde(default)]
        quantile_loss_param: QuantileLossParam,
    },
    #[serde(rename = "survival:cox")]
    SurvivalCox,
    #[serde(rename = "reg:gamma")]
    RegGamma,
    #[serde(rename = "multi:softprob")]
    MultiSoftprob {
        #[serde(default)]
        softmax_multiclass_param: SoftmaxMulticlassParam,
    },
    #[serde(rename = "multi:softmax")]
    MultiSoftmax {
        #[serde(default)]
        softmax_multiclass_param: SoftmaxMulticlassParam,
    },
    #[serde(rename = "rank:pairwise")]
    RankPairwise {
        #[serde(default)]
        lambdarank_param: LambdarankParam,
    },
    #[serde(rename = "rank:ndcg")]
    RankNdcg {
        #[serde(default)]
        lambdarank_param: LambdarankParam,
    },
    #[serde(rename = "rank:map")]
    RankMap {
        #[serde(default)]
        lambda_rank_param: LambdaRankParam,
    },
    #[serde(rename = "survival:aft")]
    SurvivalAft {
        #[serde(default)]
        aft_loss_param: AftLossParam,
    },
    #[serde(rename = "binary:hinge")]
    BinaryHinge,
}

impl Objective {
    /// Get the objective name as it appears in XGBoost JSON.
    pub fn name(&self) -> &'static str {
        match self {
            Objective::RegSquaredError { .. } => "reg:squarederror",
            Objective::RegPseudohuberError { .. } => "reg:pseudohubererror",
            Objective::RegSquaredLogError { .. } => "reg:squaredlogerror",
            Objective::RegLinear { .. } => "reg:linear",
            Objective::RegLogistic { .. } => "reg:logistic",
            Objective::BinaryLogistic { .. } => "binary:logistic",
            Objective::BinaryLogitRaw { .. } => "binary:logitraw",
            Objective::CountPoisson { .. } => "count:poisson",
            Objective::RegTweedie { .. } => "reg:tweedie",
            Objective::RegAbsoluteError => "reg:absoluteerror",
            Objective::RegQuantileError { .. } => "reg:quantileerror",
            Objective::SurvivalCox => "survival:cox",
            Objective::RegGamma => "reg:gamma",
            Objective::MultiSoftprob { .. } => "multi:softprob",
            Objective::MultiSoftmax { .. } => "multi:softmax",
            Objective::RankPairwise { .. } => "rank:pairwise",
            Objective::RankNdcg { .. } => "rank:ndcg",
            Objective::RankMap { .. } => "rank:map",
            Objective::SurvivalAft { .. } => "survival:aft",
            Objective::BinaryHinge => "binary:hinge",
        }
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerModelParam {
    #[serde(deserialize_with = "deserialize_base_score")]
    pub base_score: f32,
    #[serde(rename = "num_class")]
    #[serde_as(as = "DisplayFromStr")]
    pub n_class: i64,
    #[serde(rename = "num_feature")]
    #[serde_as(as = "DisplayFromStr")]
    pub n_features: i64,
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_num_target")]
    pub num_target: i64,
    #[serde(deserialize_with = "deserialize_bool_any")]
    #[serde(default = "default_boost_from_average")]
    pub boost_from_average: bool,
}

impl Default for LearnerModelParam {
    fn default() -> Self {
        Self {
            base_score: 0.5,
            n_class: 1,
            n_features: 0,
            num_target: 1,
            boost_from_average: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Learner {
    #[serde(default)]
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub feature_types: Vec<FeatureType>,
    pub gradient_booster: GradientBooster,
    pub objective: Objective,
    pub learner_model_param: LearnerModelParam,
}

// =============================================================================
// Top-level XGBoost model
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XgbModel {
    pub version: [u32; 3],
    pub learner: Learner,
}

impl XgbModel {
    /// Load a model from a JSON file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::compat::xgboost::XgbModel;
    ///
    /// let model = XgbModel::from_file("model.json")?;
    /// let forest = model.to_forest()?;
    /// ```
    pub fn from_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }

    /// Parse a model from a serde_json Value.
    pub fn from_value(value: &Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value.clone())
    }
}

// =============================================================================
// Public API
// =============================================================================

// TODO: Implement conversion to native types once they exist

pub fn test_parse_model(value: &Value) -> Result<XgbModel, serde_json::Error> {
    serde_json::from_value(value.clone())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn base_score_parses_number_string_array_and_bracketed() {
        let v_num = json!({"base_score": 1.5, "num_class": "1", "num_feature": "0"});
        let p_num: LearnerModelParam = serde_json::from_value(v_num).unwrap();
        assert_eq!(p_num.base_score, 1.5);

        let v_str = json!({"base_score": "1.5", "num_class": "1", "num_feature": "0"});
        let p_str: LearnerModelParam = serde_json::from_value(v_str).unwrap();
        assert_eq!(p_str.base_score, 1.5);

        let v_arr = json!({"base_score": [1.5], "num_class": "1", "num_feature": "0"});
        let p_arr: LearnerModelParam = serde_json::from_value(v_arr).unwrap();
        assert_eq!(p_arr.base_score, 1.5);

        let v_bracketed = json!({"base_score": "[1.5E0]", "num_class": "1", "num_feature": "0"});
        let p_bracketed: LearnerModelParam = serde_json::from_value(v_bracketed).unwrap();
        assert_eq!(p_bracketed.base_score, 1.5);
    }

    #[test]
    fn boost_from_average_accepts_various_types() {
        let v_bool = json!({"base_score": 0.5, "num_class": "1", "num_feature": "0", "boost_from_average": true});
        let p_bool: LearnerModelParam = serde_json::from_value(v_bool).unwrap();
        assert!(p_bool.boost_from_average);

        let v_int = json!({"base_score": 0.5, "num_class": "1", "num_feature": "0", "boost_from_average": 1});
        let p_int: LearnerModelParam = serde_json::from_value(v_int).unwrap();
        assert!(p_int.boost_from_average);

        let v_str = json!({"base_score": 0.5, "num_class": "1", "num_feature": "0", "boost_from_average": "1"});
        let p_str: LearnerModelParam = serde_json::from_value(v_str).unwrap();
        assert!(p_str.boost_from_average);

        let v_false_str = json!({"base_score": 0.5, "num_class": "1", "num_feature": "0", "boost_from_average": "0"});
        let p_false_str: LearnerModelParam = serde_json::from_value(v_false_str).unwrap();
        assert!(!p_false_str.boost_from_average);
    }

    #[test]
    fn lambdarank_flags_accept_string_and_number() {
        let v1 = json!({"lambdarank_num_pair_per_sample": "-1", "fix_list_weight": "0.0", "lambdarank_pair_method": "topk", "lambdarank_unbiased": "1", "lambdarank_bias_norm": "1.0", "ndcg_exp_gain": "1"});
        let p1: LambdarankParam = serde_json::from_value(v1).unwrap();
        assert!(p1.lambdarank_unbiased);
        assert!(p1.ndcg_exp_gain);

        let v0 = json!({"lambdarank_num_pair_per_sample": "-1", "fix_list_weight": "0.0", "lambdarank_pair_method": "topk", "lambdarank_unbiased": 0, "lambdarank_bias_norm": "1.0", "ndcg_exp_gain": 0});
        let p0: LambdarankParam = serde_json::from_value(v0).unwrap();
        assert!(!p0.lambdarank_unbiased);
        assert!(!p0.ndcg_exp_gain);
    }
}
