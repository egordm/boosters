//! LightGBM text model format parser.
//!
//! Parses LightGBM's text model format (`.txt` files saved via `save_model()`).
//! This is a line-based format with key=value pairs.

use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Error types
// =============================================================================

/// Error type for LightGBM model parsing.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid value for {field}: {message}")]
    InvalidValue {
        field: &'static str,
        message: String,
    },
    #[error("array size mismatch for {field}: expected {expected}, got {actual}")]
    ArraySizeMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("unexpected end of input while parsing {context}")]
    UnexpectedEnd { context: String },
    #[error("invalid tree format: {0}")]
    InvalidTreeFormat(String),
}

// =============================================================================
// Decision type bitfield
// =============================================================================

/// Missing value handling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingType {
    /// No special missing value handling
    #[default]
    None = 0,
    /// Treat zeros as missing
    Zero = 1,
    /// Treat NaN as missing
    NaN = 2,
}

impl MissingType {
    fn from_bits(bits: u8) -> Self {
        match bits {
            0 => MissingType::None,
            1 => MissingType::Zero,
            2 => MissingType::NaN,
            _ => MissingType::None, // Default for unknown values
        }
    }
}

/// Parsed decision type from LightGBM's bitfield.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionType {
    /// True if this is a categorical split
    pub is_categorical: bool,
    /// True if missing values go left
    pub default_left: bool,
    /// Missing value handling mode
    pub missing_type: MissingType,
}

impl DecisionType {
    /// Parse from LightGBM's decision_type bitfield.
    ///
    /// Bit layout:
    /// - Bit 0: categorical flag (1 = categorical)
    /// - Bit 1: default_left flag (1 = left)
    /// - Bits 2-3: missing type (0=None, 1=Zero, 2=NaN)
    pub fn from_i8(value: i8) -> Self {
        let v = value as u8;
        DecisionType {
            is_categorical: (v & 1) != 0,
            default_left: (v & 2) != 0,
            missing_type: MissingType::from_bits((v >> 2) & 3),
        }
    }
}

// =============================================================================
// Parsed tree structure
// =============================================================================

/// A parsed LightGBM tree.
#[derive(Debug, Clone)]
pub struct LgbTree {
    /// Number of leaves in this tree
    pub num_leaves: usize,
    /// Number of categorical features used in splits
    pub num_cat: usize,
    /// Feature index for each internal node (size: num_leaves - 1)
    pub split_feature: Vec<i32>,
    /// Split gain for each internal node (size: num_leaves - 1)
    pub split_gain: Vec<f32>,
    /// Threshold for each internal node (size: num_leaves - 1)
    pub threshold: Vec<f64>,
    /// Decision type bitfield for each internal node (size: num_leaves - 1)
    pub decision_type: Vec<i8>,
    /// Left child index for each internal node (negative = leaf) (size: num_leaves - 1)
    pub left_child: Vec<i32>,
    /// Right child index for each internal node (negative = leaf) (size: num_leaves - 1)
    pub right_child: Vec<i32>,
    /// Output value for each leaf (size: num_leaves)
    pub leaf_value: Vec<f64>,
    /// Sample count for each leaf (size: num_leaves)
    pub leaf_count: Vec<i32>,
    /// Weight (sum of hessians) for each leaf (size: num_leaves)
    pub leaf_weight: Vec<f64>,
    /// Shrinkage (learning rate) applied to this tree
    pub shrinkage: f64,
    /// Whether this tree has linear models at leaves
    pub is_linear: bool,
    /// Categorical split boundaries (size: num_cat + 1 if num_cat > 0)
    pub cat_boundaries: Vec<i32>,
    /// Categorical split threshold bitset
    pub cat_threshold: Vec<u32>,
}

impl Default for LgbTree {
    fn default() -> Self {
        Self {
            num_leaves: 0,
            num_cat: 0,
            split_feature: Vec::new(),
            split_gain: Vec::new(),
            threshold: Vec::new(),
            decision_type: Vec::new(),
            left_child: Vec::new(),
            right_child: Vec::new(),
            leaf_value: Vec::new(),
            leaf_count: Vec::new(),
            leaf_weight: Vec::new(),
            shrinkage: 1.0,
            is_linear: false,
            cat_boundaries: Vec::new(),
            cat_threshold: Vec::new(),
        }
    }
}

// =============================================================================
// Objective parsing
// =============================================================================

/// Parsed objective function information.
#[derive(Debug, Clone)]
pub enum LgbObjective {
    /// Regression with L2 loss
    Regression,
    /// Regression with L1 loss
    RegressionL1,
    /// Binary classification with logloss
    Binary { sigmoid: f64 },
    /// Multiclass classification with softmax
    Multiclass { num_class: usize },
    /// One-vs-all multiclass
    MulticlassOva { num_class: usize },
    /// Unknown objective (raw string preserved)
    Unknown(String),
}

impl LgbObjective {
    /// Parse from LightGBM objective string.
    ///
    /// Examples:
    /// - "regression"
    /// - "binary sigmoid:1"
    /// - "multiclass num_class:3"
    pub fn parse(s: &str) -> Self {
        let parts: Vec<&str> = s.split_whitespace().collect();
        let name = parts.first().map(|s| *s).unwrap_or("");

        match name {
            "regression" => LgbObjective::Regression,
            "regression_l1" => LgbObjective::RegressionL1,
            "binary" => {
                // Parse sigmoid parameter if present
                let sigmoid = parts
                    .iter()
                    .find_map(|p| {
                        p.strip_prefix("sigmoid:")
                            .and_then(|v| v.parse::<f64>().ok())
                    })
                    .unwrap_or(1.0);
                LgbObjective::Binary { sigmoid }
            }
            "multiclass" => {
                let num_class = parts
                    .iter()
                    .find_map(|p| {
                        p.strip_prefix("num_class:")
                            .and_then(|v| v.parse::<usize>().ok())
                    })
                    .unwrap_or(2);
                LgbObjective::Multiclass { num_class }
            }
            "multiclassova" => {
                let num_class = parts
                    .iter()
                    .find_map(|p| {
                        p.strip_prefix("num_class:")
                            .and_then(|v| v.parse::<usize>().ok())
                    })
                    .unwrap_or(2);
                LgbObjective::MulticlassOva { num_class }
            }
            _ => LgbObjective::Unknown(s.to_string()),
        }
    }
}

// =============================================================================
// Model header
// =============================================================================

/// Parsed LightGBM model header.
#[derive(Debug, Clone)]
pub struct LgbHeader {
    /// Model format version (e.g., "v4")
    pub version: String,
    /// Number of classes (1 for regression, 2+ for classification)
    pub num_class: usize,
    /// Number of trees per boosting iteration
    pub num_tree_per_iteration: usize,
    /// Label column index
    pub label_index: i32,
    /// Maximum feature index used (0-based)
    pub max_feature_idx: usize,
    /// Objective function
    pub objective: Option<LgbObjective>,
    /// Whether to average output across iterations
    pub average_output: bool,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Feature metadata (min:max for numerical, categories for categorical)
    pub feature_infos: Vec<String>,
}

impl Default for LgbHeader {
    fn default() -> Self {
        Self {
            version: String::new(),
            num_class: 1,
            num_tree_per_iteration: 1,
            label_index: 0,
            max_feature_idx: 0,
            objective: None,
            average_output: false,
            feature_names: Vec::new(),
            feature_infos: Vec::new(),
        }
    }
}

// =============================================================================
// Full model
// =============================================================================

/// A parsed LightGBM model.
#[derive(Debug, Clone)]
pub struct LgbModel {
    /// Model header with metadata
    pub header: LgbHeader,
    /// All trees in the model
    pub trees: Vec<LgbTree>,
}

impl LgbModel {
    /// Load a model from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ParseError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_string(&content)
    }

    /// Parse a model from a string.
    pub fn from_string(content: &str) -> Result<Self, ParseError> {
        let mut lines = content.lines().peekable();

        // Parse header
        let header = parse_header(&mut lines)?;

        // Parse trees
        let mut trees = Vec::new();
        while let Some(line) = lines.peek() {
            if line.starts_with("Tree=") {
                lines.next(); // consume the Tree=N line
                let tree = parse_tree(&mut lines)?;
                trees.push(tree);
            } else if *line == "end of trees" {
                break;
            } else {
                lines.next(); // skip other lines (e.g., tree_sizes)
            }
        }

        Ok(LgbModel { header, trees })
    }

    /// Number of trees in the model.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of classes (1 for regression/binary, k for k-class multiclass).
    pub fn num_class(&self) -> usize {
        self.header.num_class
    }

    /// Number of output groups (1 for regression/binary, num_class for multiclass).
    pub fn num_groups(&self) -> usize {
        if self.header.num_class <= 1 {
            1
        } else {
            self.header.num_class
        }
    }

    /// Number of features.
    pub fn num_features(&self) -> usize {
        self.header.max_feature_idx + 1
    }
}

// =============================================================================
// Parsing helpers
// =============================================================================

/// Parse header section until first Tree= line.
fn parse_header(
    lines: &mut std::iter::Peekable<std::str::Lines>,
) -> Result<LgbHeader, ParseError> {
    let mut header = LgbHeader::default();
    let mut kv = HashMap::new();

    // Skip model type line (e.g., "tree")
    if let Some(line) = lines.next() {
        if !line.contains('=') {
            // This is the model type line, skip it
        } else {
            // This line has key=value, parse it
            if let Some((k, v)) = line.split_once('=') {
                kv.insert(k.to_string(), v.to_string());
            }
        }
    }

    // Parse key=value pairs until Tree= or empty line
    while let Some(line) = lines.peek() {
        if line.starts_with("Tree=") || line.starts_with("tree_sizes=") || line.is_empty() {
            if line.starts_with("tree_sizes=") {
                lines.next(); // consume tree_sizes line
            }
            break;
        }

        let line = lines.next().unwrap();
        if let Some((key, value)) = line.split_once('=') {
            kv.insert(key.to_string(), value.to_string());
        } else if line == "average_output" {
            header.average_output = true;
        }
    }

    // Extract required fields
    header.version = kv.get("version").cloned().unwrap_or_default();

    header.num_class = kv
        .get("num_class")
        .and_then(|v| v.parse().ok())
        .ok_or(ParseError::MissingField("num_class"))?;

    header.num_tree_per_iteration = kv
        .get("num_tree_per_iteration")
        .and_then(|v| v.parse().ok())
        .unwrap_or(header.num_class.max(1));

    header.label_index = kv
        .get("label_index")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    header.max_feature_idx = kv
        .get("max_feature_idx")
        .and_then(|v| v.parse().ok())
        .ok_or(ParseError::MissingField("max_feature_idx"))?;

    if let Some(obj) = kv.get("objective") {
        header.objective = Some(LgbObjective::parse(obj));
    }

    if let Some(names) = kv.get("feature_names") {
        header.feature_names = names.split(' ').map(|s| s.to_string()).collect();
    }

    if let Some(infos) = kv.get("feature_infos") {
        header.feature_infos = infos.split(' ').map(|s| s.to_string()).collect();
    }

    Ok(header)
}

/// Parse a single tree section.
fn parse_tree(lines: &mut std::iter::Peekable<std::str::Lines>) -> Result<LgbTree, ParseError> {
    let mut tree = LgbTree::default();
    let mut kv = HashMap::new();

    // Parse key=value pairs until next Tree= or end
    while let Some(line) = lines.peek() {
        if line.starts_with("Tree=") || line.starts_with("end of trees") || line.is_empty() {
            if line.is_empty() {
                lines.next(); // consume empty line
            }
            break;
        }

        let line = lines.next().unwrap();
        if let Some((key, value)) = line.split_once('=') {
            kv.insert(key.to_string(), value.to_string());
        }
    }

    // Parse required fields
    tree.num_leaves = kv
        .get("num_leaves")
        .and_then(|v| v.parse().ok())
        .ok_or(ParseError::MissingField("num_leaves"))?;

    tree.num_cat = kv
        .get("num_cat")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    tree.shrinkage = kv
        .get("shrinkage")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);

    tree.is_linear = kv
        .get("is_linear")
        .and_then(|v| v.parse::<i32>().ok())
        .map(|v| v != 0)
        .unwrap_or(false);

    // Single-leaf tree has no splits
    if tree.num_leaves <= 1 {
        tree.leaf_value = kv
            .get("leaf_value")
            .map(|v| parse_double_array(v))
            .transpose()?
            .unwrap_or_else(|| vec![0.0]);
        return Ok(tree);
    }

    let num_splits = tree.num_leaves - 1;

    // Parse arrays
    tree.split_feature = kv
        .get("split_feature")
        .map(|v| parse_int_array(v))
        .transpose()?
        .ok_or(ParseError::MissingField("split_feature"))?;
    validate_array_size("split_feature", &tree.split_feature, num_splits)?;

    tree.split_gain = kv
        .get("split_gain")
        .map(|v| parse_float_array(v))
        .transpose()?
        .unwrap_or_else(|| vec![0.0; num_splits]);

    tree.threshold = kv
        .get("threshold")
        .map(|v| parse_double_array(v))
        .transpose()?
        .ok_or(ParseError::MissingField("threshold"))?;
    validate_array_size("threshold", &tree.threshold, num_splits)?;

    tree.decision_type = kv
        .get("decision_type")
        .map(|v| parse_i8_array(v))
        .transpose()?
        .unwrap_or_else(|| vec![0; num_splits]);

    tree.left_child = kv
        .get("left_child")
        .map(|v| parse_int_array(v))
        .transpose()?
        .ok_or(ParseError::MissingField("left_child"))?;
    validate_array_size("left_child", &tree.left_child, num_splits)?;

    tree.right_child = kv
        .get("right_child")
        .map(|v| parse_int_array(v))
        .transpose()?
        .ok_or(ParseError::MissingField("right_child"))?;
    validate_array_size("right_child", &tree.right_child, num_splits)?;

    tree.leaf_value = kv
        .get("leaf_value")
        .map(|v| parse_double_array(v))
        .transpose()?
        .ok_or(ParseError::MissingField("leaf_value"))?;
    validate_array_size("leaf_value", &tree.leaf_value, tree.num_leaves)?;

    tree.leaf_count = kv
        .get("leaf_count")
        .map(|v| parse_int_array(v))
        .transpose()?
        .unwrap_or_else(|| vec![0; tree.num_leaves]);

    tree.leaf_weight = kv
        .get("leaf_weight")
        .map(|v| parse_double_array(v))
        .transpose()?
        .unwrap_or_else(|| vec![0.0; tree.num_leaves]);

    // Parse categorical data if present
    if tree.num_cat > 0 {
        tree.cat_boundaries = kv
            .get("cat_boundaries")
            .map(|v| parse_int_array(v))
            .transpose()?
            .unwrap_or_default();

        tree.cat_threshold = kv
            .get("cat_threshold")
            .map(|v| parse_u32_array(v))
            .transpose()?
            .unwrap_or_default();
    }

    Ok(tree)
}

fn parse_int_array(s: &str) -> Result<Vec<i32>, ParseError> {
    s.split_whitespace()
        .map(|v| {
            v.parse().map_err(|_| ParseError::InvalidValue {
                field: "array",
                message: format!("invalid integer: {}", v),
            })
        })
        .collect()
}

fn parse_i8_array(s: &str) -> Result<Vec<i8>, ParseError> {
    s.split_whitespace()
        .map(|v| {
            v.parse().map_err(|_| ParseError::InvalidValue {
                field: "array",
                message: format!("invalid i8: {}", v),
            })
        })
        .collect()
}

fn parse_u32_array(s: &str) -> Result<Vec<u32>, ParseError> {
    s.split_whitespace()
        .map(|v| {
            v.parse().map_err(|_| ParseError::InvalidValue {
                field: "array",
                message: format!("invalid u32: {}", v),
            })
        })
        .collect()
}

fn parse_float_array(s: &str) -> Result<Vec<f32>, ParseError> {
    s.split_whitespace()
        .map(|v| {
            v.parse().map_err(|_| ParseError::InvalidValue {
                field: "array",
                message: format!("invalid float: {}", v),
            })
        })
        .collect()
}

fn parse_double_array(s: &str) -> Result<Vec<f64>, ParseError> {
    s.split_whitespace()
        .map(|v| {
            v.parse().map_err(|_| ParseError::InvalidValue {
                field: "array",
                message: format!("invalid double: {}", v),
            })
        })
        .collect()
}

fn validate_array_size<T>(field: &'static str, arr: &[T], expected: usize) -> Result<(), ParseError> {
    if arr.len() != expected {
        return Err(ParseError::ArraySizeMismatch {
            field,
            expected,
            actual: arr.len(),
        });
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_decision_type() {
        // Numerical, default right, no missing handling
        let dt = DecisionType::from_i8(0);
        assert!(!dt.is_categorical);
        assert!(!dt.default_left);
        assert_eq!(dt.missing_type, MissingType::None);

        // Numerical, default left, no missing handling
        let dt = DecisionType::from_i8(2);
        assert!(!dt.is_categorical);
        assert!(dt.default_left);
        assert_eq!(dt.missing_type, MissingType::None);

        // Categorical, default right
        let dt = DecisionType::from_i8(1);
        assert!(dt.is_categorical);
        assert!(!dt.default_left);

        // Numerical, default right, missing=Zero (bits 2-3 = 01 = 1 << 2 = 4)
        let dt = DecisionType::from_i8(4);
        assert!(!dt.is_categorical);
        assert!(!dt.default_left);
        assert_eq!(dt.missing_type, MissingType::Zero);

        // Numerical, default right, missing=NaN (bits 2-3 = 10 = 2 << 2 = 8)
        let dt = DecisionType::from_i8(8);
        assert!(!dt.is_categorical);
        assert!(!dt.default_left);
        assert_eq!(dt.missing_type, MissingType::NaN);
    }

    #[test]
    fn parse_objective() {
        assert!(matches!(LgbObjective::parse("regression"), LgbObjective::Regression));
        
        let binary = LgbObjective::parse("binary sigmoid:1");
        assert!(matches!(binary, LgbObjective::Binary { sigmoid } if (sigmoid - 1.0).abs() < 1e-6));

        let multiclass = LgbObjective::parse("multiclass num_class:3");
        assert!(matches!(multiclass, LgbObjective::Multiclass { num_class: 3 }));
    }

    #[test]
    fn parse_small_tree_model() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/lightgbm/inference/small_tree/model.txt");
        
        let model = LgbModel::from_file(&path).expect("Failed to parse model");
        
        assert_eq!(model.header.version, "v4");
        assert_eq!(model.header.num_class, 1);
        assert_eq!(model.num_trees(), 3);
        assert_eq!(model.num_features(), 5);

        // Check first tree structure
        let tree0 = &model.trees[0];
        assert_eq!(tree0.num_leaves, 4);
        assert_eq!(tree0.num_cat, 0);
        assert_eq!(tree0.split_feature.len(), 3);
        assert_eq!(tree0.leaf_value.len(), 4);
        assert!((tree0.shrinkage - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_binary_classification_model() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/lightgbm/inference/binary_classification/model.txt");
        
        let model = LgbModel::from_file(&path).expect("Failed to parse model");
        
        assert_eq!(model.header.num_class, 1); // Binary uses num_class=1
        assert!(matches!(model.header.objective, Some(LgbObjective::Binary { .. })));
        assert!(model.num_trees() > 0);
    }

    #[test]
    fn parse_multiclass_model() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/lightgbm/inference/multiclass/model.txt");
        
        let model = LgbModel::from_file(&path).expect("Failed to parse model");
        
        assert_eq!(model.header.num_class, 3);
        assert_eq!(model.header.num_tree_per_iteration, 3);
        assert!(matches!(model.header.objective, Some(LgbObjective::Multiclass { num_class: 3 })));
        assert!(model.num_trees() > 0);
        // Should have trees % 3 == 0 for multiclass
        assert_eq!(model.num_trees() % 3, 0);
    }

    #[test]
    fn parse_float_array_rejects_invalid_values() {
        let err = parse_float_array("1.0 2.0 nope 3.0").unwrap_err();
        assert!(matches!(err, ParseError::InvalidValue { field: "array", .. }));
    }

    #[test]
    fn validate_array_size_reports_mismatch() {
        let err = validate_array_size("test", &[1u32, 2u32], 3).unwrap_err();
        assert!(matches!(err, ParseError::ArraySizeMismatch { field: "test", expected: 3, actual: 2 }));
    }
}
