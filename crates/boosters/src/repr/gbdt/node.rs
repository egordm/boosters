//! Tree node types.

/// Type of split in a decision tree node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SplitType {
    /// Numeric split: go left if value < threshold
    #[default]
    Numeric = 0,
    /// Categorical split: go left if value NOT in category set
    Categorical = 1,
}

impl From<u8> for SplitType {
    fn from(value: u8) -> Self {
        match value {
            0 => SplitType::Numeric,
            _ => SplitType::Categorical,
        }
    }
}

impl From<i32> for SplitType {
    fn from(value: i32) -> Self {
        match value {
            0 => SplitType::Numeric,
            _ => SplitType::Categorical,
        }
    }
}
