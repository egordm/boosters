//! Evaluation set abstraction.

use crate::data::Dataset;

/// Named evaluation dataset.
#[derive(Debug, Clone, Copy)]
pub struct EvalSet<'a> {
    pub name: &'a str,
    pub dataset: &'a Dataset,
}

impl<'a> EvalSet<'a> {
    pub fn new(name: &'a str, dataset: &'a Dataset) -> Self {
        Self { name, dataset }
    }
}
