//! Shared inference types and utilities.
//!
//! This module contains types shared across GBDT and GBLinear inference:
//! - [`PredictionOutput`]: Row-major prediction storage
//! - Output transforms (sigmoid, softmax)

mod output;
mod predictions;

pub use output::PredictionOutput;
pub use predictions::{PredictionKind, Predictions};

// =============================================================================
// Output Transforms
// =============================================================================

/// Apply sigmoid transform in-place: `x = 1 / (1 + exp(-x))`
#[inline]
pub fn sigmoid_inplace(output: &mut [f32]) {
    for x in output.iter_mut() {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

/// Apply softmax transform in-place to a single row of logits.
#[inline]
pub fn softmax_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    
    // Find max for numerical stability
    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for x in row.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    
    // Normalize
    if sum > 0.0 {
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
}

/// Apply softmax transform to each row of a prediction output.
pub fn softmax_rows(output: &mut PredictionOutput) {
    let num_groups = output.num_groups();
    if num_groups <= 1 {
        return;
    }
    
    for row_idx in 0..output.num_rows() {
        softmax_inplace(output.row_mut(row_idx));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let mut data = vec![0.0, 2.0, -2.0];
        sigmoid_inplace(&mut data);
        
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 0.8807970779778823).abs() < 1e-6);
        assert!((data[2] - 0.11920292202211755).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let mut row = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut row);
        
        // Check they sum to 1
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check relative ordering preserved
        assert!(row[0] < row[1]);
        assert!(row[1] < row[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let mut row = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut row);
        
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
