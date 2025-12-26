//! Shared Arrow `RecordBatch` conversion logic.
//!
//! This is used by both the Arrow IPC and Parquet loaders.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, Int32Array};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use ndarray::Array2;

use super::error::DatasetLoadError;

pub(super) struct LoadedBatches {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    rows: usize,
    cols: usize,
}

impl LoadedBatches {
    pub(super) fn new(
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> Result<Self, DatasetLoadError> {
        let (rows, cols) = infer_shape(&schema, &batches)?;
        Ok(Self {
            schema,
            batches,
            rows,
            cols,
        })
    }

    /// Convert to row-major Array2<f32> with shape (n_samples, n_features).
    pub(super) fn to_row_matrix_f32(&self) -> Result<Array2<f32>, DatasetLoadError> {
        batches_to_row_matrix(&self.batches, &self.schema, self.rows, self.cols)
    }

    /// Convert to column-major Array2<f32> with shape (n_features, n_samples).
    pub(super) fn to_col_matrix_f32(&self) -> Result<Array2<f32>, DatasetLoadError> {
        batches_to_col_matrix(&self.batches, &self.schema, self.rows, self.cols)
    }

    pub(super) fn to_raw_f32(
        &self,
    ) -> Result<(Vec<f32>, Vec<f32>, usize, usize), DatasetLoadError> {
        let features =
            extract_features_row_major(&self.batches, &self.schema, self.rows, self.cols)?;
        let targets = extract_targets(&self.batches, &self.schema)?;
        Ok((features, targets, self.rows, self.cols))
    }
}

fn infer_shape(
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<(usize, usize), DatasetLoadError> {
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let cols = count_feature_columns(schema)?;
    Ok((rows, cols))
}

/// Count number of feature columns (either wide `x_0..x_{d-1}` or FixedSizeList `x`).
fn count_feature_columns(schema: &Schema) -> Result<usize, DatasetLoadError> {
    // Check for FixedSizeList `x` column first
    if let Ok(field) = schema.field_with_name("x") {
        if let DataType::FixedSizeList(_, size) = field.data_type() {
            return Ok(*size as usize);
        }
    }

    // Otherwise count x_0, x_1, ... columns
    let mut max_idx: Option<usize> = None;
    for field in schema.fields() {
        if let Some(rest) = field.name().strip_prefix("x_") {
            if let Ok(idx) = rest.parse::<usize>() {
                max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
            }
        }
    }

    match max_idx {
        Some(m) => Ok(m + 1),
        None => Err(DatasetLoadError::MissingColumn(
            "no feature columns found (expected x_0..x_N or x FixedSizeList)".into(),
        )),
    }
}

fn extract_targets(batches: &[RecordBatch], schema: &Schema) -> Result<Vec<f32>, DatasetLoadError> {
    if schema.field_with_name("y").is_err() {
        return Err(DatasetLoadError::MissingColumn("y".into()));
    }
    extract_float32_column(batches, "y")
}

fn extract_float32_column(
    batches: &[RecordBatch],
    name: &str,
) -> Result<Vec<f32>, DatasetLoadError> {
    let mut values = Vec::new();
    for batch in batches {
        let col = batch
            .column_by_name(name)
            .ok_or_else(|| DatasetLoadError::MissingColumn(name.into()))?;

        if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
            values.extend(arr.iter().map(|v| v.unwrap_or(f32::NAN)));
            continue;
        }

        // Also accept Int32 (common for labels)
        if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
            values.extend(arr.iter().map(|v| v.unwrap_or(0) as f32));
            continue;
        }

        return Err(DatasetLoadError::UnsupportedType {
            column: name.into(),
            expected: "Float32 or Int32".into(),
            got: format!("{:?}", col.data_type()),
        });
    }
    Ok(values)
}

fn extract_features_row_major(
    batches: &[RecordBatch],
    schema: &Schema,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, DatasetLoadError> {
    // FixedSizeList is already row-major
    if schema.field_with_name("x").ok().is_some() {
        let mut all_values = Vec::with_capacity(rows * cols);
        for batch in batches {
            let col = batch
                .column_by_name("x")
                .ok_or_else(|| DatasetLoadError::MissingColumn("x".into()))?;

            let list_array = col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| DatasetLoadError::UnsupportedType {
                    column: "x".into(),
                    expected: "FixedSizeList<Float32>".into(),
                    got: format!("{:?}", col.data_type()),
                })?;

            let values = list_array
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| DatasetLoadError::UnsupportedType {
                    column: "x".into(),
                    expected: "Float32".into(),
                    got: "other".into(),
                })?;

            all_values.extend(values.iter().map(|v| v.unwrap_or(f32::NAN)));
        }
        return Ok(all_values);
    }

    // Wide columns: read each column and interleave
    let mut columns: Vec<Vec<f32>> = Vec::with_capacity(cols);
    for i in 0..cols {
        let col_name = format!("x_{}", i);
        columns.push(extract_float32_column(batches, &col_name)?);
    }

    let mut result = Vec::with_capacity(rows * cols);
    for row_idx in 0..rows {
        for col in &columns {
            result.push(col[row_idx]);
        }
    }
    Ok(result)
}

/// Convert record batches to row-major Array2 with shape (n_samples, n_features).
fn batches_to_row_matrix(
    batches: &[RecordBatch],
    schema: &Schema,
    rows: usize,
    cols: usize,
) -> Result<Array2<f32>, DatasetLoadError> {
    let features = extract_features_row_major(batches, schema, rows, cols)?;
    Array2::from_shape_vec((rows, cols), features)
        .map_err(|e| DatasetLoadError::Schema(format!("Shape error: {}", e)))
}

/// Convert record batches to column-major Array2 with shape (n_features, n_samples).
fn batches_to_col_matrix(
    batches: &[RecordBatch],
    schema: &Schema,
    rows: usize,
    cols: usize,
) -> Result<Array2<f32>, DatasetLoadError> {
    if schema.field_with_name("x").ok().is_some() {
        // FixedSizeList: read row-major and transpose
        let row_major = extract_features_row_major(batches, schema, rows, cols)?;
        let mut col_major = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                col_major[col * rows + row] = row_major[row * cols + col];
            }
        }
        return Array2::from_shape_vec((cols, rows), col_major)
            .map_err(|e| DatasetLoadError::Schema(format!("Shape error: {}", e)));
    }

    // Wide columns: each column is already contiguous in Arrow
    let mut col_major = Vec::with_capacity(rows * cols);
    for i in 0..cols {
        let col_name = format!("x_{}", i);
        let col_values = extract_float32_column(batches, &col_name)?;
        col_major.extend(col_values);
    }
    Array2::from_shape_vec((cols, rows), col_major)
        .map_err(|e| DatasetLoadError::Schema(format!("Shape error: {}", e)))
}
