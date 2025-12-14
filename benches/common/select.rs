/// Small selection helpers used by benchmark suites.
///
/// These are intentionally kept in bench-only code to avoid expanding the
/// public library API with one-off slicing utilities.

/// Select a subset of rows from row-major features.
///
/// Returns a new row-major buffer with `row_indices.len()` rows.
pub fn select_rows_row_major(
	features_row_major: &[f32],
	rows: usize,
	cols: usize,
	row_indices: &[usize],
) -> Vec<f32> {
	assert_eq!(features_row_major.len(), rows * cols);
	let mut out = Vec::with_capacity(row_indices.len() * cols);
	for &r in row_indices {
		assert!(r < rows);
		let start = r * cols;
		out.extend_from_slice(&features_row_major[start..start + cols]);
	}
	out
}

/// Select targets by indices.
pub fn select_targets(targets: &[f32], row_indices: &[usize]) -> Vec<f32> {
	let mut out = Vec::with_capacity(row_indices.len());
	for &r in row_indices {
		out.push(targets[r]);
	}
	out
}
