use std::fmt;

/// Standard benchmark id component formatting.
///
/// Keeps naming consistent across suites.
pub fn shape_id(rows: usize, cols: usize) -> ShapeId {
	ShapeId { rows, cols }
}

#[derive(Clone, Copy)]
pub struct ShapeId {
	pub rows: usize,
	pub cols: usize,
}

impl fmt::Display for ShapeId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "rows={} cols={}", self.rows, self.cols)
	}
}
