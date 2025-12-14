/// Common dataset shapes used across benchmarks.

#[derive(Debug, Clone, Copy)]
pub struct DatasetShape {
	pub name: &'static str,
	pub rows: usize,
	pub cols: usize,
}

pub const PERFORMANCE_SHAPES: &[DatasetShape] = &[
	DatasetShape {
		name: "small",
		rows: 10_000,
		cols: 50,
	},
	DatasetShape {
		name: "narrow",
		rows: 100_000,
		cols: 20,
	},
	DatasetShape {
		name: "wide",
		rows: 50_000,
		cols: 2_000,
	},
	DatasetShape {
		name: "tall",
		rows: 1_000_000,
		cols: 50,
	},
];

pub const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];
