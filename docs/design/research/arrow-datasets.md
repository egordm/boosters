# Arrow datasets for benchmarks + quality (research)

## Goal

Use a standard, well-supported on-disk format for:

- performance benchmarks (load once, benchmark in-memory)
- quality evaluation harness (same train/valid/test splits across libraries)

…without inventing a bespoke serialization format.

## Candidate formats

### Arrow IPC / Feather (recommended for “data interchange”)

- Pros
  - Native Arrow representation; easy round-trip from Python/R
  - Fast load; zero-copy-ish for some buffers
  - Works well with columnar layouts
- Cons
  - Rust Arrow crates are large; compile times and dependency footprint

Notes:
- “Feather” is effectively Arrow IPC file format; in practice we can treat it as “Arrow IPC file”.

### Parquet (recommended for “storage”)

- Pros
  - Ubiquitous; good compression; excellent for large datasets
  - Strong tooling in Python
- Cons
  - More moving parts (encodings, row groups); typically heavier dependency surface
  - Might not be ideal for the tightest bench loops unless we always convert once to in-memory arrays

### CSV (not recommended)

- Pros: trivial
- Cons: huge, slow, parsing overhead; encourages IO-in-timed-region mistakes

## How it fits booste-rs

We primarily want:

1. A stable schema for `X` and `y` plus optional `weight`.
2. Deterministic split metadata (seed, indices or masks).
3. A Rust loader that produces existing in-memory representations (`RowMatrix`, `ColMatrix`, etc.)

## Schema sketch

- Table/RecordBatch columns:
  - `x_<i>` float32 (dense columnar)
  - `y` float32 (regression) or int32 (classification label)
  - optional: `w` float32

Metadata (Arrow schema / file metadata):

- `task`: regression | binary | multiclass
- `n_classes`: int
- `split_seed`: u64
- `split`: train/valid/test row index lists (optional)

## Open questions

- Do we standardize on wide table columns (`x_0..x_{d-1}`) or a single `FixedSizeList<Float32>` column?
- Do we store splits as separate files (train/valid/test) or as metadata indices?
- How strict should we be about float32 vs float64 (LightGBM bindings use f64)?
