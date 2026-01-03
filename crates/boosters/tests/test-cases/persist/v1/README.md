# Native Persist Fixtures (v1)

This directory contains test fixtures in the native boosters `.bstr.json` format.

## Structure

```
v1/
├── gbtree/          # GBDT models (tree-based)
│   ├── inference/   # Inference test cases
│   └── training/    # Training test cases
├── gblinear/        # Linear models
│   └── inference/   # Inference test cases
├── dart/            # DART models
│   └── inference/   # Inference test cases
└── lightgbm/        # Converted LightGBM models
```

## File Naming

- `<name>.model.bstr.json` - Model in native JSON format
- `<name>.input.json` - Input features for testing
- `<name>.expected.json` - Expected predictions

## Generation

These fixtures were generated using:

```bash
uv run boosters-datagen bstr
```

Or generate all test cases:

```bash
uv run boosters-datagen all
```

## Usage

```rust
use boosters::persist::Model;

let model = Model::load_json("path/to/model.bstr.json")?;
let gbdt = model.into_gbdt().unwrap();
```
