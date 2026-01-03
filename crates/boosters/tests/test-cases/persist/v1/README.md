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

These fixtures were generated from XGBoost/LightGBM test cases using:

```bash
uv run python packages/boosters-datagen/scripts/generate_persist_fixtures.py
```

## Usage

```rust
use boosters::persist::{load_json_file, JsonEnvelope};

let envelope: JsonEnvelope = load_json_file("path/to/model.bstr.json")?;
let model = envelope.into_model();
```
