# Boosters Data Generation

Test data generation for the boosters library.

## Usage

```bash
# Generate all test cases
boosters-datagen all

# Generate XGBoost test cases only
boosters-datagen xgboost

# Generate LightGBM test cases only
boosters-datagen lightgbm
```

## Output

Test cases are generated in `tests/test-cases/`:

```text
tests/test-cases/
├── xgboost/
│   ├── gbtree/
│   │   ├── inference/    # Model + input + expected predictions
│   │   └── training/     # Train data + labels + config
│   ├── gblinear/
│   │   ├── inference/
│   │   └── training/
│   └── dart/
│       └── inference/
└── lightgbm/
    └── inference/
```

## License

MIT
