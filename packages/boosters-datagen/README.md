# Boosters Data Generation

Test data generation for the boosters library.

## Usage

```bash
# Generate all test cases (XGBoost, LightGBM, and native format)
boosters-datagen all

# Generate XGBoost test cases only
boosters-datagen xgboost

# Generate LightGBM test cases only
boosters-datagen lightgbm

# Generate native .bstr.json fixtures from existing test cases
boosters-datagen bstr
```

## Output

Test cases are generated in `crates/boosters/tests/test-cases/`:

```text
tests/test-cases/
├── xgboost/              # XGBoost JSON format (source)
│   ├── gbtree/
│   │   ├── inference/    # Model + input + expected predictions
│   │   └── training/     # Train data + labels + config
│   ├── gblinear/
│   │   ├── inference/
│   │   └── training/
│   └── dart/
│       └── inference/
├── lightgbm/             # LightGBM format (source)
│   └── inference/
├── benchmark/            # Benchmark models (XGBoost/LightGBM)
└── persist/              # Native boosters format
    ├── v1/               # Schema version 1 test fixtures
    │   ├── gbtree/
    │   ├── gblinear/
    │   ├── dart/
    │   └── lightgbm/
    └── benchmark/        # Native benchmark models
```

## Workflow

1. Run `boosters-datagen xgboost` and `boosters-datagen lightgbm` to generate
   source test cases in their native formats
2. Run `boosters-datagen bstr` to convert those cases to native `.bstr.json` format
3. Or run `boosters-datagen all` to do everything in one step

The native fixtures in `persist/` are what the Rust tests use.

## License

MIT
