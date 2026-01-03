# ruff: noqa: INP001

"""Generate .bstr.json fixtures from existing XGBoost test cases.

This script converts existing XGBoost JSON test cases to native boosters format.
Run from repository root: uv run python packages/boosters-datagen/scripts/generate_persist_fixtures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

from boosters.convert import lightgbm_to_json_bytes, xgboost_to_json_bytes

console = Console()

REPO_ROOT = Path(__file__).parents[3]
XGBOOST_CASES_DIR = REPO_ROOT / "crates/boosters/tests/test-cases/xgboost"
LIGHTGBM_CASES_DIR = REPO_ROOT / "crates/boosters/tests/test-cases/lightgbm"
BENCHMARK_CASES_DIR = REPO_ROOT / "crates/boosters/tests/test-cases/benchmark"
PERSIST_DIR = REPO_ROOT / "crates/boosters/tests/test-cases/persist/v1"
PERSIST_BENCHMARK_DIR = REPO_ROOT / "crates/boosters/tests/test-cases/persist/benchmark"


def convert_xgboost_cases() -> None:
    """Convert all XGBoost test cases to native format."""
    categories = ["gbtree", "gblinear", "dart"]
    subcategories = ["inference", "training"]

    for category in categories:
        cat_dir = XGBOOST_CASES_DIR / category
        if not cat_dir.exists():
            continue

        for subcat in subcategories:
            subcat_dir = cat_dir / subcat
            if not subcat_dir.exists():
                continue

            output_dir = PERSIST_DIR / category / subcat
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find all model files
            for model_file in sorted(subcat_dir.glob("*.model.json")):
                name = model_file.name.replace(".model.json", "")
                console.print(f"  Converting {category}/{subcat}/{name}...")

                # Convert model to native format
                try:
                    json_bytes = xgboost_to_json_bytes(model_file)
                    output_model = output_dir / f"{name}.model.bstr.json"
                    output_model.write_bytes(json_bytes)
                except Exception as e:
                    console.print(f"    ERROR: {e}")
                    continue

                # Copy input and expected files
                input_file = subcat_dir / f"{name}.input.json"
                expected_file = subcat_dir / f"{name}.expected.json"

                if input_file.exists():
                    shutil.copy(input_file, output_dir / f"{name}.input.json")

                if expected_file.exists():
                    shutil.copy(expected_file, output_dir / f"{name}.expected.json")


def convert_lightgbm_cases() -> None:
    """Convert all LightGBM test cases to native format."""
    if not LIGHTGBM_CASES_DIR.exists():
        console.print("  No LightGBM test cases found, skipping")
        return

    # Similar structure to XGBoost
    output_dir = PERSIST_DIR / "lightgbm"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_file in sorted(LIGHTGBM_CASES_DIR.rglob("*.model.json")):
        # Get relative path for naming
        rel_path = model_file.relative_to(LIGHTGBM_CASES_DIR)
        name = model_file.stem.replace(".model", "")

        # Create output subdirectory matching input structure
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        console.print(f"  Converting lightgbm/{rel_path.parent}/{name}...")

        try:
            json_bytes = lightgbm_to_json_bytes(model_file)
            output_model = output_subdir / f"{name}.model.bstr.json"
            output_model.write_bytes(json_bytes)
        except Exception as e:
            console.print(f"    ERROR: {e}")
            continue

        # Copy input and expected files
        input_file = model_file.parent / f"{name}.input.json"
        expected_file = model_file.parent / f"{name}.expected.json"

        if input_file.exists():
            shutil.copy(input_file, output_subdir / f"{name}.input.json")

        if expected_file.exists():
            shutil.copy(expected_file, output_subdir / f"{name}.expected.json")


def create_readme() -> None:
    """Create README explaining the fixtures."""
    readme = PERSIST_DIR / "README.md"
    readme.write_text("""\
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
""")


def convert_benchmark_models() -> None:
    """Convert benchmark models to native format.

    These are used by Criterion benchmarks and need to be available without
    the xgboost-compat feature flag.
    """
    if not BENCHMARK_CASES_DIR.exists():
        console.print("  No benchmark models found, skipping")
        return

    # Clean and create output directory
    if PERSIST_BENCHMARK_DIR.exists():
        shutil.rmtree(PERSIST_BENCHMARK_DIR)
    PERSIST_BENCHMARK_DIR.mkdir(parents=True)

    # Convert XGBoost models (.model.json)
    for model_file in sorted(BENCHMARK_CASES_DIR.glob("*.model.json")):
        name = model_file.stem.replace(".model", "")
        console.print(f"  Converting benchmark/{name} (XGBoost)...")

        try:
            json_bytes = xgboost_to_json_bytes(model_file)
            output_model = PERSIST_BENCHMARK_DIR / f"{name}.model.bstr.json"
            output_model.write_bytes(json_bytes)
        except Exception as e:
            console.print(f"    ERROR: {e}")
            continue

    # Convert LightGBM models (.lgb.txt)
    for model_file in sorted(BENCHMARK_CASES_DIR.glob("*.lgb.txt")):
        name = model_file.stem.replace(".lgb", "")
        console.print(f"  Converting benchmark/{name} (LightGBM)...")

        try:
            json_bytes = lightgbm_to_json_bytes(model_file)
            output_model = PERSIST_BENCHMARK_DIR / f"{name}.model.bstr.json"
            output_model.write_bytes(json_bytes)
        except Exception as e:
            console.print(f"    ERROR: {e}")
            continue

    # Copy timing files (used for benchmark setup)
    for timing_file in sorted(BENCHMARK_CASES_DIR.glob("*.timing.json")):
        shutil.copy(timing_file, PERSIST_BENCHMARK_DIR / timing_file.name)


def main() -> None:
    """Generate native persist fixtures from XGBoost/LightGBM test cases."""
    console.print("Generating native persist fixtures...")

    # Clean output directory
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
    PERSIST_DIR.mkdir(parents=True)

    console.print("\nConverting XGBoost test cases:")
    convert_xgboost_cases()

    console.print("\nConverting LightGBM test cases:")
    convert_lightgbm_cases()

    console.print("\nConverting benchmark models:")
    convert_benchmark_models()

    console.print("\nCreating README...")
    create_readme()

    # Count generated files
    model_count = len(list(PERSIST_DIR.rglob("*.model.bstr.json")))
    bench_count = len(list(PERSIST_BENCHMARK_DIR.rglob("*.model.bstr.json")))
    console.print(f"\nDone! Generated {model_count} test fixtures and {bench_count} benchmark fixtures.")


if __name__ == "__main__":
    main()
