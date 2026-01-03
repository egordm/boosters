"""Generate native .bstr.json fixtures from XGBoost/LightGBM test cases.

This module converts existing test cases to native boosters format using
the Python conversion utilities from boosters.convert.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from boosters.convert import lightgbm_to_json_bytes, xgboost_to_json_bytes
from rich.console import Console

console = Console()

# Base directories (relative to package location)
_PKG_ROOT = Path(__file__).parents[4]  # booste-rs repo root
XGBOOST_CASES_DIR = _PKG_ROOT / "crates/boosters/tests/test-cases/xgboost"
LIGHTGBM_CASES_DIR = _PKG_ROOT / "crates/boosters/tests/test-cases/lightgbm"
BENCHMARK_CASES_DIR = _PKG_ROOT / "crates/boosters/tests/test-cases/benchmark"
PERSIST_DIR = _PKG_ROOT / "crates/boosters/tests/test-cases/persist/v1"
PERSIST_BENCHMARK_DIR = _PKG_ROOT / "crates/boosters/tests/test-cases/persist/benchmark"


def _convert_xgboost_cases() -> int:
    """Convert all XGBoost test cases to native format.

    Returns number of models converted.
    """
    categories = ["gbtree", "gblinear", "dart"]
    subcategories = ["inference", "training"]
    count = 0

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

            for model_file in sorted(subcat_dir.glob("*.model.json")):
                name = model_file.name.replace(".model.json", "")

                try:
                    json_bytes = xgboost_to_json_bytes(model_file)
                    output_model = output_dir / f"{name}.model.bstr.json"
                    output_model.write_bytes(json_bytes)
                    count += 1

                    # Copy input and expected files
                    input_file = subcat_dir / f"{name}.input.json"
                    expected_file = subcat_dir / f"{name}.expected.json"

                    if input_file.exists():
                        shutil.copy(input_file, output_dir / f"{name}.input.json")
                    if expected_file.exists():
                        shutil.copy(expected_file, output_dir / f"{name}.expected.json")

                except Exception as e:
                    console.print(f"  [red]ERROR[/red]: {category}/{subcat}/{name}: {e}")

    return count


def _convert_lightgbm_cases() -> int:
    """Convert all LightGBM test cases to native format.

    Returns number of models converted.
    """
    if not LIGHTGBM_CASES_DIR.exists():
        return 0

    output_dir = PERSIST_DIR / "lightgbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for model_file in sorted(LIGHTGBM_CASES_DIR.rglob("*.model.json")):
        rel_path = model_file.relative_to(LIGHTGBM_CASES_DIR)
        name = model_file.stem.replace(".model", "")

        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        try:
            json_bytes = lightgbm_to_json_bytes(model_file)
            output_model = output_subdir / f"{name}.model.bstr.json"
            output_model.write_bytes(json_bytes)
            count += 1

            # Copy input and expected files
            input_file = model_file.parent / f"{name}.input.json"
            expected_file = model_file.parent / f"{name}.expected.json"

            if input_file.exists():
                shutil.copy(input_file, output_subdir / f"{name}.input.json")
            if expected_file.exists():
                shutil.copy(expected_file, output_subdir / f"{name}.expected.json")

        except Exception as e:
            console.print(f"  [red]ERROR[/red]: lightgbm/{rel_path.parent}/{name}: {e}")

    return count


def _convert_benchmark_models() -> int:
    """Convert benchmark models to native format.

    Returns number of models converted.
    """
    if not BENCHMARK_CASES_DIR.exists():
        return 0

    if PERSIST_BENCHMARK_DIR.exists():
        shutil.rmtree(PERSIST_BENCHMARK_DIR)
    PERSIST_BENCHMARK_DIR.mkdir(parents=True)
    count = 0

    # Convert XGBoost models (.model.json)
    for model_file in sorted(BENCHMARK_CASES_DIR.glob("*.model.json")):
        name = model_file.stem.replace(".model", "")

        try:
            json_bytes = xgboost_to_json_bytes(model_file)
            output_model = PERSIST_BENCHMARK_DIR / f"{name}.model.bstr.json"
            output_model.write_bytes(json_bytes)
            count += 1
        except Exception as e:
            console.print(f"  [red]ERROR[/red]: benchmark/{name}: {e}")

    # Convert LightGBM models (.lgb.txt)
    for model_file in sorted(BENCHMARK_CASES_DIR.glob("*.lgb.txt")):
        name = model_file.stem.replace(".lgb", "")

        try:
            json_bytes = lightgbm_to_json_bytes(model_file)
            output_model = PERSIST_BENCHMARK_DIR / f"{name}.model.bstr.json"
            output_model.write_bytes(json_bytes)
            count += 1
        except Exception as e:
            console.print(f"  [red]ERROR[/red]: benchmark/{name}: {e}")

    # Copy timing files
    for timing_file in sorted(BENCHMARK_CASES_DIR.glob("*.timing.json")):
        shutil.copy(timing_file, PERSIST_BENCHMARK_DIR / timing_file.name)

    return count


def _create_readme() -> None:
    """Create README for fixtures directory."""
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
""")


def generate_native_fixtures() -> None:
    """Generate all native .bstr.json fixtures.

    This is the main entry point called from the CLI.
    """
    # Clean output directory
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
    PERSIST_DIR.mkdir(parents=True)

    console.print("  Converting XGBoost test cases...")
    xgb_count = _convert_xgboost_cases()

    console.print("  Converting LightGBM test cases...")
    lgb_count = _convert_lightgbm_cases()

    console.print("  Converting benchmark models...")
    bench_count = _convert_benchmark_models()

    console.print("  Creating README...")
    _create_readme()

    console.print(
        f"\n[green]✓ Generated {xgb_count} XGBoost, {lgb_count} LightGBM, "
        f"and {bench_count} benchmark fixtures[/green]"
    )
