#!/usr/bin/env python3
"""Set version for all packages in the workspace."""

import re
import sys
from pathlib import Path


def set_version(version: str) -> None:
    """Update version in all package configuration files."""
    if not version:
        print("Error: --version is required", file=sys.stderr)
        sys.exit(1)

    # Update packages/boosters-python/pyproject.toml
    pyproject = Path("packages/boosters-python/pyproject.toml")
    content = pyproject.read_text()
    content = re.sub(r'^version = ".*"$', f'version = "{version}"', content, flags=re.MULTILINE)
    pyproject.write_text(content)
    print(f"Updated {pyproject} to version {version}")

    # Update packages/boosters-python/Cargo.toml
    cargo = Path("packages/boosters-python/Cargo.toml")
    content = cargo.read_text()
    # Only update the first version line (package version, not dependency versions)
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("version = "):
            lines[i] = f'version = "{version}"'
            break
    cargo.write_text("\n".join(lines))
    print(f"Updated {cargo} to version {version}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Set version for all packages")
    parser.add_argument("--version", required=True, help="Version to set")
    args = parser.parse_args()
    set_version(args.version)
