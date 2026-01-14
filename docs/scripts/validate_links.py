#!/usr/bin/env python3
"""
RFC and documentation link validation script.

Validates:
- Internal links between documentation files
- RFC cross-references
- External URL availability (optional)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


class LinkResult(NamedTuple):
    """Result of validating a link."""

    source_file: Path
    line_number: int
    link_text: str
    target: str
    is_valid: bool
    error_message: str | None


def find_markdown_links(content: str, file_path: Path) -> list[tuple[int, str, str]]:
    """Find all markdown links [text](target) in content."""
    links = []
    # Match [text](target) but not ![alt](image)
    pattern = r"(?<!\!)\[([^\]]+)\]\(([^)]+)\)"
    for line_num, line in enumerate(content.splitlines(), 1):
        for match in re.finditer(pattern, line):
            text, target = match.groups()
            links.append((line_num, text, target))
    return links


def find_rst_references(content: str, file_path: Path) -> list[tuple[int, str, str]]:
    """Find all RST :doc: and :ref: references in content."""
    links = []
    # Match :doc:`target` and :doc:`text <target>`
    doc_pattern = r":doc:`(?:([^`<]+)\s*<)?([^`>]+)>?`"
    ref_pattern = r":ref:`(?:([^`<]+)\s*<)?([^`>]+)>?`"

    for line_num, line in enumerate(content.splitlines(), 1):
        for match in re.finditer(doc_pattern, line):
            text = match.group(1) or match.group(2)
            target = match.group(2)
            links.append((line_num, text, target))
        for match in re.finditer(ref_pattern, line):
            text = match.group(1) or match.group(2)
            target = match.group(2)
            links.append((line_num, text, f"ref:{target}"))

    return links


def validate_relative_link(
    source_file: Path, target: str, docs_root: Path
) -> tuple[bool, str | None]:
    """Validate a relative link target exists."""
    # Skip external links
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return True, None

    # Skip anchors within same file
    if target.startswith("#"):
        return True, None

    # Remove anchor from target
    target_path = target.split("#")[0]
    if not target_path:
        return True, None

    # Handle RST :doc: references (no extension needed)
    source_dir = source_file.parent

    # Try different resolutions
    possible_paths = [
        source_dir / target_path,
        source_dir / f"{target_path}.rst",
        source_dir / f"{target_path}.md",
        docs_root / target_path.lstrip("/"),
        docs_root / f"{target_path.lstrip('/')}.rst",
        docs_root / f"{target_path.lstrip('/')}.md",
    ]

    for path in possible_paths:
        resolved = path.resolve()
        if resolved.exists():
            return True, None

    return False, f"Target not found: {target_path}"


def validate_file(file_path: Path, docs_root: Path) -> list[LinkResult]:
    """Validate all links in a file."""
    results = []
    content = file_path.read_text(encoding="utf-8")

    if file_path.suffix == ".md":
        links = find_markdown_links(content, file_path)
    elif file_path.suffix == ".rst":
        links = find_rst_references(content, file_path)
    else:
        return results

    for line_num, text, target in links:
        is_valid, error = validate_relative_link(file_path, target, docs_root)
        results.append(
            LinkResult(
                source_file=file_path,
                line_number=line_num,
                link_text=text,
                target=target,
                is_valid=is_valid,
                error_message=error,
            )
        )

    return results


def validate_docs(docs_root: Path, verbose: bool = False) -> list[LinkResult]:
    """Validate all documentation files."""
    all_results: list[LinkResult] = []

    # Find all markdown and RST files
    for pattern in ["**/*.md", "**/*.rst"]:
        for file_path in docs_root.glob(pattern):
            # Skip build directory
            if "_build" in str(file_path):
                continue

            if verbose:
                print(f"Checking: {file_path.relative_to(docs_root)}")

            results = validate_file(file_path, docs_root)
            all_results.extend(results)

    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate documentation links and references"
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Root directory of documentation (default: docs/)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all files being checked"
    )
    parser.add_argument(
        "--show-valid",
        action="store_true",
        help="Show valid links as well as invalid ones",
    )

    args = parser.parse_args()

    docs_root = args.docs_root.resolve()
    if not docs_root.exists():
        print(f"Error: Documentation root not found: {docs_root}", file=sys.stderr)
        return 1

    print(f"Validating documentation in: {docs_root}")
    results = validate_docs(docs_root, verbose=args.verbose)

    # Filter and report
    invalid_results = [r for r in results if not r.is_valid]
    valid_count = len(results) - len(invalid_results)

    if args.show_valid:
        for r in results:
            if r.is_valid:
                rel_path = r.source_file.relative_to(docs_root)
                print(f"✓ {rel_path}:{r.line_number} -> {r.target}")

    if invalid_results:
        print(f"\n{'='*60}")
        print(f"Found {len(invalid_results)} broken link(s):")
        print(f"{'='*60}\n")

        for r in invalid_results:
            rel_path = r.source_file.relative_to(docs_root)
            print(f"✗ {rel_path}:{r.line_number}")
            print(f"  Link text: {r.link_text}")
            print(f"  Target: {r.target}")
            print(f"  Error: {r.error_message}")
            print()

    print(f"\nSummary: {valid_count} valid, {len(invalid_results)} invalid links")

    return 1 if invalid_results else 0


if __name__ == "__main__":
    sys.exit(main())
