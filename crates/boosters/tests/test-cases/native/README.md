# Native Format Test Fixtures

This directory contains `.bstr` files for testing format stability.

## Files

| File | Description |
|------|-------------|
| `simple_forest.bstr` | Single tree, regression, with base_score |
| `multi_tree_forest.bstr` | Two trees, regression, mixed depths |
| `categorical_forest.bstr` | Single tree with categorical split |
| `multiclass_forest.bstr` | Three trees for 3-class classification |
| `simple_linear.bstr` | Linear model, 3 features, 1 group |
| `multioutput_linear.bstr` | Linear model, 2 features, 2 groups |

## Regenerating Fixtures

```bash
cargo test --features storage --test native_format generate_fixtures -- --ignored
```

## Format Versioning

These fixtures are generated with format version 1.0. When the format version
changes, regenerate fixtures and update this document.

Format version: **1.0**
Generated: 2025-06-18
