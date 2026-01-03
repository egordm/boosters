# Fuzz Testing

This directory contains fuzz targets for the boosters persistence layer.

## Prerequisites

Install cargo-fuzz (requires nightly Rust):

```sh
cargo install cargo-fuzz
```

## Running Fuzz Tests

### Binary Reader Fuzz

Tests the robustness of the binary model reader:

```sh
cd crates/boosters/fuzz
cargo +nightly fuzz run fuzz_binary_read
```

Run for a specific duration:

```sh
cargo +nightly fuzz run fuzz_binary_read -- -max_total_time=3600  # 1 hour
```

### Reproducing Crashes

If a crash is found, the input will be saved to `artifacts/fuzz_binary_read/`.
Reproduce with:

```sh
cargo +nightly fuzz run fuzz_binary_read artifacts/fuzz_binary_read/<crash-file>
```

### Coverage

Generate coverage report:

```sh
cargo +nightly fuzz coverage fuzz_binary_read
```
