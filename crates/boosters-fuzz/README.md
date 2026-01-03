# Fuzz Testing

This crate contains `cargo-fuzz` targets for boosters.

## Prerequisites

- Nightly Rust toolchain
- `cargo-fuzz`: `cargo install cargo-fuzz`

## Running

```sh
cd crates/boosters-fuzz
cargo +nightly fuzz run fuzz_binary_read
```

Run for a bounded time:

```sh
cd crates/boosters-fuzz
cargo +nightly fuzz run fuzz_binary_read -- -max_total_time=3600
```
