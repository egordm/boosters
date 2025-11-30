# booste-rs Documentation

## Contents

| Section | Description |
|---------|-------------|
| [design/](./design/) | Design documentation, RFCs, research |
| [benchmarks/](./benchmarks/) | Performance benchmark results |

## Quick Links

- [Architecture Overview](./design/README.md)
- [RFC Index](./design/rfcs/README.md)
- [Implementation Roadmap](./design/ROADMAP.md)
- [Latest Benchmarks](./benchmarks/2024-11-29-m38-performance-validation.md)

## Performance Summary

As of 2024-11-29 on Apple M1 Pro:

| Benchmark | booste-rs | XGBoost C++ | Speedup |
|-----------|-----------|-------------|---------|
| Single-row | 1.24 µs | 11.6 µs | **9.4x** |
| 1K batch | 1.07 ms | 1.38 ms | **1.29x** |
| 10K batch (8 threads) | 1.58 ms | 5.00 ms | **3.18x** |
