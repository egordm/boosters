# Benchmark Template

Use this template when documenting benchmark results.

---

# YYYY-MM-DD: [Title]

## Goal

<!-- What question are we trying to answer? What hypothesis are we testing? -->

## Summary

<!-- One paragraph describing key findings and conclusions -->

## Environment

| Property | Value |
|----------|-------|
| CPU | <!-- e.g., Apple M1 Pro, 10 cores --> |
| Rust | <!-- e.g., 1.82.0 --> |
| Commit | <!-- git short hash --> |

## Results

### [Benchmark Name]

<!-- Use tables for comparing results -->

| Configuration | Time | Throughput | Notes |
|---------------|------|------------|-------|
| Baseline | | | |
| Variant A | | | |
| Variant B | | | |

### [Another Benchmark]

<!-- Add more sections as needed -->

## Analysis

<!-- Explain the results. Why did X perform better than Y? -->

## Conclusions

<!-- Key takeaways and recommendations -->

## Reproducing

```bash
# Commands to reproduce these benchmarks
cargo bench --bench prediction -- "benchmark_name"
```
