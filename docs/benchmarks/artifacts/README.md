# Benchmark / quality artifacts

This folder is for **generated outputs** produced by benchmark and quality harness runs (e.g. JSON metrics snapshots).

- These files are **not tracked in git** by default to avoid noisy diffs and large commits.
- The benchmark report in `docs/benchmarks/2025-12-14-benchmark-report.md` contains the commands needed to reproduce them.

If you want to keep a particular artifact as a “golden” reference, either:

- copy it elsewhere under a tracked location (e.g. `docs/benchmarks/golden/`), or
- update `.gitignore` to un-ignore that specific file.
