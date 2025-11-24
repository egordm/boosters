# QuantileDMatrix

QuantileDMatrix is the staging area that converts floating-point feature values into **histogram-friendly bins**.

## What problem is it solving?
- Books arrive at the library with varying measurements (page count, weight). Histograms need **uniformly spaced bins** per feature, but the raw data can be skewed, sparse, or streamed from disk.
- QuantileDMatrix answers: *"How do we turn each incoming book into compact bin ids without storing every measurement and while keeping cuts consistent across CPU and GPU?"*

## What kinds of data benefit?
- Works best for **wide numeric ranges or mixed-scale features**: quantile sketching finds stable cut points even when units differ (e.g., [0, 1] probabilities next to [$10^6$] dollar values).
- Handles **sparse inputs** because sketches sample only the non-missing entries per feature; empty columns simply end up with a single "missing" bin.
- Especially useful when the dataset **does not fit in device memory**: it streams rows from disk or iterators and still reaches the same cut quality.

## Mental model (ELI5)
The librarian scans each feature (page count, weight, etc.) across all incoming books and marks bin boundaries. Later, each book only remembers which bin it falls into, not the exact measurement.

```
incoming books     ──►  sketch per feature      ──►  HistogramCuts (shared blueprint)
book1: pg=120, wt=0.5   feature 0 (page count)      bins: [0-100, 100-200, 200+]
book2: pg=350, wt=1.2   feature 1 (weight)          bins: [0-0.5, 0.5-1.0, 1.0+]
...
```

## Why it exists (ELI13)
`QuantileDMatrix` standardizes how XGBoost discovers bin boundaries and feeds quantized batches to both `GHistIndexMatrix` (CPU) and `EllpackPage` (GPU). 

**Why quantile bins?** If you used uniform spacing (equal-width bins), most books might land in one bin (e.g., 90% have 100-200 pages) while other bins stay empty. Quantile sketching ensures **each bin gets roughly equal book counts** (equal-population), making histograms informative even with skewed data.

For example, if page counts follow a normal distribution with mean=150, quantile binning creates **narrower bins near the mean** (where most books cluster) and **wider bins in the tails** (where books are sparse). Each bin still captures ~10% of books.

It keeps only:
1. The sketches/cuts per feature (`HistogramCuts`).
2. A thin wrapper that can produce quantized pages on demand.

## Detailed flow (ELI-grad)
1. **Sketch accumulation**: `DeviceSketch` or `SketchOnDMatrix` samples every feature stream to maintain quantile summaries tolerant to duplicates and heavy tails.
2. **Cut materialization**: sketches merge into `HistogramCuts`, which encode `Ptrs` (feature offsets), `Values` (cut boundaries), and `MinVals` (baseline for missing bins).
3. **Batch emission**: downstream consumers call `GetBatches<GHistIndexMatrix>` or `GetBatches<EllpackPage>`; the matrix translates raw floats into small bin ids using the shared cuts.

## When to pick QuantileDMatrix
- **Large-on-disk tabular data** where you cannot load everything into RAM/GPU but still need consistent bins.
- **Datasets with skewed distributions** (long tails, log-like behavior). Quantile bins adapt automatically, unlike uniform binning.
- **Sparse CSR/COO feeds**. Completely empty columns stay compact because no sketch records are stored.

Avoid it when you specifically need the legacy `approx` tree method; quantile DMatrix is optimized for `hist`, `gpu_hist`, and predictor paths that consume histogram bins.

## Minimal pseudo-code
```
QuantileDMatrix Build(stream, max_bin):
  sketches = InitSketches(num_features, max_bin)
  for batch in stream:
    DeviceSketch(sketches, batch)
  cuts = sketches.MergeToCuts()
  return QuantileDMatrix(cuts)

// Later
for page in QuantileDMatrix.GetBatches<GHistIndexMatrix>():
  consume(page)
```

## Implementation breadcrumbs
- Core: `src/data/quantile_dmatrix.{h,cc}`
- External-memory flavor: `src/data/extmem_quantile_dmatrix.*`
- Shared sketch utilities: `src/common/hist_util.h`

## Practical notes
- The structure itself never stores raw floats; it only retains the `HistogramCuts` plus light metadata.
- Downstream matrices cache results so repeated training epochs do not recompute sketches.

