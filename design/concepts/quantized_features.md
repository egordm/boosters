# Quantized Features

## ELI13 (Explain like I'm 13)
Quantized features means we replace continuous numbers (like 3.14) with a small set of buckets (like 0..15). Instead of storing raw floats for every value, we store an integer that says which bucket the value belongs to. This makes the model faster and smaller because comparing bucket numbers and using precomputed thresholds is cheaper and easier for the computer.

## ELI Grad (Graduate-level explanation)
Quantization maps continuous feature values into discrete bins (indices), often using histogram cuts computed during training. A common representation is a small-integer index per feature per sample indicating the bin. Quantization reduces memory bandwidth and enables high-throughput histogram-based split finding and prediction traversal: splitting decisions compare the quantized bin index for a feature against a threshold bin index, which can be implemented as a single integer comparison rather than float comparison.

### Benefits
- Compact storage (u8/u16) vs f32/f64.
- Improved cache locality and memory throughput.
- Enables SIMD/vectorized operations and GPU-friendly memory coalescing.
- Deterministic behavior across platforms when using shared bin indices.

### Example in memory
Raw float row (3 features):
```
[f1=1.23, f2=0.124, f3=42.1]
```
After quantization with 16 bins each (4-bit) we store:
```
bins_u8_row: [3, 0, 15]  // u8 per feature
```
If we pack many rows as bytes, we get a contiguous array of u8 bin indices which is simpler to copy to GPU.

### Data structure sketch
- `FeatureQuantization`: per-feature histogram cuts (Vec<f32>)
- `QuantizedRow`: contiguous `Vec<u8>` of length `n_features` or block-packed arrays for CSR-like layouts

### Implementation note
During prediction we can either use original feature values (fallback) or the quantized indices if offered by DMatrix. When quantized, prediction kernel only needs to use integer comparisons.

## Training vs Inference
- Training: Quantization usually originates in training (histogram cut points). Building good quantiles/histogram-based cuts is a training-time activity; training benefits from streaming or dynamic quantization for histogram updates. Training can operate on raw floats and compute quantization cuts during the fit stage.
- Inference: Once cuts are computed, you can quantize and pack feature values offline and use compact integer representations for fast prediction on CPU/GPU. Using quantized values on inference avoids float-to-bin computation at runtime and improves throughput.
- Notes: The quantization mapping (cuts) must be saved alongside the model so inference uses the same bins. Some adaptive quantization strategies may require re-quantization when deploying to different data distributions.
