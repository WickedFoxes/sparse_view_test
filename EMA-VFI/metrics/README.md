# Metrics
Metrics for image restoration based on pytorch, metrics for other vision tasks will be added later.

## Convention
### Input
- Base : Image shape (C, H, W), type float32/64 in range [0, 1]
- Batchwise : Image shape (B, C, H, W), type float32/64 in range [0, 1]
- AKLD / KLD : Input is not noise map (synthetic, real) $\rightarrow$ images (synthetic, real, clean)
### Output
- Float32/64 scalar

### Name
```python
metrics
├── metric_name.py # module
|   ├── def metric_name # implementation target
|   └── def batchwise_metric_name # wrapped by _batch_wise decorator
├── _batch_wise.py # decorator for batchwise metric
└── _batch_wise.py # decorator for batchwise metric
```
File name and function name should be same.