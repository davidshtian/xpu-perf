# Layer Norm Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_norm_ops/neuron/layer_norm.json`

## System Information

| Attribute | Value |
|-----------|-------|
| Device Name | trn2.3xlarge |
| Device Count | 4 |
| Device Memory | 24576.0 MB |
| Neuron Device Count | 1 |
| Neuron Core Count | 4 |
| Torch Version | 2.9.0+cu128 |
| Torch XLA Version | 2.9.0 |
| NeuronX CC Version | 2.23.6484.0+3b612583 |
| Torch NeuronX Version | 2.9.0.2.12.22436+0f1dac25 |

## Performance Results

| Op   | Dtype    | Shape | Latency(μs) | Mem BW(GB/s) |
|------|----------|-------|-------------|--------------|
| layer_norm | float32  | 1024×256                       |          47.758 |       43.955 |
| layer_norm | float32  | 1024×512                       |          45.149 |       92.990 |
| layer_norm | float32  | 1024×1024                      |          45.738 |      183.583 |
| layer_norm | bfloat16 | 1024×256                       |          46.981 |       22.341 |
| layer_norm | bfloat16 | 1024×512                       |          46.564 |       45.082 |
| layer_norm | bfloat16 | 1024×1024                      |          41.392 |      101.431 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 41.392 μs | dtype=bfloat16 shape=1024×1024 |
| Worst latency | 47.758 μs | dtype=float32 shape=1024×256 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 41.392 | 46.981 | 44.979 |
| float32 | 45.149 | 47.758 | 46.215 |

### Memory Bandwidth

- **Peak**: 183.583 GB/s — dtype=float32 shape=1024×1024
- **Average**: 81.564 GB/s
- **Min**: 22.341 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
