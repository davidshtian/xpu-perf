# Rms Norm Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_norm_ops/neuron/rms_norm.json`

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
| rms_norm | float32  | 1024×256                       |          37.354 |       56.170 |
| rms_norm | float32  | 1024×512                       |          37.704 |      111.297 |
| rms_norm | float32  | 1024×1024                      |          36.738 |      228.446 |
| rms_norm | bfloat16 | 1024×256                       |          48.921 |       21.445 |
| rms_norm | bfloat16 | 1024×512                       |          44.689 |       46.951 |
| rms_norm | bfloat16 | 1024×1024                      |          49.408 |       84.933 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 36.738 μs | dtype=float32 shape=1024×1024 |
| Worst latency | 49.408 μs | dtype=bfloat16 shape=1024×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 44.689 | 49.408 | 47.673 |
| float32 | 36.738 | 37.704 | 37.265 |

### Memory Bandwidth

- **Peak**: 228.446 GB/s — dtype=float32 shape=1024×1024
- **Average**: 91.540 GB/s
- **Min**: 21.445 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
