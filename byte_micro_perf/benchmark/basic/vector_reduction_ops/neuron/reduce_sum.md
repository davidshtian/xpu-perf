# Reduce Sum Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_reduction_ops/neuron/reduce_sum.json`

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
| reduce_sum | float32  | 1024×256                       |          16.146 |       65.199 |
| reduce_sum | float32  | 1024×512                       |          16.383 |      128.257 |
| reduce_sum | float32  | 1024×1024                      |          16.388 |      256.184 |
| reduce_sum | bfloat16 | 1024×256                       |          15.926 |       33.049 |
| reduce_sum | bfloat16 | 1024×512                       |          16.155 |       65.033 |
| reduce_sum | bfloat16 | 1024×1024                      |          16.008 |      131.138 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 15.926 μs | dtype=bfloat16 shape=1024×256 |
| Worst latency | 16.388 μs | dtype=float32 shape=1024×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 15.926 | 16.155 | 16.030 |
| float32 | 16.146 | 16.388 | 16.306 |

### Memory Bandwidth

- **Peak**: 256.184 GB/s — dtype=float32 shape=1024×1024
- **Average**: 113.143 GB/s
- **Min**: 33.049 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
