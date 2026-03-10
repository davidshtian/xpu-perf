# Reduce Max Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_reduction_ops/neuron/reduce_max.json`

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
| reduce_max | float32  | 1024×256                       |          17.663 |       59.828 |
| reduce_max | float32  | 1024×512                       |          17.237 |      122.138 |
| reduce_max | float32  | 1024×1024                      |          17.939 |      234.269 |
| reduce_max | bfloat16 | 1024×256                       |          20.651 |       25.686 |
| reduce_max | bfloat16 | 1024×512                       |          17.160 |       61.462 |
| reduce_max | bfloat16 | 1024×1024                      |          17.410 |      120.806 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 17.160 μs | dtype=bfloat16 shape=1024×512 |
| Worst latency | 20.651 μs | dtype=bfloat16 shape=1024×256 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 17.160 | 20.651 | 18.407 |
| float32 | 17.237 | 17.939 | 17.613 |

### Memory Bandwidth

- **Peak**: 234.269 GB/s — dtype=float32 shape=1024×1024
- **Average**: 104.031 GB/s
- **Min**: 25.686 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
