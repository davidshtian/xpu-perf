# Add Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_linear_ops/neuron/add.json`

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
| add | float32  | 256×1024                       |       1,126.101 |        2.793 |
| add | float32  | 512×1024                       |       1,073.265 |        5.862 |
| add | float32  | 1024×1024                      |       1,090.355 |       11.540 |
| add | bfloat16 | 256×1024                       |         958.008 |        1.642 |
| add | bfloat16 | 512×1024                       |       1,099.311 |        2.862 |
| add | bfloat16 | 1024×1024                      |       1,026.745 |        6.128 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 958.008 μs | dtype=bfloat16 shape=256×1024 |
| Worst latency | 1,126.101 μs | dtype=float32 shape=256×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 958.008 | 1,099.311 | 1,028.021 |
| float32 | 1,073.265 | 1,126.101 | 1,096.574 |

### Memory Bandwidth

- **Peak**: 11.540 GB/s — dtype=float32 shape=1024×1024
- **Average**: 5.138 GB/s
- **Min**: 1.642 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
