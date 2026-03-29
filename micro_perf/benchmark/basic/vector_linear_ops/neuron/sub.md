# Sub Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_linear_ops/neuron/sub.json`

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
| sub | float32  | 256×1024                       |         587.394 |        5.355 |
| sub | float32  | 512×1024                       |         484.139 |       12.995 |
| sub | float32  | 1024×1024                      |         511.598 |       24.595 |
| sub | bfloat16 | 256×1024                       |         445.800 |        3.528 |
| sub | bfloat16 | 512×1024                       |         542.543 |        5.798 |
| sub | bfloat16 | 1024×1024                      |         477.899 |       13.165 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 445.800 μs | dtype=bfloat16 shape=256×1024 |
| Worst latency | 587.394 μs | dtype=float32 shape=256×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 445.800 | 542.543 | 488.747 |
| float32 | 484.139 | 587.394 | 527.710 |

### Memory Bandwidth

- **Peak**: 24.595 GB/s — dtype=float32 shape=1024×1024
- **Average**: 10.906 GB/s
- **Min**: 3.528 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
