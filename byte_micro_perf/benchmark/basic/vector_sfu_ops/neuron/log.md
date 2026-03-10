# Log Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_sfu_ops/neuron/log.json`

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
| log | float32  | 256×1024                       |         573.263 |        3.658 |
| log | float32  | 512×1024                       |         579.033 |        7.244 |
| log | float32  | 1024×1024                      |         496.307 |       16.902 |
| log | bfloat16 | 256×1024                       |         477.022 |        2.198 |
| log | bfloat16 | 512×1024                       |         574.352 |        3.651 |
| log | bfloat16 | 1024×1024                      |         440.002 |        9.532 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 440.002 μs | dtype=bfloat16 shape=1024×1024 |
| Worst latency | 579.033 μs | dtype=float32 shape=512×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 440.002 | 574.352 | 497.125 |
| float32 | 496.307 | 579.033 | 549.534 |

### Memory Bandwidth

- **Peak**: 16.902 GB/s — dtype=float32 shape=1024×1024
- **Average**: 7.198 GB/s
- **Min**: 2.198 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
