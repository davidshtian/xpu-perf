# Cos Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_sfu_ops/neuron/cos.json`

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
| cos | float32  | 256×1024                       |         551.535 |        3.802 |
| cos | float32  | 512×1024                       |         600.814 |        6.981 |
| cos | float32  | 1024×1024                      |         604.894 |       13.868 |
| cos | bfloat16 | 256×1024                       |         568.001 |        1.846 |
| cos | bfloat16 | 512×1024                       |         599.828 |        3.496 |
| cos | bfloat16 | 1024×1024                      |         580.283 |        7.228 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 551.535 μs | dtype=float32 shape=256×1024 |
| Worst latency | 604.894 μs | dtype=float32 shape=1024×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 568.001 | 599.828 | 582.704 |
| float32 | 551.535 | 604.894 | 585.748 |

### Memory Bandwidth

- **Peak**: 13.868 GB/s — dtype=float32 shape=1024×1024
- **Average**: 6.204 GB/s
- **Min**: 1.846 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
