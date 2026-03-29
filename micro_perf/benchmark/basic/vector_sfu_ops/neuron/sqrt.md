# Sqrt Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_sfu_ops/neuron/sqrt.json`

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
| sqrt | float32  | 256×1024                       |         399.843 |        5.245 |
| sqrt | float32  | 512×1024                       |         529.569 |        7.920 |
| sqrt | float32  | 1024×1024                      |         467.651 |       17.938 |
| sqrt | bfloat16 | 256×1024                       |         454.699 |        2.306 |
| sqrt | bfloat16 | 512×1024                       |         464.711 |        4.513 |
| sqrt | bfloat16 | 1024×1024                      |         543.661 |        7.715 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 399.843 μs | dtype=float32 shape=256×1024 |
| Worst latency | 543.661 μs | dtype=bfloat16 shape=1024×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 454.699 | 543.661 | 487.690 |
| float32 | 399.843 | 529.569 | 465.688 |

### Memory Bandwidth

- **Peak**: 17.938 GB/s — dtype=float32 shape=1024×1024
- **Average**: 7.606 GB/s
- **Min**: 2.306 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
