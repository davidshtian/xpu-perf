# Mul Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_linear_ops/neuron/mul.json`

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
| mul | float32  | 256×1024                       |         551.323 |        5.706 |
| mul | float32  | 512×1024                       |         578.491 |       10.876 |
| mul | float32  | 1024×1024                      |         597.417 |       21.062 |
| mul | bfloat16 | 256×1024                       |         639.857 |        2.458 |
| mul | bfloat16 | 512×1024                       |         599.637 |        5.246 |
| mul | bfloat16 | 1024×1024                      |         502.474 |       12.521 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 502.474 μs | dtype=bfloat16 shape=1024×1024 |
| Worst latency | 639.857 μs | dtype=bfloat16 shape=256×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 502.474 | 639.857 | 580.656 |
| float32 | 551.323 | 597.417 | 575.744 |

### Memory Bandwidth

- **Peak**: 21.062 GB/s — dtype=float32 shape=1024×1024
- **Average**: 9.645 GB/s
- **Min**: 2.458 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
