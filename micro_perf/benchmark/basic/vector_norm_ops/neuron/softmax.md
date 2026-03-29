# Softmax Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_norm_ops/neuron/softmax.json`

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
| softmax | float32  | 1024×256                       |          19.913 |      105.315 |
| softmax | float32  | 1024×512                       |          16.889 |      248.341 |
| softmax | float32  | 1024×1024                      |          16.796 |      499.432 |
| softmax | bfloat16 | 1024×256                       |          17.379 |       60.337 |
| softmax | bfloat16 | 1024×512                       |          16.771 |      125.043 |
| softmax | bfloat16 | 1024×1024                      |          16.823 |      249.317 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 16.771 μs | dtype=bfloat16 shape=1024×512 |
| Worst latency | 19.913 μs | dtype=float32 shape=1024×256 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 16.771 | 17.379 | 16.991 |
| float32 | 16.796 | 19.913 | 17.866 |

### Memory Bandwidth

- **Peak**: 499.432 GB/s — dtype=float32 shape=1024×1024
- **Average**: 214.631 GB/s
- **Min**: 60.337 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
