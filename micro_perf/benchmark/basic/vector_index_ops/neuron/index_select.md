# Index Select Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_index_ops/neuron/index_select.json`

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
| index_select | float32  | src=1024×128 dst=1024×128      |          17.501 |       60.384 |
| index_select | float32  | src=1024×256 dst=1024×256      |          17.216 |      122.293 |
| index_select | float32  | src=1024×512 dst=1024×512      |          17.323 |      242.595 |
| index_select | float32  | src=1024×1024 dst=1024×1024    |          20.729 |      405.075 |
| index_select | bfloat16 | src=1024×128 dst=1024×128      |          17.279 |       30.816 |
| index_select | bfloat16 | src=1024×256 dst=1024×256      |          16.917 |       62.467 |
| index_select | bfloat16 | src=1024×512 dst=1024×512      |          17.383 |      121.118 |
| index_select | bfloat16 | src=1024×1024 dst=1024×1024    |          17.365 |      242.015 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 16.917 μs | dtype=bfloat16 shape=src=1024×256 dst=1024×256 |
| Worst latency | 20.729 μs | dtype=float32 shape=src=1024×1024 dst=1024×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 16.917 | 17.383 | 17.236 |
| float32 | 17.216 | 20.729 | 18.192 |

### Memory Bandwidth

- **Peak**: 405.075 GB/s — dtype=float32 shape=src=1024×1024 dst=1024×1024
- **Average**: 160.845 GB/s
- **Min**: 30.816 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
