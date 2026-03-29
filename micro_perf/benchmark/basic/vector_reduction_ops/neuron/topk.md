# Topk Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_reduction_ops/neuron/topk.json`

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
| topk | float32  | 1024×256                       |          19.773 |      106.061 |
| topk | float32  | 1024×256                       |          19.662 |      106.661 |
| topk | float32  | 1024×512                       |          19.784 |      212.005 |
| topk | float32  | 1024×512                       |          21.821 |      192.213 |
| topk | float32  | 1024×1024                      |          19.271 |      435.299 |
| topk | float32  | 1024×1024                      |          19.599 |      428.008 |
| topk | bfloat16 | 1024×256                       |          21.657 |       48.418 |
| topk | bfloat16 | 1024×256                       |          19.756 |       53.075 |
| topk | bfloat16 | 1024×512                       |          32.382 |       64.763 |
| topk | bfloat16 | 1024×512                       |          19.665 |      106.646 |
| topk | bfloat16 | 1024×1024                      |          19.764 |      212.224 |
| topk | bfloat16 | 1024×1024                      |          19.644 |      213.514 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 19.271 μs | dtype=float32 shape=1024×1024 |
| Worst latency | 32.382 μs | dtype=bfloat16 shape=1024×512 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 19.644 | 32.382 | 22.145 |
| float32 | 19.271 | 21.821 | 19.985 |

### Memory Bandwidth

- **Peak**: 435.299 GB/s — dtype=float32 shape=1024×1024
- **Average**: 181.574 GB/s
- **Min**: 48.418 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
