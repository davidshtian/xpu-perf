# Embedding Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_index_ops/neuron/embedding.json`

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
| embedding | float32  | src=1024×128 dst=1024×128      |          17.439 |       60.597 |
| embedding | float32  | src=1024×256 dst=1024×256      |          20.384 |      103.287 |
| embedding | float32  | src=1024×512 dst=1024×512      |          17.970 |      233.864 |
| embedding | float32  | src=1024×1024 dst=1024×1024    |          18.132 |      463.085 |
| embedding | bfloat16 | src=1024×128 dst=1024×128      |          17.517 |       30.397 |
| embedding | bfloat16 | src=1024×256 dst=1024×256      |          21.435 |       49.300 |
| embedding | bfloat16 | src=1024×512 dst=1024×512      |          17.815 |      118.178 |
| embedding | bfloat16 | src=1024×1024 dst=1024×1024    |          21.513 |      195.350 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 17.439 μs | dtype=float32 shape=src=1024×128 dst=1024×128 |
| Worst latency | 21.513 μs | dtype=bfloat16 shape=src=1024×1024 dst=1024×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 17.517 | 21.513 | 19.570 |
| float32 | 17.439 | 20.384 | 18.481 |

### Memory Bandwidth

- **Peak**: 463.085 GB/s — dtype=float32 shape=src=1024×1024 dst=1024×1024
- **Average**: 156.757 GB/s
- **Min**: 30.397 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
