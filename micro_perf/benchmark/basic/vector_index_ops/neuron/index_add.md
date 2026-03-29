# Index Add Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_index_ops/neuron/index_add.json`

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
| index_add | float32  | src=1024×256 dst=1024×256      |         940.108 |        3.355 |
| index_add | float32  | src=1024×512 dst=1024×512      |         921.370 |        6.837 |
| index_add | float32  | src=1024×1024 dst=1024×1024    |       1,002.086 |       12.565 |
| index_add | bfloat16 | src=1024×256 dst=1024×256      |         694.707 |        2.276 |
| index_add | bfloat16 | src=1024×512 dst=1024×512      |         721.466 |        4.372 |
| index_add | bfloat16 | src=1024×1024 dst=1024×1024    |         813.820 |        7.741 |
| index_add | float32  | src=512×1024 dst=512×1024      |         571.006 |       11.025 |
| index_add | float32  | src=1024×1024 dst=1024×1024    |         840.472 |       14.981 |
| index_add | bfloat16 | src=512×1024 dst=512×1024      |         580.129 |        5.430 |
| index_add | bfloat16 | src=1024×1024 dst=1024×1024    |         797.489 |        7.899 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 571.006 μs | dtype=float32 shape=src=512×1024 dst=512×1024 |
| Worst latency | 1,002.086 μs | dtype=float32 shape=src=1024×1024 dst=1024×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 580.129 | 813.820 | 721.522 |
| float32 | 571.006 | 1,002.086 | 855.008 |

### Memory Bandwidth

- **Peak**: 14.981 GB/s — dtype=float32 shape=src=1024×1024 dst=1024×1024
- **Average**: 7.648 GB/s
- **Min**: 2.276 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
