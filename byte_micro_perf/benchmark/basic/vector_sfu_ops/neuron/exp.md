# Exp Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_sfu_ops/neuron/exp.json`

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
| exp | float32  | 256×1024                       |         371.568 |        5.644 |
| exp | float32  | 512×1024                       |         498.304 |        8.417 |
| exp | float32  | 1024×1024                      |         369.339 |       22.713 |
| exp | bfloat16 | 256×1024                       |         331.037 |        3.168 |
| exp | bfloat16 | 512×1024                       |         487.176 |        4.305 |
| exp | bfloat16 | 1024×1024                      |         461.646 |        9.086 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 331.037 μs | dtype=bfloat16 shape=256×1024 |
| Worst latency | 498.304 μs | dtype=float32 shape=512×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 331.037 | 487.176 | 426.620 |
| float32 | 369.339 | 498.304 | 413.070 |

### Memory Bandwidth

- **Peak**: 22.713 GB/s — dtype=float32 shape=1024×1024
- **Average**: 8.889 GB/s
- **Min**: 3.168 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
