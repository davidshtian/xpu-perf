# Div Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_sfu_ops/neuron/div.json`

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
| div | float32  | 256×1024                       |         498.184 |        6.314 |
| div | float32  | 512×1024                       |         607.831 |       10.351 |
| div | float32  | 1024×1024                      |         530.537 |       23.717 |
| div | bfloat16 | 256×1024                       |         384.352 |        4.092 |
| div | bfloat16 | 512×1024                       |         523.931 |        6.004 |
| div | bfloat16 | 1024×1024                      |         515.035 |       12.216 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 384.352 μs | dtype=bfloat16 shape=256×1024 |
| Worst latency | 607.831 μs | dtype=float32 shape=512×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 384.352 | 523.931 | 474.439 |
| float32 | 498.184 | 607.831 | 545.517 |

### Memory Bandwidth

- **Peak**: 23.717 GB/s — dtype=float32 shape=1024×1024
- **Average**: 10.449 GB/s
- **Min**: 4.092 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
