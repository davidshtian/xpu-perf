# Sin Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_sfu_ops/neuron/sin.json`

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
| sin | float32  | 256×1024                       |         569.243 |        3.684 |
| sin | float32  | 512×1024                       |         610.944 |        6.865 |
| sin | float32  | 1024×1024                      |         577.097 |       14.536 |
| sin | bfloat16 | 256×1024                       |         581.471 |        1.803 |
| sin | bfloat16 | 512×1024                       |         585.294 |        3.583 |
| sin | bfloat16 | 1024×1024                      |         574.272 |        7.304 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 569.243 μs | dtype=float32 shape=256×1024 |
| Worst latency | 610.944 μs | dtype=float32 shape=512×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 574.272 | 585.294 | 580.346 |
| float32 | 569.243 | 610.944 | 585.761 |

### Memory Bandwidth

- **Peak**: 14.536 GB/s — dtype=float32 shape=1024×1024
- **Average**: 6.296 GB/s
- **Min**: 1.803 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
