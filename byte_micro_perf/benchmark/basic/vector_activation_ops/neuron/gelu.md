# Gelu Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_activation_ops/neuron/gelu.json`

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
| gelu | float32  | 1×1024                         |          28.963 |        0.283 |
| gelu | float32  | 16×1024                        |          27.682 |        4.735 |
| gelu | float32  | 64×1024                        |          33.021 |       15.878 |
| gelu | float32  | 256×1024                       |          32.107 |       65.318 |
| gelu | float32  | 1024×1024                      |          27.630 |      303.602 |
| gelu | float16  | 1×1024                         |          28.368 |        0.144 |
| gelu | float16  | 16×1024                        |          27.787 |        2.359 |
| gelu | float16  | 64×1024                        |          27.669 |        9.474 |
| gelu | float16  | 256×1024                       |          28.420 |       36.896 |
| gelu | float16  | 1024×1024                      |          28.255 |      148.447 |
| gelu | bfloat16 | 1×1024                         |          28.823 |        0.142 |
| gelu | bfloat16 | 16×1024                        |          27.429 |        2.389 |
| gelu | bfloat16 | 64×1024                        |          29.035 |        9.029 |
| gelu | bfloat16 | 256×1024                       |          28.138 |       37.265 |
| gelu | bfloat16 | 1024×1024                      |          32.397 |      129.464 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 27.429 μs | dtype=bfloat16 shape=16×1024 |
| Worst latency | 33.021 μs | dtype=float32 shape=64×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 27.429 | 32.397 | 29.164 |
| float16 | 27.669 | 28.420 | 28.100 |
| float32 | 27.630 | 33.021 | 29.881 |

### Memory Bandwidth

- **Peak**: 303.602 GB/s — dtype=float32 shape=1024×1024
- **Average**: 51.028 GB/s
- **Min**: 0.142 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
