# Silu Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_activation_ops/neuron/silu.json`

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
| silu | float32  | 1×1024                         |          29.032 |        0.282 |
| silu | float32  | 16×1024                        |          28.025 |        4.677 |
| silu | float32  | 64×1024                        |          33.045 |       15.866 |
| silu | float32  | 256×1024                       |          28.145 |       74.513 |
| silu | float32  | 1024×1024                      |          28.514 |      294.197 |
| silu | float16  | 1×1024                         |          37.649 |        0.109 |
| silu | float16  | 16×1024                        |          32.851 |        1.995 |
| silu | float16  | 64×1024                        |          32.956 |        7.954 |
| silu | float16  | 256×1024                       |          32.672 |       32.094 |
| silu | float16  | 1024×1024                      |          32.841 |      127.717 |
| silu | bfloat16 | 1×1024                         |          32.198 |        0.127 |
| silu | bfloat16 | 16×1024                        |          33.002 |        1.986 |
| silu | bfloat16 | 64×1024                        |          34.392 |        7.622 |
| silu | bfloat16 | 256×1024                       |          32.221 |       32.543 |
| silu | bfloat16 | 1024×1024                      |          32.512 |      129.008 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 28.025 μs | dtype=float32 shape=16×1024 |
| Worst latency | 37.649 μs | dtype=float16 shape=1×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 32.198 | 34.392 | 32.865 |
| float16 | 32.672 | 37.649 | 33.794 |
| float32 | 28.025 | 33.045 | 29.352 |

### Memory Bandwidth

- **Peak**: 294.197 GB/s — dtype=float32 shape=1024×1024
- **Average**: 48.713 GB/s
- **Min**: 0.109 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
