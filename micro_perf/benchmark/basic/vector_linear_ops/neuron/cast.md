# Cast Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/vector_linear_ops/neuron/cast.json`

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
| cast | float32  | 256×1024                       |          18.583 |       84.641 |
| cast | float32  | 512×1024                       |          21.588 |      145.714 |
| cast | float32  | 1024×1024                      |          17.946 |      350.587 |
| cast | bfloat16 | 256×1024                       |          17.710 |       88.812 |
| cast | bfloat16 | 512×1024                       |          20.341 |      154.652 |
| cast | bfloat16 | 1024×1024                      |          17.786 |      353.733 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 17.710 μs | dtype=bfloat16 shape=256×1024 |
| Worst latency | 21.588 μs | dtype=float32 shape=512×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 17.710 | 20.341 | 18.612 |
| float32 | 17.946 | 21.588 | 19.372 |

### Memory Bandwidth

- **Peak**: 353.733 GB/s — dtype=bfloat16 shape=1024×1024
- **Average**: 196.356 GB/s
- **Min**: 84.641 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
