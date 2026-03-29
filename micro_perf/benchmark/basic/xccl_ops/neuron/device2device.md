# Device2Device Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/xccl_ops/neuron/device2device.json`

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
| device2device | float32  | 256×1024                       |          18.809 |      111.497 |
| device2device | float32  | 512×1024                       |          15.553 |      269.683 |
| device2device | float32  | 1024×1024                      |          15.810 |      530.589 |
| device2device | bfloat16 | 256×1024                       |          15.634 |       67.071 |
| device2device | bfloat16 | 512×1024                       |          15.518 |      135.141 |
| device2device | bfloat16 | 1024×1024                      |          15.465 |      271.206 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 15.465 μs | dtype=bfloat16 shape=1024×1024 |
| Worst latency | 18.809 μs | dtype=float32 shape=256×1024 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 15.465 | 15.634 | 15.539 |
| float32 | 15.553 | 18.809 | 16.724 |

### Memory Bandwidth

- **Peak**: 530.589 GB/s — dtype=float32 shape=1024×1024
- **Average**: 230.865 GB/s
- **Min**: 67.071 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
