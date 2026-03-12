# All Reduce Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-12
**Backend:** NEURON
**Device:** Device 0,1
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0,1 --workload workloads/basic/xccl_ops/neuron/all_reduce.json`

## System Information

| Attribute | Value |
|-----------|-------|
| Device Name | trn2.3xlarge |
| Device Count | 4 |
| Device Memory | 24576.0 MB |
| Neuron Device Count | 1 |
| Neuron Core Count | 4 |
| logical-neuroncore-config | 2 |
| Torch Version | 2.9.0+cu128 |
| Torch XLA Version | 2.9.0 |
| NeuronX CC Version | 2.23.6484.0+3b612583 |
| Torch NeuronX Version | 2.9.0.2.12.22436+0f1dac25 |

## Performance Results

| world_size | dtype | batch_size | dim_size | latency (µs) | algo_size (MB) | algo_bw (GB/s) | bus_bw (GB/s) |
|-----------|-------|-----------|---------|-------------|---------------|----------------|---------------|
| 2 | float32 | 256 | 1024 | 22.835 | 1.000 | 45.919 | 45.919 |
| 2 | float32 | 512 | 1024 | 22.192 | 2.000 | 94.499 | 94.499 |
| 2 | float32 | 1024 | 1024 | 20.689 | 4.000 | 202.729 | 202.729 |
| 2 | bfloat16 | 256 | 1024 | 19.916 | 0.500 | 26.325 | 26.325 |
| 2 | bfloat16 | 512 | 1024 | 20.382 | 1.000 | 51.447 | 51.447 |
| 2 | bfloat16 | 1024 | 1024 | 19.981 | 2.000 | 104.957 | 104.957 |

> world_size=4 cases skipped: trn2.3xlarge with logical-neuroncore-config=2 exposes only 2 logical devices (indices 0 and 1). 4-rank XLA distributed all_reduce is not supported on this instance via torch-xla.

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
- world_size=2 maps to 2 logical NeuronCores (each = 2 physical NCs, logical-neuroncore-config=2)
- Latency is stable at 20–23 µs across all cases
- float32 peak bus bandwidth: 202.7 GB/s (batch_size=1024)
- bfloat16 peak bus bandwidth: 105.0 GB/s (batch_size=1024)
