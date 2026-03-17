# Reduce-Scatter Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-17
**Backend:** NEURON
**Devices:** 64 devices (0-63)
**Device Type:** trn2.48xlarge
**Test Command:** `python3 launch.py --backend NEURON --device 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 --workload workloads/basic/xccl_ops/neuron/reduce_scatter.json`

## System Information

| Attribute | Value |
|-----------|-------|
| Device Name | trn2.48xlarge |
| Device Count | 64 |
| Device Memory | 24576.0 MB (per device) |
| Neuron Device Count | 16 |
| Neuron Core Count | 64 |
| Torch Version | 2.9.0+cu128 |
| Torch XLA Version | 2.9.0 |
| NeuronX CC Version | 2.23.6484.0+3b612583 |
| Torch NeuronX Version | 2.9.0.2.12.22436+0f1dac25 |

## Performance Results

| World Size | Dtype    | Batch Size | Dim Size | Latency(μs) | Algo BW(GB/s) | Bus BW(GB/s) |
|------------|----------|------------|----------|-------------|---------------|--------------|
| 64         | float32  | 256        | 1024     | 643.091     | 1.631         | 1.605        |
| 64         | float32  | 512        | 1024     | 671.648     | 3.122         | 3.074        |
| 64         | float32  | 1024       | 1024     | 581.558     | 7.212         | 7.099        |
| 64         | bfloat16 | 256        | 1024     | 689.832     | 0.760         | 0.748        |
| 64         | bfloat16 | 512        | 1024     | 648.369     | 1.617         | 1.592        |
| 64         | bfloat16 | 1024       | 1024     | 708.710     | 2.959         | 2.913        |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 581.558 μs | dtype=float32 batch_size=1024 |
| Worst latency | 708.710 μs | dtype=bfloat16 batch_size=1024 |
| Average latency | 657.201 μs | across all configurations |

### Performance by Data Type

| Dtype    | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|----------|-----------------|-----------------|-----------------|
| bfloat16 | 648.369         | 708.710         | 682.304         |
| float32  | 581.558         | 671.648         | 632.099         |

### Algorithm Bandwidth

- **Peak**: 7.212 GB/s — dtype=float32 batch_size=1024
- **Average**: 2.884 GB/s
- **Min**: 0.760 GB/s — dtype=bfloat16 batch_size=256

### Bus Bandwidth

- **Peak**: 7.099 GB/s — dtype=float32 batch_size=1024
- **Average**: 2.839 GB/s
- **Min**: 0.748 GB/s — dtype=bfloat16 batch_size=256

## Notes

- Tested on trn2.48xlarge (16 Neuron devices, 64 NeuronCores, 24576 MB HBM per device)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
- World size = 64 (all devices participating in collective communication)
- Reduce-scatter combines reduction with scattering results across ranks
- **Performance Note**: Reduce-scatter shows significantly higher latency (~600μs) compared to other collective operations (~20-25μs)
- The low bandwidth may indicate optimization opportunities or implementation limitations in the XLA backend

## Comparison with Other Collectives

| Operation       | Latency Range | Algo BW Peak   |
|-----------------|---------------|----------------|
| All-Reduce      | 21-27 μs      | 158.026 GB/s   |
| All-Gather      | 21-26 μs      | 195.562 GB/s   |
| Reduce-Scatter  | 581-709 μs    | 7.212 GB/s     |

**Key Observation**: Reduce-scatter is significantly slower (~25-30x higher latency) than all-reduce and all-gather operations on 64 devices.

## Known Limitations

- Single-node 64-device configuration requires staggered initialization
- XLA backend requires task_world_size == world_size (no sub-group support)
- EFA (Elastic Fabric Adapter) warnings are expected and do not affect functionality
- Performance characteristics suggest potential optimization opportunities
