# All-Reduce Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-17
**Backend:** NEURON
**Devices:** 64 devices (0-63)
**Device Type:** trn2.48xlarge
**Test Command:** `python3 launch.py --backend NEURON --device 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 --workload workloads/basic/xccl_ops/neuron/all_reduce.json`

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
| 64         | float32  | 256        | 1024     | 25.213      | 41.589        | 81.879       |
| 64         | float32  | 512        | 1024     | 26.257      | 79.871        | 157.246      |
| 64         | float32  | 1024       | 1024     | 26.542      | 158.026       | 311.113      |
| 64         | bfloat16 | 256        | 1024     | 21.119      | 24.825        | 48.874       |
| 64         | bfloat16 | 512        | 1024     | 21.907      | 47.866        | 94.236       |
| 64         | bfloat16 | 1024       | 1024     | 24.556      | 85.401        | 168.134      |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 21.119 μs | dtype=bfloat16 batch_size=256 |
| Worst latency | 26.542 μs | dtype=float32 batch_size=1024 |
| Average latency | 24.266 μs | across all configurations |

### Performance by Data Type

| Dtype    | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|----------|-----------------|-----------------|-----------------|
| bfloat16 | 21.119          | 24.556          | 22.527          |
| float32  | 25.213          | 26.542          | 26.004          |

### Algorithm Bandwidth

- **Peak**: 158.026 GB/s — dtype=float32 batch_size=1024
- **Average**: 72.930 GB/s
- **Min**: 24.825 GB/s — dtype=bfloat16 batch_size=256

### Bus Bandwidth

- **Peak**: 311.113 GB/s — dtype=float32 batch_size=1024
- **Average**: 143.580 GB/s
- **Min**: 48.874 GB/s — dtype=bfloat16 batch_size=256

### Bandwidth Scaling by Batch Size

| Batch Size | float32 Algo BW | bfloat16 Algo BW | Speedup (fp32/bf16) |
|------------|-----------------|------------------|---------------------|
| 256        | 41.589 GB/s     | 24.825 GB/s      | 1.68x               |
| 512        | 79.871 GB/s     | 47.866 GB/s      | 1.67x               |
| 1024       | 158.026 GB/s    | 85.401 GB/s      | 1.85x               |

## Optimization Details

### Initialization Strategy

Due to NCCL bootstrap bottleneck with 64 concurrent devices, a staggered initialization approach was implemented:

```python
# core/backend.py:initialize_ccl()
if rank > 0:
    time.sleep(rank * 1.0)  # Stagger initialization by 1 second per rank
```

This prevents connection queue overflow at the root node and ensures stable initialization across all 64 devices.

**Tradeoffs:**
- ✅ Stable initialization (no connection errors)
- ✅ Successful completion of all test cases
- ⚠️ Increased initialization time (~63 seconds for all ranks)

### Configuration Parameters

- **Timeout**: 600 seconds (10 minutes)
- **Master Address**: localhost
- **Device Port**: 49373
- **NCCL Backend**: xla (Neuron XLA)

## Notes

- Tested on trn2.48xlarge (16 Neuron devices, 64 NeuronCores, 24576 MB HBM per device)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
- World size = 64 (all devices participating in collective communication)
- All-reduce operation uses ring algorithm with bandwidth scaling factor of (n-1)/n where n=64
- bfloat16 shows better latency but lower bandwidth compared to float32
- Larger batch sizes achieve better bandwidth utilization

## Known Limitations

- Single-node 64-device configuration requires staggered initialization
- XLA backend requires task_world_size == world_size (no sub-group support)
- EFA (Elastic Fabric Adapter) warnings are expected and do not affect functionality
