# All-Gather Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-17
**Backend:** NEURON
**Devices:** 64 devices (0-63)
**Device Type:** trn2.48xlarge
**Test Command:** `python3 launch.py --backend NEURON --device 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 --workload workloads/basic/xccl_ops/neuron/all_gather.json`

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
| 64         | float32  | 256        | 1024     | 22.870      | 45.850        | 45.133       |
| 64         | float32  | 512        | 1024     | 26.066      | 80.455        | 79.197       |
| 64         | float32  | 1024       | 1024     | 21.447      | 195.562       | 192.507      |
| 64         | bfloat16 | 256        | 1024     | 25.961      | 20.195        | 19.880       |
| 64         | bfloat16 | 512        | 1024     | 25.271      | 41.492        | 40.844       |
| 64         | bfloat16 | 1024       | 1024     | 21.564      | 97.254        | 95.734       |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 21.447 μs | dtype=float32 batch_size=1024 |
| Worst latency | 26.066 μs | dtype=float32 batch_size=512 |
| Average latency | 23.862 μs | across all configurations |

### Performance by Data Type

| Dtype    | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|----------|-----------------|-----------------|-----------------|
| bfloat16 | 21.564          | 25.961          | 24.265          |
| float32  | 21.447          | 26.066          | 23.461          |

### Algorithm Bandwidth

- **Peak**: 195.562 GB/s — dtype=float32 batch_size=1024
- **Average**: 80.135 GB/s
- **Min**: 20.195 GB/s — dtype=bfloat16 batch_size=256

### Bus Bandwidth

- **Peak**: 192.507 GB/s — dtype=float32 batch_size=1024
- **Average**: 78.866 GB/s
- **Min**: 19.880 GB/s — dtype=bfloat16 batch_size=256

### Bandwidth Scaling by Batch Size

| Batch Size | float32 Algo BW | bfloat16 Algo BW | Speedup (fp32/bf16) |
|------------|-----------------|------------------|---------------------|
| 256        | 45.850 GB/s     | 20.195 GB/s      | 2.27x               |
| 512        | 80.455 GB/s     | 41.492 GB/s      | 1.94x               |
| 1024       | 195.562 GB/s    | 97.254 GB/s      | 2.01x               |

## Optimization Details

### Initialization Strategy

Same as all_reduce, a staggered initialization approach was used to handle 64 concurrent devices:

```python
# core/backend.py:initialize_ccl()
if rank > 0:
    time.sleep(rank * 1.0)  # Stagger initialization by 1 second per rank
```

**Tradeoffs:**
- ✅ Stable initialization across all 64 devices
- ✅ No connection errors during NCCL bootstrap
- ⚠️ Increased initialization time (~63 seconds)

### Configuration Parameters

- **Timeout**: 600 seconds (10 minutes)
- **Master Address**: localhost
- **Device Port**: 49373
- **NCCL Backend**: xla (Neuron XLA)

## Notes

- Tested on trn2.48xlarge (16 Neuron devices, 64 NeuronCores, 24576 MB HBM per device)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
- World size = 64 (all devices participating in collective communication)
- All-gather collects data from all ranks into every rank
- Bus bandwidth calculation: algo_bw × (world_size - 1) / world_size
- float32 achieves ~2x higher bandwidth than bfloat16 across all batch sizes
- Larger batch sizes show significantly better bandwidth utilization

## Comparison with All-Reduce

| Operation  | Peak Latency | Peak Algo BW | Peak Bus BW |
|------------|--------------|--------------|-------------|
| All-Gather | 21.447 μs    | 195.562 GB/s | 192.507 GB/s |
| All-Reduce | 21.119 μs    | 158.026 GB/s | 311.113 GB/s |

**Observations:**
- All-gather shows slightly higher algorithm bandwidth due to one-directional data flow
- All-reduce achieves higher bus bandwidth due to bidirectional ring algorithm
- Similar latency characteristics between the two operations

## Known Limitations

- Single-node 64-device configuration requires staggered initialization
- XLA backend requires task_world_size == world_size (no sub-group support)
- EFA (Elastic Fabric Adapter) warnings are expected and do not affect functionality
