# Gemm Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-09
**Backend:** NEURON
**Device:** Device 0
**Device Type:** trn2.3xlarge
**Test Command:** `python launch.py --backend NEURON --device 0 --workload workloads/basic/tensor_gemm_ops/neuron/gemm.json`

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

| Op   | Dtype    | Shape | Latency(μs) | TFLOPS | Mem BW(GB/s) |
|------|----------|-------|-------------|--------|--------------|
| gemm | float32  | M=1024 K=4096 N=4096           |       1,432.278 |   23.990 |       70.282 |
| gemm | float32  | M=4096 K=4096 N=4096           |       4,116.277 |   33.389 |       48.910 |
| gemm | float32  | M=1024 K=8192 N=8192           |       4,113.150 |   33.415 |       81.578 |
| gemm | float32  | M=4096 K=8192 N=8192           |      15,852.370 |   34.680 |       33.867 |
| gemm | float32  | M=1024 K=1024 N=8192           |         896.558 |   19.162 |       79.530 |
| gemm | float32  | M=4096 K=1024 N=8192           |       2,312.322 |   29.719 |       79.811 |
| gemm | float16  | M=1024 K=4096 N=4096           |         698.743 |   49.174 |       72.032 |
| gemm | float16  | M=4096 K=4096 N=4096           |       1,466.460 |   93.722 |       68.644 |
| gemm | float16  | M=1024 K=8192 N=8192           |       1,485.828 |   92.500 |      112.915 |
| gemm | float16  | M=4096 K=8192 N=8192           |       4,353.789 |  126.271 |       61.656 |
| gemm | float16  | M=1024 K=1024 N=8192           |         620.373 |   27.693 |       57.468 |
| gemm | float16  | M=4096 K=1024 N=8192           |         948.150 |   72.477 |       97.321 |
| gemm | bfloat16 | M=1024 K=4096 N=4096           |         697.366 |   49.271 |       72.174 |
| gemm | bfloat16 | M=4096 K=4096 N=4096           |       1,474.179 |   93.231 |       68.284 |
| gemm | bfloat16 | M=1024 K=8192 N=8192           |       1,483.526 |   92.643 |      113.090 |
| gemm | bfloat16 | M=4096 K=8192 N=8192           |       4,300.377 |  127.839 |       62.421 |
| gemm | bfloat16 | M=1024 K=1024 N=8192           |         552.824 |   31.077 |       64.490 |
| gemm | bfloat16 | M=4096 K=1024 N=8192           |         909.070 |   75.593 |      101.504 |

## Performance Analysis

### Latency Summary

| Metric | Value | Configuration |
|--------|-------|---------------|
| Best latency | 552.824 μs | dtype=bfloat16 shape=M=1024 K=1024 N=8192 |
| Worst latency | 15,852.370 μs | dtype=float32 shape=M=4096 K=8192 N=8192 |

### Performance by Data Type

| Dtype | Min Latency(μs) | Max Latency(μs) | Avg Latency(μs) |
|-------|-----------------|-----------------|-----------------|
| bfloat16 | 552.824 | 4,300.377 | 1,569.557 |
| float16 | 620.373 | 4,353.789 | 1,595.557 |
| float32 | 896.558 | 15,852.370 | 4,787.159 |

### Memory Bandwidth

- **Peak**: 113.090 GB/s — dtype=bfloat16 shape=M=1024 K=8192 N=8192
- **Average**: 74.776 GB/s
- **Min**: 33.867 GB/s

## Notes

- Tested on trn2.3xlarge (1 Neuron device, 4 NeuronCores, 24576 MB HBM)
- All results use torch-neuronx 2.9.0.2.12.22436+0f1dac25
