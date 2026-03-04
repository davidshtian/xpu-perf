# Index Select Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-04
**Backend:** NEURON
**Device:** Device 20
**Device Type:** trn2.48xlarge
**Test Command:** `python launch.py --backend NEURON --device 20 --workload workloads/basic/vector_index_ops/neuron/index_select.json`

## System Information

| Attribute | Value |
|-----------|-------|
| Device Name | trn2.48xlarge |
| Device Count | 64 |
| Device Memory | 24576.0 MB |
| Neuron Device Count | 16 |
| Neuron Core Count | 64 |
| Torch Version | 2.9.0+cu128 |
| Torch XLA Version | 2.9.0 |
| NeuronX CC Version | 2.23.6484.0+3b612583 |
| Torch NeuronX Version | 2.9.0.2.12.22436+0f1dac25 |

## Performance Results

| Op           | Dtype    | Src Batch | Dst Batch | Dim Size | Latency(μs) | Mem BW(GB/s) |
|--------------|----------|-----------|-----------|----------|-------------|--------------|
| index_select | float32  | 1024      | 1024      | 128      | 97.78       | 10.808       |
| index_select | float32  | 1024      | 1024      | 256      | 49.98       | 42.119       |
| index_select | float32  | 1024      | 1024      | 512      | 29.58       | 142.058      |
| index_select | float32  | 1024      | 1024      | 1024     | 29.59       | 283.722      |
| index_select | bfloat16 | 1024      | 1024      | 128      | 312.98      | 1.701        |
| index_select | bfloat16 | 1024      | 1024      | 256      | 156.55      | 6.750        |
| index_select | bfloat16 | 1024      | 1024      | 512      | 90.80       | 23.186       |
| index_select | bfloat16 | 1024      | 1024      | 1024     | 58.64       | 71.667       |

## Performance Analysis

### Key Findings

**Best Performance:**
- **Float32 @ dim_size=1024**: Latency 29.59 μs, Memory BW 283.72 GB/s (highest bandwidth)
- **Float32 @ dim_size=512**: Latency 29.58 μs, Memory BW 142.06 GB/s
- **Float32 @ dim_size=256**: Latency 49.98 μs, Memory BW 42.12 GB/s

**Performance by Data Type:**
- **Float32**: Consistently better performance with latencies 29.59-97.78 μs
- **BFloat16**: Higher latencies (58.64-312.98 μs) but acceptable bandwidth at larger dimensions

**Performance by Dimension Size:**
- **Larger dim_size = Better Performance**: Memory bandwidth scales significantly with dimension size
- **dim_size=1024**: 283.72 GB/s (float32) vs 71.67 GB/s (bfloat16)
- **dim_size=128**: 10.81 GB/s (float32) vs 1.70 GB/s (bfloat16)

### Trends

1. **Dimension Size Impact**: Performance improves dramatically as dimension size increases
2. **Float32 Advantage**: Float32 provides 3-4x better memory bandwidth than bfloat16
3. **Latency Consistency**: Float32 maintains stable latency (~30μs) for dim_size ≥ 512

### Recommendations

1. **For Maximum Performance**: Use float32 with dim_size ≥ 512
2. **For Memory Efficiency**: dim_size=1024 provides optimal memory bandwidth utilization
3. **BFloat16 Consideration**: Acceptable for dim_size=1024 if memory savings are critical (71.67 GB/s)
