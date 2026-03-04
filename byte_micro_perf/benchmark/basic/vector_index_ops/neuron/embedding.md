# Embedding Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-04
**Backend:** NEURON
**Device:** Device 10
**Device Type:** trn2.48xlarge
**Test Command:** `python launch.py --backend NEURON --device 10 --workload workloads/basic/vector_index_ops/neuron/embedding.json`

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

| Op        | Dtype    | Src Batch | Dst Batch | Dim Size | Latency(μs) | Mem BW(GB/s) |
|-----------|----------|-----------|-----------|----------|-------------|--------------|
| embedding | float32  | 1024      | 1024      | 128      | 95.22       | 11.098       |
| embedding | float32  | 1024      | 1024      | 256      | 51.84       | 40.611       |
| embedding | float32  | 1024      | 1024      | 512      | 29.41       | 142.892      |
| embedding | float32  | 1024      | 1024      | 1024     | 19.85       | 423.102      |
| embedding | bfloat16 | 1024      | 1024      | 128      | 350.43      | 1.519        |
| embedding | bfloat16 | 1024      | 1024      | 256      | 149.66      | 7.061        |
| embedding | bfloat16 | 1024      | 1024      | 512      | 70.77       | 29.750       |
| embedding | bfloat16 | 1024      | 1024      | 1024     | 35.29       | 119.080      |

## Performance Analysis

### Key Findings

**Best Performance:**
- **Float32 @ dim_size=1024**: Latency 19.85 μs, Memory BW 423.10 GB/s (highest bandwidth)
- **Float32 @ dim_size=512**: Latency 29.41 μs, Memory BW 142.89 GB/s

**Performance by Data Type:**
- **Float32**: Consistently lower latency (19.85-95.22 μs) and higher memory bandwidth
- **BFloat16**: Higher latency (35.29-350.43 μs) but still achieves good bandwidth at larger dim sizes

**Performance by Dimension Size:**
- **Larger dim_size = Better Performance**: Memory bandwidth increases significantly with dimension size
- **dim_size=1024**: 423.10 GB/s (float32) vs 119.08 GB/s (bfloat16)
- **dim_size=128**: 11.10 GB/s (float32) vs 1.52 GB/s (bfloat16)

### Trends

1. **Dimension Size Impact**: Performance improves dramatically as dimension size increases from 128 to 1024
2. **Float32 Advantage**: Float32 provides 2-4x better performance than bfloat16 across all configurations
3. **Memory Bandwidth**: Scales well with dimension size, reaching peak of 423 GB/s

### Recommendations

1. **For Maximum Performance**: Use float32 with dim_size ≥ 512
2. **For Large Embeddings**: dim_size=1024 provides optimal memory bandwidth utilization
3. **Trade-off Consideration**: If memory is limited, bfloat16 at dim_size=1024 still provides 119 GB/s bandwidth
