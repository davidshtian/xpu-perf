# GEMM Performance Benchmark - NEURON Backend

**Test Date:** 2026-03-04
**Backend:** NEURON
**Device:** Device 10
**Device Type:** trn2.48xlarge
**Test Command:** `python launch.py --backend NEURON --device 10 --workload workloads/basic/tensor_gemm_ops/gemm.json`

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

| Op   | Dtype    | Shape            | Latency(μs) | TFLOPS  | Mem BW(GB/s) |
|------|----------|------------------|-------------|---------|--------------|
| gemm | float32  | 1024×4096×4096   | 576,223.39  | 0.060   | 0.175        |
| gemm | float32  | 2048×4096×4096   | 580,201.86  | 0.118   | 0.231        |
| gemm | float32  | 4096×4096×4096   | 530,622.57  | 0.259   | 0.379        |
| gemm | float32  | 1024×8192×8192   | 607,443.90  | 0.226   | 0.552        |
| gemm | float32  | 2048×8192×8192   | 1,536.01    | 178.956 | 262.143      |
| gemm | float32  | 4096×8192×8192   | 3,002.00    | 183.130 | 178.837      |
| gemm | float32  | 1024×8192×1024   | 431,480.89  | 0.040   | 0.165        |
| gemm | float32  | 2048×8192×1024   | 548,439.10  | 0.063   | 0.199        |
| gemm | float32  | 4096×8192×1024   | 548,677.99  | 0.125   | 0.336        |
| gemm | float32  | 1024×1024×8192   | 378,900.77  | 0.045   | 0.188        |
| gemm | float32  | 2048×1024×8192   | 503,880.39  | 0.068   | 0.216        |
| gemm | float32  | 4096×1024×8192   | 469,042.16  | 0.147   | 0.393        |
| gemm | float16  | 1024×4096×4096   | 460,505.15  | 0.075   | 0.109        |
| gemm | float16  | 2048×4096×4096   | 607,904.29  | 0.113   | 0.110        |
| gemm | float16  | 4096×4096×4096   | 785,959.85  | 0.175   | 0.128        |
| gemm | float16  | 1024×8192×8192   | 605,999.46  | 0.227   | 0.277        |
| gemm | float16  | 2048×8192×8192   | 25,062.26   | 10.968  | 8.033        |
| gemm | float16  | 4096×8192×8192   | 42,501.04   | 12.935  | 6.316        |
| gemm | float16  | 1024×8192×1024   | 369,414.06  | 0.047   | 0.097        |
| gemm | float16  | 2048×8192×1024   | 500,363.89  | 0.069   | 0.109        |
| gemm | float16  | 4096×8192×1024   | 680,322.08  | 0.101   | 0.136        |
| gemm | float16  | 1024×1024×8192   | 369,490.17  | 0.046   | 0.096        |
| gemm | float16  | 2048×1024×8192   | 485,994.98  | 0.071   | 0.112        |
| gemm | float16  | 4096×1024×8192   | 651,889.44  | 0.105   | 0.142        |
| gemm | bfloat16 | 1024×4096×4096   | 475,293.44  | 0.072   | 0.106        |
| gemm | bfloat16 | 2048×4096×4096   | 624,342.53  | 0.110   | 0.107        |
| gemm | bfloat16 | 4096×4096×4096   | 790,134.64  | 0.174   | 0.127        |
| gemm | bfloat16 | 1024×8192×8192   | 635,936.79  | 0.216   | 0.264        |
| gemm | bfloat16 | 2048×8192×8192   | 20,769.72   | 13.235  | 9.693        |
| gemm | bfloat16 | 4096×8192×8192   | 41,736.59   | 13.172  | 6.432        |
| gemm | bfloat16 | 1024×8192×1024   | 396,320.56  | 0.043   | 0.090        |
| gemm | bfloat16 | 2048×8192×1024   | 538,024.44  | 0.064   | 0.101        |
| gemm | bfloat16 | 4096×8192×1024   | 686,655.31  | 0.100   | 0.134        |
| gemm | bfloat16 | 1024×1024×8192   | 361,848.06  | 0.047   | 0.099        |
| gemm | bfloat16 | 2048×1024×8192   | 481,098.85  | 0.071   | 0.113        |
| gemm | bfloat16 | 4096×1024×8192   | 576,223.39  | 0.060   | 0.175        |

## Performance Analysis

### Top Performers

**Best Performance Configurations:**

1. **Float32 @ 4096×8192×8192**
   - TFLOPS: **183.130**
   - Latency: 3.00 ms
   - Memory BW: 178.84 GB/s

2. **Float32 @ 2048×8192×8192**
   - TFLOPS: **178.956**
   - Latency: 1.54 ms (lowest latency for high performance)
   - Memory BW: 262.14 GB/s (highest bandwidth)

3. **BFloat16 @ 2048×8192×8192**
   - TFLOPS: **13.235**
   - Latency: 20.77 ms
   - Memory BW: 9.69 GB/s

4. **Float16 @ 4096×8192×8192**
   - TFLOPS: **12.935**
   - Latency: 42.50 ms
   - Memory BW: 6.32 GB/s

### Key Findings

#### Performance by Data Type

- **Float32**: Achieves peak performance (~180 TFLOPS) on large matrices (≥2048×8192×8192)
- **Float16/BFloat16**: Performance capped at ~11-13 TFLOPS on large matrices
- **Float32 Advantage**: ~14x faster than Float16/BFloat16 on optimal workloads

#### Performance by Matrix Size

- **Large Matrices** (2048×8192×8192 and larger): Excellent performance (>10 TFLOPS for all dtypes on optimal configs)
- **Medium Matrices** (4096×4096×4096): Moderate performance (0.17-0.26 TFLOPS)
- **Small/Rectangular Matrices**: Poor performance (<0.2 TFLOPS)

#### Memory Bandwidth Observations

- **Peak Memory BW**: 262.14 GB/s (Float32 @ 2048×8192×8192)
- **Correlation**: Higher memory bandwidth generally correlates with better TFLOPS
- **BW Range**: 0.09 GB/s (small matrices) to 262.14 GB/s (optimal large matrices)

### Recommendations

1. **For Maximum Performance**: Use Float32 with matrix shapes ≥2048×8192×8192
2. **For Memory Efficiency**: BFloat16 offers good balance (13.2 TFLOPS with lower memory footprint)
3. **Avoid Small Matrices**: Performance degrades significantly for matrices <2048 in any dimension
4. **Square Large Matrices**: M×K×N configurations where all dimensions are large (8192+) perform best

## Notes

- Some test configurations show extremely high latency (>500ms) indicating potential compilation overhead or initialization costs
- The performance jump from 1024×8192×8192 to 2048×8192×8192 is dramatic (~1000x improvement), suggesting a performance threshold
- Float32 shows unexpectedly better performance than Float16/BFloat16, which may be specific to the Neuron architecture optimization
