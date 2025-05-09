# fmm

Fast Matrix Multiply

Matrixes should be multiple of size 8x8x8

# Benchmarks

## Comparison with OpenBLAS

### Enviroment
| parameter | value |
|-|-|
| arch | x86-64 |
| cpu | AMD Ryzen 5 3500u (Zen+) |
| L1D cache size | 32 KiB (per core) |
| os | Linux fedora-41 6.13.11 |
| compiler | gcc 14.2.1 |
| compiler options | -march=native -Ofast |
| matrix size | 32x32x32 |
| std::mt19937_64 seed | 1 |
| OpenBLAS verision | 0.3.26 | 

### results

fmm: $(205 \pm 66) \cdot 10^3$ cpu cycles

OpenBLAS: $(203 \pm 62) \cdot 10^3$ cpu cycles

fmm slower than OpenBLAS by 0.9%. 