#ifndef FMM_H_
#define FMM_H_

#if defined (__cplusplus)
extern "C" {
#endif // __cplusplus

#include <stdbool.h>
#include <stddef.h>

//    k            n         n
// m      *   k        = m 
// 
void FmmMatMulf(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k);
void FmmMatMuld(const double* matA, const double* matB, double* matC, size_t size_m, size_t size_n, size_t size_k);

#if defined (__cplusplus)
}
#endif // __cplusplus

#endif // FMM_H_

