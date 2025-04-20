#include "fmm.h"

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

// static ---------------------------------------------------------------------

static size_t TranslateInd(size_t n, size_t m, size_t N) {
    return m * N + n;
}

static void FmmMatMulfNaive(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k);
static void FmmMatMulfKernel(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k);

static void FmmKernel2x8(const float* A_start, const float* B_start, float* C_start, size_t size_m, size_t size_n, size_t size_k);
static void FmmMicroKernel2x8x8(const float* A_start, const float* B_start, float* C_start, size_t size_m, size_t size_n, size_t size_k);

    // global ---------------------------------------------------------------------

//    k            n         n
// m      *   k        = m 
// 
void FmmMatMulf(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k) {
#if defined (NAIVE)
    FmmMatMulfNaive(matA, matB, matC, size_m, size_n, size_k);    
#elif defined (KERNEL)
    FmmMatMulfKernel(matA, matB, matC, size_m, size_n, size_k);
#else 
    FmmMatMulfKernel(matA, matB, matC, size_m, size_n, size_k);    
#endif // NAIVE TRANSPOSE
}

// static ---------------------------------------------------------------------

static void FmmMatMulfNaive(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k) {
    for (size_t im = 0; im < size_m; im++) {
        for (size_t in = 0; in < size_n; in++) {
            matC[size_n * im + in] = 0;
            // inner loop
            for (size_t ik = 0; ik < size_k; ik++) {
                matC[TranslateInd(in, im, size_n)] += matA[TranslateInd(ik, im, size_k)] * matB[TranslateInd(in, ik, size_n)];
            }
        }
    }
}

static const size_t block_size_m = 2;
static const size_t block_size_n = 8;
static const size_t block_size_k = 8;

static void FmmMatMulfKernel(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k) {
    memset(matC, 0, size_m * size_n * sizeof(*matC));

    for (size_t im = 0; im < size_m; im += block_size_m) {
        for (size_t in = 0; in < size_n; in += block_size_n) {
            FmmKernel2x8(
                &matA[TranslateInd(0, im, size_k)], 
                &matB[TranslateInd(in, 0, size_n)], 
                &matC[TranslateInd(in, im, size_n)], 
                size_m, size_n, size_k
            );
        }
    }
}

static void FmmKernel2x8(const float* A_start, const float* B_start, float* C_start, size_t size_m, size_t size_n, size_t size_k) {    
    for (size_t k = 0; k < size_k; k += block_size_k) {
        FmmMicroKernel2x8x8(
            &A_start[TranslateInd(k, 0, size_k)], 
            &B_start[TranslateInd(0, k, size_n)], 
            &C_start[TranslateInd(0, 0, size_n)], 
            size_m, size_n, size_k
        );
    }
}

static void FmmMicroKernel2x8x8(const float* A_start, const float* B_start, float* C_start, size_t size_m, size_t size_n, size_t size_k) {
    __m256 lineA1 = _mm256_loadu_ps(&A_start[TranslateInd(0, 0, size_k)]);
    __m256 lineA2 = _mm256_loadu_ps(&A_start[TranslateInd(0, 1, size_k)]);

    __m256 lineC1 = _mm256_loadu_ps(&C_start[TranslateInd(0, 0, size_n)]);
    __m256 lineC2 = _mm256_loadu_ps(&C_start[TranslateInd(0, 1, size_n)]);

    __m256 lineB;
    float elemA1;
    float elemA2;

    union {
        float f;
        int i;
    } convert;

    union {
        __m256 f;
        __m256i i;
    } convert_v;

#define MUL(k_) \
    lineB = _mm256_loadu_ps(&B_start[TranslateInd(0, k_, size_n)]); \
    convert_v.f = lineA1; \
    convert.i = _mm256_extract_epi32(convert_v.i, k_); \
    elemA1 = convert.f; \
    convert_v.f = lineA2; \
    convert.i = _mm256_extract_epi32(convert_v.i, k_); \
    elemA2 = convert.f; \
    lineC1 += lineB * elemA1; \
    lineC2 += lineB * elemA2; \

    MUL(0)
    MUL(1)
    MUL(2)
    MUL(3)
    MUL(4)
    MUL(5)
    MUL(6)
    MUL(7)

    _mm256_store_ps(&C_start[TranslateInd(0, 0, size_n)], lineC1);
    _mm256_store_ps(&C_start[TranslateInd(0, 1, size_n)], lineC2);
}
