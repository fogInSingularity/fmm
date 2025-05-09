#include "fmm.h"

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#include <immintrin.h>

// static ---------------------------------------------------------------------

static size_t TranslateInd(size_t n, size_t m, size_t N) {
    return m * N + n;
}

static float I32ToF32(int val) {
    float conv;
    memcpy(&conv, &val, sizeof(val));
    return conv;
}

static int F32ToI32(float val) {
    int conv;
    memcpy(&conv, &val, sizeof(val));
    return conv;
}

static __m256i F256ToI256(__m256 val) {
    return _mm256_castps_si256(val);
}

static __m256 I256ToF256(__m256i val) {
    return _mm256_castsi256_ps(val);
}

static void FmmMatMulfNaive(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k);
static void FmmMatMulfKernel(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k);

static void FmmKernel(const float* A_start, const float* B_start, float* C_start, size_t size_m, size_t size_n, size_t size_k);
static void FmmMicroKernel(const float* A_start, const float* B_start, float* C_start, size_t size_m, size_t size_n, size_t size_k);

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

static const size_t block_size_m = 8;
static const size_t block_size_n = 8;
static const size_t block_size_k = 8;

static void FmmMatMulfKernel(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k) {
    memset(matC, 0, size_m * size_n * sizeof(*matC));

    for (size_t im = 0; im < size_m; im += block_size_m) {
        for (size_t in = 0; in < size_n; in += block_size_n) {
            FmmKernel(
                &matA[TranslateInd(0, im, size_k)], 
                &matB[TranslateInd(in, 0, size_n)], 
                &matC[TranslateInd(in, im, size_n)], 
                size_m, size_n, size_k
            );
        }
    }
}

static void FmmKernel(const float* A_start, const float* B_start, float* C_start, size_t size_m, size_t size_n, size_t size_k) {    
    for (size_t k = 0; k < size_k; k += block_size_k) {
        FmmMicroKernel(
            &A_start[TranslateInd(k, 0, size_k)], 
            &B_start[TranslateInd(0, k, size_n)], 
            &C_start[TranslateInd(0, 0, size_n)], 
            size_m, size_n, size_k
        );
    }
}

static void FmmMicroKernel(const float* A_start, const float* B_start, float* C_start, size_t size_m, size_t size_n, size_t size_k) {

// n = 8

// #define MUL(k_, ind_) \
//     lineB ## ind_ = _mm256_loadu_ps(&B_start[TranslateInd(0, k_, size_n)]); \
//     elemA ## ind_ = I32ToF32(_mm256_extract_epi32(F256ToI256(lineA ## ind_), k_)); \
//     lineC ## ind_ += (lineB ## ind_)  * (elemA ## ind_); \

// #define LINE_UNROLL(ind_) \
//     __m256 lineA ## ind_ = _mm256_loadu_ps(&A_start[TranslateInd(0, ind_, size_k)]); \
//     __m256 lineC ## ind_ = _mm256_loadu_ps(&C_start[TranslateInd(0, ind_, size_n)]); \
//     __m256 lineB ## ind_; \
//     float elemA ## ind_; \
//     MUL(0, ind_) \
//     MUL(1, ind_) \
//     MUL(2, ind_) \
//     MUL(3, ind_) \
//     MUL(4, ind_) \
//     MUL(5, ind_) \
//     MUL(6, ind_) \
//     MUL(7, ind_) \
//     _mm256_storeu_ps(&C_start[TranslateInd(0, ind_, size_n)], lineC ## ind_);

//     LINE_UNROLL(0)
//     LINE_UNROLL(1)
//     LINE_UNROLL(2)
//     LINE_UNROLL(3)
//     LINE_UNROLL(4)
//     LINE_UNROLL(5)
//     LINE_UNROLL(6)
//     LINE_UNROLL(7)


#define MUL(k_) \
    lineB = _mm256_loadu_ps(&B_start[TranslateInd(0, k_, size_n)]); \
    elemA = I32ToF32(_mm256_extract_epi32(F256ToI256(lineA), k_)); \
    lineC += lineB * elemA; \

    for (size_t i = 0; i < block_size_m; i++) {
        __m256 lineA = _mm256_loadu_ps(&A_start[TranslateInd(0, i, size_k)]); 
        __m256 lineC = _mm256_loadu_ps(&C_start[TranslateInd(0, i, size_n)]); 
        __m256 lineB;
        float elemA; 
        MUL(0) 
        MUL(1) 
        MUL(2) 
        MUL(3) 
        MUL(4) 
        MUL(5) 
        MUL(6) 
        MUL(7) 
        _mm256_storeu_ps(&C_start[TranslateInd(0, i, size_n)], lineC);
    }

// n = 16


// #define MUL(k_) \
//     lineB1 = _mm256_loadu_ps(&B_start[TranslateInd(0, k_, size_n)]); \
//     lineB2 = _mm256_loadu_ps(&B_start[TranslateInd(block_size_n / 2, k_, size_n)]); \
//     elemA = I32ToF32(_mm256_extract_epi32(F256ToI256(lineA), k_)); \
//     lineC1 += lineB1 * elemA; \
//     lineC2 += lineB2 * elemA; \

//     for (size_t i = 0; i < block_size_m; i++) {
//         __m256 lineA = _mm256_loadu_ps(&A_start[TranslateInd(0, i, size_k)]); 
//         __m256 lineC1 = _mm256_loadu_ps(&C_start[TranslateInd(0, i, size_n)]); 
//         __m256 lineC2 = _mm256_loadu_ps(&C_start[TranslateInd(block_size_n / 2, i, size_n)]); 
//         __m256 lineB1;
//         __m256 lineB2;
//         float elemA; 
//         MUL(0) 
//         MUL(1) 
//         MUL(2) 
//         MUL(3) 
//         MUL(4) 
//         MUL(5) 
//         MUL(6) 
//         MUL(7) 
//         _mm256_storeu_ps(&C_start[TranslateInd(0, i, size_n)], lineC1);
//         _mm256_storeu_ps(&C_start[TranslateInd(block_size_n / 2, i, size_n)], lineC2);
//     }

}
