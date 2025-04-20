#include "fmm.h"

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

// static ---------------------------------------------------------------------

static size_t TranslateInd(size_t x, size_t y, size_t width) {
    return y * width + x;
}

static void FmmMatMulfNaive(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k);
static void FmmMatMulfTranspose(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k);
static void FmmMatMulfKernel(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k);

static float* FmmMatTransposef(const float* mat, size_t M, size_t N);
static void FmmKernel4x4(const float* A_start, const float* BT_start, float* C_start, size_t size_m, size_t size_n, size_t size_k);

// global ---------------------------------------------------------------------

//    k            n         n
// m      *   k        = m 
// 
void FmmMatMulf(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k) {
#if defined (NAIVE)
    FmmMatMulfNaive(matA, matB, matC, size_m, size_n, size_k);    
#elif defined (TRANSPOSE)
    FmmMatMulfTranspose(matA, matB, matC, size_m, size_n, size_k);    
#elif defined (KERNEL)
    FmmMatMulfKernel(matA, matB, matC, size_m, size_n, size_k);
#else 
    FmmMatMulfNaive(matA, matB, matC, size_m, size_n, size_k);    
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

static void FmmMatMulfTranspose(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k) {
    float* matBT = FmmMatTransposef(matB, size_k, size_n);

    for (size_t im = 0; im < size_m; im++) {
        for (size_t in = 0; in < size_n; in++) {
            matC[size_n * im + in] = 0;
            // inner loop
            for (size_t ik = 0; ik < size_k; ik++) {
                matC[TranslateInd(in, im, size_n)] += matA[TranslateInd(ik, im, size_k)] * matBT[TranslateInd(ik, in, size_k)];
            }
        }
    }
}

static const size_t block_size_x = 4;
static const size_t block_size_y = 4;

static void FmmMatMulfKernel(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k) {
    float* matBT = FmmMatTransposef(matB, size_k, size_n);

    memset(matC, 0, size_m * size_n * sizeof(*matC));

    for (size_t im = 0; im < size_m; im += block_size_y) {
        for (size_t in = 0; in < size_n; in += block_size_x) {
            FmmKernel4x4(&matA[TranslateInd(0, im, size_k)], &matBT[TranslateInd(0, in, size_k)], &matC[TranslateInd(in, im, size_n)], size_m, size_n, size_k);
        }
    }
}

static void FmmKernel4x4(const float* A_start, const float* BT_start, float* C_start, size_t size_m, size_t size_n, size_t size_k) {
    for (size_t ik = 0; ik < size_k; ik++) { // for every index there sub matrix
        for (size_t im = 0; im < block_size_y; im++) {
            for (size_t in = 0; in < block_size_x; in++) {
                C_start[TranslateInd(in, im, size_n)] += A_start[TranslateInd(ik, im, size_k)] * BT_start[TranslateInd(ik, in, size_k)];
            }
        }
    }
}

// hight = M, width = N
static float* FmmMatTransposef(const float* mat, size_t M, size_t N) {
    float* matT = calloc(M * N, sizeof(*mat));

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            matT[TranslateInd(i, j, M)] = mat[TranslateInd(j, i, N)];
        }
    }

    return matT;
}

// void FmmMatMuld(const double* matA, const double* matB, double* matC, size_t size_m, size_t size_n, size_t size_k) {
//     for (size_t im = 0; im < size_m; im++) {
//         for (size_t in = 0; in < size_n; in++) {
//             matC[size_n * im + in] = 0;
//             // inner loop
//             for (size_t ik = 0; ik < size_k; ik++) {
//                 matC[TranslateInd(in, im, size_n)] += matA[TranslateInd(ik, im, size_k)] * matB[TranslateInd(in, ik, size_n)];
//             }
//         }
//     }
// }
