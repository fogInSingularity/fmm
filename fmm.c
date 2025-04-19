#include "fmm.h"

#include <stdio.h>
#include <stddef.h>
#include <string.h>

static size_t TranslateInd(size_t x, size_t y, size_t width) {
    return y * width + x;
}

//    k            n         n
// m      *   k        = m 
// 
void FmmMatMulf(const float* matA, const float* matB, float* matC, size_t size_m, size_t size_n, size_t size_k) {
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

void FmmMatMuld(const double* matA, const double* matB, double* matC, size_t size_m, size_t size_n, size_t size_k) {
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
