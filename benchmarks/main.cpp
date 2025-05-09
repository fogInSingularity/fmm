#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cstddef>
#include <random>
#include <cmath>
#include <string>

#include <immintrin.h>
#include <sched.h>
#include <unistd.h>

#include <openblas/cblas.h>

#include "fmm.h"

static void BlasWraperf(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    blasint iM = static_cast<blasint>(M);
    blasint iN = static_cast<blasint>(N);
    blasint iK = static_cast<blasint>(K);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, iM, iN, iK, 1, A, iK, B, iN, 0, C, iN);
}

static bool FloatEq(float a, float b) {
    constexpr float eps = 0.003f;
    return std::abs(a - b) < eps;
}

static void BindToCore(size_t core) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(core, &cpu_set);

    auto res = sched_setaffinity(0, sizeof(cpu_set), &cpu_set);
    if (res != 0) {
        perror("cant bind to core:");
    }
}

static void UpPriority() {
    errno = 0;
    auto res = nice(-20);
    if (res != 0 && errno != 0) {
        perror("cant set priority");
    }
}

// ./fmm-test [cmp|openblas|fmm] [<seed>|random] <M> <N> <K>  
int main(const int argc, const char* argv[]) {
    if (argc < 6) {
        std::cerr << "Not enought args:" << std::endl;
        std::cerr << argv[0] << " [cmp|openblas|fmm}] [<seed>|random] <M> <N> <K>" << std::endl;
        return EXIT_FAILURE;
    }

    BindToCore(0);
    UpPriority();

    bool is_cmp = false;
    bool is_openblas = false;
    bool is_fmm = false;
    if (strcmp(argv[1], "cmp") == 0) {
        is_cmp = true;
    } else if (strcmp(argv[1], "openblas") == 0) {
        is_openblas = true;
    } else if (strcmp(argv[1], "fmm") == 0) {
        is_fmm = true;
    } else {
        std::cerr << argv[0] << " [cmp|openblas|fmm}] <M> <N> <K>" << std::endl;
        return EXIT_FAILURE;
    }

    unsigned default_seed = 0;
    bool use_random = false;

    if (strcmp(argv[2], "random") == 0) {
        use_random = true;
    } else {
        default_seed = static_cast<unsigned>(std::stoul(std::string{argv[2]}));
    }

    size_t size_m = std::stoull(std::string{argv[3]});
    size_t size_n = std::stoull(std::string{argv[4]});
    size_t size_k = std::stoull(std::string{argv[5]});
 
    float* A = new float[size_m * size_k];
    float* B = new float[size_k * size_n];

    unsigned seed = 0;
    if (use_random) {
        std::random_device rdev{};
        seed = rdev();
    } else {
        seed = default_seed;
    }

    std::mt19937_64 gen{seed};
    std::normal_distribution<float> distr{1, 0.01f};
    for (size_t i = 0; i < size_m * size_k; i++) {
        A[i] = distr(gen);
    }

    for (size_t i = 0; i < size_k * size_n; i++) {
        B[i] = distr(gen);
    }

    if (is_cmp) {
        std::cerr << "seed: " << seed << std::endl;
        float* C = new float[size_m * size_n];
        float* D = new float[size_m * size_n];

        FmmMatMulf(A, B, C, size_m, size_n, size_k);
        BlasWraperf(A, B, D, size_m, size_n, size_k);

        for (size_t i = 0; i < size_m * size_k; i++ ) {
            if (!FloatEq(C[i], D[i])) {
                std::cout << "C and D dif: i: " << i << ", val: " << "C: " << C[i] << ", D: " << D[i] << "\n";
            }
        }
        
        delete[] C;
        delete[] D;
    } else if (is_openblas) {
        float* C = new float[size_m * size_n];

        unsigned aux = 0;
        auto start = __rdtscp(&aux);
        BlasWraperf(A, B, C, size_m, size_n, size_k);
        auto end = __rdtscp(&aux);

        std::cout << end - start << std::endl;

        delete[] C;
    } else if (is_fmm) {
        float* C = new float[size_m * size_n];

        unsigned aux = 0;
        auto start = __rdtscp(&aux);
        BlasWraperf(A, B, C, size_m, size_n, size_k);
        auto end = __rdtscp(&aux);

        std::cout << end - start << std::endl;

        delete[] C;
    }

    delete[] A;
    delete[] B;
}