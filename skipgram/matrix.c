#include "stdio.h"
#include <xmmintrin.h>
#include <immintrin.h>


const float dot(const float v1[], const float v2[], unsigned n) {
    __m128 u = {0};

    for (unsigned i = 0; i < n; i += 4) {
        __m128 w = _mm_load_ps(&v1[i]);
        __m128 x = _mm_load_ps(&v2[i]);

        x = _mm_mul_ps(w, x);
        u = _mm_add_ps(u, x);
    }
    __attribute__((aligned(16))) float t[4] = {0};
    _mm_store_ps(t, u);
    return t[0] + t[1] + t[2] + t[3];
}
