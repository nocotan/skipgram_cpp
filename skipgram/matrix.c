#include "stdio.h"
#include <xmmintrin.h>
#include <immintrin.h>


const float dot_sse(const float v1[], const float v2[], unsigned n) {
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

void mmul_sse(const float * a, const float * b, float * r) {
  __m128 a_line, b_line, r_line;
  for (int i=0; i<16; i+=4) {

    a_line = _mm_load_ps(a);
    b_line = _mm_set1_ps(b[i]);
    r_line = _mm_mul_ps(a_line, b_line);
    for (int j=1; j<4; j++) {
      a_line = _mm_load_ps(&a[j*4]);
      b_line = _mm_set1_ps(b[i+j]);

      r_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line);
    }
    _mm_store_ps(&r[i], r_line);
  }
}
