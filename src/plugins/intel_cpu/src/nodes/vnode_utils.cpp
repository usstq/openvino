// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnode_utils.hpp"

#include <cmath>
#include <limits>
#include <cstring>
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif
namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

inline void scale_add_reduce_max(float* a, const float scale, const float* b, const size_t size, float& max) {
#if defined(HAVE_AVX512F)
    auto v_max = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm512_set1_ps(scale);
    auto v_a = v_max;
    auto v_b = v_max;
    size_t i = 0;
    size_t len = size;
    // process vector body
    while (len > 15) {
        v_a = _mm512_loadu_ps(a + i);
        v_b = _mm512_loadu_ps(b + i);
        v_a = _mm512_fmadd_ps(v_a, v_scale, v_b);
        v_max = _mm512_max_ps(v_max, v_a);
        _mm512_storeu_ps(a + i, v_a);
        len -= 16;
        i += 16;
    }

    // process tails
    if (len > 0) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_b = _mm512_maskz_loadu_ps(mask, b + i);
        v_a = _mm512_fmadd_ps(v_a, v_scale, v_b);
        v_max = _mm512_mask_max_ps(v_max, mask, v_a, v_max);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }

    max = _mm512_reduce_max_ps(v_max);
#else
    for (size_t i = 0; i < size; i++) {
        a[i] *= scale;
        a[i] += b[i];
        max = a[i] > max ? a[i] : max;
    }
#endif
}
#if defined(HAVE_AVX512F)
inline void exp_ps_avx512(__m512 & src) {
    static __m512 exp_ln_flt_min_f = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));    // log(FLT_MIN)
    static __m512 exp_ln_flt_max_f = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));    // log(FLT_MAX)
    static __m512 exp_log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));          // log2(e)
    static __m512 half = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f000000));                // 0.5f
    static __m512 ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));                // ln(2)
    static __m512 one = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f800000));                 // 1.0f
    static __m512i exponent_bias = _mm512_set1_epi32(0x0000007f);                           // 127
    static constexpr int n_mantissa_bits = 23;
    static __m512 exp_pol1 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f7ffffb));            // p1 = 0.999999701f
    static __m512 exp_pol2 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3efffee3));            // p2 = 0.499991506f
    static __m512 exp_pol3 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3e2aad40));            // p3 = 0.166676521f
    static __m512 exp_pol4 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3d2b9d0d));            // p4 = 0.0418978221f
    static __m512 exp_pol5 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3c07cfce));            // p5 = 0.00828929059f
    static __m512 two = _mm512_castsi512_ps(_mm512_set1_epi32(0x40000000));                 // 2
    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r)       // simplify the exp(n*ln(2)) expression

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    auto zero_mask = _mm512_cmp_ps_mask(src, exp_ln_flt_min_f, _CMP_LT_OS);

    // clip src
    src = _mm512_min_ps(src, exp_ln_flt_max_f);
    src = _mm512_max_ps(src, exp_ln_flt_min_f);

    // aux1 : r
    auto aux1 = src;

    // calculate exp(x)
    // fx = x * log2(e) + 0.5
    src = _mm512_mul_ps(src, exp_log2ef);
    src = _mm512_add_ps(src, half);

    // tmp = floorf(fx)
    src = _mm512_floor_ps(src);

    // aux1 = x - fx * ln2
    aux1 = _mm512_fnmadd_ps(src, ln2f, aux1);
    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    src = _mm512_sub_ps(src, one);
    auto aux2_i = _mm512_cvtps_epi32(src);
    aux2_i = _mm512_add_epi32(aux2_i, exponent_bias);
    aux2_i = _mm512_slli_epi32(aux2_i, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    auto zero = _mm512_setzero_ps();
    auto aux2 = _mm512_mask_blend_ps(zero_mask, _mm512_castsi512_ps(aux2_i), zero);

    // compute polynomial
    src = exp_pol5;
    src = _mm512_fmadd_ps(src, aux1, exp_pol4);
    src = _mm512_fmadd_ps(src, aux1, exp_pol3);
    src = _mm512_fmadd_ps(src, aux1, exp_pol2);
    src = _mm512_fmadd_ps(src, aux1, exp_pol1);
    src = _mm512_fmadd_ps(src, aux1, one);

    // y = y * 2^n
    src = _mm512_mul_ps(src, aux2);
    src = _mm512_mul_ps(src, two);
}
#endif

inline void exp_reduce_sum(float* a, const float max, const size_t size, float& sum) {
#if defined(HAVE_AVX512F)
    size_t len = size;
    size_t i = 0;
    __m512 v_a;
    auto v_sum = _mm512_set1_ps(0.0f);
    while (len > 15) {
        v_a = _mm512_loadu_ps(a + i);
        exp_ps_avx512(v_a);
        v_sum = _mm512_add_ps(v_sum, v_a);
        _mm512_storeu_ps(a + i, v_a);
        i += 16;
        len -= 16;
    }

    if (len > 0) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        exp_ps_avx512(v_a);
        v_sum = _mm512_mask_add_ps(v_sum, mask, v_a, v_sum);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }
    sum = _mm512_reduce_add_ps(v_sum);
#else
    for (size_t i = 0; i < size; i++) {
        a[i] = exp(a[i] - max);
        sum += a[i];
    }
#endif
}

inline void multiply_scalar(float* a, const float val, const size_t size) {
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
    size_t i = 0;
    size_t len = size;
    while (len > 15) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        _mm512_storeu_ps(a + i, v_a);
        len -= 16;
        i += 16;
    }
    if (len > 0) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }
#else
    for (size_t i = 0; i < size; i++) {
        a[i] *= val;
    }
#endif
}

void scale_add_softmax(float* a, float scale, float* mask, size_t len, size_t total_size) {
    float max = std::numeric_limits<float>::lowest();
    scale_add_reduce_max(a, scale, mask, len, max);
    float sum = 0.0f;
    // exp sum
    exp_reduce_sum(a, max, len, sum);
    // divide sum
    float scalar = 1.0f / sum;
    multiply_scalar(a, scalar, len);
    // apply causual mask with zeros
    memset(a + len, 0, sizeof(float) * (total_size - len));
}
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine