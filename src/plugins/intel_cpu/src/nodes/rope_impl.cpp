// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_impl.hpp"

#include <iostream>
#include <utility>

#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

#if defined(HAVE_AVX2)
inline __m256i get_mask(int N7) {
    static __m256i mask[] = {
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),
        _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
        _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),
        _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
        _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),
    };
    return _mm256_loadu_si256(&mask[N7]);
}
#endif

void rope_impl(const float* src, const float* sin, const float* cos, float* dst, size_t rotary_dims) {
#if defined(HAVE_AVX2)
    size_t i = 0;
    // std::cout << "rope_impl AVX2" << std::endl;

    auto half_rotary_dims = rotary_dims / 2;
    // for (; i < half_rotary_dims; i++)
    //    dst[i] = cos[i] * src[i] + sin[i] * (-src[i + half_rotary_dims]);

    // process vector body
    auto* src2 = src + half_rotary_dims;
    while (i + 8 <= half_rotary_dims) {
        auto v_src = _mm256_loadu_ps(src + i);
        auto v_cos = _mm256_loadu_ps(cos + i);
        auto v_sin = _mm256_loadu_ps(sin + i);
        auto v_src2 = _mm256_loadu_ps(src2 + i);
        auto v_a = _mm256_mul_ps(v_src, v_cos);
        auto v_b = _mm256_mul_ps(v_src2, v_sin);
        v_a = _mm256_sub_ps(v_a, v_b);
        _mm256_storeu_ps(dst + i, v_a);
        i += 8;
    }

    if (i < half_rotary_dims) {
        auto mask = get_mask(half_rotary_dims - i);
        auto v_src = _mm256_maskload_ps(src + i, mask);
        auto v_cos = _mm256_maskload_ps(cos + i, mask);
        auto v_sin = _mm256_maskload_ps(sin + i, mask);
        auto v_src2 = _mm256_maskload_ps(src2 + i, mask);
        auto v_a = _mm256_mul_ps(v_src, v_cos);
        auto v_b = _mm256_mul_ps(v_src2, v_sin);
        v_a = _mm256_sub_ps(v_a, v_b);
        _mm256_maskstore_ps(dst + i, mask, v_a);
    }

    i = half_rotary_dims;

    // for (; i < rotary_dims; i++)
    //    dst[i] = cos[i] * src[i] + sin[i] * (src[i - half_rotary_dims]);
    src2 = src - half_rotary_dims;
    while (i + 8 <= rotary_dims) {
        auto v_src = _mm256_loadu_ps(src + i);
        auto v_cos = _mm256_loadu_ps(cos + i);
        auto v_sin = _mm256_loadu_ps(sin + i);
        auto v_src2 = _mm256_loadu_ps(src2 + i);
        auto v_a = _mm256_mul_ps(v_src, v_cos);
        auto v_b = _mm256_mul_ps(v_src2, v_sin);
        v_a = _mm256_add_ps(v_a, v_b);
        _mm256_storeu_ps(dst + i, v_a);
        i += 8;
    }

    if (i < rotary_dims) {
        auto mask = get_mask(rotary_dims - i);
        auto v_src = _mm256_maskload_ps(src + i, mask);
        auto v_cos = _mm256_maskload_ps(cos + i, mask);
        auto v_sin = _mm256_maskload_ps(sin + i, mask);
        auto v_src2 = _mm256_maskload_ps(src2 + i, mask);
        auto v_a = _mm256_mul_ps(v_src, v_cos);
        auto v_b = _mm256_mul_ps(v_src2, v_sin);
        v_a = _mm256_add_ps(v_a, v_b);
        _mm256_maskstore_ps(dst + i, mask, v_a);
    }
#else
    size_t i = 0;
    auto half_rotary_dims = rotary_dims / 2;
    for (; i < half_rotary_dims; i++) {
        dst[i] = cos[i] * src[i] + sin[i] * (-src[i + half_rotary_dims]);
    }
    for (; i < rotary_dims; i++) {
        dst[i] = cos[i] * src[i] + sin[i] * (src[i - half_rotary_dims]);
    }
#endif
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
