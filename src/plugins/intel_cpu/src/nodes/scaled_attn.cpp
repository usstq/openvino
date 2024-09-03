// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.h"

#include "common/arbitrary_order_desc_creator.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/op/mha.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/util/common_util.hpp"
#include "shape_inference/custom/scaled_attn.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OV_CPU_WITH_MLAS
#    include "mlas/sgemm.hpp"
#endif

#ifdef OV_CPU_WITH_ACL
#     include "kernels/acl/gemm_kernel.hpp"
#endif

#include "utils/plain_tensor.hpp"
#include "kernels/scaled_attn/softmax.hpp"
#include "kernels/scaled_attn/mha_single_token.hpp"
#include "kernels/scaled_attn/attn_memcpy.hpp"
#include "kernels/scaled_attn/attn_quant.hpp"
#include "kernels/x64/brgemm_kernel.hpp"
#include "kernels/x64/rope_kernel.hpp"
#include "nodes/common/cpu_convert.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace ov::Extensions::Cpu::XARCH;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::kernel;

namespace ov {
namespace intel_cpu {
namespace node {

struct ScaledDotProductAttentionKey {
    ov::element::Type rtPrecision;

    size_t hash() const;
    bool operator==(const ScaledDotProductAttentionKey& rhs) const;
};

size_t ScaledDotProductAttentionKey::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, rtPrecision.hash());

    return seed;
}

bool ScaledDotProductAttentionKey::operator==(const ScaledDotProductAttentionKey& rhs) const {
    auto retVal = rtPrecision == rhs.rtPrecision;

    return retVal;
}

static std::shared_ptr<kernel::JitKernelBase> createJitKernel(const jit_rotary_compile_params& param, bool check_vec_size2 = false) {
    std::shared_ptr<kernel::JitKernelBase> res;

    MAYBE_UNUSED(param);
    MAYBE_UNUSED(check_vec_size2);

#if defined(OPENVINO_ARCH_X86_64)

    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        bool flag = true;
        if (check_vec_size2) {
            auto vec_size = jit_rotary_kernel<dnnl::impl::cpu::x64::avx512_core>::vec_size;
            if (param.rotary_ndims % (vec_size * 2) != 0)
                flag = false;
        }
        if (flag)
            res = std::make_shared<jit_rotary_kernel<dnnl::impl::cpu::x64::avx512_core>>(param);
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        bool flag = true;
        if (check_vec_size2) {
            auto vec_size = jit_rotary_kernel<dnnl::impl::cpu::x64::avx2>::vec_size;
            if (param.rotary_ndims % (vec_size * 2) != 0)
                flag = false;
        }
        if (flag)
            res = std::make_shared<jit_rotary_kernel<dnnl::impl::cpu::x64::avx2>>(param);
    }

    if (res)
        res->create_kernel();

#endif // OPENVINO_ARCH_X86_64

    return res;
}

static void execJitKernel(const std::shared_ptr<kernel::JitKernelBase>& ker, const void* src, void* dst, const float* cos, const float* sin) {
    MAYBE_UNUSED(ker);
    MAYBE_UNUSED(src);
    MAYBE_UNUSED(dst);
    MAYBE_UNUSED(cos);
    MAYBE_UNUSED(sin);

#if defined(OPENVINO_ARCH_X86_64)

    jit_rotary_call_args call_args;
    call_args.src = src;
    call_args.cos = cos;
    call_args.sin = sin;
    call_args.dst = dst;
    (*ker)(&call_args);

#endif // OPENVINO_ARCH_X86_64
}

// default implementation: reference
template <ScaledDotProductAttention::KernelTypes KType, typename T>
struct MHAKernel {
    const GraphContext::CPtr context;
    MHAKernel() = delete;
    explicit MHAKernel(GraphContext::CPtr ctx) : context(ctx) {}

    template <typename D>
    float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
        float result = 0;
        if (stride_b == 1) {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        } else {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
        return result;
    }

    void softmax(float* a, int len) {
        float max = *std::max_element(a, a + len);
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            a[i] = exp(a[i] - max);
            sum += a[i];
        }
        float scale = 1.0f / sum;
        for (int i = 0; i < len; i++) {
            a[i] *= scale;
        }
    }

    template <typename D>
    void accumulate(float* acc, const D* v, int len, float weight = 1.0f) {
        for (int i = 0; i < len; i++) {
            acc[i] += static_cast<float>(v[i]) * weight;
        }
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, q_len, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto Hk = present_key.size(1);
        size_t h_each_group_len = H / Hk;
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            std::vector<float> attn_score(kv_len);
            std::vector<float> word_vec(head_size, 0.0f);

            for (size_t m = 0; m < q_len; m++) {
                // dot-product to get attention scores
                auto* q = query.ptr<T>(b, h, m, 0);
                // how many key/values can be accessed causally
                auto ncausal = kv_len;
                // no causall mask is set and it's not fused into attention_mask
                if (auto_causal)
                    ncausal = kv_len - q_len + m + 1;
                for (size_t n = 0; n < ncausal; n++) {
                    auto* k = &present_key.at<T>({b, h / h_each_group_len, n, 0}, true);
                    attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;

                    // apply alibi tensor
                    if (alibi_mask)
                        attn_score[n] += alibi_mask.at<float>({b, h, m, n}, true);

                    // apply attention mask (maybe combined with causal_mask)
                    if (attention_mask)
                        attn_score[n] += attention_mask.at<T>({b, h, m, n}, true);

                    // apply causal_mask
                    if (causal_mask) {
                        bool is_zero = causal_mask.at<uint8_t>({b, h, m, n}, true) == 0;
                        if (select_nfltmax_at_0) {
                            if (is_zero)
                                attn_score[n] = -FLT_MAX;
                        } else {
                            if (!is_zero) {
                                attn_score[n] = -FLT_MAX;
                            }
                        }
                    }
                }

                // softmax
                softmax(&attn_score[0], ncausal);

                // linearly combine value
                word_vec.assign(head_size, 0.0f);
                for (size_t n = 0; n < ncausal; n++) {
                    auto* v = &present_value.at<T>({b, h / h_each_group_len, n, 0}, true);
                    accumulate(word_vec.data(), v, head_size, attn_score[n]);
                }

                // output [B, L1, H*head_size]
                auto* out = has_out_transpose ? &output_emb.at<T>({b, m, h * head_size}) : &output_emb.at<T>({b, h, m});
                std::copy(word_vec.begin(), word_vec.end(), out);
            }
        });
    }
};

template <typename T>
struct MHAKernel<ScaledDotProductAttention::KT_ONEDNN, T> {
    // q: [B, H, q_len, S]
    // k: [B, H, kv_len, S]
    // v: [B, H, kv_len, S]
    const GraphContext::CPtr context;
    dnnl::memory::desc out_md;
    PlainTensor fp32_out;
    PlainTensor qk_scratch_a;
    PlainTensor qk_scratch_b;
    PlainTensor wv_scratch_a;
    PlainTensor wv_scratch_b;
    PlainTensor weight_score;
    std::vector<size_t> wsp;
    size_t wsp_size_per_thread = 0;
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    struct brgemmKey {
        size_t M;
        size_t N;
        size_t K;
        size_t lda;
        size_t ldb;
        size_t ldc;
        bool b_transposed;
        ov::element::Type in_type;
        size_t hash() const {
            using namespace dnnl::impl;
            using namespace dnnl::impl::primitive_hashing;
            size_t seed = 0;
            seed = hash_combine(seed, M);
            seed = hash_combine(seed, N);
            seed = hash_combine(seed, K);
            seed = hash_combine(seed, lda);
            seed = hash_combine(seed, ldb);
            seed = hash_combine(seed, ldc);
            seed = hash_combine(seed, b_transposed);
            seed = hash_combine(seed, in_type.hash());
            return seed;
        }
        bool operator==(const brgemmKey& rhs) const {
            return (rhs.M == M) && (rhs.N == N) && (rhs.K == K) && (rhs.lda == lda) && (rhs.ldb == ldb) &&
                   (rhs.ldc == ldc) && (rhs.b_transposed == b_transposed) && (rhs.in_type == in_type);
        }
    };

    std::shared_ptr<BrgemmKernel> qk_gemm_ptr = nullptr;
    std::shared_ptr<BrgemmKernel> wv_gemm_ptr = nullptr;

    MHAKernel() = delete;
    explicit MHAKernel(GraphContext::CPtr ctx)
        : context(ctx) {}

    dnnl::memory::dims make_dnnl_dims(const std::vector<size_t>& dims) {
        dnnl::memory::dims dnnl_dims(dims.size());
        for (size_t i = 0; i < dims.size(); i++)
            dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
        return dnnl_dims;
    }

    void prepare_brgemm_prim(dnnl::stream strm, PlainTensor& query, PlainTensor& present_key, bool has_out_transpose) {
        auto in_type = precision_of<T>::value;
        auto qkv_dt = in_type == ov::element::f32 ? dt::f32 : dt::bf16;
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto Hk = present_key.size(1);
        brgemmKey qk_key = {q_len, kv_len, head_size, query.stride(2), present_key.stride(2), kv_len, true, in_type};

        auto builder = [](const brgemmKey& key) -> std::shared_ptr<BrgemmKernel> {
            return std::make_shared<BrgemmKernel>(key.M,
                                                  key.N,
                                                  key.K,
                                                  key.lda,
                                                  key.ldb,
                                                  key.ldc,
                                                  key.b_transposed,
                                                  key.in_type);
        };

        auto cache = this->context->getParamsCache();
        auto qk_result = cache->getOrCreate(qk_key, builder);
        if (!qk_result.first) {
            OPENVINO_THROW("ScaledDotProductAttention 1st token qk gemm creation fails");
        }

        qk_gemm_ptr = qk_result.first;
        if (has_out_transpose)
            out_md = dnnl::memory::desc(make_dnnl_dims({B, q_len, H, head_size}), qkv_dt, tag::abcd);
        else
            out_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, head_size}), qkv_dt, tag::abcd);

        size_t ldc_index = 2;
        if (has_out_transpose) {
            ldc_index = 1;
        }
        brgemmKey wv_key = {q_len,
                            head_size,
                            kv_len,
                            kv_len * (in_type == ov::element::Type_t::f32 ? 1 : 2),
                            present_key.stride(2),
                            static_cast<size_t>(out_md.get_strides()[ldc_index]),
                            false,
                            in_type};

        auto wv_result = cache->getOrCreate(wv_key, builder);
        if (!wv_result.first) {
            OPENVINO_THROW("ScaledDotProductAttention 1st token wv gemm creation fails");
        }

        wv_gemm_ptr = wv_result.first;

        size_t nthr = static_cast<size_t>(parallel_get_max_threads());

        // wsp is used to compute beta when K is blocked
        wsp_size_per_thread = wv_gemm_ptr->get_wsp_size();
        wsp.resize(nthr * wsp_size_per_thread);

        // allocate scratch a/b, notice get_scratch_a_size/get_scratch_b_size returns in bytes
        size_t data_size = sizeof(T);
        qk_scratch_a.resize<T>({nthr, qk_gemm_ptr->get_scratch_a_size() / data_size});
        wv_scratch_a.resize<T>({nthr, wv_gemm_ptr->get_scratch_a_size() / data_size});

        qk_scratch_b.resize<T>({B, Hk, qk_gemm_ptr->get_scratch_b_size() / data_size});
        wv_scratch_b.resize<T>({B, Hk, wv_gemm_ptr->get_scratch_b_size() / data_size});
        const size_t m_block_size = qk_gemm_ptr->get_mblk_size();
        weight_score.resize<float>({static_cast<size_t>(parallel_get_max_threads()), H, m_block_size, kv_len});
        if (has_out_transpose) {
            fp32_out.resize<float>({B, q_len, H, head_size});
        } else {
            fp32_out.resize<float>({B, H, q_len, head_size});
        }
        return;
    }

    void execute_brgemm(PlainTensor& query,
                        PlainTensor& present_key,
                        PlainTensor& present_value,
                        const PlainTensor& alibi_mask,
                        const PlainTensor& attention_mask,
                        PlainTensor& output_emb,
                        bool has_out_transpose,
                        bool auto_causal,
                        float d_scale = 0.0f) {
        const auto B = query.size(0);
        const auto H = query.size(1);
        const auto q_len = query.size(2);
        const auto head_size = query.size(3);
        const auto Hk = present_key.size(1);
        const auto kv_len = present_key.size(2);
        size_t h_each_group_len = H / Hk;
        const size_t m_block_size = qk_gemm_ptr->get_mblk_size();
        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;
        bool is_bf16 = precision_of<T>::value == ov::element::bf16;
        // packed k, v
        parallel_for2d(B, Hk, [&](size_t b, size_t h) {
            T* k_ptr = &present_key.at<T>({b, h, 0, 0});
            T* v_ptr = &present_value.at<T>({b, h, 0, 0});
            qk_gemm_ptr->copy_buffer_b(k_ptr, &qk_scratch_b.at<T>({b, h, 0}));
            if (is_bf16)
                wv_gemm_ptr->copy_buffer_b(v_ptr, &wv_scratch_b.at<T>({b, h, 0}));
        });

        // attention
        parallel_for3d(B, H, m_blocks, [&](size_t ithr, size_t b, size_t h, size_t m_blk) {
            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;
            size_t tid = parallel_get_thread_num();
            T* q_ptr = &query.at<T>({b, h, m_start, 0});
            float* c_ptr = weight_score.ptr<float>(ithr, h, 0, 0);
            T* k_ptr = &qk_scratch_b.at<T>({b, h / h_each_group_len, 0});
            qk_gemm_ptr->executeGemm(m_cnt < m_block_size,
                                     q_ptr,
                                     k_ptr,
                                     c_ptr,
                                     wsp.data() + tid * wsp_size_per_thread,
                                     qk_scratch_a ? &qk_scratch_a.at<T>({tid, 0}) : nullptr);
            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at<float>({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }

            uint8_t* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = reinterpret_cast<uint8_t*>(&attention_mask.at<T>({b, h, 0, 0}, true));
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2) * sizeof(T);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at<uint8_t>({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }
            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
                auto score = weight_score.ptr<float>(ithr, h, m - m_start);
                attn_softmax(score,
                            reinterpret_cast<T*>(score),
                            d_scale,
                            alibi_ptr + m * alibi_stride,
                            attn_mask_ptr + m * attn_mask_stride,
                            cmask_ptr + m * cmask_stride,
                            select_nfltmax_at_0,
                            ncausal,
                            kv_len,
                            precision_of<T>::value,
                            precision_of<T>::value);
            }
            auto* w_ptr = reinterpret_cast<T*>(weight_score.ptr<float>(ithr, h, 0, 0));
            float* fp32_out_ptr;
            if (is_bf16) {
                fp32_out_ptr = has_out_transpose ? &fp32_out.at<float>({b, m_start, h, 0}) : &fp32_out.at<float>({b, h, m_start, 0});
            } else {
                fp32_out_ptr = has_out_transpose ? &output_emb.at<float>({b, m_start, h * head_size}) : &output_emb.at<float>({b, h, m_start, 0});
            }
            T* v_ptr = is_bf16 ? &wv_scratch_b.at<T>({b, h / h_each_group_len, 0})
                               : &present_value.at<T>({b, h / h_each_group_len, 0, 0});
            wv_gemm_ptr->executeGemm(m_cnt < m_block_size,
                                     w_ptr,
                                     v_ptr,
                                     fp32_out_ptr,
                                     wsp.data() + tid * wsp_size_per_thread,
                                     wv_scratch_a ? &wv_scratch_a.at<T>({tid, 0}) : nullptr);
            if (is_bf16) {
                if (has_out_transpose) {
                    attn_memcpy2d_kernel(&fp32_out.at<float>({b, m_start, h, 0}),
                                         &output_emb.at<T>({b, m_start, h * head_size}),
                                         ov::element::f32,
                                         ov::element::bf16,
                                         fp32_out.stride(1),
                                         output_emb.stride(1),
                                         head_size,
                                         m_cnt);
                } else {
                    attn_memcpy2d_kernel(&fp32_out.at<float>({b, h, m_start, 0}),
                                         &output_emb.at<T>({b, h, m_start, 0}),
                                         ov::element::f32,
                                         ov::element::bf16,
                                         0,
                                         0,
                                         m_cnt * head_size,
                                         1);
                }
            }
        });
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0 = false;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi          [B, H, q_len, kv_len]
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto head_size = query.size(3);
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        prepare_brgemm_prim(strm, query, present_key, has_out_transpose);
        execute_brgemm(query,
                       present_key,
                       present_value,
                       alibi_mask,
                       attention_mask,
                       output_emb,
                       has_out_transpose,
                       auto_causal,
                       d_scale);
    }
};

#ifdef OV_CPU_WITH_ACL
template <>
struct MHAKernel<ScaledDotProductAttention::KT_ACL, float> {
    const GraphContext::CPtr context;
    size_t m_block_size;

    MHAKernel() = delete;
    explicit MHAKernel(GraphContext::CPtr ctx): context(ctx) {
        m_block_size = 512;
        select_nfltmax_at_0 = false;
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto h_group_num = present_key.size(1);
        size_t h_each_group_len = H / h_group_num;

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        auto k_stride_s = present_key.stride(3);

        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            float* q_ptr = &query.at<float>({b, h, m_start, 0});
            float* k_ptr = &present_key.at<float>({b, h / h_each_group_len, 0, 0});
            float* v_ptr = &present_value.at<float>({b, h / h_each_group_len, 0, 0});

            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at<float>({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }
            uint8_t* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = reinterpret_cast<uint8_t*>(&attention_mask.at<float>({b, h, 0, 0}, true));
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2) * sizeof(float);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at<uint8_t>({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }

            arm_compute::Tensor qkTensor;
            arm_compute::TensorInfo qkInfo;

            bool b_transpose = false;
            if (k_stride_s == 1)
                b_transpose = true;
            GemmKernel qk_gemm(m_cnt, head_size, kv_len, b_transpose);

            arm_compute::Strides qStrides({query.stride_bytes(3), query.stride_bytes(2)});
            arm_compute::Strides kStrides({present_key.stride_bytes(3), present_key.stride_bytes(2)});
            qk_gemm.executeGemm(reinterpret_cast<void *>(q_ptr),
                                reinterpret_cast<void *>(k_ptr),
                                qkInfo,
                                qkTensor,
                                qStrides,
                                kStrides);

            auto qk = reinterpret_cast<float*>(qkTensor.buffer());


            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
                attn_softmax(qk + (m - m_start) * kv_len,
                             qk + (m - m_start) * kv_len,
                             d_scale,
                             alibi_ptr + m * alibi_stride,
                             attn_mask_ptr + m * attn_mask_stride,
                             cmask_ptr + m * cmask_stride,
                             select_nfltmax_at_0,
                             ncausal,
                             kv_len,
                             ov::element::f32,
                             ov::element::f32);
            }
            arm_compute::TensorInfo outInfo;
            arm_compute::Tensor outTensor;

            auto out = has_out_transpose ? &output_emb.at<float>({b, m_start, h * head_size}) : &output_emb.at<float>({b, h, m_start});
            auto strides = arm_compute::Strides({output_emb.stride_bytes(1), output_emb.stride_bytes(2)});
            GemmKernel out_gemm(m_cnt, kv_len, head_size);

            arm_compute::Strides vStrides({present_value.stride_bytes(3), present_value.stride_bytes(2)});
            out_gemm.executeGemm(qkTensor.buffer(),
                                 reinterpret_cast<void *>(v_ptr),
                                 outInfo,
                                 outTensor,
                                 qkInfo.strides_in_bytes(),
                                 vStrides,
                                 nullptr,
                                 1.0,
                                 0.0,
                                 &strides,
                                 reinterpret_cast<void*>(out));
            qkTensor.allocator()->free();
        });
    }
};
#endif

#ifdef OV_CPU_WITH_MLAS
template <>
struct MHAKernel<ScaledDotProductAttention::KT_MLAS, float> {
    const GraphContext::CPtr context;
    size_t m_block_size;
    // buffer to hold qk temp
    std::vector<PlainTensor> qk_buffers;

    MHAKernel() = delete;
    explicit MHAKernel(GraphContext::CPtr ctx): context(ctx) {
        m_block_size = 4;
        select_nfltmax_at_0 = false;
        qk_buffers.resize(parallel_get_max_threads());
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto h_group_num = present_key.size(1);
        size_t h_each_group_len = H / h_group_num;

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        auto k_stride_s = present_key.stride(3);

        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            auto thread_id = parallel_get_thread_num();
            if (thread_id < 0)
                OPENVINO_THROW("The calling thread isn't initialized!");
            auto& qk_buf = qk_buffers[thread_id];

            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            auto kv_len_cache_align = (((kv_len * sizeof(float)) + 63) / 64 * 64) / sizeof(float);
            qk_buf.resize<float>({m_block_size, kv_len_cache_align});
            const float* q_ptr = &query.at<float>({b, h, m_start, 0});
            const float* k_ptr = &present_key.at<float>({b, h / h_each_group_len, 0, 0});
            const float* v_ptr = &present_value.at<float>({b, h / h_each_group_len, 0, 0});

            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at<float>({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }
            uint8_t* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = reinterpret_cast<uint8_t*>(&attention_mask.at<float>({b, h, 0, 0}, true));
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2) * sizeof(float);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at<uint8_t>({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }

            float* qk = &(qk_buf.at<float>({0, 0}));
            auto qk_m_stride = qk_buf.stride(0);

            if (k_stride_s == 1)
                mlas_sgemm("N",
                           "T",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(2),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);
            else
                mlas_sgemm("N",
                           "N",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(3),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);

            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
                attn_softmax(qk + (m - m_start) * qk_m_stride,
                            qk + (m - m_start) * qk_m_stride,
                            d_scale,
                            alibi_ptr + m * alibi_stride,
                            attn_mask_ptr + m * attn_mask_stride,
                            cmask_ptr + m * cmask_stride,
                            select_nfltmax_at_0,
                            ncausal,
                            kv_len,
                            ov::element::f32,
                            ov::element::f32);
            }
            mlas_sgemm("N",
                       "N",
                       m_cnt,
                       head_size,
                       kv_len,
                       1.0f,
                       qk,
                       qk_m_stride,
                       v_ptr,
                       present_value.stride(2),
                       0.f,
                       has_out_transpose ? &output_emb.at<float>({b, m_start, h * head_size}) : &output_emb.at<float>({b, h, m_start}),
                       has_out_transpose ? output_emb.stride(1) : output_emb.stride(2),
                       1);
        });
    }
};
#endif

// 2nd token case : only 1 token in query
struct MHASingleToken {
    PlainTensor m_attn_w;
    PlainTensor m_temp;
    PlainTensor m_head_sum;

    MHASingleToken() {}

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // alibi
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, L1, H, S]
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    const PlainTensor& beams,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale,
                    const PlainTensor& k_scale_zp,
                    const PlainTensor& v_scale_zp) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        size_t kv_len;
        kv_len = present_key.size(2);

        // aligned to cache line (64bytes=16*sizeof(float)) to avoid false sharing
        m_attn_w.resize<float>({B, H, q_len, (kv_len + 15) / 16 * 16});
        mha_single_token(query, present_key, present_value, alibi_mask, attention_mask, beams,
            output_emb, m_attn_w, m_temp, has_out_transpose, auto_causal, d_scale, k_scale_zp, v_scale_zp, m_head_sum);
    }
};

template <typename T>
struct ScaledDotProductAttention::MHAExecutor : public ScaledDotProductAttention::Executor {
    GraphContext::CPtr context;
    PlainTensor attn_buf;          // f32[[B|1],[H|1], L1|1, L0+L1]
    std::shared_ptr<kernel::JitKernelBase> m_rotaryKernel;

    MHAKernel<KT_ONEDNN, T> kernel;
    MHASingleToken kernel_single_token;

    MHAExecutor(GraphContext::CPtr ctx) : context(ctx), kernel(context) {}

    void rope_concat(const PlainTensor& q_input, const PlainTensor& k_input, const PlainTensor& v_input,
        PlainTensor& cos_tab, PlainTensor& sin_tab, PlainTensor& present_key, PlainTensor& present_value, size_t rotary_dims, size_t kv_pos) {
        if (!m_rotaryKernel) {
            jit_rotary_compile_params jcp;
            jcp.src_prc = precision_of<T>::value;
            jcp.dst_prc = precision_of<T>::value;
            jcp.rotary_ndims = rotary_dims;
            jcp.interleave = false;
            m_rotaryKernel = createJitKernel(jcp);
        }

        auto batch_size = k_input.size(0);
        auto head_cnt = k_input.size(1);
        auto seq_len = k_input.size(2);
        auto feature_size = k_input.size(3);
        auto kvcache_precision = present_key.get_precision();

        // TODO: support chatglm ... pattern
        parallel_for3d(batch_size, head_cnt, seq_len, [&](size_t b, size_t h, size_t p) {
            auto cos_pos = p + kv_pos;
            auto* q = q_input.ptr<T>(b, h, p);
            auto* k = k_input.ptr<T>(b, h, p);
            auto* v = v_input.ptr<T>(b, h, p);
            auto* cos = cos_tab.ptr<float>(cos_pos, 0);
            auto* sin = sin_tab.ptr<float>(cos_pos, 0);

            if (m_rotaryKernel) {
                execJitKernel(m_rotaryKernel, q, q, cos, sin);
                execJitKernel(m_rotaryKernel, k, k, cos, sin);
            } else {
                auto half_rotary_dims = rotary_dims / 2;
                for (size_t i = 0; i < half_rotary_dims; i++) {
                    auto src0 = q[i];
                    auto src1 = q[i + half_rotary_dims];
                    q[i] = cos[i] * src0 - sin[i] * src1;
                    q[i + half_rotary_dims] = cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0;
                }
                for (size_t i = 0; i < half_rotary_dims; i++) {
                    auto src0 = k[i];
                    auto src1 = k[i + half_rotary_dims];
                    k[i] = cos[i] * src0 - sin[i] * src1;
                    k[i + half_rotary_dims] = cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0;
                }
            }
            if (kvcache_precision == ov::element::u8) {
                auto* k_cache = reinterpret_cast<float*>(present_key.ptr<uint8_t>(b, h, p + kv_pos));
                attn_quant_u8(k, present_key.ptr<uint8_t>(b, h, p + kv_pos) + 8, feature_size, k_cache[0], k_cache[1], k_input.get_precision());
                auto* v_cache = reinterpret_cast<float*>(present_value.ptr<uint8_t>(b, h, p + kv_pos));
                attn_quant_u8(v, present_value.ptr<uint8_t>(b, h, p + kv_pos) + 8, feature_size, v_cache[0], v_cache[1], k_input.get_precision());
            } else {
                attn_memcpy2d_kernel(k, present_key.ptr_v(b, h, p + kv_pos), k_input.get_precision(), kvcache_precision, 0, 0, feature_size, 1);
                attn_memcpy2d_kernel(v, present_value.ptr_v(b, h, p + kv_pos), k_input.get_precision(), kvcache_precision, 0, 0, feature_size, 1);
            }
        });
    }

    void execute(dnnl::stream strm, const Config& config, const std::vector<MemoryPtr>& inputs, const MemoryPtr output,
                 const MemoryPtr, const MemoryPtr, const MemoryPtr, const PlainTensor&, const PlainTensor&) override {
        PlainTensor q_input;                        // fx[B, L1, H * S]
        PlainTensor k_input;
        PlainTensor v_input;
        PlainTensor kv_cache;                       // [2 * layer_num, max_kv_len, B, Hk, S]
        PlainTensor present_key, present_value;     // [B, Hk, max_kv_len, S]
        PlainTensor beam_input;
        PlainTensor beam_table;                     // i32[B, max_kv_len]
        PlainTensor attn_mask;                      // fx[B, L0 + L1]
        PlainTensor cos_tab;                        // f32[max_kv_len, rotary_dims]
        PlainTensor sin_tab;                        // f32[max_kv_len, rotary_dims]
        PlainTensor output_emb(output);         // fx[B, L1, H * S]
        float scale_input = 0.0f;
        size_t B, L1, L0, S, H, Hk;

        // init
        kv_cache.reset(inputs[0]);
        beam_input.reset(inputs[1]);
        attn_mask.reset(inputs[2]);
        cos_tab.reset(inputs[3]);
        sin_tab.reset(inputs[4]);
        S = kv_cache.size(4);
        H = config.config_mha.n_head;
        Hk = config.config_mha.num_kv_heads;
        if (Hk == 0)
            Hk = H;
        if (inputs.size() == 6) {
            PlainTensor qkv;
            qkv.reset(inputs[5]);
            q_input = qkv.slice(2, 0, H * S);
            k_input = qkv.slice(2, H * S, (H + Hk) * S);
            v_input = qkv.slice(2, (H + Hk) * S, (H + 2 * Hk) * S);
        } else {
            q_input.reset(inputs[5]);
            k_input.reset(inputs[6]);
            v_input.reset(inputs[7]);
        }

        B = q_input.size(0);
        L1 = q_input.size(1);
        L0 = attn_mask.size(1) - L1;
        auto layer_id = config.config_mha.layer_id;

        q_input.assert_dims({B, L1, H * S});
        k_input.assert_dims({B, L1, Hk * S});
        v_input.assert_dims({B, L1, Hk * S});
        //kv_cache.assert_dims({0, 0, Hk, 0, S}, true);
        // greedy-search will not use beam input
        if (beam_input.m_dims[0])
            beam_table = beam_input;
        attn_mask.assert_dims({B, L0 + L1});
        cos_tab.assert_dims({0, static_cast<size_t>(config.config_mha.rotary_dims)}, true);
        sin_tab.assert_dims({0, static_cast<size_t>(config.config_mha.rotary_dims)}, true);
        output_emb.assert_dims({B, L1, H * S});
        if (inputs.size() == 6) {
            size_t strides[4] = {q_input.m_strides[0], q_input.m_strides[1], S, 1};
            PlainTensor q_tmp, k_tmp, v_tmp;
            q_tmp.resize({B, L1, H, S}, q_input.ptr<T>(), strides);
            q_input = q_tmp.permute({0, 2, 1, 3});
            k_tmp.resize({B, L1, Hk, S}, k_input.ptr<T>(), strides);
            k_input = k_tmp.permute({0, 2, 1, 3});
            v_tmp.resize({B, L1, Hk, S}, v_input.ptr<T>(), strides);
            v_input = v_tmp.permute({0, 2, 1, 3});
        } else {
            q_input = q_input.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
            k_input = k_input.reshape({B, L1, Hk, S}).permute({0, 2, 1, 3});
            v_input = v_input.reshape({B, L1, Hk, S}).permute({0, 2, 1, 3});
        }
        attn_mask = attn_mask.reshape({B, 1, 1, L0 + L1});

        // [2 * layer_num, max_kv_len, B_max, Hk, S] => [L0+L1, B_max, Hk, S]
        present_key = kv_cache.slice(0, 2 * layer_id, 2 * layer_id).slice(0, 0, L0 + L1);
        present_value = kv_cache.slice(0, 2 * layer_id + 1, 2 * layer_id + 1).slice(0, 0, L0 + L1);
        auto B_max = present_key.size(1);
        if (B != B_max) {
            OPENVINO_ASSERT(B_max % B == 0, "max batch must be multiple of current batch");
            auto num_beams = B_max / B;
            // [L, B_Max(B*num_beams), H, S]
            // make (num_beams - 1) batch item as padding
            present_key.m_dims[1] /= num_beams;     // = B_max/num_beams
            present_key.m_strides[1] *= num_beams;

            present_value.m_dims[1] /= num_beams;
            present_value.m_strides[1] *= num_beams;
        }

        // L,B,H,S => B,H,L,S
        present_key = present_key.permute({1,2,0,3});
        present_value = present_value.permute({1,2,0,3});

        // rope & concat
        rope_concat(q_input, k_input, v_input, cos_tab, sin_tab, present_key, present_value, config.config_mha.rotary_dims, L0);

        // attention
        if (L1 > 1) {
            // multi-token version
            kernel(strm, q_input, k_input, v_input, {}, attn_mask, output_emb, true, true, scale_input);
        } else {
            // 1-token version
            // for second token, using a special AVX2/AVX512 float path:
            //  1, in matrix mutiply, using AMX is not efficency because the M dimension of A will alway be 1
            //  2, using float will save the repack cost which typically is required for bf16/int8 opt
            //  3, using dot product can leverage the SIMD while easily adapt to indirect kv cache
            PlainTensor k_scale_zp, v_scale_zp;
            kernel_single_token(q_input, present_key, present_value, {}, attn_mask,
                output_emb, beam_table, true, true, scale_input, k_scale_zp, v_scale_zp);
        }
    }
};

template <ScaledDotProductAttention::KernelTypes KType, typename T>
struct ScaledDotProductAttention::AttentionExecutor : public ScaledDotProductAttention::Executor {
    GraphContext::CPtr context;
    PlainTensor attn_buf;          // f32[[B|1],[H|1], L1|1, L0+L1]

    MHAKernel<KType, T> kernel;
    MHASingleToken kernel_single_token;

    AttentionExecutor(GraphContext::CPtr ctx) : context(ctx), kernel(context) {}

    void prepare_attn_mask(MemoryPtr attn_input) {
        attn_buf.resize<float>(attn_input->getStaticDims());
        auto p = attn_input->getDataAs<uint8_t>();
        for (size_t i = 0; i < attn_input->getSize(); i++)
            attn_buf.ptr<float>()[i] = p[i] ? 0.0f : -FLT_MAX;
    }

    void execute(dnnl::stream strm, const Config& config, const std::vector<MemoryPtr>& inputs, const MemoryPtr output,
                 const MemoryPtr presentk_input, const MemoryPtr presentv_input, const MemoryPtr beam_input,
                 const PlainTensor& k_scale_zp, const PlainTensor& v_scale_zp) override {
        bool has_out_transpose = config.config.output_BLHxS;
        bool fuse_causal_attn = config.config.fuse_causal_attn;
        bool is_causal = config.config.is_causal;
        bool fuse_concat = config.config.fuse_concat;
        auto input_num = inputs.size();
        PlainTensor present_key, present_value;
        PlainTensor q_input;           // f32[B, H, L1, S]
        PlainTensor k_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
        PlainTensor v_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
        PlainTensor beam_table;        // i32[B, max_kvLen]
        PlainTensor attn_mask;
        PlainTensor output_emb(output);
        float scale_input = 0.0f;
        size_t B, L1, L0, S;

        q_input.reset(inputs[0]);
        k_input.reset(inputs[1]);
        v_input.reset(inputs[2]);
        present_key.reset(presentk_input);
        present_value.reset(presentv_input);
        if (beam_input)
            beam_table.reset(beam_input);
        if (input_num > 3) {
            // attn_mask
            if (inputs[3]->getDesc().getPrecision() == ov::element::u8) {
                // bool->f32
                prepare_attn_mask(inputs[3]);
                attn_mask = attn_buf;
            } else {
                attn_mask.reset(inputs[3]);
            }
            // if has scale, attn_mask must be present
            if (input_num > 4) {
                scale_input = *inputs[4]->getDataAs<float>();
            }
        }

        // q: [B, H, L1, S]
        const auto & permute_axes = config.config.permute_axes;
        if (!permute_axes.empty()) {
            q_input = q_input.permute(permute_axes);
            k_input = k_input.permute(permute_axes);
            v_input = v_input.permute(permute_axes);
            present_key = present_key.permute(permute_axes);
            present_value = present_value.permute(permute_axes);
        }
        B = q_input.size(0);
        L1 = q_input.size(2);
        S = q_input.size(3);
        L0 = present_key.size(2) - L1;
        auto Hk = k_input.size(1);

        if (fuse_concat) {
            k_input.assert_dims({B, Hk, L1, S});
            v_input.assert_dims({B, Hk, L1, S});
        } else {
            k_input.assert_dims({B, Hk, L0 + L1, S});
            v_input.assert_dims({B, Hk, L0 + L1, S});
        }
        present_key.assert_dims({B, Hk, L0 + L1, S});
        present_value.assert_dims({B, Hk, L0 + L1, S});
        if (beam_table)
            beam_table.assert_dims({B, L0 + L1});

        bool auto_causal;
        bool use_attn_mask;
        if (fuse_causal_attn) {
            assert(attn_mask);
            attn_mask.assert_dims({B, 1, L1, L0 + L1});
            auto_causal = true;
            use_attn_mask = true;
        } else {
            if (is_causal) {
                auto_causal = true;
                use_attn_mask = false;
            } else {
                // no attn_mask but has scale, there is a 1-d fake attn_mask
                if (input_num > 3 && attn_mask.m_rank > 1) {
                    assert(attn_mask);
                    // spec requires at least 3, but torch sl test does use rank 2
                    if (attn_mask.m_rank == 2)
                        attn_mask = attn_mask.reshape({1, 1, attn_mask.m_dims[0], attn_mask.m_dims[1]});
                    else if (attn_mask.m_rank == 3)
                        attn_mask = attn_mask.reshape({1, attn_mask.m_dims[0], attn_mask.m_dims[1], attn_mask.m_dims[2]});
                    auto_causal = false;
                    use_attn_mask = true;
                } else {
                    auto_causal = false;
                    use_attn_mask = false;
                }
            }
        }

        // second token, or first token with pastkv fusing
        bool use_one_token = L1 == 1 || (fuse_concat && L0 > 0);
        if (!use_one_token) {
            // multi-token version
            kernel(strm, q_input, k_input, v_input, {}, use_attn_mask ? attn_mask : PlainTensor(),
                   output_emb, has_out_transpose, auto_causal, scale_input);
        } else {
            // 1-token version
            // for second token, using a special AVX2/AVX512 float path:
            //  1, in matrix mutiply, using AMX is not efficency because the M dimension of A will alway be 1
            //  2, using float will save the repack cost which typically is required for bf16/int8 opt
            //  3, using dot product can leverage the SIMD while easily adapt to indirect kv cache
            kernel_single_token(q_input, present_key, present_value, {}, use_attn_mask ? attn_mask : PlainTensor(),
                output_emb, beam_table, has_out_transpose, auto_causal, scale_input, k_scale_zp, v_scale_zp);
        }
    }
};

ScaledDotProductAttention::ScaledDotProductAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, SDPAShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const ov::op::v13::ScaledDotProductAttention>(op);
    if (node) {
        m_config.config.is_causal = node->get_causal();
    } else {
        const auto node = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op);
        if (node) {
            m_config.config = node->get_config();
        } else {
            const auto node = std::dynamic_pointer_cast<const ov::op::v15::MultiHeadAttention>(op);
            m_config.config_mha = node->get_config();
            m_config.mha_valid = true;
        }
    }
}

void ScaledDotProductAttention::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto rtPrecision = getRuntimePrecision();
    NodeConfig config;
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto input_num = getOriginalInputsNumber(); 
    config.inConfs.resize(input_num);
    config.outConfs.resize(getOriginalOutputsNumber());
    if (!m_config.mha_valid) {
        // q, k, v
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            rtPrecision, getInputShapeAtPort(0)));
        config.inConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            rtPrecision, getInputShapeAtPort(1)));
        config.inConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            rtPrecision, getInputShapeAtPort(2)));
        auto orginSDPInputNumber = input_num - (m_config.config.fuse_concat ? 3 : 0);

        auto nextPortIdx = 3;
        if (orginSDPInputNumber > 3) {
            // attn_mask
            if (getOriginalInputPrecisionAtPort(nextPortIdx) == ov::element::u8) {
                config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                    ov::element::u8, getInputShapeAtPort(nextPortIdx)));
            } else {
                config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                    rtPrecision, getInputShapeAtPort(nextPortIdx)));
            }
            nextPortIdx++;
        }
        if (orginSDPInputNumber > 4) {
            config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                ov::element::f32, getInputShapeAtPort(nextPortIdx)));
        }

        if (m_config.config.fuse_concat) {
            // beam_idx
            config.inConfs[orginSDPInputNumber + 0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                ov::element::i32, getInputShapeAtPort(orginSDPInputNumber + 0)));

            // Since the InputMemory nodes are simple proxy for the state memory as well as the init subgraph memory,
            // it doesn't make sense to set the real KV cache precision, since we don't need any precision conversions
            // provided by the common graph logic. We set precisions equal to the precisions of the state nodes to avoid
            // reorder insertion in between MemoryInputSDPA and SDPA nodes.

            auto past_k_input_mem_precision = getParentEdgeAt(orginSDPInputNumber + 1)->getParent()->getOriginalOutputPrecisionAtPort(0);
            // pastk
            config.inConfs[orginSDPInputNumber + 1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                past_k_input_mem_precision, getInputShapeAtPort(orginSDPInputNumber + 1)));

            auto past_v_input_mem_precision = getParentEdgeAt(orginSDPInputNumber + 2)->getParent()->getOriginalOutputPrecisionAtPort(0);
            // pastv
            config.inConfs[orginSDPInputNumber + 2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                past_v_input_mem_precision, getInputShapeAtPort(orginSDPInputNumber + 2)));

            config.outConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                past_k_input_mem_precision, getOutputShapeAtPort(1)));
            config.outConfs[1].inPlace(-1);
            config.outConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                past_v_input_mem_precision, getOutputShapeAtPort(2)));
            config.outConfs[2].inPlace(-1);
        }
    } else {
        // kvcache
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(0)));
        // beam table
        config.inConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            ov::element::i32, getInputShapeAtPort(1)));
        // attn_mask
        config.inConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            rtPrecision, getInputShapeAtPort(2)));
        // cos_tab
        config.inConfs[3].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            ov::element::f32, getInputShapeAtPort(3)));
        // sin_tab
        config.inConfs[4].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            ov::element::f32, getInputShapeAtPort(4)));
        if (input_num == 6) {
            // qkv
            config.inConfs[5].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                rtPrecision, getInputShapeAtPort(5)));
        } else {
            OPENVINO_ASSERT(input_num == 8, "input number should be 8 for MultiHeadAttention");
            // q, k, v
            config.inConfs[5].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                rtPrecision, getInputShapeAtPort(5)));
            config.inConfs[6].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                rtPrecision, getInputShapeAtPort(6)));
            config.inConfs[7].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                rtPrecision, getInputShapeAtPort(7)));
        }
    }
    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getOutputShapeAtPort(0)));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref_any);
}

void ScaledDotProductAttention::createPrimitive() {
    if (m_config.config.fuse_concat) {
        auto desc = getSelectedPrimitiveDescriptor();
        if (desc == nullptr)
            OPENVINO_THROW("has unidentified preferable primitive descriptor");
    }
    auto rtPrecision = getRuntimePrecision();

    ScaledDotProductAttentionKey key = {rtPrecision};

    auto builder = [&](const ScaledDotProductAttentionKey& key) -> std::shared_ptr<Executor> {
        std::shared_ptr<Executor> executor = nullptr;
        if (m_config.mha_valid) {
#ifdef OPENVINO_ARCH_X86_64
            if (rtPrecision == ov::element::bf16) {
                executor = std::make_shared<MHAExecutor<ov::bfloat16>>(context);
            } else {
                executor = std::make_shared<MHAExecutor<float>>(context);
            }
#endif
            return executor;
        }

        if (rtPrecision == ov::element::bf16) {
#ifdef OPENVINO_ARCH_X86_64
            executor = std::make_shared<AttentionExecutor<KT_ONEDNN, ov::bfloat16>>(context);
#endif
        } else {
#ifdef OV_CPU_WITH_ACL
            executor = std::make_shared<AttentionExecutor<KT_ACL, float>>(context);
#elif defined(OV_CPU_WITH_MLAS)
            executor = std::make_shared<AttentionExecutor<KT_MLAS, float>>(context);
#elif defined(OPENVINO_ARCH_X86_64)
            if (with_cpu_x86_avx512_core()) {
                executor = std::make_shared<AttentionExecutor<KT_ONEDNN, float>>(context);
            } else {
                executor = std::make_shared<AttentionExecutor<KT_REF, float>>(context);
            }
#else
            executor = std::make_shared<AttentionExecutor<KT_REF, float>>(context);
#endif
        }
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    if (!result.first) {
        OPENVINO_THROW("ScaledDotProductAttention AttentionExecutor creation fails with precision " + rtPrecision.to_string());
    }
    m_executor = result.first;
}

void ScaledDotProductAttention::execute(dnnl::stream strm) {
    auto orginSDPInputNumber = getOriginalInputsNumber() - (m_config.config.fuse_concat ? 3 : 0);
    std::vector<MemoryPtr> inputs(orginSDPInputNumber);
    auto output = getDstMemoryAtPort(0);
    MemoryPtr presentk_input, presentv_input, beam_input;
    for (size_t i = 0; i < orginSDPInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    PlainTensor k_scale_zp, v_scale_zp;
    if (!m_config.mha_valid) {
        if (m_config.config.fuse_concat) {
            CPU_NODE_ASSERT(m_k_state && m_v_state, "has null input states");
            // initialization will be also completed in this func
            gatherConcatPastkv(inputs[1], inputs[2], getSrcMemoryAtPort(orginSDPInputNumber));

            presentk_input = m_k_state->internal_state_mem();
            presentv_input = m_v_state->internal_state_mem();
            beam_input = m_k_state->hidden_state_mem();
            k_scale_zp = m_k_state->get_scale_zp();
            v_scale_zp = m_v_state->get_scale_zp();
        } else {
            presentk_input = inputs[1];
            presentv_input = inputs[2];
        }
    }
    m_executor->execute(strm, m_config, inputs, output, presentk_input, presentv_input, beam_input, k_scale_zp, v_scale_zp);
}

bool ScaledDotProductAttention::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto mha = std::dynamic_pointer_cast<const op::v15::MultiHeadAttention>(op);
        if (!std::dynamic_pointer_cast<const ov::op::v13::ScaledDotProductAttention>(op) &&
            !std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op) &&
            !mha) {
            errorMessage = "Only ScaledDotProductAttention or ScaledDotProductAttentionWithKVCache or MultiHeadAttention operation are supported";
            return false;
        }
        if (!mha) {
            // expect shape of q: [B, H, L, S]
            auto inRank = op->get_input_partial_shape(0).size();
            if (inRank != 4u) {
                errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank);
                return false;
            }
            int orgSDPAInput = static_cast<int>(op->get_input_size());
            const auto node = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op);
            if (node) {
                if (node->get_config().fuse_concat) {
                    orgSDPAInput -= 3;
                }
            }
            if (orgSDPAInput > 3) {
                inRank = op->get_input_partial_shape(3).size();
                if (inRank > 4u) {
                    errorMessage = "Doesn't support 'attention mask' with rank: " + std::to_string(inRank);
                    return false;
                }
            }
            // using mha should be better for static shapes
            if (!op->is_dynamic()) {
                errorMessage = "Only run in dynamic mode";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

void ScaledDotProductAttention::assignState(const std::shared_ptr<VariableStateKVcache>& state, int idx) {
    auto inputNumber = getOriginalInputsNumber();
    if (inputNumber - 2 == static_cast<size_t>(idx)) {
        m_k_state = state;
    } else if (inputNumber - 1 == static_cast<size_t>(idx)) {
        m_v_state = state;
    } else {
        OPENVINO_THROW(
            "Unexpected idx ", idx , " for a state in a node with type: ", getTypeStr(), " and name ", getName());
    }
}

template <typename T>
std::vector<T> permute_axes(const std::vector<T>& shape, const std::vector<size_t>& order) {
    std::vector<T> results(shape.size());
    for (size_t i = 0; i < order.size(); i++) {
        results[i] = shape[order[i]];
    }
    return results;
}

void ScaledDotProductAttention::resetBeamTablePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx) {
    std::vector<size_t> order = {0, 1, 2, 3};
    if (!m_config.config.permute_axes.empty()) {
        order = m_config.config.permute_axes;
    }
    // order aims to permute input shape to BHLS
    // real_order aims to permute input shape to LBHS
    const std::vector<size_t> real_order = getKVCacheOrder();
    PlainTensor beam_idx, old_beam_table_k;
    auto old_hidden_state_k = m_k_state->hidden_state_mem();
    beam_idx.reset(mem_beam_idx);

    auto inputNumber = getOriginalInputsNumber();
    auto&& v_dims = getParentEdgeAt(inputNumber - 1)->getMemory().getStaticDims();
    size_t L0 = v_dims.at(order[2]);
    auto B_state = v_dims.at(order[0]);
    old_beam_table_k.reset(old_hidden_state_k);

    PlainTensor cur_k;
    PlainTensor cur_v;
    cur_k.reset(mem_cur_k);
    cur_v.reset(mem_cur_v);
    cur_k = cur_k.permute(order);
    cur_v = cur_v.permute(order);
    auto B = cur_k.size(0);
    auto H = cur_k.size(1);
    auto L1 = cur_k.size(2);
    auto S = cur_k.size(3);
    auto reverse = [&order] (const std::vector<size_t>& cur) {
        std::vector<size_t> result(cur.size());
        for (size_t i = 0; i < cur.size(); i++) {
            result[order[i]] = cur[i];
        }
        return result;
    };

    // 1. check beam idx if it's valid
    auto* table = beam_idx.ptr<int32_t>();
    for (size_t i = 0; i < B; i++) {
        OPENVINO_ASSERT(static_cast<size_t>(table[i]) < B_state, "beam_idx[", i, "]=", table[i],
            " should less than batch of previous pastkv: ", B_state);
    }

    // 2. resize pastkv
    ov::element::Type kvcache_precision = m_k_state->internal_desc()->getPrecision();
    {
        // shape is the shape used by the original model which maybe different from BHLS, reverse here is to permute BHLS to original model shape.
        // BHLS is the stated input shape of SDPA, however internally we use LBHS for KV-cache storage.
        // real_order is used to permute the original shape to LBHS
        std::vector<size_t> shape = reverse({B, H, (L0 + L1) * 2, S});
        auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(kvcache_precision,
                                                               Shape(shape),
                                                               permute_axes(shape, real_order),
                                                               real_order);
        auto new_internal_mem_k = std::make_shared<Memory>(getEngine(), mem_desc);
        auto new_internal_mem_v = std::make_shared<Memory>(getEngine(), mem_desc);

        PlainTensor new_pastk, new_pastv, old_past_k, old_past_v;
        new_pastk.reset(new_internal_mem_k);
        new_pastv.reset(new_internal_mem_v);
        new_pastk = new_pastk.permute(order);
        new_pastv = new_pastv.permute(order);
        if (L0 > 0) {
            auto old_internal_mem_k = m_k_state->internal_state_mem();
            auto old_internal_mem_v = m_v_state->internal_state_mem();
            old_past_k.reset(old_internal_mem_k);
            old_past_v.reset(old_internal_mem_v);
            old_past_k = old_past_k.permute(order);
            old_past_v = old_past_v.permute(order);
            parallel_for3d(B, H, L0, [&](size_t b, size_t h, size_t m) {
                auto idx = static_cast<size_t>(table[b]);
                auto b_kv = static_cast<size_t>(old_beam_table_k.at<int32_t>({idx, m}));
                memcpy(&new_pastk.at<char>({b, h, m}),
                       &old_past_k.at<char>({b_kv, h, m}),
                       S * old_past_k.m_element_size);
                memcpy(&new_pastv.at<char>({b, h, m}),
                       &old_past_v.at<char>({b_kv, h, m}),
                       S * old_past_v.m_element_size);
            });
        }
        if (kvcache_precision == ov::element::u8) {
            auto& old_scale_zp_k = m_k_state->get_scale_zp();
            auto& old_scale_zp_v = m_v_state->get_scale_zp();
            PlainTensor new_scale_zp_k, new_scale_zp_v;
            std::vector<size_t> shape = reverse({B, H, (L0 + L1) * 2, 2});
            std::vector<size_t> real_shape = permute_axes(shape, real_order);
            new_scale_zp_k.resize<float>(real_shape);
            new_scale_zp_v.resize<float>(real_shape);
            if (L0 > 0) {
                parallel_for2d(L0, B, [&](size_t m, size_t b) {
                    auto idx = static_cast<size_t>(table[b]);
                    for (size_t h = 0; h < H; h++) {
                        auto b_kv = static_cast<size_t>(old_beam_table_k.at<int32_t>({idx, m}));
                        new_scale_zp_k.at<float>({m, b, h, 0}) = old_scale_zp_k.at<float>({m, b_kv, h, 0});
                        new_scale_zp_k.at<float>({m, b, h, 1}) = old_scale_zp_k.at<float>({m, b_kv, h, 1});
                        new_scale_zp_v.at<float>({m, b, h, 0}) = old_scale_zp_v.at<float>({m, b_kv, h, 0});
                        new_scale_zp_v.at<float>({m, b, h, 1}) = old_scale_zp_v.at<float>({m, b_kv, h, 1});
                    }
                });
            }

            m_k_state->set_scale_zp(new_scale_zp_k);
            m_v_state->set_scale_zp(new_scale_zp_v);
        }

        std::vector<size_t> new_shape = reverse({B, H, (L0 + L1), S});
        // Get the shape of physical layout using real order
        auto strides = mem_desc->getStrides();
        mem_desc = std::make_shared<CpuBlockedMemoryDesc>(kvcache_precision,
                                                          Shape(new_shape),
                                                          permute_axes(new_shape, real_order),
                                                          real_order,
                                                          0,
                                                          VectorDims{},
                                                          strides);
        new_internal_mem_k->redefineDesc(mem_desc);
        new_internal_mem_v->redefineDesc(mem_desc);
        if (kvcache_precision == ov::element::u8) {
            // past_k's shape is BHLS, internal layout LBHS
            // scale_zp's shape is LBHS, internal layout LBHS
            attn_quantkv(cur_k, cur_v,
                new_pastk.slice(2, L0, L0 + L1), new_pastv.slice(2, L0, L0 + L1),
                m_k_state->get_scale_zp().slice(0, L0, L0 + L1), m_v_state->get_scale_zp().slice(0, L0, L0 + L1));
        } else {
            attn_memcpy(cur_k, cur_v, new_pastk.slice(2, L0, L0 + L1), new_pastv.slice(2, L0, L0 + L1));
        }

        m_k_state->assign_internal_state(new_internal_mem_k);
        m_v_state->assign_internal_state(new_internal_mem_v);
        m_k_state->assign_internal_state_max_size(B * H * (L0 + L1) * 2 * S);
        m_v_state->assign_internal_state_max_size(B * H * (L0 + L1) * 2 * S);
    }
    // 3. create beam table
    {
        auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, Shape{B, (L0 + L1) * 2});

        auto new_hidden_state_k = std::make_shared<Memory>(getEngine(), mem_desc);
        auto new_hidden_state_v = std::make_shared<Memory>(getEngine(), mem_desc);
        PlainTensor new_beam_table_k, new_beam_table_v;
        new_beam_table_k.reset(new_hidden_state_k);
        new_beam_table_v.reset(new_hidden_state_v);

        for (size_t b = 0; b < B; b++) {
            for (size_t l = 0; l < L0 + L1; l++) {
                new_beam_table_k.at<int32_t>({b, l}) = b;
                new_beam_table_v.at<int32_t>({b, l}) = b;
            }
        }

        std::vector<size_t> new_shape{B, (L0 + L1)};
        mem_desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32,
            Shape(new_shape),
            new_shape,
            VectorDims{0, 1},
            0,
            VectorDims{},
            mem_desc->getStrides());
        new_hidden_state_k->redefineDesc(mem_desc);
        new_hidden_state_v->redefineDesc(mem_desc);

        m_k_state->assign_hidden_state(new_hidden_state_k);
        m_v_state->assign_hidden_state(new_hidden_state_v);
        m_k_state->assign_hidden_state_max_size(B * (L0 + L1) * 2);
        m_v_state->assign_hidden_state_max_size(B * (L0 + L1) * 2);
    }
}

void ScaledDotProductAttention::gatherConcatPastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v, const MemoryPtr& mem_beam_idx) {
    PlainTensor cur_k;
    cur_k.reset(mem_cur_k);
    auto inputNumber = getOriginalInputsNumber();
    auto&& v_dims = getParentEdgeAt(inputNumber - 1)->getMemory().getStaticDims();
    size_t B_state;
    if (!m_config.config.permute_axes.empty()) {
        cur_k = cur_k.permute(m_config.config.permute_axes);
        B_state = v_dims.at(m_config.config.permute_axes[0]);
    } else {
        B_state = v_dims.at(0);
    }

    auto B = cur_k.size(0);
    auto L1 = cur_k.size(2);
    if (B != B_state) {
        resetBeamTablePastkv(mem_cur_k, mem_cur_v, mem_beam_idx);
        return;
    }

    updateBeamTable(mem_beam_idx, L1);
    updatePastkv(mem_cur_k, mem_cur_v);
}

// Update beam table using beam_idx. For first token, beam table is like [[0, 0, 0, ...], [1, 1, 1, ...], ...],
//   for second token, beam table is updated using gather(beam_table, beam_idx) then appending [0, 1, 2, ...] to the end for itself.
void ScaledDotProductAttention::updateBeamTable(const MemoryPtr& mem_beam_idx, size_t L1) {
    std::vector<size_t> order = {0, 1, 2, 3};
    if (!m_config.config.permute_axes.empty()) {
        order = m_config.config.permute_axes;
    }
    PlainTensor beam_idx, beam_table_k, beam_table_v;
    auto hidden_state_k = m_k_state->hidden_state_mem();
    auto hidden_state_v = m_v_state->hidden_state_mem();
    beam_idx.reset(mem_beam_idx);

    auto B = beam_idx.size(0);
    auto is_reset = m_k_state->is_reset_state() || m_v_state->is_reset_state();
    auto inputNumber = getOriginalInputsNumber();
    auto&& v_dims = getParentEdgeAt(inputNumber - 1)->getMemory().getStaticDims();
    size_t L0 = v_dims.at(order[2]);
    auto B_state = v_dims.at(order[0]);
    OPENVINO_ASSERT(m_k_state->is_reset_state() == m_v_state->is_reset_state(),
        "KV state must be reset simultaneously, please also reset state for ",
        (m_k_state->is_reset_state() ? m_v_state->get_name() : m_k_state->get_name()));
    OPENVINO_ASSERT(B == B_state, "beam idx batch: ", B, " is not equal to batch of state: ", B_state);
    OPENVINO_ASSERT(B * (L0 + L1) > 0, "B or (L0+L1) is zero, B: ", B, ", L0: ", L0, ", L1: ", L1);
    // resize buffer
    bool need_redefine = true;
    if (B * (L0 + L1) > m_k_state->hidden_state_max_size()) {
        auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, Shape{B, (L0 + L1) * 2});

        auto new_hidden_state_k = std::make_shared<Memory>(getEngine(), mem_desc);
        auto new_hidden_state_v = std::make_shared<Memory>(getEngine(), mem_desc);
        PlainTensor new_beam_table_k, new_beam_table_v;
        new_beam_table_k.reset(new_hidden_state_k);
        new_beam_table_v.reset(new_hidden_state_v);
        if (L0 > 0 && !is_reset) {
            beam_table_k.reset(hidden_state_k);
            beam_table_v.reset(hidden_state_v);
            for (size_t b = 0; b < B; b++) {
                std::memcpy(new_beam_table_k.ptr<int32_t>(b), beam_table_k.ptr<int32_t>(b), sizeof(int32_t) * L0);
                std::memcpy(new_beam_table_v.ptr<int32_t>(b), beam_table_v.ptr<int32_t>(b), sizeof(int32_t) * L0);
            }
        }
        m_k_state->assign_hidden_state(new_hidden_state_k);
        m_v_state->assign_hidden_state(new_hidden_state_v);
        m_k_state->assign_hidden_state_max_size(B * (L0 + L1) * 2);
        m_v_state->assign_hidden_state_max_size(B * (L0 + L1) * 2);
        hidden_state_k = new_hidden_state_k;
        hidden_state_v = new_hidden_state_v;
        beam_table_k = new_beam_table_k;
        beam_table_v = new_beam_table_v;
    }  else if (is_reset) {
        // when reset and not resize, just reset the desc
        need_redefine = false;
        auto size = m_k_state->hidden_state_max_size();
        auto max_l = size / B;
        VectorDims strides(2);
        strides[0] = max_l;
        strides[1] = 1;
        std::vector<size_t> new_shape{B, (L0 + L1)};
        auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32,
            Shape(new_shape),
            new_shape,
            VectorDims{0, 1},
            0,
            VectorDims{},
            strides);
        hidden_state_k->redefineDesc(mem_desc);
        hidden_state_v->redefineDesc(mem_desc);
    }
    if (need_redefine) {
        std::vector<size_t> new_shape{B, (L0 + L1)};
        auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32,
            Shape(new_shape),
            new_shape,
            VectorDims{0, 1},
            0,
            VectorDims{},
            hidden_state_k->getDescWithType<BlockedMemoryDesc>()->getStrides());
        hidden_state_k->redefineDesc(mem_desc);
        hidden_state_v->redefineDesc(mem_desc);
    }

    if (!beam_table_k) {
        beam_table_k.reset(hidden_state_k);
        beam_table_v.reset(hidden_state_v);
    }

    // first token
    if (L0 == 0 || is_reset) {
        for (size_t b = 0; b < B; b++) {
            for (size_t l = 0; l < L0 + L1; l++) {
                beam_table_k.at<int32_t>({b, l}) = b;
                beam_table_v.at<int32_t>({b, l}) = b;
            }
        }
        return;
    }

    // beam order is like [0, 1, 2,...]
    bool no_reorder = true;
    for (size_t i = 0; i < B; i++) {
        if (beam_idx.ptr<int32_t>()[i] != static_cast<int32_t>(i)) {
            no_reorder = false;
            break;
        }
    }

    // reorder
    if (!no_reorder) {
        auto* table = beam_idx.ptr<int32_t>();
        // beam table is same for both k,v state
        for (size_t i = 0; i < B; i++) {
            std::memcpy(beam_table_k.ptr<int32_t>(i),
                        beam_table_v.ptr<int32_t>(table[i]),
                        sizeof(int32_t) * L0);
        }
        for (size_t i = 0; i < B; i++) {
            std::memcpy(beam_table_v.ptr<int32_t>(i),
                        beam_table_k.ptr<int32_t>(i),
                        sizeof(int32_t) * L0);
        }
    }
    // second token itself
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < L1; j++) {
            beam_table_k.at<int32_t>({i, L0 + j}) = i;
            beam_table_v.at<int32_t>({i, L0 + j}) = i;
        }
    }
}

// Update pastkv using cur_k, cur_v, simply append cur_k, cur_v to the end of pastkv in the state.
void ScaledDotProductAttention::updatePastkv(const MemoryPtr& mem_cur_k, const MemoryPtr& mem_cur_v) {
    // L, B, H, S -> [2, 0, 1, 3] -> B, H, L, S
    std::vector<size_t> order = {0, 1, 2, 3};
    if (!m_config.config.permute_axes.empty()) {
        order = m_config.config.permute_axes;
    }
    // order aims to pemute input to B, H, L, S, but the real layout of past key value here is L, B, H, S
    std::vector<size_t> real_order = {order[2], order[0], order[1], order[3]};
    PlainTensor cur_k, past_k;
    PlainTensor cur_v, past_v;
    cur_k.reset(mem_cur_k);
    cur_v.reset(mem_cur_v);
    cur_k = cur_k.permute(order);
    cur_v = cur_v.permute(order);
    auto B = cur_k.size(0);
    auto H = cur_k.size(1);
    auto L1 = cur_k.size(2);
    auto S = cur_k.size(3);
    auto reverse = [&order] (const std::vector<size_t>& cur) {
        std::vector<size_t> result(cur.size());
        for (size_t i = 0; i < cur.size(); i++) {
            result[order[i]] = cur[i];
        }
        return result;
    };
    auto internal_mem_k = m_k_state->internal_state_mem();
    auto internal_mem_v = m_v_state->internal_state_mem();

    auto is_reset = m_k_state->is_reset_state();
    auto inputNumber = getOriginalInputsNumber();
    auto&& v_dims = getParentEdgeAt(inputNumber - 1)->getMemory().getStaticDims();
    size_t L0 = v_dims.at(order[2]);
    auto B_state = v_dims.at(order[0]);
    OPENVINO_ASSERT(B == B_state, "pastkv batch: ", B, " is not equal to batch of state: ", B_state);
    OPENVINO_ASSERT(B * (L0 + L1) > 0, "B or (L0+L1) is zero, B: ", B, ", L0: ", L0, ", L1: ", L1);
    // resize buffer
    ov::element::Type kvcache_precision = m_k_state->internal_desc()->getPrecision();
    bool need_redefine = true;
    if (B * H * (L0 + L1) * S > m_k_state->internal_state_max_size()) {
        // new_shape is the shape used by the original model which maybe different from BHLS, reverse here is to permute BHLS to original model shape.
        // BHLS is the stated input shape of SDPA, however internally we use LBHS for KV-cache storage.
        // real_order is used to permute the original shape to LBHS
        std::vector<size_t> new_shape = reverse({B, H, (L0 + L1) * 2, S});
        auto real_shape = permute_axes(new_shape, real_order);
        auto mem_desc =
            std::make_shared<CpuBlockedMemoryDesc>(kvcache_precision, Shape(new_shape), real_shape, real_order);

        auto new_internal_mem_k = std::make_shared<Memory>(getEngine(), mem_desc);
        auto new_internal_mem_v = std::make_shared<Memory>(getEngine(), mem_desc);

        PlainTensor new_pastk, new_pastv;
        new_pastk.reset(new_internal_mem_k);
        new_pastv.reset(new_internal_mem_v);
        new_pastk = new_pastk.permute(order);
        new_pastv = new_pastv.permute(order);
        if (L0 > 0 && !is_reset) {
            past_k.reset(internal_mem_k);
            past_v.reset(internal_mem_v);
            past_k = past_k.permute(order);
            past_v = past_v.permute(order);
            attn_memcpy(past_k, past_v, new_pastk, new_pastv);
        }
        internal_mem_k = new_internal_mem_k;
        internal_mem_v = new_internal_mem_v;
        past_k = new_pastk;
        past_v = new_pastv;
        m_k_state->assign_internal_state(new_internal_mem_k);
        m_v_state->assign_internal_state(new_internal_mem_v);
        m_k_state->assign_internal_state_max_size(2 * (L0 + L1) * B * H * S);
        m_v_state->assign_internal_state_max_size(2 * (L0 + L1) * B * H * S);
        if (kvcache_precision == ov::element::u8) {
            auto& old_scale_zp_k = m_k_state->get_scale_zp();
            auto& old_scale_zp_v = m_v_state->get_scale_zp();
            PlainTensor new_scale_zp_k, new_scale_zp_v;
            std::vector<size_t> shape = reverse({B, H, (L0 + L1) * 2, 2});
            std::vector<size_t> real_shape = permute_axes(shape, real_order);
            new_scale_zp_k.resize<float>(real_shape);
            new_scale_zp_v.resize<float>(real_shape);
            if (L0 > 0 && !is_reset) {
                parallel_for(L0, [&](size_t m) {
                    memcpy(new_scale_zp_k.ptr<float>(m), old_scale_zp_k.ptr<float>(m), sizeof(float) * B * H * 2);
                    memcpy(new_scale_zp_v.ptr<float>(m), old_scale_zp_v.ptr<float>(m), sizeof(float) * B * H * 2);
                });
            }

            m_k_state->set_scale_zp(new_scale_zp_k);
            m_v_state->set_scale_zp(new_scale_zp_v);
        }
    } else if (is_reset) {
        // when reset and not resize, just reset the desc
        need_redefine = false;
        // new_shape is the shape used by the original model which maybe different from BHLS, reverse here is to permute BHLS to original model shape.
        // BHLS is the stated input shape of SDPA, however internally we use LBHS for KV-cache storage.
        // real_order is used to permute the original shape to LBHS
        std::vector<size_t> new_shape = reverse({B, H, (L0 + L1), S});
        VectorDims strides(new_shape.size(), 1);
        auto real_shape = permute_axes(new_shape, real_order);
        for (size_t i = 2; i <= real_shape.size(); i++) {
            strides[real_shape.size() - i] = strides[real_shape.size() - (i-1)] * real_shape[real_shape.size() - (i-1)];
        }
        auto mem_desc = std::make_shared<CpuBlockedMemoryDesc>(kvcache_precision,
            Shape(new_shape),
            real_shape,
            real_order,
            0,
            VectorDims{},
            strides);
        internal_mem_k->redefineDesc(mem_desc);
        internal_mem_v->redefineDesc(mem_desc);
        if (kvcache_precision == ov::element::u8) {
            auto& old_scale_zp_k = m_k_state->get_scale_zp();
            auto& old_scale_zp_v = m_v_state->get_scale_zp();
            // only dim0, dim1 need change
            // LBHS
            old_scale_zp_k.m_strides[0] = H * B * 2;
            old_scale_zp_k.m_strides[1] = H * 2;
            old_scale_zp_v.m_strides[0] = H * B * 2;
            old_scale_zp_v.m_strides[1] = H * 2;
        }
    }
    if (need_redefine) {
        // new_shape is the shape used by the original model which maybe different from BHLS, reverse here is to permute BHLS to original model shape.
        // BHLS is the stated input shape of SDPA, however internally we use LBHS for KV-cache storage.
        // real_order is used to permute the original shape to LBHS
        std::vector<size_t> new_shape = reverse({B, H, (L0 + L1), S});
        auto real_shape = permute_axes(new_shape, real_order);
        auto mem_desc =
            std::make_shared<CpuBlockedMemoryDesc>(kvcache_precision,
                                                   Shape(new_shape),
                                                   real_shape,
                                                   real_order,
                                                   0,
                                                   VectorDims{},
                                                   internal_mem_k->getDescWithType<BlockedMemoryDesc>()->getStrides());
        internal_mem_k->redefineDesc(mem_desc);
        internal_mem_v->redefineDesc(mem_desc);
    }

    if (!past_k) {
        past_k.reset(internal_mem_k);
        past_v.reset(internal_mem_v);
        past_k = past_k.permute(order);
        past_v = past_v.permute(order);
    }
    if (L0 > 0 && is_reset) {
        auto inputNumber = getOriginalInputsNumber();
        auto k_mem = getSrcMemoryAtPort(inputNumber - 2);
        auto v_mem = getSrcMemoryAtPort(inputNumber - 1);
        auto&& k_shape = k_mem->getShape();
        auto&& v_shape = v_mem->getShape();
        if (!k_shape.hasZeroDims() && !v_shape.hasZeroDims()) {
            PlainTensor init_k, init_v;
            init_k.reset(k_mem);
            init_v.reset(v_mem);
            init_k = init_k.permute(order);
            init_v = init_v.permute(order);
            if (kvcache_precision == ov::element::u8) {
                attn_quantkv(init_k, init_v, past_k, past_v, m_k_state->get_scale_zp(), m_v_state->get_scale_zp());
            } else {
                attn_memcpy(init_k, init_v, past_k, past_v);
            }
        }
    }

    if (kvcache_precision == ov::element::u8) {
        // past_k's shape is BHLS, internal layout LBHS
        // scale_zp's shape is LBHS, internal layout LBHS
        attn_quantkv(cur_k, cur_v,
            past_k.slice(2, L0, L0 + L1), past_v.slice(2, L0, L0 + L1),
            m_k_state->get_scale_zp().slice(0, L0, L0 + L1), m_v_state->get_scale_zp().slice(0, L0, L0 + L1));
    } else {
        attn_memcpy(cur_k, cur_v, past_k.slice(2, L0, L0 + L1), past_v.slice(2, L0, L0 + L1));
    }
}

ov::element::Type ScaledDotProductAttention::getKVCachePrecision() {
    ov::element::Type kvcache_precision;
    auto rtPrecision = getRuntimePrecision();
    auto kvCachePrecisionHint = context->getConfig().kvCachePrecision;
    bool enableKVCacheFP16 = m_config.config.fuse_concat && mayiuse(cpu_isa_t::avx2) &&
        rtPrecision != ov::element::bf16 && kvCachePrecisionHint == ov::element::f16;
    kvcache_precision = enableKVCacheFP16 ? ov::element::f16 : rtPrecision;
    bool use_int8_kv_cache_precision = kvCachePrecisionHint == ov::element::u8;
    if (use_int8_kv_cache_precision)
        kvcache_precision = ov::element::u8;
    else
        kvcache_precision = enableKVCacheFP16 ? ov::element::f16 : rtPrecision;

    return kvcache_precision;
}

ov::element::Type ScaledDotProductAttention::getRuntimePrecision() const {
    auto rtPrecision = getOriginalInputPrecisionAtPort(m_config.mha_valid ? 5 : 0);
    // bf16 should be enabled only when platform supports
    if (rtPrecision == ov::element::bf16 && ov::with_cpu_x86_bfloat16()) {
        rtPrecision = ov::element::bf16;
    } else {
        rtPrecision = ov::element::f32;
    }
    return rtPrecision;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
