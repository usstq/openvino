
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <dnnl_extension_utils.h>
#include <node.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "vnode_utils.hpp"
#include "openvino/core/parallel.hpp"

#ifdef OV_CPU_WITH_LLM
#include "llm_emb_gpt.hpp"
#include "llm_mha_gpt.hpp"
#include "llm_mm.hpp"
#endif
#include "utils/plain_tensor.hpp"
#include "gemm/ov_cpu_gemm.h"
namespace ov {
namespace intel_cpu {
namespace node {

//============================ kernels ============================
enum KernelTypes { KT_REF, KT_LLMDNN, KT_MLAS };

// default implementation: reference
template <KernelTypes KType, typename T>
struct RoPE_kernel {
    RoPE_kernel() = default;

    void operator()(PlainTensor<T>& cur_query,  // B,L,H,S
                    PlainTensor<T>& cur_key,    // B,L,H,S
                    PlainTensor<T>& cur_value,  // B,L,H,S
                    PlainTensor<T>& past_key,
                    PlainTensor<T>& past_value,
                    PlainTensor<T>& query_emb,  // B,H,L,S
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    PlainTensor<float>& rotary_emb_cos,
                    PlainTensor<float>& rotary_emb_sin) {
        auto B = cur_query.size(0);
        auto L1 = cur_query.size(1);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto rotary_dims = rotary_emb_cos.size(3);
        auto half_rotary_dims = rotary_dims / 2;
        // copy past kv into present
        parallel_for2d(B, H, [&](size_t b, size_t h) {
            memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(T) * L0 * S);
            memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(T) * L0 * S);
        });

        // rotary embedding for word vector at each position p
        // meanwhile concat is performed
        parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
            p += L0;
            auto* q = &cur_query.at({b, p - L0, h, 0});
            auto* k = &cur_key.at({b, p - L0, h, 0});
            auto* v = &cur_value.at({b, p - L0, h, 0});
            auto* present_k = &present_key.at({b, h, p, 0});    // f32[B, H, L0+L1, 64]
            auto* present_v = &present_value.at({b, h, p, 0});  // f32[B, H, L0+L1, 64]
            auto* q_embed = &query_emb.at({b, h, p - L0, 0});
            // q_embed = (q * cos) + (rotate_half(q) * sin)
            // k_embed = (k * cos) + (rotate_half(k) * sin)
            auto* cos = &rotary_emb_cos({0, 0, p, 0});
            auto* sin = &rotary_emb_sin({0, 0, p, 0});

            size_t s = 0;
            for (; s < half_rotary_dims; s++) {
                q_embed[s] = cos[s] * q[s] + sin[s] * (-q[s + half_rotary_dims]);
                present_k[s] = cos[s] * k[s] + sin[s] * (-k[s + half_rotary_dims]);
                present_v[s] = v[s];
            }
            for (; s < rotary_dims; s++) {
                q_embed[s] = cos[s] * q[s] + sin[s] * (q[s - half_rotary_dims]);
                present_k[s] = cos[s] * k[s] + sin[s] * (k[s - half_rotary_dims]);
                present_v[s] = v[s];
            }
            for (; s < S; s++) {
                q_embed[s] = q[s];
                present_k[s] = k[s];
                present_v[s] = v[s];
            }
        });
    }
};

#ifdef OV_CPU_WITH_LLM
template<typename DT>
llmdnn::tensor Convert2LLMTensor(const PlainTensor<DT>& src) {
    llmdnn::tensor dst;
    dst.m_capacity = 0;
    dst.m_rank = src.m_rank;
    dst.m_ptr = src.m_ptr.get();
    memcpy(dst.m_dims, src.m_dims, sizeof(size_t) * src.m_rank);
    dst.m_element_size = sizeof(DT);
    dst.m_dtype = llmdnn::precision_of<DT>::value;
    for (size_t i = 0; i < src.m_rank; i++) {
        dst.m_strides[i] = src.m_strides[i] * sizeof(DT);
    }
    return std::move(dst);
}

// specialization on llmdnn
template <>
struct RoPE_kernel<KT_LLMDNN, ov::bfloat16> {
    RoPE_kernel() = default;

    void operator()(PlainTensor<ov::bfloat16>& cur_query,   // B,L,H,S
                    PlainTensor<ov::bfloat16>& cur_key,     // B,L,H,S
                    PlainTensor<ov::bfloat16>& cur_value,   // B,L,H,S
                    PlainTensor<ov::bfloat16>& past_key,    // f32[B, H, L0, S]
                    PlainTensor<ov::bfloat16>& past_value,  // f32[B, H, L0, S]
                    PlainTensor<ov::bfloat16>& query_emb,
                    PlainTensor<ov::bfloat16>& present_key,
                    PlainTensor<ov::bfloat16>& present_value,
                    PlainTensor<float>& rotary_emb_cos,
                    PlainTensor<float>& rotary_emb_sin) {
        llmdnn::emb_gpt(Convert2LLMTensor(cur_query),               // B,L,H,S
                        Convert2LLMTensor(cur_key),                 // B,L,H,S
                        Convert2LLMTensor(cur_value),               // B,L,H,S
                        Convert2LLMTensor(past_key),                // f32[B, H, L0, S]
                        Convert2LLMTensor(past_value),              // f32[B, H, L0, S]
                        Convert2LLMTensor(query_emb),
                        Convert2LLMTensor(present_key),
                        Convert2LLMTensor(present_value),
                        Convert2LLMTensor(rotary_emb_cos),
                        Convert2LLMTensor(rotary_emb_sin),
                        llmdnn::tensor());
    }
};
#endif

// default implementation: reference
template <KernelTypes KType, typename T>
struct MHA_kernel {
    MHA_kernel() = default;

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

    PlainTensor<float>* p_alibi = nullptr;

    void set_alibi(PlainTensor<float>& alibi) {
        p_alibi = &alibi;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, L1, S]
    // present_key   [B, H, L0+L1, S]  stride of last dim maybe > 1
    // present_value [B, H, L0+L1, S]
    // attention_mask [B, 1, L1, L0 + L1]
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<T>& query,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<T>& output_emb,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        std::vector<float> attn_score(kv_len, 0.0f);
        std::vector<float> word_vec(head_size, 0.0f);
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                // auto key = &present_key.at({b, h, 0, 0});
                // auto value = &present_value.at({b, h, 0, 0});
                // auto output = &output_emb.at({b, 0, h * head_size});
                for (size_t m = 0; m < q_len; m++) {
                    // dot-product to get attention scores
                    auto* q = &query.at({b, h, m, 0});
                    // how many key/values can be accessed causally
                    auto ncausal = kv_len - q_len + m + 1;
                    float *attn_mask = nullptr;
                    if (attention_mask) {
                        auto attn_mask_qlen = attention_mask.size(2);
                        attn_mask = &attention_mask.at({b, 0, std::min(m, attn_mask_qlen - 1), 0});
                        if (attn_mask_qlen > 1) {
                            // this imply attn mask is combined with causal mask
                            ncausal = kv_len;
                        }
                    } else {
                        ncausal = kv_len;
                    }
                    for (size_t n = 0; n < ncausal; n++) {
                        auto* k = &present_key.at({b, h, n, 0});
                        attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;
                        if (p_alibi)
                            attn_score[n] += p_alibi->at({h, 0, n});
                        // apply attention mask (maybe combined with causal_mask)
                        if (attn_mask)
                            attn_score[n] += attn_mask[n];
                    }

                    // softmax
                    softmax(&attn_score[0], ncausal);

                    // linearly combine value
                    word_vec.assign(head_size, 0.0f);
                    for (size_t n = 0; n < ncausal; n++) {
                        auto* v = &present_value.at({b, h, n, 0});
                        accumulate(word_vec.data(), v, head_size, attn_score[n]);
                    }

                    // output [B, L1, H*head_size]
                    auto* out = &output_emb.at({b, m, h * head_size});
                    std::copy(word_vec.begin(), word_vec.end(), out);
                }
            }
        }
    }
};

template <>
struct MHA_kernel<KT_MLAS, float> {
    MHA_kernel() = default;
    PlainTensor<float>* p_alibi = nullptr;

    void set_alibi(PlainTensor<float>& alibi) {
        p_alibi = &alibi;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, L1, S]
    // present_key   [B, H, L0+L1, S]  stride of last dim maybe > 1
    // present_value [B, H, L0+L1, S]
    // attention_mask [B, 1, L1, L0 + L1]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<float>& query,
                    PlainTensor<float>& present_key,
                    PlainTensor<float>& present_value,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<float>& output_emb,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        size_t attn_mask_qlen = 0;
        if (attention_mask) {
            attn_mask_qlen = attention_mask.size(2);
        }
        auto update_len = kv_len - q_len;
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        // initialize temp buffer for qk matmul and  qkv matmul
        size_t num_threads = parallel_get_num_threads();
        if (p_qk_buffer == nullptr) {
            p_qk_buffer.reset(new PlainTensor<float>());
        }
        p_qk_buffer->resize({num_threads, q_len, kv_len});
        if (p_dst_buffer == nullptr) {
            p_dst_buffer.reset(new PlainTensor<float>());
        }
        p_dst_buffer->resize({num_threads, q_len, head_size});
        auto k_stride_s = present_key.stride(3);
        parallel_for2d(B, H, [&](size_t b, size_t h) {
            size_t thread_id = static_cast<size_t>(parallel_get_thread_num());
            const float* q_ptr = &query.at({b, h, 0, 0});
            const float* k_ptr = &present_key.at({b, h, 0, 0});
            const float* v_ptr = &present_value.at({b, h, 0, 0});
            float* qk = &(p_qk_buffer->at({thread_id, 0, 0}));
            float* dst = &(p_dst_buffer->at({thread_id, 0, 0}));
            if (k_stride_s == 1)
                ov_sgemm("N", "T", q_len, kv_len, head_size, 1.0f, q_ptr, query.stride(2), k_ptr, present_key.stride(2), 0.f, qk, kv_len, 1);
            else
                ov_sgemm("N", "N", q_len, kv_len, head_size, 1.0f, q_ptr, query.stride(2), k_ptr, present_key.stride(2), 0.f, qk, kv_len, 1);

            float * mask = nullptr;
            for (size_t m = 0; m < q_len; m++) {
                size_t offset = m * kv_len;
                // apply attention mask
                // sofmax
                auto ncausal = update_len + m + 1;
                if (attn_mask_qlen > 0) {
                    mask = &attention_mask.at({b, 0, std::min(m, attn_mask_qlen - 1), 0});
                    if (attn_mask_qlen > 1) {
                        // this imply attn mask is combined with causal mask
                        ncausal = kv_len;
                    }
                } else {
                    // no attention mask, means no causal mask too
                    ncausal = kv_len;
                }
                InferenceEngine::Extensions::Cpu::XARCH::scale_add_softmax(&qk[offset],
                                                                           d_scale,
                                                                           mask,
                                                                           p_alibi ? &p_alibi->at({h, 0, 0}) : nullptr,
                                                                           ncausal,
                                                                           kv_len);
            }
            ov_sgemm("N",
                     "N",
                     q_len,
                     head_size,
                     kv_len,
                     1.0f,
                     qk,
                     kv_len,
                     v_ptr,
                     present_value.stride(2),
                     0.f,
                     &output_emb.at({b, 0, h * head_size}),
                     output_emb.stride(1),
                     1);
            // output [B, L1, H*S]
            //for (size_t m = 0; m < q_len; m++) {
            //    const float* src = dst + m * head_size;
            //    std::copy(src, src + head_size, &output_emb.at({b, m, h * head_size}));
            //}
        });
    }
    // buffer to hold qk temp
    std::unique_ptr<PlainTensor<float>> p_qk_buffer = nullptr;
    // buffer to hold qkv output
    std::unique_ptr<PlainTensor<float>> p_dst_buffer = nullptr;
};

#ifdef OV_CPU_WITH_LLM
template <>
struct MHA_kernel<KT_LLMDNN, ov::bfloat16> {
    llmdnn::mha_gpt m_kernel;
    bool m_kernel_initialized = false;
    MHA_kernel() {}

    PlainTensor<float>* p_alibi = nullptr;

    void set_alibi(PlainTensor<float>& alibi) {
        p_alibi = &alibi;
    }

    void operator()(PlainTensor<ov::bfloat16>& query,
                    PlainTensor<ov::bfloat16>& present_key,
                    PlainTensor<ov::bfloat16>& present_value,
                    const PlainTensor<float>& attention_mask,  // [batch, 1, query_seq_len, key_seq_len]
                    PlainTensor<ov::bfloat16>& attn_output,
                    float d_scale = 0.0f) {
        auto head_size = query.size(3);
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        llmdnn::tensor alibi;
        if (p_alibi) {
            auto B = query.size(0);
            auto H = present_key.size(1);
            alibi = Convert2LLMTensor(*p_alibi).reshape({B, H, 1, p_alibi->size(2)});
        }
        m_kernel.exec(Convert2LLMTensor(query),
                      Convert2LLMTensor(present_key),
                      Convert2LLMTensor(present_value),
                      Convert2LLMTensor(attn_output),
                      Convert2LLMTensor(attention_mask),
                      alibi,
                      d_scale,
                      p_alibi == nullptr);
    }
};
#endif

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov