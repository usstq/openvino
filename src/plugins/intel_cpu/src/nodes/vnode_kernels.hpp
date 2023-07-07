
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
// specialization on llmdnn
template <>
struct RoPE_kernel<KT_LLMDNN, ov::bfloat16> {
    RoPE_kernel() = default;

    llmdnn::emb_gpt m_kernel_emb;
    bool m_kernel_initialized = false;

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
        auto B = cur_query.size(0);
        auto L1 = cur_query.size(1);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto rotary_dims = rotary_emb_cos.size(3);
        if (!m_kernel_initialized) {
            if (!m_kernel_emb.create(llmdnn::emb_gpt::create_param{
                    .num_heads = H,
                    .head_size = S,
                    .head_size_aligned = S,  // better to aligned to 64 bytes for best performance, apply for qkv
                    .qkv_precision = llmdnn::data_type_t::dnnl_bf16,
                    .dst_precision = llmdnn::data_type_t::dnnl_bf16,
                    .rotary_dims = rotary_dims,
                    .use_position2d = false,
                })) {
                IE_THROW() << __func__ << " llmdnn::emb_gpt::create failed " << std::endl;
            }
            m_kernel_initialized = true;
        }

        m_kernel_emb.exec(llmdnn::emb_gpt::exec_param{
            .batch = B,
            .query_seq_len = L1,
            .past_seq_len = L0,
            .q = reinterpret_cast<uint8_t*>(cur_query.data()),  // [B, L1, H, S]
            .k = reinterpret_cast<uint8_t*>(cur_key.data()),    // [B, L1, H, S]
            .v = reinterpret_cast<uint8_t*>(cur_value.data()),  // [B, L1, H, S]
            .ldq = cur_query.stride(2),
            .ldk = cur_key.stride(2),
            .ldv = cur_value.stride(2),
            .query_dst = reinterpret_cast<uint8_t*>(query_emb.data()),  // rotary embbeding dst
            .layer_past_key_src = past_key.get_batched_ptrs(),          // past key src
            .layer_past_value_src = past_value.get_batched_ptrs(),      // past value src
            .layer_past_key_dst =
                present_key.get_batched_ptrs(),  // past key dst, if layer_past_key_src!=layer_past_key_dst,
                                                 // will copy layer_past_key_src to layer_past_key_dst
            .layer_past_value_dst =
                present_value.get_batched_ptrs(),  // past value dst, if layer_past_value!=layer_past_value_dst,
                                                   // will copy layer_past_value to layer_past_value_dst
            .cos = rotary_emb_cos.data(),
            .sin = rotary_emb_sin.data(),
            .position2d_ids = nullptr,          // shape: [batch, 2, query_seq_len]
            .head_stride_in_kv = (L0 + L1) * S  // kv stride for next head; kv may be preallocated a big buffer
        });
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
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<T>& query,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    PlainTensor<float>& attention_mask,
                    PlainTensor<T>& output_emb) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        std::vector<float> attn_score(kv_len, 0.0f);
        std::vector<float> word_vec(head_size, 0.0f);
        float d_scale = 1.0f / sqrt(head_size);

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
                    auto attn_mask_qlen = attention_mask.size(2);
                    auto* attn_mask = &attention_mask.at({b, 0, std::min(m, attn_mask_qlen - 1), 0});
                    if (attn_mask_qlen > 1) {
                        // this imply attn mask is combined with causal mask
                        ncausal = kv_len;
                    }
                    for (size_t n = 0; n < ncausal; n++) {
                        auto* k = &present_key.at({b, h, n, 0});
                        attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;
                        if (p_alibi)
                            attn_score[n] += p_alibi->at({h, 0, n});
                        // apply attention mask (maybe combined with causal_mask)
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
                    PlainTensor<float>& attention_mask,
                    PlainTensor<float>& output_emb) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto attn_mask_qlen = attention_mask.size(2);
        auto update_len = kv_len - q_len;
        float d_scale = 1.0f / sqrt(head_size);
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
                ov_sgemm("N", "T", q_len, kv_len, head_size, 1.0f, q_ptr, head_size, k_ptr, head_size, 0.f, qk, kv_len, 1);
            else
                ov_sgemm("N", "N", q_len, kv_len, head_size, 1.0f, q_ptr, head_size, k_ptr, kv_len, 0.f, qk, kv_len, 1);

            for (size_t m = 0; m < q_len; m++) {
                size_t offset = m * kv_len;
                // apply attention mask
                // sofmax
                auto ncausal = update_len + m + 1;
                if (attn_mask_qlen > 1) {
                    // this imply attn mask is combined with causal mask
                    ncausal = kv_len;
                }
                auto* mask = &attention_mask.at({b, 0, std::min(m, attn_mask_qlen - 1), 0});
                InferenceEngine::Extensions::Cpu::XARCH::scale_add_softmax(&qk[offset],
                                                                           d_scale,
                                                                           mask,
                                                                           p_alibi ? &p_alibi->at({h, 0, 0}) : nullptr,
                                                                           ncausal,
                                                                           kv_len);
            }
            ov_sgemm("N", "N", q_len, head_size, kv_len, 1.0f, qk, kv_len, v_ptr, head_size, 0.f, dst, head_size, 1);
            // output [B, L1, H*S]
            for (size_t m = 0; m < q_len; m++) {
                const float* src = dst + m * head_size;
                std::copy(src, src + head_size, &output_emb.at({b, m, h * head_size}));
            }
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
                    PlainTensor<float>& attention_mask,  // [batch, 1, query_seq_len, key_seq_len]
                    PlainTensor<ov::bfloat16>& attn_output) {
        int max_position_embeddings = 2048;
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto is_causal_in_attention = attention_mask.size(2) > 1;
        if (!m_kernel_initialized) {
            if (!m_kernel.create(llmdnn::mha_gpt::create_param{
                    .num_heads = H,
                    .head_size = head_size,
                    .head_size_aligned =
                        head_size,  // better to aligned to 64 bytes for best performance, apply for qkv
                    .max_seq_len = static_cast<int>(
                        max_position_embeddings),  // max seq length for computing the size of matmul tmp result
                    .normal_factor = 1.0f / sqrt(head_size),
                    .qkv_precision = llmdnn::data_type_t::dnnl_bf16,
                    .dst_precision = llmdnn::data_type_t::dnnl_bf16,
                    .is_bloom = p_alibi != nullptr,
                })) {
                IE_THROW() << __func__ << " llmdnn::mha_gpt::create failed " << std::endl;
            }
            m_kernel_initialized = true;
        }
        auto batched_ptrs_key = present_key.get_batched_ptrs();
        auto batched_ptrs_value = present_value.get_batched_ptrs();
        m_kernel.exec(llmdnn::mha_gpt::exec_param{
            .batch = B,
            .query_seq_len = q_len,
            .key_seq_len = kv_len,
            .is_causal_in_attention =
                is_causal_in_attention,  // causal mask is fused in attention mask: chatglm uses it.
            .q = reinterpret_cast<uint8_t*>(
                query.data()),        // q buffer, compact, shape: [batch, num_heads, query_seq_len, head_size]
            .k = batched_ptrs_key,    // k buffer, k[N] stands different batch which may be discreted
                                      //      k[0] shape: [batch, num_heads, key_seq_len, head_size]
            .v = batched_ptrs_value,  // v buffer, v[N] stands different batch which may be discreted
                                      //      v[0] shape: [batch, num_heads, value_seq_len, head_size]
            .attention_mask = attention_mask.data(),  // attention mask, attention_mask[0] shape:
                                                      //      [batch, 1, 1, key_seq_len], when
                                                      //      is_causal_in_attention is false
            //      [batch, 1, query_seq_len, key_seq_len], when is_causal_in_attention is true
            .attn_output = reinterpret_cast<uint8_t*>(
                attn_output.data()),  // output, compact, shape: [batch, query_seq_len, num_heads
                                      // * head_size]
            .head_stride_in_kv =
                (kv_len)*head_size,  // kv stride for next head; kv may be preallocated a big buffer
                                     // expected quant schema:
                                     //   q,k,v use per tensor quant, attn_output may use per tensor/channel quant
            .alibi = p_alibi ? p_alibi->data() : nullptr,
            .q_dequant = 1.0f,
            .k_dequant = 1.0f,
            .v_dequant = 1.0f,
            .qk_quant = 1.0f,
            .qkv_quant = {}  // size==1 per tensor, size==head_size per channel
        });
    }
};
#endif

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov