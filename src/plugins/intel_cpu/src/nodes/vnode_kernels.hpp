
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

    void operator()(PlainTensor<T>& qkv_input,
                    PlainTensor<T>& past_key,
                    PlainTensor<T>& past_value,
                    PlainTensor<T>& query_emb,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    PlainTensor<float>& rotary_emb_cos,
                    PlainTensor<float>& rotary_emb_sin) {
        auto B = qkv_input.get_dims()[0];
        auto L1 = qkv_input.get_dims()[1];
        auto H = past_key.get_dims()[1];
        auto L0 = past_key.get_dims()[2];
        auto S = past_key.get_dims()[3];
        auto rotary_dims = rotary_emb_cos.get_dims()[3];
        auto half_rotary_dims = rotary_dims / 2;
        // copy past kv into present
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(T) * L0 * S);
            }
        }

        // rotary embedding for word vector at each position p
        // meanwhile concat is performed
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t p = L0; p < L0 + L1; p++) {
                    auto* q = &qkv_input.at({b, p - L0, (h * 3 + 0) * S + 0});
                    auto* k = &qkv_input.at({b, p - L0, (h * 3 + 1) * S + 0});
                    auto* v = &qkv_input.at({b, p - L0, (h * 3 + 2) * S + 0});
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
                }
            }
        }
    }
};

#ifdef OV_CPU_WITH_LLM
// specialization on llmdnn
template <>
struct RoPE_kernel<KT_LLMDNN, ov::bfloat16> {
    RoPE_kernel() = default;

    llmdnn::emb_gpt m_kernel_emb;
    bool m_kernel_initialized = false;

    void operator()(PlainTensor<ov::bfloat16>& qkv_input,   // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
                    PlainTensor<ov::bfloat16>& past_key,    // f32[B, H, L0, S]
                    PlainTensor<ov::bfloat16>& past_value,  // f32[B, H, L0, S]
                    PlainTensor<ov::bfloat16>& query_emb,
                    PlainTensor<ov::bfloat16>& present_key,
                    PlainTensor<ov::bfloat16>& present_value,
                    PlainTensor<float>& rotary_emb_cos,
                    PlainTensor<float>& rotary_emb_sin) {
        auto B = qkv_input.get_dims()[0];
        auto L1 = qkv_input.get_dims()[1];
        auto H = past_key.get_dims()[1];
        auto L0 = past_key.get_dims()[2];
        auto S = past_key.get_dims()[3];
        auto rotary_dims = rotary_emb_cos.get_dims()[3];
        if (!m_kernel_initialized) {
            std::cout << "::::::::::::::;  " << __func__ << " llmdnn::emb_gpt " << std::endl;
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
            .q = reinterpret_cast<uint8_t*>(&qkv_input.at({0, 0, 0})),      // [B, L1,  H*3*S]
            .k = reinterpret_cast<uint8_t*>(&qkv_input.at({0, 0, S})),      // [B, L1,  H*3*S]
            .v = reinterpret_cast<uint8_t*>(&qkv_input.at({0, 0, 2 * S})),  // [B, L1,  H*3*S]
            .ldq = 3 * S,
            .ldk = 3 * S,
            .ldv = 3 * S,
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
    float dot_product(const D* a, const D* b, int len) {
        float result = 0;
        for (int i = 0; i < len; i++)
            result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
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

    // Q, K, V is ready, do attention
    // query         [B, H, L1, S]
    // present_key   [B, H, L0+L1, S]
    // present_value [B, H, L0+L1, S]
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<T>& query,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    PlainTensor<float>& attention_mask,
                    PlainTensor<T>& output_emb) {
        auto& query_dims = query.get_dims();
        auto B = query_dims[0];
        auto H = query_dims[1];
        auto q_len = query_dims[2];
        auto head_size = query_dims[3];
        auto& present_key_dims = present_key.get_dims();
        auto kv_len = present_key_dims[2];
        std::vector<float> attn_score(kv_len, 0.0f);
        std::vector<float> word_vec(head_size, 0.0f);
        float d_scale = 1.0f / sqrt(head_size);

        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                auto cquery = &query.at({b, h, 0, 0});
                auto key = &present_key.at({b, h, 0, 0});
                auto value = &present_value.at({b, h, 0, 0});
                auto output = &output_emb.at({b, 0, h * head_size});
                for (size_t m = 0; m < q_len; m++) {
                    // dot-product to get attention scores
                    auto* q = cquery + m * head_size;
                    // how many key/values can be accessed causally
                    auto ncausal = kv_len - q_len + m + 1;
                    for (size_t n = 0; n < ncausal; n++) {
                        auto* k = key + n * head_size;
                        attn_score[n] = dot_product(q, k, head_size) * d_scale;
                    }
                    // apply causal_mask
                    // apply attention mask
                    // softmax
                    softmax(&attn_score[0], ncausal);

                    // linearly combine value
                    word_vec.assign(head_size, 0.0f);
                    for (size_t n = 0; n < ncausal; n++) {
                        auto* v = value + n * head_size;
                        accumulate(word_vec.data(), v, head_size, attn_score[n]);
                    }

                    // output [B, L1, H*head_size]
                    auto* out = output + m * H * head_size;
                    std::copy(word_vec.begin(), word_vec.end(), out);
                }
            }
        }
    }
};

template <KernelTypes KType, typename T>
struct GPT2_MHA_kernel {
    GPT2_MHA_kernel() = default;
    template <typename D>
    float dot_product(const D* a, const D* b, int len) {
        float result = 0;
        for (int i = 0; i < len; i++)
            result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
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

    // Q, K, V is ready, do attention
    // qkv           [B, L1, 3*(H*S)]
    // curKey   [B, H, L0+L1, S]
    // curValue [B, H, L0+L1, S]
    // attnMask [B, 1, 1, L0+L1]
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<T>& qkv,
                 PlainTensor<T>& curKey,
                 PlainTensor<T>& curValue,
                 PlainTensor<float>& attnMask,
                 PlainTensor<T>& output,
                 float* qk,
                 float* dst,
                 size_t batchIndex,
                 size_t headIndex) {
        const auto& qkvDims = qkv.get_dims();
        const auto& curKeyDims = curKey.get_dims();
        auto L1 = qkvDims[1];
        auto S = curKeyDims[3];
        auto L0 = curKeyDims[2] - L1;
        // use dot-product to save memory cost
        float* attn_score = qk;
        std::fill(attn_score, attn_score + L0 + L1, 0.0f);
        float* word_vec = dst;
        std::fill(word_vec, word_vec + S, 0.0f);
        float d_scale = 1.0f / sqrt(S);
        // Q x K^T =[L1, d] * [L0 + L1, d]^T
        for (size_t m = 0; m < L1; m++) {
            // dot-product to get attention scores
            T* q = &qkv.at({batchIndex, m, 0}) + headIndex * S;
            auto* mask = &attnMask.at({batchIndex, 0, 0, 0});
            // causal_mask is applied by above tril-matrix
            // apply attention mask
            for (size_t n = 0; n <= L0 + m; n++) {
                T* k = &curKey.at({batchIndex, headIndex, n, 0});
                attn_score[n] = dot_product(q, k, S) * d_scale;
                attn_score[n] += mask[n];
            }
            // sofmax
            softmax(&attn_score[0], L0 + m + 1);

            // linearly combine value
            std::fill(word_vec, word_vec + S, 0.0f);
            for (size_t n = 0; n <= L0 + m; n++)
                accumulate(word_vec, &curValue.at({batchIndex, headIndex, n, 0}), S, attn_score[n]);

            // output [B, L1, H*S]
            std::copy(word_vec, word_vec + S, &output.at({batchIndex, m, headIndex * S}));
        }
    }
};

template <>
struct GPT2_MHA_kernel<KT_MLAS, float> {
    GPT2_MHA_kernel() = default;
    float dot_product(const float* a, const float* b, int len) {
        float result = 0;
        for (int i = 0; i < len; i++)
            result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        return result;
    }

    void softmax(float* a, size_t len, size_t total_size) {
        float max = *std::max_element(a, a + len);
        float sum = 0.0f;
        for (size_t i = 0; i < len; i++) {
            a[i] = exp(a[i] - max);
            sum += a[i];
        }
        float scale = 1.0f / sum;
        for (size_t i = 0; i < len; i++) {
            a[i] *= scale;
        }
        for (size_t i = len; i < total_size; i++) {
            a[i] = 0.0f;
        }
    }

    void accumulate(float* acc, const float* v, int len, float weight = 1.0f) {
        for (int i = 0; i < len; i++) {
            acc[i] += static_cast<float>(v[i]) * weight;
        }
    }

    // Q, K, V is ready, do attention
    // qkv           [B, L1, 3*(H*S)]
    // curKey   [B, H, L0+L1, S]
    // curValue [B, H, L0+L1, S]
    // attnMask [B, 1, 1, L0+L1]
    // output_emb    [B, L1, H*S]
    // qk        [L1, L0+L1]
    // dst       [L1, S]
    void operator()(PlainTensor<float>& qkv,
                 PlainTensor<float>& curKey,
                 PlainTensor<float>& curValue,
                 PlainTensor<float>& attnMask,
                 PlainTensor<float>& output,
                 float* qk,
                 float* dst,
                 size_t batchIndex,
                 size_t headIndex) {
        const auto& qkvDims = qkv.get_dims();
        const auto& curKeyDims = curKey.get_dims();
        auto L1 = qkvDims[1];
        auto S = curKeyDims[3];
        auto L0 = curKeyDims[2] - L1;
        auto qkvSize = qkvDims[2];
        // use dot-product to save memory cost
        std::vector<float> word_vec(S, 0.0f);
        float d_scale = 1.0f / sqrt(S);
        // Q x K^T =[L1, d] * [L0 + L1, d]^T
        const float* qBasePtr = &qkv.at({batchIndex, 0, 0}) + headIndex * S;
        const float* kBasePtr = &curKey.at({batchIndex, headIndex, 0, 0});
        const float* vBasePtr = &curValue.at({batchIndex, headIndex, 0, 0});
        ov_sgemm("N", "T", L1, L0 + L1, S, 1.0f, qBasePtr, qkvSize, kBasePtr, S, 0.f, qk, L0 + L1, 1);
        // iterate m
        auto* mask = &attnMask.at({batchIndex, 0, 0, 0});
        for (size_t m = 0; m < L1; m++) {
            size_t offset = m * (L0 + L1);
            // apply attention mask
            for (size_t n = 0; n <= L0 + m; n++) {
                qk[offset + n] *= d_scale;
                qk[offset + n] += mask[n];
            }
            // sofmax
            softmax(&qk[offset], L0 + m + 1, L0 + L1);
        }

        ov_sgemm("N", "N", L1, S, L0 + L1, 1.0f, qk, L0 + L1, vBasePtr, S, 0.f, dst, S, 1);

        // output [B, L1, H*S]
        for (size_t m = 0; m < L1; m++) {
            const float * src = dst + m * S;
            std::copy(src, src + S, &output.at({batchIndex, m, headIndex * S}));
        }
    }
    // Q, K, V is ready, do attention
    // qkv        [B, L1, 3*(H*S)]
    // past_key   [B, H, L0, S]
    // past_value [B, H, L0, S]
    // attn_mask  [B, 1, 1, L0+L1]
    // output_emb [B, L1, H*S]
    // void operator()(PlainTensor<float>& qkv,
    //              PlainTensor<float>& past_key,
    //              PlainTensor<float>& past_value,
    //              PlainTensor<float>& attn_mask,
    //              float* qk,
    //              PlainTensor<float>& output,
    //              size_t batchIndex,
    //              size_t headIndex) {
    //     const auto& qkv_dims = qkv.get_dims();
    //     const auto& past_key_dims = past_key.get_dims();
    //     auto L1 = qkv_dims[1];
    //     auto S = past_key_dims[3];
    //     auto L0 = past_key_dims[2];
    //     auto qkvSize = qkv_dims[2];
    //     auto hiddenSize = qkvSize / 3; // [H * S]
    //     // use dot-product to save memory cost
    //     std::vector<float> attn_score(L0 + L1, 0.0f);
    //     std::vector<float> word_vec(S, 0.0f);
    //     float d_scale = 1.0f / sqrt(S);
    //     // Q x K^T =[L1, d] * [L0 + L1, d]^T
    //     const float* qBasePtr = &qkv.at({batchIndex, 0, 0}) + headIndex * S;
    //     const float* kBasePtr = &qkv.at({batchIndex, 0, hiddenSize}) + headIndex * S;
    //     const float* pastKPtr = &qkv.at({batchIndex, headIndex, 0, 0});
    //     // split Q x K^T into [L1,d] *[L0,d]^T + [L1,d] *[L1,d]^T
    //     ov_sgemm("N", "T", L1, L0, S, 1.0f, qBasePtr, qkvSize, pastKPtr, S, 0.f, qk, L0 + L1);
    //     ov_sgemm("N", "T", L1, L1, S, 1.0f, qBasePtr, qkvSize, kBasePtr, qkvSize, 0.f, qk + L1, L0 + L1);
    // }
};

#ifdef OV_CPU_WITH_LLM
template <>
struct MHA_kernel<KT_LLMDNN, ov::bfloat16> {
    llmdnn::mha_gpt m_kernel;
    bool m_kernel_initialized = false;
    MHA_kernel() {}

    void operator()(PlainTensor<ov::bfloat16>& query,
                    PlainTensor<ov::bfloat16>& present_key,
                    PlainTensor<ov::bfloat16>& present_value,
                    PlainTensor<float>& attention_mask,
                    PlainTensor<ov::bfloat16>& attn_output) {
        int max_position_embeddings = 2048;
        auto& query_dims = query.get_dims();
        auto B = query_dims[0];
        auto H = query_dims[1];
        auto q_len = query_dims[2];
        auto head_size = query_dims[3];
        auto& present_key_dims = present_key.get_dims();
        auto kv_len = present_key_dims[2];
        if (!m_kernel_initialized) {
            std::cout << "::::::::::::::;  " << __func__ << " llmdnn::mha_gpt " << std::endl;
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
            .is_causal_in_attention = false,  // causal mask is fused in attention mask: chatglm uses it.
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