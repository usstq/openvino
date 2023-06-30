
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

#include "llm_emb_gpt.hpp"
#include "llm_mha_gpt.hpp"
#include "llm_mm.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {

template <typename TI, typename TO>
void matmul(bool transA, bool transB, int M, int N, int K, TI* pA, int lda, TI* pB, int ldb, TO* pC, int ldc) {
    size_t stride_a0 = transA ? 1 : lda;
    size_t stride_b0 = transB ? 1 : ldb;
    size_t stride_a1 = transA ? lda : 1;
    size_t stride_b1 = transB ? ldb : 1;

    for (int m = 0; m < M; m++) {
        TO* Cm = pC + m * ldc;
        TI* Am = pA + m * stride_a0;
        for (int n = 0; n < N; n++) {
            TI* Bn = pB + n * stride_b1;
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += Am[k * stride_a1] * Bn[k * stride_b0];
            }
            // if (bias) sum += bias[n];
            // if (act) sum = act(sum);
            Cm[n] = sum;
        }
    }
}

template <typename T>
inline float dot_product(const T* a, const T* b, int len) {
    float result = 0;
    for (int i = 0; i < len; i++)
        result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    return result;
}

inline void softmax(float* a, int len) {
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

template <typename T>
inline void accumulate(float* acc, const T* v, int len, float weight = 1.0f) {
    for (int i = 0; i < len; i++) {
        acc[i] += static_cast<float>(v[i]) * weight;
    }
}

// vnode executor with inputs/outputs described in PlainTensor
struct vnode_executor {
    std::vector<PlainTensorBase*> inputs;
    std::vector<PlainTensorBase*> outputs;
    int use_ref;

    vnode_executor() {
        use_ref = std::getenv("USE_REF") ? atoi(std::getenv("USE_REF")) : 0;
    }

    void register_inputs() {}
    template <typename T0, typename... Ts>
    void register_inputs(PlainTensor<T0>& in0, PlainTensor<Ts>&... ins) {
        inputs.push_back(&in0);
        register_inputs(ins...);
    }

    void register_outputs() {}
    template <typename T0, typename... Ts>
    void register_outputs(PlainTensor<T0>& out0, PlainTensor<Ts>&... outs) {
        outputs.push_back(&out0);
        register_outputs(outs...);
    }

    void update_inputs(Node* node) {
        int ii = 0;
        for (auto& inp : inputs) {
            inp->reset(node->getParentEdgeAt(ii++)->getMemoryPtr());
        }
    }

    void update_outputs(Node* node) {
        int oo = 0;
        for (auto& outp : outputs) {
            outp->reset(node->getChildEdgeAt(oo++)->getMemoryPtr());
        }
    }

    virtual void exec(Node* node, dnnl::stream strm) = 0;

    // Q, K, V is ready, do attention
    // query         [B, H, L1, S]
    // present_key   [B, H, L0+L1, S]
    // present_value [B, H, L0+L1, S]
    // output_emb    [B, L1, H*S]
    template <typename T>
    void ref_mha(PlainTensor<T>& query,
                 PlainTensor<T>& present_key,
                 PlainTensor<T>& present_value,
                 PlainTensor<T>& output_emb) {
        auto& query_dims = query.get_dims();
        auto B = query_dims[0];
        auto H = query_dims[1];
        auto L1 = query_dims[2];
        auto S = query_dims[3];
        auto& present_key_dims = present_key.get_dims();
        auto L0 = present_key_dims[2] - L1;

        // use dot-product to save memory cost
        std::vector<float> attn_score(L0 + L1, 0.0f);
        std::vector<float> word_vec(S, 0.0f);
        float d_scale = 1.0f / sqrt(S);
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t m = 0; m < L1; m++) {
                    // dot-product to get attention scores
                    ov::bfloat16* q = &query.at({b, h, m, 0});
                    for (size_t n = 0; n <= L0 + m; n++) {
                        ov::bfloat16* k = &present_key.at({b, h, n, 0});
                        attn_score[n] = dot_product(q, k, S) * d_scale;
                    }
                    // apply causal_mask
                    // apply attention mask
                    // softmax
                    softmax(&attn_score[0], L0 + m + 1);

                    // linearly combine value
                    word_vec.assign(S, 0.0f);
                    for (size_t n = 0; n <= L0 + m; n++)
                        accumulate(word_vec.data(), &present_value.at({b, h, n, 0}), S, attn_score[n]);

                    // output [B, L1, H*S]
                    std::copy(word_vec.begin(), word_vec.end(), &output_emb.at({b, m, h * S}));
                }
            }
        }
    }
};

struct gpt2_attention_executor : public vnode_executor {
    PlainTensor<bfloat16> qkv_input;    // "f32[?,?,2304]"
    PlainTensor<bfloat16> past_key;     // "f32[?,12,?,64]"
    PlainTensor<bfloat16> past_value;   // "f32[?,12,?,64]"
    PlainTensor<uint8_t> Constant_174;  // "u8[1,1,1024,1024]"
    PlainTensor<float> attention_mask;  // "f32[?,1,1,?]"

    PlainTensor<ov::bfloat16> output_emb;     // f32[B, L1, 512]
    PlainTensor<ov::bfloat16> present_key;    // f32[B, H, L0+L1, 64]
    PlainTensor<ov::bfloat16> present_value;  // f32[B, H, L0+L1, 64]

    gpt2_attention_executor(Node* node) {
        std::cout << node->getName() << " creates gpt2_attention_executor" << std::endl;
        register_inputs(qkv_input, past_key, past_value, Constant_174, attention_mask);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<ov::bfloat16> query;

    void exec(Node* node, dnnl::stream strm) override {
        update_inputs(node);

        auto& dims_past = past_key.get_dims();
        auto& dims_qkv = qkv_input.get_dims();
        auto B = dims_past[0];
        auto H = dims_past[1];   // 12
        auto L0 = dims_past[2];  // number of tokens have been encoded
        auto S = dims_past[3];   // 64
        auto L1 = dims_qkv[1];

        qkv_input.assert_dims({B, L1, 3*(H*S)});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});
        attention_mask.assert_dims({B, 1, 1, L0 + L1});

        std::vector<VectorDims> outputShapes;
        outputShapes.push_back(VectorDims{B, L1, H * S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        node->redefineOutputMemory(outputShapes);

        update_outputs(node);

        query.resize({B, H, L1, S});

        DEBUG_LOG(" B=", B, " H=", H, " S=", S, " L0=", L0, " L1=", L1);

        //qkv_input.max_repr_len = 99999;
        DEBUG_LOG("qkv_input=", qkv_input.repr(256, 8));

        // concat pask_key/value & k/v into present_key/value
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(bfloat16) * L0 * S);
                memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(bfloat16) * L0 * S);

                for (size_t p = 0; p < L1; p++) {
                    auto * q = &qkv_input.at({b, p, h*S});
                    auto * k = &qkv_input.at({b, p, (H + h)*S});
                    auto * v = &qkv_input.at({b, p, (2*H + h)*S});
                    memcpy(&query.at({b, h, p, 0}), q, sizeof(bfloat16) * S);
                    memcpy(&present_key.at({b, h, L0 + p, 0}), k, sizeof(bfloat16) * S);
                    memcpy(&present_value.at({b, h, L0 + p, 0}), v, sizeof(bfloat16) * S);
                }
            }
        }
        DEBUG_LOG(" attention_mask=", attention_mask);
        ref_mha(query, present_key, present_value, output_emb);
    }
};

struct gptneox_attention_executor : public vnode_executor {
    PlainTensor<int32_t> input_ids;       // i32[B, L1]
    PlainTensor<float> attention_mask;    // f32[B, 1, 1, L0 + L1]
    PlainTensor<bfloat16> past_key;       // f32[B, H, L0, S]
    PlainTensor<bfloat16> past_value;     // f32[B, H, L0, S]
    PlainTensor<bfloat16> past_0_key;     // f32[B, H, L0, S]
    PlainTensor<bfloat16> qkv_input;      // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
    PlainTensor<uint8_t> attention_bias;  // u8[1,1,2048,2048]
    PlainTensor<float> rotary_emb_cos;    // f32[1,1,2048,16]
    PlainTensor<float> rotary_emb_sin;    // f32[1,1,2048,16]

    PlainTensor<ov::bfloat16> output_emb;     // f32[B, L1, 512]
    PlainTensor<ov::bfloat16> present_key;    // f32[B, H, L0+L1, 64]
    PlainTensor<ov::bfloat16> present_value;  // f32[B, H, L0+L1, 64]

    gptneox_attention_executor(Node* node) {
        std::cout << node->getName() << " creates gptneox_attention_executor" << std::endl;
        register_inputs(input_ids,
                        attention_mask,
                        past_key,
                        past_value,
                        past_0_key,
                        qkv_input,
                        attention_bias,
                        rotary_emb_cos,
                        rotary_emb_sin);
        register_outputs(output_emb, present_key, present_value);
    }

    bool m_kernel_initialized;
    PlainTensor<ov::bfloat16> m_query_emb;  // query with embed
    llmdnn::emb_gpt m_kernel_emb;
    llmdnn::mha_gpt m_kernel_mha;

    size_t B;
    size_t H;
    size_t L0;
    size_t L1;
    size_t S;
    size_t rotary_dims;
    size_t max_position_embeddings;

    void exec(Node* node, dnnl::stream strm) override {
        update_inputs(node);

        auto& dims_past = past_key.get_dims();
        auto& dims_cur = qkv_input.get_dims();
        B = dims_past[0];
        H = dims_past[1];   // 8
        L0 = dims_past[2];  // number of tokens to be encoded
        S = dims_past[3];   // 64
        L1 = dims_cur[1];
        rotary_dims = rotary_emb_cos.get_dims()[3];
        max_position_embeddings = attention_bias.get_dims()[2];

        std::vector<VectorDims> outputShapes;
        outputShapes.push_back(VectorDims{B, L1, H * S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        node->redefineOutputMemory(outputShapes);

        update_outputs(node);

        input_ids.assert_dims({B, L1});
        attention_mask.assert_dims({B, 1, 1, L0 + L1});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});
        past_0_key.assert_dims({B, H, L0, S});
        qkv_input.assert_dims({B, L1, H * 3 * S});
        attention_bias.assert_dims({1, 1, max_position_embeddings, max_position_embeddings});
        rotary_emb_cos.assert_dims({1, 1, max_position_embeddings, rotary_dims});
        rotary_emb_sin.assert_dims({1, 1, max_position_embeddings, rotary_dims});

        DEBUG_LOG(" B=", B, " H=", H, " S=", S, " L0=", L0, " L1=", L1);

        m_query_emb.resize<ov::bfloat16>({B, H, L1, S});

        if (!m_kernel_initialized) {
            if ((use_ref & 1) == 0) {
                std::cout << "::::::::::::::;  " << __func__ << " llmdnn::emb_gpt " << std::endl;
                if (!m_kernel_emb.create(llmdnn::emb_gpt::create_param{
                        .num_heads = H,
                        .head_size = S,
                        .head_size_aligned = S,  // better to aligned to 64 bytes for best performance, apply for qkv
                        .max_seq_len =
                            max_position_embeddings,  // max seq length for computing the size of matmul tmp result
                        .qkv_precision = llmdnn::data_type_t::dnnl_bf16,
                        .dst_precision = llmdnn::data_type_t::dnnl_bf16,
                        .rotary_emb_base = 10000,
                        .rotary_pct = 0.25f,
                        .use_position2d = false,
                    })) {
                    IE_THROW() << __func__ << " llmdnn::emb_gpt::create failed " << std::endl;
                }
            }
            if ((use_ref & 2) == 0) {
                std::cout << "::::::::::::::;  " << __func__ << " llmdnn::mha_gpt " << std::endl;
                if (!m_kernel_mha.create(llmdnn::mha_gpt::create_param{
                        .num_heads = H,
                        .head_size = S,
                        .head_size_aligned = S,  // better to aligned to 64 bytes for best performance, apply for qkv
                        .max_seq_len =
                            max_position_embeddings,  // max seq length for computing the size of matmul tmp result
                        .normal_factor = 1.0f / sqrt(S),
                        .qkv_precision = llmdnn::data_type_t::dnnl_bf16,
                        .dst_precision = llmdnn::data_type_t::dnnl_bf16,
                    })) {
                    IE_THROW() << __func__ << " llmdnn::mha_gpt::create failed " << std::endl;
                }
            }
            m_kernel_initialized = true;
        }

        // exec_ref_rope(); exec_ref_mha(); return;
        auto batched_ptrs_key = present_key.get_batched_ptrs<ov::bfloat16>();
        auto batched_ptrs_value = present_value.get_batched_ptrs<ov::bfloat16>();
        if (use_ref & 1) {
            exec_ref_rope();
        } else {
            m_kernel_emb.exec(llmdnn::emb_gpt::exec_param{
                .batch = B,
                .query_seq_len = L1,
                .past_seq_len = L0,
                .qkv = qkv_input.data<uint8_t>(),  // shape: [batch, query_seq_len, 3 * hidden size]   [B, L1, H*3*S]
                .query_dst = m_query_emb.data<uint8_t>(),                             // rotary embbeding dst
                .layer_past_key_src = past_key.get_batched_ptrs<ov::bfloat16>(),      // past key src
                .layer_past_value_src = past_value.get_batched_ptrs<ov::bfloat16>(),  // past value src
                .layer_past_key_dst = batched_ptrs_key,  // past key dst, if layer_past_key_src!=layer_past_key_dst,
                                                         // will copy layer_past_key_src to layer_past_key_dst
                .layer_past_value_dst =
                    batched_ptrs_value,             // past value dst, if layer_past_value!=layer_past_value_dst,
                                                    // will copy layer_past_value to layer_past_value_dst
                .position2d_ids = nullptr,          // shape: [batch, 2, query_seq_len]
                .head_stride_in_kv = (L0 + L1) * S  // kv stride for next head; kv may be preallocated a big buffer
            });
        }

        DEBUG_LOG(" m_query_emb=", m_query_emb);

        if (use_ref & 2) {
            ref_mha(m_query_emb, present_key, present_value, output_emb);
        } else {
            m_kernel_mha.exec(llmdnn::mha_gpt::exec_param{
                .batch = B,
                .query_seq_len = L1,
                .key_seq_len = L0 + L1,
                .is_causal_in_attention = false,  // causal mask is fused in attention mask: chatglm uses it.
                .q = m_query_emb
                         .data<uint8_t>(),  // q buffer, compact, shape: [batch, num_heads, query_seq_len, head_size]
                .k = batched_ptrs_key,      // k buffer, k[N] stands different batch which may be discreted
                                            //      k[0] shape: [batch, num_heads, key_seq_len, head_size]
                .v = batched_ptrs_value,    // v buffer, v[N] stands different batch which may be discreted
                                            //      v[0] shape: [batch, num_heads, value_seq_len, head_size]
                .attention_mask = attention_mask.data<float>(),  // attention mask, attention_mask[0] shape:
                                                                 //      [batch, 1, 1, key_seq_len], when
                                                                 //      is_causal_in_attention is false
                //      [batch, 1, query_seq_len, key_seq_len], when is_causal_in_attention is true
                .attn_output = output_emb.data<uint8_t>(),  // output, compact, shape: [batch, query_seq_len, num_heads
                                                            // * head_size]
                .head_stride_in_kv =
                    (L0 + L1) * S,  // kv stride for next head; kv may be preallocated a big buffer
                                    // expected quant schema:
                                    //   q,k,v use per tensor quant, attn_output may use per tensor/channel quant
                .q_dequant = 1.0f,
                .k_dequant = 1.0f,
                .v_dequant = 1.0f,
                .qk_quant = 1.0f,
                .qkv_quant = {}  // size==1 per tensor, size==head_size per channel
            });
        }
    }

    void exec_ref_rope() {
        DEBUG_LOG(" attention_mask=", attention_mask);
        DEBUG_LOG(" qkv_input=", qkv_input);  // bfloat16[1,6,1536]
        // DEBUG_LOG(" rotary_emb_cos=", rotary_emb_cos);  // float[1,1,2048,16]
        // DEBUG_LOG(" rotary_emb_sin=", rotary_emb_sin);
        //  rotary embedding on q/k of qkv_input, this is done in-place
        //    position id is range: [L0, L0+L1)
        //  concat (past_key, cur_key)
        //  matmul W=Q*K'
        //  apply causal_mask
        //  apply attention mask
        //  softmax
        //  concat (past_value, cur_value)
        //  matmul W*V
        //

        PlainTensor<ov::bfloat16> ref_q_emb;
        ref_q_emb.resize({B, H, L1, S});  // bfloat16[1,8,6,64]

        // calculate reference
        auto half_rotary_dims = rotary_dims / 2;
        auto rotate_half = [&](bfloat16* q, int s) -> float {
            if (s < half_rotary_dims)
                return static_cast<float>(-q[s + half_rotary_dims]);
            return static_cast<float>(q[s - half_rotary_dims]);
        };

        // copy past kv into present
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(bfloat16) * L0 * S);
                memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(bfloat16) * L0 * S);
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
                    auto* q_embed = &m_query_emb.at({b, h, p - L0, 0});
                    // q_embed = (q * cos) + (rotate_half(q) * sin)
                    // k_embed = (k * cos) + (rotate_half(k) * sin)
                    auto* cos = &rotary_emb_cos({0, 0, p, 0});
                    auto* sin = &rotary_emb_sin({0, 0, p, 0});
                    for (size_t s = 0; s < S; s++) {
                        if (s < rotary_dims) {
                            q_embed[s] = cos[s] * q[s] + sin[s] * rotate_half(q, s);
                            present_k[s] = cos[s] * k[s] + sin[s] * rotate_half(k, s);
                        } else {
                            q_embed[s] = q[s];
                            present_k[s] = k[s];
                        }
                        present_v[s] = v[s];
                    }
                }
            }
        }
        // DEBUG_LOG(" ref_q_emb=", ref_q_emb);
    }
};

inline std::function<std::shared_ptr<vnode_executor>(Node* node)> vnode_executor_creator(std::string name) {
    if (name == "gpt2_attention") {
        return [](Node* node) {
            return std::make_shared<gpt2_attention_executor>(node);
        };
    }
    if (name == "gptneox_attention") {
        return [](Node* node) {
            return std::make_shared<gptneox_attention_executor>(node);
        };
    }
    return nullptr;
}

}  // namespace intel_cpu
}  // namespace ov