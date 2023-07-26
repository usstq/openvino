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

#include "openvino/core/parallel.hpp"
#include "utils/plain_tensor.hpp"
#include "vnode_kernels.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

//============================ executors ============================
// executor is a bridge between CPU node and kernels, contains some glue logic
//
struct vnode_executor {
    std::string signature;
    std::vector<PlainTensorBase*> inputs;
    std::vector<PlainTensorBase*> outputs;

    vnode_executor() {}

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

    virtual void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) = 0;
};


template <typename RT>
struct KVConcatKernel {
    void operator()(const PlainTensor<RT>& k_input,       // "f32[B, L1, H, S]"
                    const PlainTensor<RT>& v_input,       // "f32[B, L1, H, S]"
                    const PlainTensor<RT>& past_key,      // "f32[B, H, L0, S]"
                    const PlainTensor<RT>& past_value,    // "f32[B, H, L0, S]"
                    const PlainTensor<RT>& present_key,   // "f32[B, H, L0+L1, S]"
                    const PlainTensor<RT>& present_value  // "f32[B, H, L0+L1, S]"
    ) {
        auto B = past_key.size(0);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto L1 = k_input.size(1);
        auto past_ks_stride = past_key.stride(3);
        parallel_for3d(2, B, H, [&](size_t kv_id, size_t b, size_t h) {
            if (kv_id == 0) {
                if (past_ks_stride == 1) {
                    if (L0 > 0) {
                        memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                    }
                    for (size_t p = 0; p < L1; p++) {
                        memcpy(&present_key.at({b, h, L0 + p, 0}), &k_input.at({b, p, h, 0}), sizeof(RT) * S);
                    }
                } else {
                    // special layout for bloom past/present_key, [B, H, S, L0]
                    if (L0 > 0) {
                        for (size_t s = 0; s < S; s++) {
                            memcpy(&present_key.at({b, h, 0, s}), &past_key.at({b, h, 0, s}), sizeof(RT) * L0);
                        }
                    }
                    for (size_t s = 0; s < S; s++) {
                        for (size_t p = 0; p < L1; p++) {
                            present_key.at({b, h, L0 + p, s}) = k_input.at({b, p, h, s});
                        }
                    }
                }
            } else {
                // past_key/value
                if (L0 > 0) {
                    memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                }
                for (size_t p = 0; p < L1; p++) {
                    memcpy(&present_value.at({b, h, L0 + p, 0}), &v_input.at({b, p, h, 0}), sizeof(RT) * S);
                }
            }
        });
    }
};

template <KernelTypes KType, typename RT>
struct opt_attention_executor : public vnode_executor {
    PlainTensor<RT> q_input;  // "f32[B, L1, H*S]"
    PlainTensor<RT> k_input;  // "f32[B, L1, H*S]"
    PlainTensor<RT> v_input;  // "f32[B, L1, H*S]"

    PlainTensor<RT> past_key;                    // "f32[B, H, L0, S]"
    PlainTensor<RT> past_value;                  // "f32[B, H, L0, S]"
    PlainTensor<float> combined_attention_mask;  // (attention + causal) : f32[B,1,q_len,kv_len]

    PlainTensor<int32_t> shape;  // "i32[3]"

    PlainTensor<RT> output_emb;     // f32[B, L1, H*S]
    PlainTensor<RT> present_key;    // "f32[B, H, L0+L1, S]"
    PlainTensor<RT> present_value;  // "f32[B, H, L0+L1, S]"

    MHA_kernel<KType, RT> kernel;
    KVConcatKernel<RT> kvconcat;

    opt_attention_executor() {
        register_inputs(q_input, k_input, v_input, past_key, past_value, combined_attention_mask, shape);
        register_outputs(output_emb, present_key, present_value);
    }

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        auto B = q_input.size(0);
        auto L1 = q_input.size(1);
        auto HS = q_input.size(2);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto H = HS / S;

        node->redefineOutputMemory({{B, L1, H * S}, {B, H, L0 + L1, S}, {B, H, L0 + L1, S}});
        update_outputs(node);

        kvconcat(k_input.reshape({B, L1, H, S}),
                 v_input.reshape({B, L1, H, S}),
                 past_key,
                 past_value,
                 present_key,
                 present_value);
        auto query = q_input.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
        kernel(query, present_key, present_value, {}, combined_attention_mask, output_emb, 1.0f);
    }
};

// RT : runtime precision(type)
template <KernelTypes KType, typename RT>
struct gpt2_attention_executor : public vnode_executor {
    PlainTensor<RT> qkv_input;          // "f32[?,?,2304]"
    PlainTensor<RT> past_key;           // "f32[?,12,?,64]"
    PlainTensor<RT> past_value;         // "f32[?,12,?,64]"
    PlainTensor<float> attention_mask;  // "f32[?,1,1,?]"

    PlainTensor<RT> output_emb;     // f32[B, L1, 512]
    PlainTensor<RT> present_key;    // f32[B, H, L0+L1, 64]
    PlainTensor<RT> present_value;  // f32[B, H, L0+L1, 64]

    MHA_kernel<KType, RT> kernel;
    KVConcatKernel<RT> kvconcat;

    gpt2_attention_executor() {
        register_inputs(qkv_input, past_key, past_value, attention_mask);
        register_outputs(output_emb, present_key, present_value);
    }

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        auto B = past_key.size(0);
        auto H = past_key.size(1);   // 12
        auto L0 = past_key.size(2);  // number of tokens have been encoded
        auto S = past_key.size(3);   // 64
        auto L1 = qkv_input.size(1);

        qkv_input.assert_dims({B, L1, 3 * (H * S)});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});
        attention_mask.assert_dims({B, 1, 1, L0 + L1});

        std::vector<VectorDims> outputShapes{VectorDims{B, L1, H * S},
                                             VectorDims{B, H, L0 + L1, S},
                                             VectorDims{B, H, L0 + L1, S}};
        node->redefineOutputMemory(outputShapes);

        update_outputs(node);

        auto qkv4d = qkv_input.reshape({B, L1, 3*H, S});
        auto q_input = qkv4d.slice(2, 0, H).permute({0, 2, 1, 3});
        kvconcat(qkv4d.slice(2, H, 2 * H),
                 qkv4d.slice(2, 2 * H, 3 * H),
                 past_key,
                 past_value,
                 present_key,
                 present_value);
        kernel(q_input, present_key, present_value, {}, attention_mask, output_emb);
    }
};

template <KernelTypes KType, typename RT>
struct gptneox_attention_executor : public vnode_executor {
    PlainTensor<RT> qkv_input;          // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
    PlainTensor<RT> past_key;           // f32[B, H, L0, S]
    PlainTensor<RT> past_value;         // f32[B, H, L0, S]
    PlainTensor<RT> past_0_key;         // f32[B, H, L0, S]
    PlainTensor<float> attention_mask;  // f32[B, 1, 1, L0 + L1]
    PlainTensor<float> rotary_emb_cos;  // f32[1,1,2048,16]
    PlainTensor<float> rotary_emb_sin;  // f32[1,1,2048,16]
    PlainTensor<int32_t> input_ids;     // i32[B, L1]

    PlainTensor<RT> output_emb;     // f32[B, L1, 512]
    PlainTensor<RT> present_key;    // f32[B, H, L0+L1, 64]
    PlainTensor<RT> present_value;  // f32[B, H, L0+L1, 64]

    MHA_kernel<KType, RT> kernel;
    RoPE_kernel<KType, RT> rope_kernel;

    gptneox_attention_executor() {
        register_inputs(qkv_input,
                        past_key,
                        past_value,
                        past_0_key,
                        attention_mask,
                        rotary_emb_cos,
                        rotary_emb_sin,
                        input_ids);
        register_outputs(output_emb, present_key, present_value);
    }

    bool m_kernel_initialized;
    PlainTensor<RT> m_query_emb;  // query with embed

    size_t B;
    size_t H;
    size_t L0;
    size_t L1;
    size_t S;
    size_t rotary_dims;
    size_t max_position_embeddings;

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        B = past_key.size(0);
        H = past_key.size(1);   // 8
        L0 = past_key.size(2);  // number of tokens to be encoded
        S = past_key.size(3);   // 64
        L1 = qkv_input.size(1);
        rotary_dims = rotary_emb_cos.size(3);
        max_position_embeddings = rotary_emb_cos.size(2);

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
        rotary_emb_cos.assert_dims({1, 1, max_position_embeddings, rotary_dims});
        rotary_emb_sin.assert_dims({1, 1, max_position_embeddings, rotary_dims});

        DEBUG_LOG(" B=", B, " H=", H, " S=", S, " L0=", L0, " L1=", L1);

        m_query_emb.resize({B, H, L1, S});

        // PlainTensor<RT> qkv_input;          // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
        auto qkv_5d = qkv_input.reshape({B, L1, H, 3, S});
        auto q_input = qkv_5d.index({{}, {}, {}, {0}, {}});
        auto k_input = qkv_5d.index({{}, {}, {}, {1}, {}});
        auto v_input = qkv_5d.index({{}, {}, {}, {2}, {}});
        rope_kernel(q_input,
                    k_input,
                    v_input,
                    past_key,
                    past_value,
                    m_query_emb,
                    present_key,
                    present_value,
                    rotary_emb_cos,
                    rotary_emb_sin);

        DEBUG_LOG(" m_query_emb=", m_query_emb);

        kernel(m_query_emb, present_key, present_value, {}, attention_mask, output_emb);
    }
};

template <KernelTypes KType, typename RT>
struct open_llama_attention_executor : public vnode_executor {
    PlainTensor<RT> q_proj;  //  "f32[B,L1,H*S]"
    PlainTensor<RT> k_proj;  //  "f32[B,L1,H*S]"
    PlainTensor<RT> v_proj;  //  "f32[B,L1,H*S]"
    PlainTensor<int32_t> self_attn_Shape;
    PlainTensor<RT> past_key;                  // f32[B, H, L0, S]
    PlainTensor<RT> past_value;                // f32[B, H, L0, S]
    PlainTensor<float> attention_causal_mask;  // f32[B, 1, L1, L0+L1]
    PlainTensor<float> rotary_emb_cos;         // f32[1,1,2048,100]
    PlainTensor<float> rotary_emb_sin;         // f32[1,1,2048,100]
    PlainTensor<int32_t> input_ids_shape;      // i32[2]
    PlainTensor<int32_t> past_key_0_shape;     // i32[4]

    PlainTensor<RT> output_emb;     // f32[B, L1, 512]
    PlainTensor<RT> present_key;    // f32[B, H, L0+L1, 64]
    PlainTensor<RT> present_value;  // f32[B, H, L0+L1, 64]

    MHA_kernel<KType, RT> kernel;
    RoPE_kernel<KType, RT> rope_kernel;

    open_llama_attention_executor() {
        register_inputs(q_proj,
                        k_proj,
                        v_proj,
                        self_attn_Shape,
                        past_key,
                        past_value,
                        attention_causal_mask,
                        rotary_emb_cos,
                        rotary_emb_sin,
                        input_ids_shape,
                        past_key_0_shape);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<RT> m_query_emb;  // query with embed

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        auto B = q_proj.size(0);
        auto L1 = q_proj.size(1);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);  // number of tokens to be encoded
        auto S = past_key.size(3);   // 64
        auto max_position_embeddings = rotary_emb_cos.size(2);
        auto rotary_dims = rotary_emb_cos.size(3);

        std::vector<VectorDims> outputShapes;
        outputShapes.push_back(VectorDims{B, L1, H * S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        node->redefineOutputMemory(outputShapes);

        update_outputs(node);

        input_ids_shape.assert_dims({2});
        past_key_0_shape.assert_dims({4});

        q_proj.assert_dims({B, L1, H * S});
        k_proj.assert_dims({B, L1, H * S});
        v_proj.assert_dims({B, L1, H * S});
        attention_causal_mask.assert_dims({B, 1, L1, L0 + L1});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});
        rotary_emb_cos.assert_dims({1, 1, max_position_embeddings, rotary_dims});
        rotary_emb_sin.assert_dims({1, 1, max_position_embeddings, rotary_dims});

        DEBUG_LOG(" B=", B, " H=", H, " S=", S, " L0=", L0, " L1=", L1);

        // std::cout << "attention_causal_mask=" << attention_causal_mask.repr(256) << std::endl;
        // asm("int3");

        m_query_emb.resize({B, H, L1, S});

        auto q_4d = q_proj.reshape({B, L1, H, S});
        auto k_4d = k_proj.reshape({B, L1, H, S});
        auto v_4d = v_proj.reshape({B, L1, H, S});
        rope_kernel(q_4d,
                    k_4d,
                    v_4d,
                    past_key,
                    past_value,
                    m_query_emb,
                    present_key,
                    present_value,
                    rotary_emb_cos,
                    rotary_emb_sin);

        DEBUG_LOG(" m_query_emb=", m_query_emb);

        kernel(m_query_emb, present_key, present_value, {}, attention_causal_mask, output_emb);
    }
};

template <KernelTypes KType, typename RT>
struct bloom_attention_executor : public vnode_executor {
    PlainTensor<RT> qkv_input;   //"f32[B, L, H*3*S]"
    PlainTensor<RT> past_key;    // f32[B*H, S, L]
    PlainTensor<RT> past_value;  // f32[B*H, L, S]
    PlainTensor<float> alibi;    // f32[B*H, 1, kv_len] will be broadcasted add to attention weights [B, H, q_len, kv_len]
    PlainTensor<uint8_t> combined_attention_mask;  // (attention + causal) : u8[B,1,q_len,kv_len]  False means add 0,
                                                   // True means set to -FLT_MAX

    PlainTensor<RT> output_emb;     // f32[B, L1, H*S]
    PlainTensor<RT> present_key;    // f32[B*H, S, L0+L1]
    PlainTensor<RT> present_value;  // f32[B*H, L0+L1, S]

    MHA_kernel<KType, RT> kernel;
    KVConcatKernel<RT> kvconcat;

    bloom_attention_executor() {
        register_inputs(qkv_input, past_key, past_value, alibi, combined_attention_mask);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<float> attention_mask;

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        auto B = qkv_input.size(0);
        auto L1 = qkv_input.size(1);
        auto H3S = qkv_input.size(2);
        auto L0 = past_value.size(1);
        auto S = past_value.size(2);
        auto H = H3S / (3 * S);

        node->redefineOutputMemory({{B, L1, H * S}, {B * H, S, L0 + L1}, {B * H, L0 + L1, S}});
        update_outputs(node);

        qkv_input.assert_dims({B, L1, H * 3 * S});
        past_key.assert_dims({B * H, S, L0});
        past_value.assert_dims({B * H, L0, S});
        alibi.assert_dims({B * H, 1, L0 + L1});
        combined_attention_mask.assert_dims({B, 1, L1, L0 + L1});

        // transform u8 mask to f32 additive mask
        attention_mask.resize({B, 1, L1, L0 + L1});
        auto* psrc = combined_attention_mask.data();
        auto* pdst = attention_mask.data();
        for (int i = 0; i < B * L1 * (L0 + L1); i++) {
            pdst[i] = psrc[i] ? -FLT_MAX : 0;
        }

        auto past_key2 = past_key.reshape({B, H, S, L0}).permute({0, 1, 3, 2});
        auto present_key2 = present_key.reshape({B, H, S, L0 + L1}).permute({0, 1, 3, 2});
        auto past_value2 = past_value.reshape({B, H, L0, S});
        auto present_value2 = present_value.reshape({B, H, L0 + L1, S});

        auto qkv_4d = qkv_input.reshape({B, L1, H, 3*S});

        kvconcat(qkv_4d.slice(3, S, 2 * S),
                 qkv_4d.slice(3, 2 * S, 3 * S),
                 past_key2,
                 past_value2,
                 present_key2,
                 present_value2);

        auto query = qkv_4d.slice(3, 0, S).permute({0, 2, 1, 3});
        kernel(query, present_key2, present_value2, alibi.reshape({B, H, 1, L0 + L1}), attention_mask, output_emb);
    }
};


template <KernelTypes KType, typename RT>
struct bloom2_attention_executor : public vnode_executor {
    PlainTensor<RT> qkv_input;   //"f32[B, qL, H*3*S]"
    PlainTensor<RT> key;         // f32[B, H, S, kL]
    PlainTensor<RT> value;       // f32[B, H, kL, S]
    PlainTensor<float> alibi;    // f32[B, H, 1, kL] will be broadcasted add to attention weights [B, H, q_len, kv_len]
    PlainTensor<uint8_t> combined_attention_mask;  // (attention + causal) : u8[B,1,q_len,kv_len]  False means add 0,
                                                   // True means set to -FLT_MAX

    PlainTensor<RT> output_emb;     // f32[B, qL, H, S]

    MHA_kernel<KType, RT> kernel;
    KVConcatKernel<RT> kvconcat;

    bloom2_attention_executor() {
        register_inputs(qkv_input, key, value, alibi, combined_attention_mask);
        register_outputs(output_emb);
    }

    PlainTensor<float> attention_mask;

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        auto B = qkv_input.size(0);
        auto qL = qkv_input.size(1);
        auto kL = key.size(3);
        auto H = static_cast<size_t>(symbol2value["H"]);
        auto S = static_cast<size_t>(symbol2value["S"]);

        node->redefineOutputMemory({{B, qL, H, S}});
        update_outputs(node);

        qkv_input.assert_dims({B, qL, H * 3 * S});
        key.assert_dims({B, H, S, kL});
        value.assert_dims({B, H, kL, S});
        alibi.assert_dims({B, H, 1, kL});
        combined_attention_mask.assert_dims({B, 1, qL, kL});

        // transform u8 mask to f32 additive mask
        attention_mask.resize({B, 1, qL, kL});
        auto* psrc = combined_attention_mask.data();
        auto* pdst = attention_mask.data();
        for (int i = 0; i < B * qL * kL; i++) {
            pdst[i] = psrc[i] ? -FLT_MAX : 0;
        }

        auto key2 = key.permute({0, 1, 3, 2});

        auto query = qkv_input.reshape({B, qL, H, 3*S}).slice(3, 0, S).permute({0, 2, 1, 3});
        auto output = output_emb.reshape({B, qL, H*S});
        kernel(query, key2, value, alibi, attention_mask, output);
    }
};

template <KernelTypes KType, typename RT>
struct whisper_enc_attention_executor : public vnode_executor {
    PlainTensor<RT> q_input;      // "f32[?,1500,384]"  B,L,H*S
    PlainTensor<RT> k_input;      // "f32[?,1500,384]"  B,L,H*S
    PlainTensor<RT> v_input;      // "f32[?,1500,384]"  B,L,H*S
    PlainTensor<int32_t> shape1;  // "i32[3]"

    PlainTensor<RT> output_emb;  // f32[?,1500,384]

    MHA_kernel<KType, RT> kernel;

    whisper_enc_attention_executor() {
        register_inputs(q_input, k_input, v_input, shape1);
        register_outputs(output_emb);
    }

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);
        auto B = q_input.size(0);
        auto L = q_input.size(1);
        auto HS = q_input.size(2);
        auto H = static_cast<size_t>(symbol2value["H"]);
        auto S = HS / H;
        node->redefineOutputMemory({{B, L, H * S}});
        update_outputs(node);

        // Q, K, V is ready, do attention
        // query         [B, H, L1, S]
        // present_key   [B, H, L0+L1, S]  stride of last dim maybe > 1
        // present_value [B, H, L0+L1, S]
        // attention_mask [B, 1, L1, L0 + L1]
        // output_emb    [B, L1, H*S]
        auto q = q_input.reshape({B, L, H, S}).permute({0, 2, 1, 3});
        auto k = k_input.reshape({B, L, H, S}).permute({0, 2, 1, 3});
        auto v = v_input.reshape({B, L, H, S}).permute({0, 2, 1, 3});
        kernel(q, k, v, {}, {}, output_emb, 1.0f);
    }
};

template <KernelTypes KType, typename RT>
struct whisper_dec_self_attn_executor : public vnode_executor {
    PlainTensor<RT> q_input;       // "f32[?,1500,384]"  B, L, H*S
    PlainTensor<RT> k_input;       // "f32[?,1500,384]"  B, L, H*S
    PlainTensor<RT> v_input;       // "f32[?,1500,384]"  B, L, H*S
    PlainTensor<float> attn_mask;  // "f32[?,1,?,?]"     B, 1, q_len, kv_len
    PlainTensor<int32_t> shape1;   // "i32[3]"

    PlainTensor<RT> output_emb;     // "f32[?,?,384]"
    PlainTensor<RT> present_key;    // "f32[?,6,?,64]" [B, H, L, S]  transposed from k_input
    PlainTensor<RT> present_value;  // "f32[?,6,?,64]" [B, H, L, S]  transposed from v_input

    MHA_kernel<KType, RT> kernel;

    whisper_dec_self_attn_executor() {
        register_inputs(q_input, k_input, v_input, attn_mask, shape1);
        register_outputs(output_emb, present_key, present_value);
    }

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);
        auto B = q_input.size(0);
        auto L = q_input.size(1);
        auto H = static_cast<size_t>(symbol2value["H"]);
        auto S = static_cast<size_t>(symbol2value["S"]);
        q_input.assert_dims({B, L, H * S});
        k_input.assert_dims({B, L, H * S});
        v_input.assert_dims({B, L, H * S});
        attn_mask.assert_dims({B, 1, L, L});
        node->redefineOutputMemory({{B, L, H * S}, {B, H, L, S}, {B, H, L, S}});
        update_outputs(node);

        parallel_for3d(B, H, L, [&](size_t b, size_t h, size_t l) {
            memcpy(&present_key.at({b, h, l, 0}), &k_input.at({b, l, h*S}), sizeof(RT) * S);
            memcpy(&present_value.at({b, h, l, 0}), &v_input.at({b, l, h*S}), sizeof(RT) * S);
        });

        // Q, K, V is ready, do attention
        // query         [B, H, L1, S]
        // present_key   [B, H, L0+L1, S]  stride of last dim maybe > 1
        // present_value [B, H, L0+L1, S]
        // attention_mask [B, 1, L1, L0 + L1]
        // output_emb    [B, L1, H*S]
        auto q = q_input.reshape({B, L, H, S}).permute({0, 2, 1, 3});
        kernel(q, present_key, present_value, {}, attn_mask, output_emb, 1.0f);
    }
};

template <KernelTypes KType, typename RT>
struct whisper_dec_enc_attn_executor : public vnode_executor {
    PlainTensor<RT> q_input;       // "f32[?,1,384]"     [B, L1, H*S]
    PlainTensor<RT> k_input;       // "f32[?,1500,384]"  [B, L0, H*S]
    PlainTensor<RT> v_input;       // "f32[?,1500,384]"  [B, L0, H*S]
    PlainTensor<int32_t> shape1;   // "i32[3]"

    PlainTensor<RT> output_emb;     // "f32[?,?,384]"  [B, L, H*S]
    PlainTensor<RT> present_key;    // "f32[?,6,?,64]" [B, H, L, S]  transposed from k_input
    PlainTensor<RT> present_value;  // "f32[?,6,?,64]" [B, H, L, S]  transposed from v_input

    MHA_kernel<KType, RT> kernel;

    whisper_dec_enc_attn_executor() {
        register_inputs(q_input, k_input, v_input, shape1);
        register_outputs(output_emb, present_key, present_value);
    }

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);
        auto B = q_input.size(0);
        auto L1 = q_input.size(1);
        auto L0 = k_input.size(1);
        auto H = static_cast<size_t>(symbol2value["H"]);
        auto S = static_cast<size_t>(symbol2value["S"]);

        q_input.assert_dims({B, L1, H * S});
        k_input.assert_dims({B, L0, H * S});
        v_input.assert_dims({B, L0, H * S});
        node->redefineOutputMemory({{B, L1, H * S}, {B, H, L0, S}, {B, H, L0, S}});
        update_outputs(node);

        parallel_for3d(B, H, L0, [&](size_t b, size_t h, size_t l) {
            memcpy(&present_key.at({b, h, l, 0}), &k_input.at({b, l, h*S}), sizeof(RT) * S);
            memcpy(&present_value.at({b, h, l, 0}), &v_input.at({b, l, h*S}), sizeof(RT) * S);
        });

        // Q, K, V is ready, do attention
        // query         [B, H, L1, S]
        // present_key   [B, H, L0+L1, S]  stride of last dim maybe > 1
        // present_value [B, H, L0+L1, S]
        // attention_mask [B, 1, L1, L0 + L1]
        // output_emb    [B, L1, H*S]
        auto q = q_input.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
        kernel(q, present_key, present_value, {}, {}, output_emb, 1.0f);
    }
};

template <KernelTypes KType, typename RT>
struct whisper_dec2_self_attn_executor : public vnode_executor {
    PlainTensor<RT> q_input;       // "f32[?,..448,384]"  [B,L,H*S]
    PlainTensor<RT> k_input;       // "f32[?,..448,384]"  [B,L,H*S]
    PlainTensor<RT> v_input;       // "f32[?,..448,384]"  [B,L,H*S]
    PlainTensor<RT> past_decoder_key;   // "f32[?,6,?,64]" [B,H,L,S]
    PlainTensor<RT> past_decoder_value; // "f32[?,6,?,64]" [B,H,L,S]
    PlainTensor<int32_t> shape1;   // "i32[3]"

    PlainTensor<RT> output_emb;     // "f32[?,..448,?]" [B,L,H*S]
    PlainTensor<RT> present_key;    // "f32[?,6,?,64]"  [B,H,L,S]
    PlainTensor<RT> present_value;  // "f32[?,6,?,64]"  [B,H,L,S]

    MHA_kernel<KType, RT> kernel;

    whisper_dec2_self_attn_executor() {
        register_inputs(q_input, k_input, v_input, past_decoder_key, past_decoder_value, shape1);
        register_outputs(output_emb, present_key, present_value);
    }

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);
        auto B = q_input.size(0);
        auto L1 = q_input.size(1);
        auto H = static_cast<size_t>(symbol2value["H"]);
        auto S = static_cast<size_t>(symbol2value["S"]);
        auto L0 = past_decoder_key.size(2);
        q_input.assert_dims({B, L1, H * S});
        k_input.assert_dims({B, L1, H * S});
        v_input.assert_dims({B, L1, H * S});
        past_decoder_key.assert_dims({B, H, L0, S});
        past_decoder_value.assert_dims({B, H, L0, S});
        node->redefineOutputMemory({{B, L1, H * S}, {B, H, L0 + L1, S}, {B, H, L0 + L1, S}});
        update_outputs(node);

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            memcpy(&present_key.at({b, h, 0, 0}), &past_decoder_key.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
            memcpy(&present_value.at({b, h, 0, 0}), &past_decoder_value.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
        });

        parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t l) {
            memcpy(&present_key.at({b, h, L0 + l, 0}), &k_input.at({b, l, h*S}), sizeof(RT) * S);
            memcpy(&present_value.at({b, h, L0 + l, 0}), &v_input.at({b, l, h*S}), sizeof(RT) * S);
        });

        // Q, K, V is ready, do attention
        // query         [B, H, L1, S]
        // present_key   [B, H, L0+L1, S]  stride of last dim maybe > 1
        // present_value [B, H, L0+L1, S]
        // attention_mask [B, 1, L1, L0 + L1]
        // output_emb    [B, L1, H*S]
        auto q = q_input.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
        kernel(q, present_key, present_value, {}, {}, output_emb, 1.0f);
    }
};


template <KernelTypes KType, typename RT>
struct whisper_dec2_enc_attn_executor : public vnode_executor {
    PlainTensor<RT> q_input;            // "f32[?,..448,384]"     [B, L1, H*S]
    PlainTensor<RT> past_encoder_key;   // "f32[?,6,?,64]" [B,H,L0,S]
    PlainTensor<RT> past_encoder_value; // "f32[?,6,?,64]" [B,H,L0,S]
    PlainTensor<int32_t> shape1;        // "i32[3]"

    PlainTensor<RT> output_emb;         // "f32[?,?,384]"  [B, L1, H*S]

    MHA_kernel<KType, RT> kernel;

    whisper_dec2_enc_attn_executor() {
        register_inputs(q_input, past_encoder_key, past_encoder_value, shape1);
        register_outputs(output_emb);
    }

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);
        auto B = q_input.size(0);
        auto L1 = q_input.size(1);
        auto L0 = past_encoder_key.size(2);
        auto H = static_cast<size_t>(symbol2value["H"]);
        auto S = static_cast<size_t>(symbol2value["S"]);

        q_input.assert_dims({B, L1, H * S});
        past_encoder_key.assert_dims({B, H, L0, S});
        past_encoder_value.assert_dims({B, H, L0, S});
        node->redefineOutputMemory({{B, L1, H * S}});
        update_outputs(node);

        // Q, K, V is ready, do attention
        // query         [B, H, L1, S]
        // present_key   [B, H, L0+L1, S]  stride of last dim maybe > 1
        // present_value [B, H, L0+L1, S]
        // attention_mask [B, 1, L1, L0 + L1]
        // output_emb    [B, L1, H*S]
        auto q = q_input.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
        kernel(q, past_encoder_key, past_encoder_value, {}, {}, output_emb, 1.0f);
    }
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov