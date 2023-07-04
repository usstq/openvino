
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

    virtual void exec(Node* node, dnnl::stream strm) = 0;
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

    gpt2_attention_executor() {
        register_inputs(qkv_input, past_key, past_value, attention_mask);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<RT> query;

    void exec(Node* node, dnnl::stream strm) override {
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

        DEBUG_LOG(" B=", B, " H=", H, " S=", S, " L0=", L0, " L1=", L1);

        // auto query = qkv_input.slice(2, 0, H * S).reshape({B, L1, H, S}).permute({0, 2, 1, 3});
        // auto query = qkv_input.reshape({B, L1, 3, H, S}).index({{}, {}, {0}, {}, {}}).permute({0, 2, 1, 3});
        // std::cout << "query=" << query << std::endl; asm("int3");
        // concat pask_key/value & k/v into present_key/value
        query.resize({B, H, L1, S});
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(RT) * L0 * S);

                for (size_t p = 0; p < L1; p++) {
                    auto* q = &qkv_input.at({b, p, h * S});
                    auto* k = &qkv_input.at({b, p, (H + h) * S});
                    auto* v = &qkv_input.at({b, p, (2 * H + h) * S});
                    memcpy(&query.at({b, h, p, 0}), q, sizeof(RT) * S);
                    memcpy(&present_key.at({b, h, L0 + p, 0}), k, sizeof(RT) * S);
                    memcpy(&present_value.at({b, h, L0 + p, 0}), v, sizeof(RT) * S);
                }
            }
        }

        kernel(query, present_key, present_value, attention_mask, output_emb);
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

    void exec(Node* node, dnnl::stream strm) override {
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

        rope_kernel(qkv_input,
                    past_key,
                    past_value,
                    m_query_emb,
                    present_key,
                    present_value,
                    rotary_emb_cos,
                    rotary_emb_sin);

        DEBUG_LOG(" m_query_emb=", m_query_emb);

        kernel(m_query_emb, present_key, present_value, attention_mask, output_emb);
    }
};

#if 0
template <KernelTypes KType, typename RT>
struct open_llama_attention_executor : public vnode_executor {
    PlainTensor<RT> q_proj;          //  "f32[B,L1,H*S]"
    PlainTensor<RT> k_proj;          //  "f32[B,L1,H*S]"
    PlainTensor<RT> v_proj;          //  "f32[B,L1,H*S]"
    PlainTensor<int32_t> self_attn_Shape;
    PlainTensor<RT> past_key;           // f32[B, 32, L0, 100]
    PlainTensor<RT> past_value;         // f32[B, H, L0, S]
    PlainTensor<RT> past_0_key;         // f32[B, H, L0, S]
    PlainTensor<float> attention_mask;  // f32[B, 1, 1, L0 + L1]
    PlainTensor<float> rotary_emb_cos;  // f32[1,1,2048,100]
    PlainTensor<float> rotary_emb_sin;  // f32[1,1,2048,100]
    PlainTensor<int32_t> input_ids;     // i32[B, L1]

    PlainTensor<RT> output_emb;     // f32[B, L1, 512]
    PlainTensor<RT> present_key;    // f32[B, H, L0+L1, 64]
    PlainTensor<RT> present_value;  // f32[B, H, L0+L1, 64]

    MHA_kernel<KType, RT> kernel;
    RoPE_kernel<KType, RT> rope_kernel;

    open_llama_attention_executor() {
        register_inputs(q_proj, k_proj, v_proj, self_attn_Shape,
                        past_key,
                        past_value,
                        past_0_key, input_ids,
                        attention_mask,
                        rotary_emb_cos,
                        rotary_emb_sin,
                        input_ids);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<RT> m_query_emb;  // query with embed

    void exec(Node* node, dnnl::stream strm) override {
        update_inputs(node);

        auto& dims_past = past_key.get_dims();
        auto& dims_cur = qkv_input.get_dims();
        auto B = q_proj.get_dims()[0];
        auto H = q_proj.get_dims()[1];   // 8
        auto L0 = past_key.size(2);  // number of tokens to be encoded
        auto S = past_key.size(3);   // 64
        auto L1 = dims_cur[1];
        auto max_position_embeddings = rotary_emb_cos.get_dims()[2];
        auto rotary_dims = rotary_emb_cos.get_dims()[3];

        std::vector<VectorDims> outputShapes;
        outputShapes.push_back(VectorDims{B, L1, H * S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        node->redefineOutputMemory(outputShapes);

        update_outputs(node);

        q_proj.assert_dims({B, L1, H * S});
        k_proj.assert_dims({B, L1, H * S});
        v_proj.assert_dims({B, L1, H * S});
        attention_mask.assert_dims({B, 1, 1, L0 + L1});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});

        qkv_input.assert_dims({B, L1, H * 3 * S});
        rotary_emb_cos.assert_dims({1, 1, max_position_embeddings, rotary_dims});
        rotary_emb_sin.assert_dims({1, 1, max_position_embeddings, rotary_dims});

        DEBUG_LOG(" B=", B, " H=", H, " S=", S, " L0=", L0, " L1=", L1);

        m_query_emb.resize({B, H, L1, S});

        rope_kernel(qkv_input,
                    past_key,
                    past_value,
                    m_query_emb,
                    present_key,
                    present_value,
                    rotary_emb_cos,
                    rotary_emb_sin);

        DEBUG_LOG(" m_query_emb=", m_query_emb);

        kernel(m_query_emb, present_key, present_value, attention_mask, output_emb);
    }
};
#endif

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov