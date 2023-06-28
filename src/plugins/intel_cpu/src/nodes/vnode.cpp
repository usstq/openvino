// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnode.h"

#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "ie_parallel.hpp"
#include "transformations/cpu_opset/common/op/vnode.hpp"

//====================used by llmdnn==========================
namespace ov {
namespace cpu {

size_t getTotalThreads() {
    return parallel_get_max_threads();
}

void TrySimpleParallelFor(const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
    parallel_for(total, fn);
}

};  // namespace cpu
};  // namespace ov
//====================used by llmdnn==========================

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool VNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    const auto vnode = std::dynamic_pointer_cast<const ov::intel_cpu::VNode>(op);
    if (!vnode) {
        errorMessage = "Only VNode operation is supported";
        return false;
    }

    auto vtype = vnode->get_vtype();
    if (vtype != "gptneox_attention") {
        errorMessage = vtype + " is not supported!";
        return false;
    }

    return true;
}

VNode::VNode(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "VNode layer with name '" + getName() + "'";

    m_vnode = std::dynamic_pointer_cast<ov::intel_cpu::VNode>(op);
    m_vtype = m_vnode->get_vtype();
    m_kernel_initialized = false;
}

void VNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != m_vnode->get_input_size())
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().size() != m_vnode->get_output_size())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

void VNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

#if 0
    PlainTensor<int32_t> input_ids(getParentEdgeAt(ii++)->getMemoryPtr());       // i32[B, L1]
    PlainTensor<float> attention_mask(getParentEdgeAt(ii++)->getMemoryPtr());  // f32[B, 1, 1, L0 + L1]
    PlainTensor<bfloat16> past_key(getParentEdgeAt(ii++)->getMemoryPtr());        // f32[B, H, L0, S]
    PlainTensor<bfloat16> past_value(getParentEdgeAt(ii++)->getMemoryPtr());      // f32[B, H, L0, S]
    PlainTensor<bfloat16> past_0_key(getParentEdgeAt(ii++)->getMemoryPtr());      // f32[B, H, L0, S]
    PlainTensor<bfloat16> qkv_input(getParentEdgeAt(ii++)->getMemoryPtr());       // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
    PlainTensor<float> attention_bias(getParentEdgeAt(ii++)->getMemoryPtr());  // u8[1,1,2048,2048]
    PlainTensor<float> rotary_emb_cos(getParentEdgeAt(ii++)->getMemoryPtr());  // f32[1,1,2048,16]
    PlainTensor<float> rotary_emb_sin(getParentEdgeAt(ii++)->getMemoryPtr());  // f32[1,1,2048,16]
#endif
    if (m_vtype == "gptneox_attention") {
        // so far we only have BF16 kernels:
        std::vector<ov::intel_cpu::PortConfigurator> inPortConfigs;
        std::vector<ov::intel_cpu::PortConfigurator> outPortConfigs;
        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::I32);  // input_ids
        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::FP32); // attention_mask

        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::BF16); // past_key
        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::BF16); // past_value
        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::BF16); // past_0_key

        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::BF16); // qkv_input
        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::U8); // attention_bias
        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::FP32); // rotary_emb_cos
        inPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::FP32); // rotary_emb_sin

        outPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::BF16); // nv_output
        outPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::BF16); // present_key
        outPortConfigs.emplace_back(LayoutType::ncsp, InferenceEngine::Precision::BF16); // present_value
        addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
        return;
    }

    std::vector<ov::intel_cpu::PortConfigurator> inPortConfigs;
    std::vector<ov::intel_cpu::PortConfigurator> outPortConfigs;
    for (size_t i = 0; i < m_vnode->get_input_size(); i++) {
        inPortConfigs.emplace_back(LayoutType::ncsp, getOriginalInputPrecisionAtPort(i));
    }
    for (size_t i = 0; i < m_vnode->get_output_size(); i++) {
        outPortConfigs.emplace_back(LayoutType::ncsp, getOriginalOutputPrecisionAtPort(i));
    }
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void VNode::execute(dnnl::stream strm) {
    if (m_vtype == "gptneox_attention") {
        gptneox_attention(strm, false);
    } else {
        IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    }
}

void VNode::executeDynamicImpl(dnnl::stream strm) {
    if (m_vtype == "gptneox_attention") {
        gptneox_attention(strm, true);
    } else {
        IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    }
}

bool VNode::created() const {
    return getType() == Type::VNode;
}

// low-runtime cost accessor class

void VNode::gptneox_attention(dnnl::stream strm, bool redefine_outputs) {
    int ii = 0;
    PlainTensor<int32_t> input_ids(getParentEdgeAt(ii++)->getMemoryPtr());     // i32[B, L1]
    PlainTensor<float> attention_mask(getParentEdgeAt(ii++)->getMemoryPtr());  // f32[B, 1, 1, L0 + L1]
    PlainTensor<bfloat16> past_key(getParentEdgeAt(ii++)->getMemoryPtr());     // f32[B, H, L0, S]
    PlainTensor<bfloat16> past_value(getParentEdgeAt(ii++)->getMemoryPtr());   // f32[B, H, L0, S]
    PlainTensor<bfloat16> past_0_key(getParentEdgeAt(ii++)->getMemoryPtr());   // f32[B, H, L0, S]
    PlainTensor<bfloat16> qkv_input(getParentEdgeAt(ii++)->getMemoryPtr());    // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
    PlainTensor<uint8_t> attention_bias(getParentEdgeAt(ii++)->getMemoryPtr());// u8[1,1,2048,2048]
    PlainTensor<float> rotary_emb_cos(getParentEdgeAt(ii++)->getMemoryPtr());  // f32[1,1,2048,16]
    PlainTensor<float> rotary_emb_sin(getParentEdgeAt(ii++)->getMemoryPtr());  // f32[1,1,2048,16]

    auto& dims_past = past_key.get_dims();
    auto& dims_cur = qkv_input.get_dims();
    auto B = dims_past[0];
    auto H = dims_past[1];   // 8
    auto L0 = dims_past[2];  // number of tokens to be encoded
    auto S = dims_past[3];   // 64
    auto L1 = dims_cur[1];

    input_ids.assert_dims({B, L1});
    attention_mask.assert_dims({B, 1, 1, L0 + L1});
    past_key.assert_dims({B, H, L0, S});
    past_value.assert_dims({B, H, L0, S});
    past_0_key.assert_dims({B, H, L0, S});
    qkv_input.assert_dims({B, L1, H * 3 * S});
    attention_bias.assert_dims({1, 1, 2048, 2048});
    rotary_emb_cos.assert_dims({1, 1, 2048, 16});
    rotary_emb_sin.assert_dims({1, 1, 2048, 16});

    if (redefine_outputs) {
        std::vector<VectorDims> outputShapes;
        // shape infer
        // f32[B,H,L0,64]    past_key/past_value
        // f32[B,L1,1536]    qkv_input

        // f32[B,L1,512]     hidden
        // f32[B,H,L0+L1,64] present_key/present_value
        outputShapes.push_back(VectorDims{B, L1, 512});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        outputShapes.push_back(VectorDims{B, H, L0 + L1, S});
        Node::redefineOutputMemory(outputShapes);
    }

    int oo = 0;
    PlainTensor<ov::bfloat16> output_emb(getChildEdgeAt(oo++)->getMemoryPtr());     // f32[B, L1, 512]
    PlainTensor<ov::bfloat16> present_key(getChildEdgeAt(oo++)->getMemoryPtr());    // f32[B, H, L0+L1, 64]
    PlainTensor<ov::bfloat16> present_value(getChildEdgeAt(oo++)->getMemoryPtr());  // f32[B, H, L0+L1, 64]
    // IE_THROW() << errorPrefix << "has inconsistent input shape and output size";

    // rotary embedding on q/k of qkv_input, this is done in-place
    //   position id is range: [L0, L0+L1)
    // concat (past_key, cur_key)
    // matmul W=Q*K'
    // apply causal_mask
    // apply attention mask
    // softmax
    // concat (past_value, cur_value)
    // matmul W*V
    //

    if (!m_kernel_initialized) {
        std::cout << "::::::::::::::;  " << __func__ << " " << m_vtype << std::endl;
        if (!m_kernel_emb.create(llmdnn::emb_gpt::create_param{
                .num_heads = H,
                .head_size = S,
                .head_size_aligned = S,  // better to aligned to 64 bytes for best performance, apply for qkv
                .max_seq_len = 2048,     // max seq length for computing the size of matmul tmp result
                .qkv_precision = llmdnn::data_type_t::dnnl_bf16,
                .dst_precision = llmdnn::data_type_t::dnnl_bf16,
                .rotary_emb_base = 10000,
                .rotary_pct = 0.25f,
                .use_position2d = false,
            })) {
            IE_THROW() << errorPrefix << " llmdnn::emb_gpt::create failed " << std::endl;
        }
        if (!m_kernel_mha.create(llmdnn::mha_gpt::create_param{
                .num_heads = H,
                .head_size = S,
                .head_size_aligned = S,  // better to aligned to 64 bytes for best performance, apply for qkv
                .max_seq_len = 2048,     // max seq length for computing the size of matmul tmp result
                .normal_factor = 0.125f,
                .qkv_precision = llmdnn::data_type_t::dnnl_bf16,
                .dst_precision = llmdnn::data_type_t::dnnl_bf16,
            })) {
            IE_THROW() << errorPrefix << " llmdnn::mha_gpt::create failed " << std::endl;
        }
        m_kernel_initialized = true;
    }

    m_query_emb.resize<ov::bfloat16>({B, H, L1, S});

    auto batched_ptrs_key = present_key.get_batched_ptrs<ov::bfloat16>();
    auto batched_ptrs_value = present_value.get_batched_ptrs<ov::bfloat16>();

    m_kernel_emb.exec(llmdnn::emb_gpt::exec_param{
        .batch = B,
        .query_seq_len = L1,
        .past_seq_len = L0,
        .qkv = qkv_input.data<uint8_t>(),          // shape: [batch, query_seq_len, 3 * hidden size]   [B, L1, H*3*S]
        .query_dst = m_query_emb.data<uint8_t>(),  // rotary embbeding dst
        .layer_past_key_src = past_key.get_batched_ptrs<ov::bfloat16>(),      // past key src
        .layer_past_value_src = past_value.get_batched_ptrs<ov::bfloat16>(),  // past value src
        .layer_past_key_dst = batched_ptrs_key,      // past key dst, if layer_past_key_src!=layer_past_key_dst, will
                                                     // copy layer_past_key_src to layer_past_key_dst
        .layer_past_value_dst = batched_ptrs_value,  // past value dst, if layer_past_value!=layer_past_value_dst,
                                                     // will copy layer_past_value to layer_past_value_dst
        .position2d_ids = nullptr,                   // shape: [batch, 2, query_seq_len]
        .head_stride_in_kv = (L0 + L1) * S           // kv stride for next head; kv may be preallocated a big buffer
    });

    m_kernel_mha.exec(llmdnn::mha_gpt::exec_param{
        .batch = B,
        .query_seq_len = L1,
        .key_seq_len = L0 + L1,
        .is_causal_in_attention = false,   // causal mask is fused in attention mask: chatglm uses it.
        .q = m_query_emb.data<uint8_t>(),  // q buffer, compact, shape: [batch, num_heads, query_seq_len, head_size]
        .k = batched_ptrs_key,             // k buffer, k[N] stands different batch which may be discreted
                                           //      k[0] shape: [batch, num_heads, key_seq_len, head_size]
        .v = batched_ptrs_value,           // v buffer, v[N] stands different batch which may be discreted
                                           //      v[0] shape: [batch, num_heads, value_seq_len, head_size]
        .attention_mask =
            attention_mask.data<float>(),  // attention mask, attention_mask[0] shape:
                                           //      [batch, 1, 1, key_seq_len], when is_causal_in_attention is false
        //      [batch, 1, query_seq_len, key_seq_len], when is_causal_in_attention is true
        .attn_output =
            output_emb.data<uint8_t>(),  // output, compact, shape: [batch, query_seq_len, num_heads * head_size]
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

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov