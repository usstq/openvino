// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnode.h"

#include <ngraph/opsets/opset1.hpp>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>

#include "transformations/cpu_opset/common/op/vnode.hpp"

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

    std::vector<ov::intel_cpu::PortConfigurator> inPortConfigs;
    std::vector<ov::intel_cpu::PortConfigurator> outPortConfigs;
    for (int i = 0; i < m_vnode->get_input_size(); i++) {
        inPortConfigs.emplace_back(LayoutType::ncsp, getOriginalInputPrecisionAtPort(i));
    }
    for (int i = 0; i < m_vnode->get_output_size(); i++) {
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

void VNode::gptneox_attention(dnnl::stream strm, bool redefine_outputs) {
    IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    int ii = 0;
    auto input_ids = getParentEdgeAt(ii++)->getMemoryPtr();      // i32[B, L0]
    auto attention_mask = getParentEdgeAt(ii++)->getMemoryPtr(); // i32[B, L0]
    auto past_key = getParentEdgeAt(ii++)->getMemoryPtr();      // f32[B, H, L0, S]
    auto past_value = getParentEdgeAt(ii++)->getMemoryPtr();    // f32[B, H, L0, S]
    auto past_0_key = getParentEdgeAt(ii++)->getMemoryPtr();    // f32[B, H, L0, S]
    auto qkv_input = getParentEdgeAt(ii++)->getMemoryPtr();     // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
    auto attention_bias = getParentEdgeAt(ii++)->getMemoryPtr();// u8[1,1,2048,2048]
    auto rotary_emb_cos = getParentEdgeAt(ii++)->getMemoryPtr();// f32[1,1,2048,16]
    auto rotary_emb_sin = getParentEdgeAt(ii++)->getMemoryPtr();// f32[1,1,2048,16]

    auto dims_input_ids = input_ids->getStaticDims();
    auto dims_past = past_key->getStaticDims();
    auto dims_cur = qkv_input->getStaticDims();
    auto B = dims_past[0];
    auto H = dims_past[1]; // 8
    auto L0 = dims_past[2]; // number of tokens to be encoded
    auto S = dims_past[3]; // 64
    auto L1 = dims_cur[1];
    assert(dims_input_ids[0] == B);
    assert(dims_input_ids[1] == L0);
    assert(dims_cur[0] == B);
    assert(dims_cur[2] == H*S*3);

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
    auto hidden = getChildEdgeAt(oo++)->getMemoryPtr();
    auto present_key = getChildEdgeAt(oo++)->getMemoryPtr();
    auto present_value = getChildEdgeAt(oo++)->getMemoryPtr();
    // IE_THROW() << errorPrefix << "has inconsistent input shape and output size";

    // rotary embedding on q/k of qkv_input
    //   position id is range: [L0, L0+L1)
    // concat (past_key, cur_key)
    // matmul W=Q*K'
    // apply causal_mask
    // apply attention mask
    // softmax
    // concat (past_value, cur_value)
    // matmul W*V
    //
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov