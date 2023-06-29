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
        m_executor = std::make_shared<gptneox_attention_executor>();
    } else {
        IE_THROW() << errorPrefix << " unsupported vnode type " << m_vtype;
    }
    std::vector<ov::intel_cpu::PortConfigurator> inPortConfigs;
    std::vector<ov::intel_cpu::PortConfigurator> outPortConfigs;
    for (auto * p : m_executor->inputs) {
        inPortConfigs.emplace_back(LayoutType::ncsp, p->get_precision());
    }
    for (auto * p : m_executor->outputs) {
        outPortConfigs.emplace_back(LayoutType::ncsp, p->get_precision());
    }
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void VNode::execute(dnnl::stream strm) {
    if (m_executor) {
        m_executor->exec(this, strm);
    } else {
        IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    }
}

void VNode::executeDynamicImpl(dnnl::stream strm) {
    if (m_executor) {
        m_executor->exec(this, strm);
    } else {
        IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    }
}

bool VNode::created() const {
    return getType() == Type::VNode;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov