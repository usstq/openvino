// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rotary_pe.h"

#include <ie_ngraph_utils.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "ie_parallel.hpp"
#include "rope_impl.hpp"
#include "utils/plain_tensor.hpp"
#include "utils/profiler.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool RPE::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    const auto RPE = std::dynamic_pointer_cast<const ov::op::internal::RPE>(op);
    if (!RPE) {
        errorMessage = "Only RPE operation is supported";
        return false;
    }

    return true;
}

RPE::RPE(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "RPE layer with name '" + getName() + "'";

    m_rpe = std::dynamic_pointer_cast<ov::op::internal::RPE>(op);
    // m_config = m_rpe->get_config();
}

void RPE::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != m_rpe->get_input_size())
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().size() < m_rpe->get_output_size())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

// default implementation: reference
template <typename T>
struct RoPE_kernel : public RPE::ExecutorBase {
    RoPE_kernel() {
        std::cout << "RoPE_kernel" << std::endl;
    }

    void exec(RPE* node) override {
        PlainTensor<float> input;
        PlainTensor<float> rotary_emb_sin;
        PlainTensor<float> rotary_emb_cos;
        PlainTensor<float> output;
        int idx = 0;
        input.reset(node->getParentEdgeAt(idx++)->getMemoryPtr());
        rotary_emb_sin.reset(node->getParentEdgeAt(idx++)->getMemoryPtr());
        rotary_emb_cos.reset(node->getParentEdgeAt(idx++)->getMemoryPtr());

        if (false) {
            std::cout << "input = " << input << std::endl;
            std::cout << "sin = " << rotary_emb_sin << std::endl;
            std::cout << "cos = " << rotary_emb_cos << std::endl;
        }

        auto B = input.size(0);
        auto H = input.size(1);
        auto L = input.size(2);
        auto S = input.size(3);
        node->redefineOutputMemory({{B, H, L, S}});
        output.reset(node->getChildEdgeAt(0)->getMemoryPtr());

        auto rotary_dims = 0;

        if (rotary_dims == 0)
            rotary_dims = S;
        auto half_rotary_dims = rotary_dims / 2;

        assert(rotary_emb_cos.size(0) == 1);
        assert(rotary_emb_cos.size(1) == 1);
        assert(rotary_emb_sin.size(0) == 1);
        assert(rotary_emb_sin.size(1) == 1);

        parallel_for3d(B, H, L, [&](size_t b, size_t h, size_t p) {
            auto* src = &input.at({b, h, p, 0});
            auto* dst = &output.at({b, h, p, 0});
            // q_embed = (q * cos) + (rotate_half(q) * sin)
            // k_embed = (k * cos) + (rotate_half(k) * sin)
            auto* cos = &rotary_emb_cos({0, 0, p, 0});  //
            auto* sin = &rotary_emb_sin({0, 0, p, 0});  //

            InferenceEngine::Extensions::Cpu::XARCH::rope_impl(src, sin, cos, dst, rotary_dims);
        });
    }
};

void RPE::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // orginal precision at input port 0 as a hint of runtime precisions
    m_runtime_precision = getOriginalInputPrecisionAtPort(0);

    m_executor = std::make_shared<RoPE_kernel<float>>();

    std::vector<ov::intel_cpu::PortConfigurator> inPortConfigs;
    std::vector<ov::intel_cpu::PortConfigurator> outPortConfigs;

    inPortConfigs.emplace_back(LayoutType::ncsp, m_runtime_precision);  // data
    inPortConfigs.emplace_back(LayoutType::ncsp, m_runtime_precision);  // sin
    inPortConfigs.emplace_back(LayoutType::ncsp, m_runtime_precision);  // cos

    outPortConfigs.emplace_back(LayoutType::ncsp, m_runtime_precision);  // data(RoPE-ed)

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void RPE::execute(dnnl::stream strm) {
    m_executor->exec(this);
}

void RPE::executeDynamicImpl(dnnl::stream strm) {
    m_executor->exec(this);
}

bool RPE::created() const {
    return getType() == Type::RPE;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov