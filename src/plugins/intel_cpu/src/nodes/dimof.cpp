// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dimof.h"

#include <ngraph/opsets/opset1.hpp>
#include <utils/shape_inference/shape_inference_cpu.hpp>

#include "transformations/cpu_opset/common/op/dimof.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

namespace {
/**
 * Implements Shape Of shape inference algorithm. The output shape is simply a 1D tensor with the size of the input
 * tensor rank.
 *
 */
class DimOfShapeInfer : public ShapeInferEmptyPads {
public:
    DimOfShapeInfer() = default;
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        IE_ASSERT(!input_shapes.empty());
        return {{VectorDims{1}}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class DimOfShapeInferFactory : public ShapeInferFactory {
public:
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<DimOfShapeInfer>();
    }
};
}  // namespace

bool DimOf::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    const auto DimOf = std::dynamic_pointer_cast<const DimOfNode>(op);
    if (!DimOf) {
        errorMessage = "Only DimOfNode operation is supported";
        return false;
    }

    return true;
}

DimOf::DimOf(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, DimOfShapeInferFactory()) {
    const auto DimOf = std::dynamic_pointer_cast<const DimOfNode>(op);
    if (!DimOf) {
        IE_THROW(NotImplemented) << "Only DimOfNode operation is supported";
        return;
    }

    errorPrefix = "DimOf layer with name '" + getName() + "' ";
    if (DimOf->get_input_partial_shape(0).size() == 0)
        IE_THROW() << errorPrefix << "gets unsupported input 0D tensor (scalar)";

    m_axis = DimOf->get_axis();
}

void DimOf::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    if (getParentEdges().size() != 1)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

void DimOf::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision precision = getOriginalInputPrecisionAtPort(0);

    const LayoutType dataFormats[4] = {LayoutType::ncsp, LayoutType::nspc, LayoutType::nCsp16c, LayoutType::nCsp8c};
    for (const auto& df : dataFormats) {
        addSupportedPrimDesc({{df, precision}}, {{LayoutType::ncsp, Precision::I32}}, impl_desc_type::ref);
    }
}

bool DimOf::isExecutable() const {
    return true;
}

void DimOf::execute(dnnl::stream strm) {
    auto inPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto outPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto inDims = inPtr->getStaticDims();
    size_t dimsCount = inDims.size();
    // output shape must be [1]
    if (outPtr->getStaticDims().size() != 1 || 1 != outPtr->getStaticDims()[0])
        IE_THROW() << errorPrefix << "has inconsistent input shape and output size";

    auto* dst = reinterpret_cast<int*>(getChildEdgeAt(0)->getMemoryPtr()->getData());

    auto axis = m_axis;
    if (axis < 0)
        axis += dimsCount;

    if (axis < 0 || axis >= dimsCount)
        IE_THROW() << errorPrefix << "axis is out of range";

    dst[0] = inDims[axis];
}

bool DimOf::created() const {
    return getType() == Type::DimOf;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
