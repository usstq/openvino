// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "ie_parallel.hpp"
#include "gather_elements.h"
#include <ngraph/opsets/opset1.hpp>
#include <precision_utils.h>
#include <utils/general_utils.h>
#include "common/cpu_memcpy.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool GatherElements::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v6::GatherElements::get_type_info_static())) {
            errorMessage = "Node is not an instance of the GatherElements operation from operation set v6.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GatherElements::GatherElements(const std::shared_ptr<ngraph::Node>& op, GraphContext::Ptr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix_ = std::string("Layer GatherElements with name '") + op->get_friendly_name() + "'";

    if (inputShapes.size() != 2 || outputShapes.size() != 1)
        IE_THROW() << errorPrefix_ << " has invalid number of input/output edges.";

    const auto dataRank = getInputShapeAtPort(dataIndex_).getRank();
    const auto indicesRank = getInputShapeAtPort(indicesIndex_).getRank();
    if (dataRank != indicesRank)
        IE_THROW() << errorPrefix_ << " has invalid input shapes. Inputs 'Data' and 'Indices' must have equal ranks.";

    auto gatherElementsOp = ov::as_type_ptr<ov::op::v6::GatherElements>(op);
    auto axis = gatherElementsOp->get_axis();
    if (axis < 0)
        axis += dataRank;
    if (axis < 0 || axis >= static_cast<int>(dataRank))
        IE_THROW() << errorPrefix_ << " has invalid axis attribute: " << axis;
    axis_ = axis;
}

void GatherElements::prepareParams() {
    const auto& dataDims = getParentEdgesAtPort(dataIndex_)[0]->getMemory().getStaticDims();
    const auto& dstDims = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();
    strideAxDst_ = 1;
    for (int i = dstDims.size() - 1; i > axis_; i--)
        strideAxDst_ *= dstDims[i];
    dstAxDim_ = dstDims[axis_];
    if (axis_ > 0) {
        strideAx1Diff_ = 1;
        for (int i = dataDims.size() - 1; i >= axis_; i--)
            strideAx1Diff_ *= dataDims[i];
        strideAx1Diff_ -= strideAxDst_ * dstDims[axis_];
    }
}

void GatherElements::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inDataPrecision = getOriginalInputPrecisionAtPort(dataIndex_);
    if (!one_of(inDataPrecision.size(),
                sizeof(PrecisionTrait<Precision::I32>::value_type),
                sizeof(PrecisionTrait<Precision::I16>::value_type),
                sizeof(PrecisionTrait<Precision::I8>::value_type))) {
        IE_THROW() << errorPrefix_ << " has unsupported 'inputData' input precision: " << inDataPrecision;
    }

    Precision indicesPrecision = getOriginalInputPrecisionAtPort(indicesIndex_);
    if (!one_of(indicesPrecision, Precision::I32, Precision::I64)) {
        IE_THROW() << errorPrefix_ << " has unsupported 'indices' input precision: " << indicesPrecision;
    }

    dataTypeSize_ = inDataPrecision.size();

    addSupportedPrimDesc({{LayoutType::ncsp, inDataPrecision},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

void GatherElements::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <typename dataType>
void GatherElements::directExecution() {
    const auto *srcData = reinterpret_cast<const dataType *>(getParentEdgeAt(dataIndex_)->getMemoryPtr()->GetPtr());
    const auto *indices = reinterpret_cast<const int *>(getParentEdgeAt(indicesIndex_)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<dataType *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const int outSize = getChildEdgesAtPort(0)[0]->getMemory().GetShape().getElementsCount();
    auto threadBody = [&](const int ithr, const int nthr) {
        int start(0lu), end(0lu);
        splitter(outSize, nthr, ithr, start, end);
        if (start >= end)
            return;

        int axStrideIt = start % strideAxDst_;
        int dstAxIdx = (start / strideAxDst_) % dstAxDim_;
        int dstShift0 = (start / strideAxDst_ / dstAxDim_) * strideAx1Diff_;

        for (size_t o = start; o < end; o++, axStrideIt++) {
            if (axStrideIt == strideAxDst_) {
                axStrideIt = 0;
                dstAxIdx++;
                if (dstAxIdx == dstAxDim_) {
                    dstAxIdx = 0;
                    dstShift0 += strideAx1Diff_;
                }
            }
            dstData[o] = srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_];
        }
    };

    parallel_nt(0, threadBody);
}

void GatherElements::execute(dnnl::stream strm) {
    switch (dataTypeSize_) {
        case sizeof(PrecisionTrait<Precision::I32>::value_type):
            return directExecution<PrecisionTrait<Precision::I32>::value_type>();
        case sizeof(PrecisionTrait<Precision::I16>::value_type):
            return directExecution<PrecisionTrait<Precision::I16>::value_type>();
        case sizeof(PrecisionTrait<Precision::I8>::value_type):
            return directExecution<PrecisionTrait<Precision::I8>::value_type>();
        default:
            return IE_THROW() << "Unsupported data type size";
    }
}

bool GatherElements::created() const {
    return getType() == Type::GatherElements;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
