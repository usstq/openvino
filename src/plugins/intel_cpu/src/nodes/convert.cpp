// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dnnl_extension_utils.h>
#include "convert.h"
#include "common/cpu_convert.h"
#include "common/blocked_desc_creator.h"
#include <ngraph/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>
#include <utils/ngraph_utils.hpp>

using namespace mkldnn;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Convert::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto convert = std::dynamic_pointer_cast<const ngraph::opset1::Convert>(op);
        if (!convert) {
            errorMessage = "Only opset1 Convert operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Convert::Convert(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Convert node with name '" + getName() + "'";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto convert = ov::as_type_ptr<const ngraph::opset1::Convert>(op);
    origPrc = details::convertPrecision(convert->get_destination_type());
}

std::vector<VectorDims> Convert::shapeInfer() const {
    return std::vector<VectorDims>{getParentEdgesAtPort(0)[0]->getMemory().getStaticDims()};
}

Convert::Convert(const Shape &shape, const InferenceEngine::Precision &inPrc, const InferenceEngine::Precision &outPrc,
                 const std::string &nodeName, const mkldnn::engine& eng, WeightsSharing::Ptr &cache)
        : Node("Convert", nodeName, eng, cache)
        , origPrc(outPrc) {
    inputShapes.push_back(shape);
    addOriginalInputPrecision(inPrc);
    outputShapes.push_back(shape);
    addOriginalOutputPrecision(outPrc);

    isDynamic = shape.isDynamic();

    errorPrefix = "Convert node with name '" + getName() + "'";
}

void Convert::getSupportedDescriptors() {
    // if tensor descriptors are set via setDescs method we need to update the inDims/outDims data
    // from correspond tensor descriptors.
    if (outputShapes.empty())
        outputShapes.push_back(output->getShape());
    if (inputShapes.empty())
        inputShapes.push_back(input->getShape());
    if (getParentEdges().size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has incorrect number of output edges";
}

bool Convert::isSupportedDesc(const MemoryDesc &desc) {
    bool isSupported = desc.getType() & MemoryDescType::Blocked;
    if (desc.getType() == MemoryDescType::DnnlBlocked)
        isSupported &= desc.as<const DnnlMemoryDesc>()->hasEmptyExtraData();
    return isSupported;
}

void Convert::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    NodeConfig config;
    PortConfig dataIn;
    PortConfig dataConfigOut;

    config.dynBatchSupport = false;

    bool canInitExternalDesc = false;
    if (input && output) {
        canInitExternalDesc = true;
        canInitExternalDesc &= isSupportedDesc(*input);
        canInitExternalDesc &= isSupportedDesc(*output);
    }

    // if input and output pointers are not null and not contain extra data, then the inp/output tensor descriptors were set using setDescs method, so
    // they should be used as the actual descriptors.
    if (canInitExternalDesc) {
        dataIn.setMemDesc(input);
        config.inConfs.push_back(dataIn);

        // inp/out layouts must be the same
        dataConfigOut.setMemDesc(config.inConfs[0].getMemDesc());
        dataConfigOut.setMemDesc(dataConfigOut.getMemDesc()->cloneWithNewPrecision(output->getPrecision()));
        config.outConfs.push_back(dataConfigOut);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    } else if (inputShapes.size() == 1 && outputShapes.size() == 1) {
        const Shape& insShape = getInputShapeAtPort(0);
        auto insPrecision = getOriginalInputPrecisionAtPort(0);
        const Shape& outputShape = getOutputShapeAtPort(0);
        auto outPrecision = getOriginalOutputPrecisionAtPort(0);

        config.inConfs.push_back(dataIn);
        config.outConfs.push_back(dataConfigOut);

        auto creators = BlockedDescCreator::getCommonCreators();
        auto range = BlockedDescCreator::makeFilteredRange(creators, insShape.getRank());

        for (auto itr = range.first; itr != range.second; ++itr) {
            config.inConfs[0].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(insPrecision, insShape)));
            config.outConfs[0].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(outPrecision, outputShape)));

            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        }
    } else {
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges";
    }
}

void Convert::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

void Convert::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();
    auto& childMem = getChildEdgeAt(0)->getMemory();

    const auto parentPaddElemCount = parentMem.GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    const auto childPaddElemCount = childMem.GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();

    if (parentPaddElemCount != childPaddElemCount)
        IE_THROW() << errorPrefix << " has different elements number in input and output buffers";

    void* srcPtr = parentMem.GetPtr();
    void* dstPtr = childMem.GetPtr();

    cpu_convert(srcPtr,
                dstPtr,
                parentMem.getDesc().getPrecision(),
                origPrc,
                childMem.getDesc().getPrecision(),
                parentPaddElemCount);
}

bool Convert::created() const {
    return getType() == Type::Convert;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
