// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_transpose_node.h"
#include "ie_parallel.hpp"

#include <algorithm>
#include <string>
#include "mkldnn_extension_utils.h"
#include <common/primitive_hashing_utils.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace {
struct TransposeKey {
    PermuteParams params;

    size_t hash() const;
    bool operator==(const TransposeKey& rhs) const;
};

size_t TransposeKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = get_vector_hash(seed, params.src_block_dims);
    seed = get_vector_hash(seed, params.dst_block_dims);
    seed = get_vector_hash(seed, params.src_block_order);
    seed = get_vector_hash(seed, params.dst_block_order);
    seed = get_vector_hash(seed, params.order);
    seed = hash_combine(seed, params.data_size);
    return seed;
}

bool TransposeKey::operator==(const TransposeKey& rhs) const {
    return (params.src_block_dims == rhs.params.src_block_dims) &&
           (params.dst_block_dims == rhs.params.dst_block_dims) &&
           (params.src_block_order == rhs.params.src_block_order) &&
           (params.dst_block_order == rhs.params.dst_block_order) && (params.order == rhs.params.order) &&
           (params.data_size == rhs.params.data_size);
}

}  // namespace

bool MKLDNNTransposeNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v1::Transpose::get_type_info_static())) {
            errorMessage = "Node is not an instance of the Transpose operation from opset1.";
            return false;
        }

        if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO: Support parameterized Order input for dynamic shapes.
            errorMessage = "Constant expected as the second input for static shapes.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTransposeNode::MKLDNNTransposeNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() == ov::op::v0::Constant::get_type_info_static()) {
        isInputOrderConst = true;
        order = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(INPUT_ORDER_IDX))->cast_vector<size_t>();

        if (order.empty()) {
            size_t rank = getInputShapeAtPort(INPUT_DATA_IDX).getRank();
            for (size_t i = 1lu; i <= rank; ++i) {
                order.emplace_back(rank - i);
            }
        }
    }
}

void MKLDNNTransposeNode::getSupportedDescriptors() {
}

void MKLDNNTransposeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    prec = getOriginalInputPrecisionAtPort(0);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[INPUT_DATA_IDX].inPlace = -1;
    config.inConfs[INPUT_DATA_IDX].constant = false;
    config.inConfs[INPUT_ORDER_IDX].constant = isInputOrderConst;
    config.inConfs[INPUT_ORDER_IDX].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            Precision::I32, getInputShapeAtPort(INPUT_ORDER_IDX));
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    const auto& inputDataShape = getInputShapeAtPort(INPUT_DATA_IDX);
    const auto& outputDataShape = getOutputShapeAtPort(0);
    if (inputDataShape.getRank() == 4 || inputDataShape.getRank() == 5) {
        config.inConfs[0].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, inputDataShape);
        config.outConfs[0].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, outputDataShape);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});

        const auto& srcDims = inputDataShape.getDims();
        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 8 == 0) {
            config.inConfs[0].desc = creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(prec, inputDataShape);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
        }

        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 16 == 0) {
            config.inConfs[0].desc = creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(prec, inputDataShape);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
        }

        if (prec == Precision::FP32 || prec == Precision::I8 || prec == Precision::U8) {
            config.inConfs[0].desc = creatorsMap.at(LayoutType::nspc)->createSharedDesc(prec, inputDataShape);
            config.outConfs[0].desc = creatorsMap.at(LayoutType::nspc)->createSharedDesc(prec, outputDataShape);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
        }
    } else {
        // general plain case
        config.inConfs[0].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, inputDataShape);
        config.outConfs[0].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, outputDataShape);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
    }
}

bool MKLDNNTransposeNode::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

bool MKLDNNTransposeNode::needPrepareParams() const {
    if (isOptimized)
        return false;
    return inputShapesModified();
}

void MKLDNNTransposeNode::prepareParams() {
    auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    params.src_block_dims = srcDesc->getBlockDims();
    auto dstDesc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    params.dst_block_dims = dstDesc->getBlockDims();
    if (!isInputOrderConst) {
        auto orderPtr = reinterpret_cast<const int32_t*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
        auto orderLen = getParentEdgeAt(0)->getMemoryPtr()->GetSize();
        params.order.assign(orderPtr, orderPtr + orderLen);
    }

    TransposeKey key = {params};

    auto engine = getEngine();
    auto builder = [&engine](const TransposeKey& key) -> std::shared_ptr<TransposeJitExecutor> {
        return std::make_shared<TransposeJitExecutor>(key.params);
    };

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    execPtr = result.first;
}

void MKLDNNTransposeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory was not allocated.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory was not allocated.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor was not set.";

    if (getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp) &&
            std::find(optimizedOrders.begin(), optimizedOrders.end(), order) != optimizedOrders.end()) {
        isOptimized = true;
        execPtr = std::make_shared<TransposeRefExecutor>();
        return;
    }

    params.data_size = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->getPrecision().size();
    if (isInputOrderConst)
        params.order = order;
    auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    params.src_block_order = srcDesc->getOrder();
    auto dstDesc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    params.dst_block_order = dstDesc->getOrder();

    if (inputShapesDefined() && isExecutable()) {
        prepareParams();
        updateLastInputDims();
    }
}

template <typename T>
static void transpose_to_0312(const int MB, const MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    const auto src_data = reinterpret_cast<const T*>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<T*>(dstMemPtr->GetPtr());

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];

    parallel_for3d(MB, DIM1, DIM2, [&](const int n, const int dim1, const int dim2) {
        for (int dim3 = 0; dim3 < DIM3; ++dim3) {
            const int src_off = n * DIM1 * DIM2 * DIM3 +
                                dim1 * DIM2 * DIM3 +
                                dim2 * DIM3 +
                                dim3;
            const int dst_off = n * DIM1 * DIM2 * DIM3 +
                                dim3 * DIM1 * DIM2 +
                                dim1 * DIM2 +
                                dim2;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template<typename T>
static void transpose_to_04123(const int MB, const MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    const auto src_data = reinterpret_cast<const T*>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<T*>(dstMemPtr->GetPtr());

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];
    const int DIM4 = srcMemPtr->getStaticDims()[4];

    parallel_for4d(MB, DIM1, DIM2, DIM3, [&](const int n, const int dim1, const int dim2, const int dim3) {
        for (int dim4 = 0; dim4 < DIM4; ++dim4) {
            const int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                                dim1 * DIM2 * DIM3 * DIM4 +
                                dim2 * DIM3 * DIM4 +
                                dim3 * DIM4 +
                                dim4;
            const int dst_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                                dim4 * DIM1 * DIM2 * DIM3 +
                                dim1 * DIM2 * DIM3 +
                                dim2 * DIM3 +
                                dim3;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template<typename T>
static void transpose_to_051234(const int MB, const MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    const auto src_data = reinterpret_cast<const T*>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<T*>(dstMemPtr->GetPtr());

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];
    const int DIM4 = srcMemPtr->getStaticDims()[4];
    const int DIM5 = srcMemPtr->getStaticDims()[5];

    parallel_for5d(MB, DIM1, DIM2, DIM3, DIM4, [&](const int n, const int dim1, const int dim2, const int dim3, const int dim4) {
        for (int dim5 = 0; dim5 < DIM5; ++dim5) {
            const int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 * DIM5 +
                                dim1 * DIM2 * DIM3 * DIM4 * DIM5 +
                                dim2 * DIM3 * DIM4 * DIM5 +
                                dim3 * DIM4 * DIM5 +
                                dim4 * DIM5 +
                                dim5;
            const int dst_off = n * DIM5 * DIM1 * DIM2 * DIM3 * DIM4 +
                                dim5 * DIM1 * DIM2 * DIM3 * DIM4 +
                                dim1 * DIM2 * DIM3 * DIM4 +
                                dim2 * DIM3 * DIM4 +
                                dim3 * DIM4 +
                                dim4;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template<typename T>
void MKLDNNTransposeNode::optimizedExecute(const int MB, const MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    switch (srcMemPtr->getStaticDims().size()) {
        case 4:
            transpose_to_0312<T>(MB, srcMemPtr, dstMemPtr);
            break;
        case 5:
            transpose_to_04123<T>(MB, srcMemPtr, dstMemPtr);
            break;
        case 6:
            transpose_to_051234<T>(MB, srcMemPtr, dstMemPtr);
            break;
        default:
            IE_THROW() << "Transpose '" << getName() << "' supports optimized execution with only 4D, 5D and 6D shapes";
    }
}

void MKLDNNTransposeNode::execute(mkldnn::stream strm) {
    if (execPtr) {
        auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto &srcMemPtr = getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr();

        int MB = 0;
        if (isDynamicNode()) {
            MB = srcMemPtr->getStaticDims()[0];
        } else {
            MB = batchToProcess();
        }

        execPtr->exec(this, srcMemPtr, dstMemPtr, MB);
    } else {
        IE_THROW() << "Could not execute Transpose node. Primitive was not created.";
    }
}

void MKLDNNTransposeNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

MKLDNNTransposeNode::TransposeJitExecutor::TransposeJitExecutor(const PermuteParams& params) {
    pKernel = std::make_shared<PermuteKernel>(params);
}

void MKLDNNTransposeNode::TransposeJitExecutor::exec(MKLDNNTransposeNode* node, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr, const int MB) {
    if (!pKernel)
        IE_THROW() << "Could not execute. Kernel for Transpose node was not compiled.";

    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());

    pKernel->execute(srcData, dstData, MB);
}

void MKLDNNTransposeNode::TransposeRefExecutor::exec(MKLDNNTransposeNode* node, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr, const int MB) {
    const size_t dataSize = srcMemPtr->getDesc().getPrecision().size();
    TransposeContext ctx = {node, srcMemPtr, dstMemPtr, MB};
    OV_SWITCH(MKLDNNPlugin, TransposeOptimizedEmitter, ctx, dataSize,
              OV_CASE(1, PrecisionTrait<Precision::U8>::value_type),
              OV_CASE(2, PrecisionTrait<Precision::U16>::value_type),
              OV_CASE(4, PrecisionTrait<Precision::I32>::value_type));
}

bool MKLDNNTransposeNode::created() const {
    return getType() == Transpose;
}

REG_MKLDNN_PRIM_FOR(MKLDNNTransposeNode, Transpose);
