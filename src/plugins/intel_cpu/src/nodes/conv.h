// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <memory>
#include <string>
#include <vector>
#include "common/dnnl_executor.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Eltwise;

class Convolution : public Node {
public:
    Convolution(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void initDescriptor(const NodeConfig& config) override;
    void selectOptimalPrimitiveDescriptor() override;
    void initSupportedPrimitiveDescriptors() override;
    void filterSupportedPrimitiveDescriptors() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }
    InferenceEngine::Precision getRuntimePrecision() const override;
    std::shared_ptr<MemoryDesc> getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    dnnl::memory getWeights() const;
    dnnl::memory getBias() const;

    size_t descInputNumbers(DnnlDesriptor desc) override {
        return getOriginalInputsNumber();
    }

    bool canBeExecutedInInt8() const;
    size_t getGroupNum() const { return groupNum; }
    //OV Legacy input zero point mechanism can support per-channel zero point.
    //Hold legacy input zero point.
    std::vector<uint8_t> legacyInputZeroPoints;
    //Hold legacy weight zero point.
    std::vector<float> legacyWeightsZeroPoints;
    //Hold legacy pre-calculated output compensation
    std::vector<int32_t> legacyOutputCompensation;
    //Hold stock per-tensor input zero point. Pass to onednn to calculate output compensation.
    std::vector<int32_t> inputZeroPoints;
    void initializeInputZeroPoints(const uint8_t* inputZpData, const size_t inputZpSize);

    const InferenceEngine::SizeVector &getWeightDims() { return weightDims; }
    const std::vector<size_t> &getStride() { return stride; }
    const std::vector<ptrdiff_t> &getDilation() { return dilation; }
    const std::vector<ptrdiff_t> &getPaddingL() { return paddingL; }
    const std::vector<ptrdiff_t> &getPaddingR() { return paddingR; }

    bool canFuse(const NodePtr& node) const override;
    bool isDepthWise() const {
        return isGrouped && 1 == groupOC && 1 == groupIC;
    }

    bool isWinograd() const { return isWino; }

    void setDynamicBatchLim(int lim) override;

protected:
    InferenceEngine::Precision fusedEltwisePrecision(const NodePtr& fusingNode) const;
    void redefineOutputMemory(const std::vector<VectorDims> &newOutputShapes) override;
    void addFusedNode(const NodePtr &fusingNode) override;
    const std::vector<impl_desc_type>& getPrimitivesPriority() override;

private:
    enum class zpType {
        None,
        PerTensor,
        PerChannel
    };
    class FusedSubgraph;
    using FusedSubgraphPtr = std::shared_ptr<FusedSubgraph>;
    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;

    class ConvolutionExecutor : public DnnlExecutor {
        public:
            ConvolutionExecutor(const dnnl::convolution_forward::primitive_desc& pd,
                                const dnnl::memory::desc& inMemDesc,
                                const dnnl::memory::desc& weightMemDesc,
                                const dnnl::memory::desc& outMemDesc,
                                const dnnl::engine& engine);
    };

    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    void addLegacyZeroPoints(dnnl::primitive_attr& attr);
    void addZeroPoints(dnnl::primitive_attr& attr);
    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims, bool useLegacyPostOps, bool initWeights = false);
    void SetPostOpsAndZeroPoints(std::vector<dnnl::primitive_attr> &attrs);
    void filterSupportedDescriptors();
    bool isPossibleToSkipInitConfig(DnnlDesriptor &desc) const;
    bool isNspcAvailable() const;
    InferenceEngine::Blob::Ptr createInternalBlob(InferenceEngine::SizeVector dims, size_t edgeNum, bool isGrouped = false);

    void updatePadding();
    MemoryDescPtr getSumMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it);
    MemoryPtr getOutputMemory() const;

    void appendLegacyZeroPointsArgs();
    void appendZeroPointsArgs();
    void initTryBrgconvFlag();

    bool withBiases;
    bool withSum;
    bool withDWConv;
    bool isGrouped;
    bool isPrimitivesPriorityDefined = false;
    bool withSumBroadcast = false;
    bool preferLegacyPostOps = false;
    bool preferLegacyZeroPoint = false;
    zpType inputZeroPointType = zpType::None;

    std::vector<size_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    InferenceEngine::SizeVector weightDims;
    InferenceEngine::SizeVector biasesDims;
    std::unordered_map<int, MemoryPtr> convPostOpsArgs[2];

    size_t dw_conv_oc;
    size_t dw_conv_ih;
    size_t dw_conv_iw;
    std::vector<size_t> dw_conv_kernel;
    std::vector<size_t> dw_conv_strides;
    dnnl::memory::data_type dw_conv_in_dt;

    size_t groupNum;
    size_t IC;
    size_t groupIC;
    size_t groupOC;

    InferenceEngine::Precision eltwisePrecision;

    const size_t X_AXIS = 0;
    const size_t Y_AXIS = 1;

    bool isWino = false;
    bool shouldTryBrgconv = false;
    std::vector<dnnl::primitive_attr> attrs;
    AttrPtr pAttr;
    bool autoPadding = false;
    FusedSubgraphPtr subgraph;
    std::unordered_map<NodePtr, std::vector<NodePtr>> fusedConstNodes;

    MemoryPtr legacyInputZeroPointsMemPtr;
    MemoryPtr legacyWeightsZeroPointsMemPtr;
    MemoryPtr legacyOutputCompensationMemPtr;
    MemoryPtr stockInputZeroPointsMemPtr;
    dnnl::memory::data_type outputDataType;
    InferenceEngine::Precision sumPrc = InferenceEngine::Precision::UNSPECIFIED;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
