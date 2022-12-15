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

class FullyConnected : public Node {
public:
    FullyConnected(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape &dims) const override;
    void getSupportedDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool canBeInPlace() const override {
        return false;
    }

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() == 3 ? 2 : 1;
    }

    const std::vector<impl_desc_type>& getPrimitivesPriority() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;

    size_t descInputNumbers(DnnlDesriptor desc) override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }

    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    // void createPrimitive() override;
    std::shared_ptr<MemoryDesc> getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    std::shared_ptr<dnnl::primitive_attr> initPrimitiveAttr() override;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

    void setDynamicBatchLim(int lim) override;

    void setMinSparseRate(float sparseRate) { minSparseRate = sparseRate; }

private:
    void createDescriptorInternal(const dnnl::memory::desc &inputDesc,
                                  const dnnl::memory::desc &outputDesc);

    VectorDims makeDummyInputDims() const;
    VectorDims makeDummyOutputDims(const VectorDims& inDims) const;

    VectorDims inDims;
    VectorDims outDims;

    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims, bool initWeights = false);

    bool withBiases = false;

    std::string errorPrefix;
    static const size_t DATA_ID = 0;
    static const size_t WEIGHTS_ID = 1;
    static const size_t BIAS_ID = 2;
    dnnl::memory::data_type outputDataType;

    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;
    bool useConv1x1 = false;
    impl_desc_type implementationTypeIP;
    MemoryDescPtr weightDescIP;
    // when weightCache is not enabled (such as stream=1), brgconv weights may change due to
    // different shapes. Weights will be cached in privateWeightCache.
    // When weightCache is enabled, it holds weight ptr reference since weightCache does not hold the
    // reference
    std::unordered_map<std::string, MemoryPtr> privateWeightCache;

    class ExecutorInnerProduct : public DnnlExecutor {
        public:
            ExecutorInnerProduct(const dnnl::inner_product_forward::primitive_desc& pd);
    };

    class ExecutorConv1x1 : public DnnlExecutor {
        public:
            ExecutorConv1x1(const dnnl::convolution_forward::primitive_desc& pd);
    };

    static DnnlDesriptor createDescriptorInternalForConv(DnnlMemoryDescCPtr inputDescPtr,
                                                         DnnlMemoryDescCPtr weightDescPtr,
                                                         DnnlMemoryDescCPtr biasDescPtr,
                                                         DnnlMemoryDescCPtr outputDescPtr);

    bool canBeExecutedInConv1x1() const;
    MemoryPtr prepareWeightMemory(const DnnlMemoryDescPtr weightDesc);

    // sparse weights
    bool useSparseWeights = false;
    int nnzCount = -1;
    float minSparseRate = 1.f;
    float weiSparseRate = 0.f;
    bool useSparseWeightsDecompression();
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
