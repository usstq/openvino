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

class Deconvolution : public Node {
public:
    Deconvolution(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void createPrimitive() override;
    void filterSupportedPrimitiveDescriptors() override;
    void filterSupportedDescriptors();
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    size_t descInputNumbers(DnnlDesriptor desc) override {
        return static_cast<size_t>(getParentEdges().size());
    }

    std::shared_ptr<MemoryDesc> getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    bool canFuse(const NodePtr& node) const override;

    const VectorDims& getWeightDims() const { return getInputShapeAtPort(1).getStaticDims(); }
    const std::vector<ptrdiff_t>& getStride() const { return stride; }

    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override { execute(strm); }
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;

    void setDynamicBatchLim(int lim) override;
    bool canBeExecutedInInt8() const;

protected:
    AttrPtr initPrimitiveAttr() override;
    AttrPtr makePrimitiveAttr(const VectorDims& dims);
    std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape& dims) const override;

private:
    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;

    class DeconvExecutorDefault : public DnnlExecutor {
        public:
            DeconvExecutorDefault(const dnnl::convolution_backward_data::primitive_desc& pd,
                                  const dnnl::memory::desc& inMemDesc,
                                  const dnnl::memory::desc& weightMemDesc,
                                  const dnnl::memory::desc& outMemDesc,
                                  const dnnl::engine& engine);
    };

    class DeconvExecutorInt8 : public DnnlExecutor {
        public:
            DeconvExecutorInt8(const dnnl::deconvolution_forward::primitive_desc& pd,
                               const dnnl::memory::desc& inMemDesc,
                               const dnnl::memory::desc& weightMemDesc,
                               const dnnl::memory::desc& outMemDesc,
                               const dnnl::engine& engine);
    };

    bool withGroups = false;
    bool isDW = false;
    bool isInt8 = false;
    bool autoPad = false;
    bool externOutShape = false;
    size_t groupNum = 1;
    size_t IC;
    size_t OC;
    std::vector<ptrdiff_t> kernel;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    ov::CoordinateDiff paddingL;
    ov::CoordinateDiff paddingR;
    ov::CoordinateDiff outputPadding;
    std::vector<int32_t> lastOutputSpatialDims;
    VectorDims int8WeightDims;
    VectorDims biasesDims;

    Shape inShape;

    AttrPtr pAttr;

    std::shared_ptr<dnnl::primitive_attr> attr;
    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims);

    VectorDims shapeInferInternal(const VectorDims &inDims, std::vector<int32_t> outSpDims) const;
    void initPaddingR(const Shape &inShape, const Shape &outShape);
    std::vector<int32_t> readOutputSpatialDims() const;
    std::pair<VectorDims, VectorDims> makeDummyInOutShape();
    bool withBiases = false;
    size_t biasPort;

    std::string errorPrefix;

    InferenceEngine::Blob::Ptr createWeiBlobAsIO(InferenceEngine::SizeVector dims);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
