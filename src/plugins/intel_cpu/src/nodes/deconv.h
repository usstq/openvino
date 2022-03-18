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
    using DefaultDeconvDescs = std::pair<std::shared_ptr<mkldnn::convolution_backward_data::desc>,
                                         std::shared_ptr<mkldnn::convolution_forward::primitive_desc>>;
    using Int8DeconvDesc = std::shared_ptr<mkldnn::deconvolution_forward::desc>;

public:
    Deconvolution(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache);

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

    std::shared_ptr<MemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    bool canFuse(const NodePtr& node) const override;

    const VectorDims& getWeightDims() const { return getInputShapeAtPort(1).getStaticDims(); }
    const std::vector<ptrdiff_t>& getStride() const { return stride; }

    void prepareParams() override;
    void execute(mkldnn::stream strm) override;
    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;

    void setDynamicBatchLim(int lim) override;

    void cleanup() override;

protected:
    AttrPtr initPrimitiveAttr() override;
    AttrPtr makePrimitiveAttr(const VectorDims& dims);

private:
    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;

    class DeconvExecutorDefault : public DnnlExecutor {
        public:
            DeconvExecutorDefault(const mkldnn::convolution_backward_data::primitive_desc& pd,
                                  const mkldnn::memory::desc& inMemDesc,
                                  const mkldnn::memory::desc& weightMemDesc,
                                  const mkldnn::memory::desc& outMemDesc,
                                  const mkldnn::engine& engine);
    };

    class DeconvExecutorInt8 : public DnnlExecutor {
        public:
            DeconvExecutorInt8(const mkldnn::deconvolution_forward::primitive_desc& pd,
                               const mkldnn::memory::desc& inMemDesc,
                               const mkldnn::memory::desc& weightMemDesc,
                               const mkldnn::memory::desc& outMemDesc,
                               const mkldnn::engine& engine);
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

    Shape inShape;

    AttrPtr pAttr;

    std::shared_ptr<mkldnn::primitive_attr> attr;
    void setPostOps(mkldnn::primitive_attr &attr, const VectorDims &dims);

    VectorDims shapeInferInternal(const VectorDims &inDims, std::vector<int32_t> outSpDims) const;
    void initPadding(std::shared_ptr<ngraph::Node> op, const Shape &inShape, const std::vector<int32_t>& outSpDims);
    void initPaddingR(const Shape &inShape, const Shape &outShape);
    std::vector<int32_t> readOutputSpatialDims() const;
    std::pair<VectorDims, VectorDims> makeDummyInOutShape();

    DefaultDeconvDescs createDescriptorInternalDefault(const mkldnn::memory::desc& in_candidate,
                                                       const mkldnn::memory::desc& wgh_candidate,
                                                       const mkldnn::memory::desc& out_candidate,
                                                       mkldnn::algorithm alg) const;
    Int8DeconvDesc createDescriptorInternalInt8(const mkldnn::memory::desc& in_candidate,
                                                const mkldnn::memory::desc& wgh_candidate,
                                                const mkldnn::memory::desc& out_candidate) const;
    std::shared_ptr<DnnlDesriptor> createDefaultMkldnnDeconvDesc(const mkldnn::memory::desc& srcDesc,
                                                                 const mkldnn::memory::desc& wghDesc,
                                                                 const mkldnn::memory::desc& dstDesc,
                                                                 bool isWinograd) const;
    std::shared_ptr<DnnlDesriptor> createInt8MkldnnDeconvDesc(const mkldnn::memory::desc& srcDesc,
                                                              const mkldnn::memory::desc& wghDesc,
                                                              const mkldnn::memory::desc& dstDesc) const;

    void createDeconvPrim(std::shared_ptr<DnnlDesriptor> desc,
                          MemoryPtr srcMemPtr,
                          MemoryPtr wghMemPtr,
                          MemoryPtr dstMemPtr,
                          AttrPtr attr,
                          impl_desc_type selectedImpl);

    std::string errorPrefix;

    bool canBeExecutedInInt8() const;
    InferenceEngine::Blob::Ptr createWeiBlobAsIO(InferenceEngine::SizeVector dims);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
