// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {
namespace node {

class AdaptivePooling : public Node {
public:
  AdaptivePooling(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int spatialDimsCount;
    mutable std::vector<Dim> spatialDimsValue = {};
    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    inline void setBinBorders(size_t *startPtr, size_t *endPtr, size_t idx, size_t inputLength, size_t outputLength);

    std::string errorPrefix;

protected:
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override { return false; };
    void executeDynamicImpl(mkldnn::stream strm) override;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
