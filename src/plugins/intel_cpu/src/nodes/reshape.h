// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <vector>
#include <memory>
#include "input.h"

namespace ov {
namespace intel_cpu {

class MKLDNNReshapeNode : public MKLDNNNode {
public:
    MKLDNNReshapeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    bool isExecutable() const override {
        return false;
    }

    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override { return false; }
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    mutable std::vector<int> lastSecondInputValues;

    std::string errorPrefix;
};

}   // namespace intel_cpu
}   // namespace ov
