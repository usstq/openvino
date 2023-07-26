// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class RPE : public Node {
public:
    RPE(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool needShapeInfer() const override {return false;};
    bool needPrepareParams() const override {return false;};
    bool isExecutable() const override { return true; }
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    struct ExecutorBase {
        virtual void exec(RPE * node) = 0;
    };

private:
    std::string errorPrefix;
    InferenceEngine::Precision m_runtime_precision;
    std::shared_ptr<ov::op::internal::RPE> m_rpe;
    std::shared_ptr<ExecutorBase> m_executor;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov