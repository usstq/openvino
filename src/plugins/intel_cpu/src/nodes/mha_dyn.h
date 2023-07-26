// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

#include "transformations/cpu_opset/x64/op/dyn_mha.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class MHADynamic : public Node {
public:
    MHADynamic(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

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
        virtual void exec(MHADynamic * node, ov::intel_cpu::MHADynamic::Config& config) = 0;
    };
    std::string errorPrefix;

private:
    ov::intel_cpu::MHADynamic::Config m_config;
    std::shared_ptr<ov::intel_cpu::MHADynamic> m_mha;
    InferenceEngine::Precision m_runtime_precision;
    std::shared_ptr<ExecutorBase> m_executor;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov