// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ie_common.h>
#include <node.h>

#include <memory>
#include <string>
#include <vector>

#include "transformations/cpu_opset/x64/op/rope.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class RoPE : public Node {
public:
    RoPE(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::RoPE;
    }
    bool needShapeInfer() const override {
        return false;
    };
    bool needPrepareParams() const override {
        return false;
    };
    bool isExecutable() const override {
        return true;
    }
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    struct Executor {
        virtual void execute(RoPE * pnode) = 0;
    };

    const RoPENode::Config& getConfig() const {
        return m_config;
    }

private:
    RoPENode::Config m_config;
    std::shared_ptr<Executor> m_executor;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
