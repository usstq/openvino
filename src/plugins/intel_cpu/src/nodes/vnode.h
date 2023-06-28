// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

#include "transformations/cpu_opset/common/op/vnode.hpp"

#include "utils/plain_tensor.hpp"

#include "llm_emb_gpt.hpp"
#include "llm_mha_gpt.hpp"
#include "llm_mm.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class VNode : public Node {
public:
    VNode(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool needShapeInfer() const override {return false;};
    bool needPrepareParams() const override {return false;};
    bool isExecutable() const override { return true; }
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void gptneox_attention(dnnl::stream strm, bool redefine_outputs);

    std::string m_vtype;
    std::string errorPrefix;
    std::shared_ptr<ov::intel_cpu::VNode> m_vnode;

    bool m_kernel_initialized;
    PlainTensor<ov::bfloat16> m_query_emb; // query with embed
    llmdnn::emb_gpt m_kernel_emb;
    llmdnn::mha_gpt m_kernel_mha;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov