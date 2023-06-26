// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class VNode : public ngraph::op::Op {
public:
    OPENVINO_OP("VNode", "cpu_plugin_opset");

    VNode() = default;

    VNode(const ngraph::OutputVector& new_args, const ngraph::Output<Node> &org, const std::string& vtype);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    std::string get_vtype() const { return m_vtype; }
    ngraph::Output<Node> get_org() { return m_org; }

    void get_internal_vnodes(ov::NodeVector & nv, ngraph::Output<Node> base);
private:
    ngraph::Output<Node> m_org;
    std::string m_vtype;
    ov::NodeVector m_nodes;
};

}   // namespace intel_cpu
}   // namespace ov
