// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class DimOfNode : public ngraph::op::Op {
public:
    OPENVINO_OP("DimOf", "cpu_plugin_opset");

    DimOfNode() = default;

    DimOfNode(const ngraph::Output<Node> &input, const int axis, bool output_scalar);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    bool get_output_scalar() const { return m_output_scalar; }
    void set_output_scalar(bool set) { m_output_scalar = set; }
    int get_axis() const { return m_axis; }

private:
    int m_axis;
    bool m_output_scalar;
};

}   // namespace intel_cpu
}   // namespace ov
