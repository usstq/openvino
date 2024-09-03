// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

class AddRMSNode : public ov::op::Op {
public:
    OPENVINO_OP("AddRMS", "cpu_plugin_opset");

    AddRMSNode() = default;

    AddRMSNode(const Output<Node>& data0,
               const Output<Node>& data1,
               const Output<Node>& gamma,
               float epsilon,
               const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    float get_epsilon() const {
        return m_epsilon;
    }

protected:
    float m_epsilon{0};
};

}   // namespace intel_cpu
}   // namespace ov
