// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "add_rms.hpp"
#include "transformations/itt.hpp"

ov::intel_cpu::AddRMSNode::AddRMSNode(const Output<Node>& data0,
                                      const Output<Node>& data1,
                                      const Output<Node>& gamma,
                                      float epsilon,
                                      const ov::element::Type) : Op({data0, data1, gamma}), m_epsilon(epsilon) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::AddRMSNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(AddRMSNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::AddRMSNode>(new_args.at(0), new_args.at(1), new_args.at(2), m_epsilon);
}

bool ov::intel_cpu::AddRMSNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(AddRMSNode_visit_attributes);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void ov::intel_cpu::AddRMSNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(AddRMSNode_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    set_output_type(1, get_input_element_type(0), get_input_partial_shape(0));
}
