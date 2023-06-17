// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dimof.hpp"
#include "transformations/itt.hpp"

ov::intel_cpu::DimOfNode::DimOfNode(const ngraph::Output<Node> &src, const int axis, bool output_scalar)
    : Op({src}), m_axis(axis), m_output_scalar(output_scalar) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::DimOfNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(DimOfNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return std::make_shared<ov::intel_cpu::DimOfNode>(new_args.at(0), m_axis, m_output_scalar);
    }

    OPENVINO_THROW("Unsupported number of arguments for FullyConnected operation");
}

void ov::intel_cpu::DimOfNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(DimOfNode_validate_and_infer_types);
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 1,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected: 1.");

    const auto src_shape = get_input_partial_shape(0);

    auto r = src_shape.rank();

    NODE_VALIDATION_CHECK(this,
        r.is_static(),
        "input rank must be static");

    auto rmax = r.get_length();

    NODE_VALIDATION_CHECK(this,
        m_axis >= -rmax && m_axis < rmax,
        "Axis ", m_axis, " is out of the tensor rank range [", -rmax, ", ", rmax, ")");

    // default: scalar
    ngraph::PartialShape output_pshape;
    if (!m_output_scalar) {
        output_pshape = ngraph::PartialShape{1};
    }

    auto output_type = ngraph::element::i32;
    set_output_type(0, output_type, output_pshape);
}

bool ov::intel_cpu::DimOfNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(FullyConnectedNode_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("output_scalar", m_output_scalar);
    return true;
}