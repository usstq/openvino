// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/pad.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "pad_shape_inference.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::Pad);

op::v1::Pad::Pad(const Output<Node>& arg,
                 const Output<Node>& pads_begin,
                 const Output<Node>& pads_end,
                 const Output<Node>& arg_pad_value,
                 PadMode pad_mode)
    : Op({arg, pads_begin, pads_end, arg_pad_value}),
      m_pad_mode{pad_mode} {
    constructor_validate_and_infer_types();
}

op::v1::Pad::Pad(const Output<Node>& arg,
                 const Output<Node>& pads_begin,
                 const Output<Node>& pads_end,
                 PadMode pad_mode)
    : Op({arg, pads_begin, pads_end, op::v0::Constant::create(arg.get_element_type(), ov::Shape{}, {0})}),
      m_pad_mode{pad_mode} {
    constructor_validate_and_infer_types();
}

CoordinateDiff op::v1::Pad::get_pads_begin() const {
    CoordinateDiff pads_begin_coord{};
    if (auto pads_begin_const = get_constant_from_source(input_value(1))) {
        pads_begin_coord = pads_begin_const->cast_vector<ptrdiff_t>();
    }
    return pads_begin_coord;
}

CoordinateDiff op::v1::Pad::get_pads_end() const {
    CoordinateDiff pads_end_coord{};
    if (auto pads_end_const = get_constant_from_source(input_value(2))) {
        pads_end_coord = pads_end_const->cast_vector<ptrdiff_t>();
    }
    return pads_end_coord;
}

bool ngraph::op::v1::Pad::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v1_Pad_visit_attributes);
    visitor.on_attribute("pad_mode", m_pad_mode);
    return true;
}

void op::v1::Pad::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v1_Pad_validate_and_infer_types);
    element::Type result_et;

    const auto& arg_element_type = get_input_element_type(0);
    const auto& pads_begin_element_type = get_input_element_type(1);
    const auto& pads_end_element_type = get_input_element_type(2);

    if (m_pad_mode == PadMode::CONSTANT && get_input_size() == 4) {
        const auto& arg_pad_element_type = get_input_element_type(3);
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(result_et, arg_element_type, arg_pad_element_type),
                              "Argument element types do not match (input arg element type: ",
                              arg_element_type,
                              ", arg_pad element type: ",
                              arg_pad_element_type,
                              ").");
    }

    NODE_VALIDATION_CHECK(this,
                          pads_begin_element_type.is_integral_number(),
                          "pads_begin must be an integral number, but is: ",
                          pads_begin_element_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_end_element_type.is_integral_number(),
                          "pads_end must be an integral number, but is: ",
                          pads_end_element_type,
                          ").");

    std::vector<PartialShape> output_shapes;
    std::vector<PartialShape> input_shapes;
    for (size_t i = 0; i < get_input_size(); i++)
        input_shapes.push_back(get_input_partial_shape(i));
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

shared_ptr<Node> op::v1::Pad::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v1_Pad_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (get_input_size() == 4) {
        return make_shared<v1::Pad>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_pad_mode);
    } else {
        return make_shared<v1::Pad>(new_args.at(0), new_args.at(1), new_args.at(2), m_pad_mode);
    }
}

bool op::v1::Pad::evaluate_pad(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    const auto& data = inputs[0];
    const auto elem_size = data->get_element_type().size();

    const char* pad_value = nullptr;
    const std::vector<char> pad_zero_value(elem_size, 0);
    if (get_input_size() == 4) {
        pad_value = inputs[3]->get_data_ptr<char>();
    } else {
        pad_value = pad_zero_value.data();
    }
    const auto& out = outputs[0];

    ngraph::runtime::reference::pad(data->get_data_ptr<char>(),
                                    pad_value,
                                    out->get_data_ptr<char>(),
                                    elem_size,
                                    data->get_shape(),
                                    out->get_shape(),
                                    get_pads_begin(),
                                    get_pads_end(),
                                    get_pad_mode());

    return true;
}

bool op::v1::Pad::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v1_Pad_evaluate);
    return evaluate_pad(outputs, inputs);
}

bool op::v1::Pad::has_evaluate() const {
    NGRAPH_OP_SCOPE(v1_Pad_has_evaluate);
    return true;
}
