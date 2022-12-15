// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/concat.hpp"

#include <memory>

#include "concat_shape_inference.hpp"
#include "dimension_tracker.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::Concat::Concat(const OutputVector& args, int64_t axis) : Op(args), m_axis(axis) {
    constructor_validate_and_infer_types();
}

op::Concat::Concat(const NodeVector& args, int64_t axis) : Concat(as_output_vector(args), axis) {}

bool op::Concat::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Concat_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::Concat::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Concat_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() >= 1, "At least one argument required.");

    element::Type inputs_et{element::dynamic};
    auto input_shapes = std::vector<PartialShape>();

    for (size_t i = 0; i < get_input_size(); ++i) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(inputs_et, inputs_et, get_input_element_type(i)),
                              "Argument element types are inconsistent.");
        const auto& input_shape = get_input_partial_shape(i);
        const auto& input_rank = input_shape.rank();

        if (input_rank.is_static() && (get_concatenation_axis() < 0)) {
            set_concatenation_axis(get_axis() < 0 ? get_axis() + input_rank.get_length() : get_axis());
        }

        const auto concat_axis = get_concatenation_axis();

        NODE_VALIDATION_CHECK(this,
                              input_shape.is_dynamic() || (0 <= concat_axis && concat_axis < input_rank.get_length()),
                              "Concatenation axis (",
                              concat_axis,
                              ") is out of bounds [",
                              -input_rank.get_length(),
                              ", ",
                              input_rank.get_length() - 1,
                              "] for ",
                              "argument ",
                              i,
                              ", which has shape ",
                              input_shape,
                              ".");

        input_shapes.push_back(input_shape);
    }

    std::vector<PartialShape> output_shapes(1, PartialShape{});

    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, inputs_et, output_shapes.front());
}

shared_ptr<Node> op::Concat::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Concat_clone_with_new_inputs);
    return make_shared<Concat>(new_args, m_axis);
}

namespace {
bool evaluate_concat(const HostTensorVector& args, const HostTensorPtr& out, int64_t concatenation_axis) {
    std::vector<const char*> arg_bufs;
    std::vector<ov::Shape> arg_shapes;
    ov::Shape out_shape(args[0]->get_shape());
    out_shape[concatenation_axis] = 0;
    for (auto& input : args) {
        arg_bufs.push_back(input->get_data_ptr<char>());
        arg_shapes.push_back(input->get_shape());
        out_shape[concatenation_axis] += arg_shapes.back()[concatenation_axis];
    }
    out->set_shape(out_shape);
    runtime::reference::concat(arg_bufs,
                               out->get_data_ptr<char>(),
                               arg_shapes,
                               out_shape,
                               concatenation_axis,
                               out->get_element_type().size());

    return true;
}
}  // namespace

bool op::Concat::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Concat_evaluate);
    NGRAPH_CHECK(!inputs.empty());
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, inputs.size()));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));
    auto concat_axis = get_axis() < 0 ? get_axis() + inputs[0]->get_shape().size() : get_axis();
    return evaluate_concat(inputs, outputs[0], concat_axis);
}

bool op::Concat::has_evaluate() const {
    OV_OP_SCOPE(v0_Concat_has_evaluate);
    return true;
}

bool op::Concat::evaluate_lower(const HostTensorVector& output_values) const {
    return default_lower_bound_evaluator(this, output_values);
}

bool op::Concat::evaluate_upper(const HostTensorVector& output_values) const {
    return default_upper_bound_evaluator(this, output_values);
}

bool op::Concat::evaluate_label(TensorLabelVector& output_labels) const {
    const auto& inputs = input_values();
    if (std::all_of(inputs.cbegin(), inputs.cend(), [](const Output<Node>& out) {
            const auto& labels = out.get_tensor().get_value_label();
            return has_no_labels(labels);
        })) {
        return false;
    }

    HostTensorVector idx_inputs;
    idx_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        auto input_label = input.get_tensor().get_value_label();
        if (input_label.empty()) {
            const auto& shape = input.get_partial_shape();
            // sanity check. at this point value propagation was successful
            NGRAPH_CHECK(shape.is_static());
            const auto& num_elements = shape_size(shape.to_shape());
            input_label.resize(num_elements, no_label);
        }
        const auto& constant = Constant::create(element::u64, input.get_shape(), input_label);
        idx_inputs.push_back(std::make_shared<HostTensor>(constant));
    }

    const auto& output_tensor = std::make_shared<HostTensor>(element::u64, get_output_shape(0));
    evaluate({output_tensor}, idx_inputs);
    output_labels[0] = std::make_shared<Constant>(output_tensor)->cast_vector<size_t>();
    return true;
}
