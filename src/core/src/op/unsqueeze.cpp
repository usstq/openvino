// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/unsqueeze.hpp"

#include <cstddef>
#include <functional>
#include <set>

#include "itt.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/validation_util.hpp"
#include "unsqueeze_shape_inference.hpp"

using namespace std;
using namespace ngraph;

op::v0::Unsqueeze::Unsqueeze(const Output<Node>& data, const Output<Node>& axes) : Op({data, axes}) {
    constructor_validate_and_infer_types();
}

void op::v0::Unsqueeze::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Unsqueeze_validate_and_infer_types);

    const auto input_shapes = get_node_input_partial_shapes(*this);
    auto output_shapes = std::vector<ov::PartialShape>(1);

    shape_infer(this, input_shapes, output_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool op::v0::Unsqueeze::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Unsqueeze_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::Unsqueeze::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Unsqueeze_clone_with_new_inputs);
    if (new_args.size() != 2) {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Unsqueeze>(new_args.at(0), new_args.at(1));
}

namespace unsqueeze {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    runtime::reference::copy(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape_size(out->get_shape()));
    return true;
}

// The evaluate cannot use shape_infer for output shape calculation as shape inference accepts
// repeated axis and evaluate not. When shape inference will changed to be compatible with `numpy` then
// evaluate and inference can use same function to calculate output shape. TODO for next version for this operator.
bool evaluate_unsqueeze(const Node* node,
                        const HostTensorPtr& arg0,
                        const HostTensorPtr& arg1,
                        const HostTensorPtr& out) {
    auto element_type = arg0->get_element_type();
    out->set_element_type(element_type);

    const auto& axes_shape = arg1->get_shape();
    ov::op::v0::check_unsqueeze_axes_rank(node, Rank(axes_shape.size()));

    const auto& data_shape = arg0->get_shape();
    const auto out_rank = static_cast<int64_t>(data_shape.size() + shape_size(axes_shape));

    // Get axes and normalize
    auto axes = read_index_vector(arg1);
    normalize_axes(node, out_rank, axes);

    // Sort in increasing order
    std::set<int64_t> axes_set(axes.begin(), axes.end());
    NGRAPH_CHECK(axes.size() == axes_set.size(), "Axes has duplicate axis.");

    auto out_shape = data_shape;
    for (int64_t axis : axes_set) {
        out_shape.insert(out_shape.begin() + axis, 1);
    }
    out->set_shape(out_shape);

    bool rc = true;
    switch (element_type) {
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, i32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, i64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, u32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, u64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, f16, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, f32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, f64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, bf16, arg0, out);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace unsqueeze

bool op::v0::Unsqueeze::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Unsqueeze_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 2));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));
    return unsqueeze::evaluate_unsqueeze(this, inputs[0], inputs[1], outputs[0]);
}

bool op::v0::Unsqueeze::has_evaluate() const {
    OV_OP_SCOPE(v0_Unsqueeze_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
    case ngraph::element::bf16:
        return true;
    default:
        break;
    }
    return false;
}

bool op::v0::Unsqueeze::evaluate_lower(const HostTensorVector& output_values) const {
    if (!get_input_tensor(1).has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v0::Unsqueeze::evaluate_upper(const HostTensorVector& output_values) const {
    if (!get_input_tensor(1).has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v0::Unsqueeze::evaluate_label(TensorLabelVector& output_labels) const {
    if (!get_input_tensor(1).has_and_set_bound())
        return false;
    return default_label_evaluator(this, output_labels);
}

bool op::v0::Unsqueeze::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    if (get_output_partial_shape(0).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    const auto& shape = get_output_shape(0);

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        output_values[0] = std::make_shared<op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}
