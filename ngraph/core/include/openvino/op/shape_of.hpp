// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief Operation that returns the shape of its input argument as a tensor.
class OPENVINO_API ShapeOf : public Op {
public:
    OPENVINO_RTTI_DECLARATION;
    ShapeOf() = default;
    /// \brief Constructs a shape-of operation.
    ShapeOf(const Output<Node>& arg, const element::Type output_type = element::i64);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    element::Type get_output_type() const {
        return m_output_type;
    }
    void set_output_type(element::Type output_type) {
        m_output_type = output_type;
    }
    // Overload collision with method on Node
    using Node::set_output_type;

    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(const HostTensorVector& output_values) const override;
    bool evaluate_upper(const HostTensorVector& output_values) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& input_values) override;

private:
    element::Type m_output_type;
};
}  // namespace v3

namespace v0 {
/// \brief Operation that returns the shape of its input argument as a tensor.
class OPENVINO_API ShapeOf : public Op {
public:
    OPENVINO_RTTI_DECLARATION;
    ShapeOf() = default;
    /// \brief Constructs a shape-of operation.
    ShapeOf(const Output<Node>& arg);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(const HostTensorVector& output_values) const override;
    bool evaluate_upper(const HostTensorVector& output_values) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& input_values) override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
