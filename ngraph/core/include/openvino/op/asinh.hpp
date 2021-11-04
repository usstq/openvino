// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief Elementwise inverse hyperbolic sin operation.
///
class OPENVINO_API Asinh : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Asinh", "opset4", util::UnaryElementwiseArithmetic, 3);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs an Asinh operation.
    Asinh() = default;
    /// \brief Constructs an Asinh operation.
    ///
    /// \param arg Output that produces the input tensor.<br>
    /// `[d1, ...]`
    ///
    /// Output `[d1, ...]`
    ///
    Asinh(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor&) override {
        return true;
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
