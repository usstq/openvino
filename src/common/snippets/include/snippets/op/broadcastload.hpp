// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <snippets/op/broadcastmove.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface BroadcastLoad
 * @brief Is generated for broadcasting by least varying dimension for non-blocked cases and the second varying dimension for blocked
 * @ingroup snippets
 */
class BroadcastLoad : public BroadcastMove {
public:
    OPENVINO_OP("BroadcastLoad", "SnippetsOpset", ngraph::snippets::op::BroadcastMove);

    BroadcastLoad(const Output<Node>& x, Shape output_shape);
    BroadcastLoad() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;
};

} // namespace op
} // namespace snippets
} // namespace ngraph