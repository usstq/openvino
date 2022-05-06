// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/op/util/attr_types.hpp>

namespace ov {
namespace intel_cpu {

class RNNPrim : public ngraph::op::Op {
public:
    OPENVINO_OP("RNNPrim", "cpu_plugin_opset");

    
    using direction = op::RecurrentSequenceDirection;
    RNNPrim() = default;

    //  cell_type: vanilla_rnn, lstm, gru, lbr_gru
    //  dir:       left2right, right2left, bidirectional_concat, bidirectional_sum
    RNNPrim(const std::string cell_type,
            const std::string dir,
            const bool batch_first,
            const ngraph::OutputVector& args);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    std::string cell_type;
    std::string dir;
    bool batch_first;
private:
    
};



}   // namespace intel_cpu
}   // namespace ov
