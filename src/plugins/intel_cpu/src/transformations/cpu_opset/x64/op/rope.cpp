// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope.hpp"

#include <algorithm>

#include "transformations/itt.hpp"

ov::intel_cpu::RoPENode::RoPENode(const OutputVector& args, const Config& cfg) : Op(args), m_config(cfg) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::RoPENode::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(RoPENode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::RoPENode>(new_args, m_config);
}

void ov::intel_cpu::RoPENode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(RoPENode_validate_and_infer_types);
    auto input_pshape = get_input_partial_shape(0);

    if (m_config.reshape_H) {
        auto reshape_X = input_pshape[2] / m_config.reshape_H;
        input_pshape = ov::PartialShape{input_pshape[0], input_pshape[1], m_config.reshape_H, reshape_X};
    }

    auto input_slice_size = m_config.slice_stop - m_config.slice_start;
    if (input_slice_size > 0) {
        input_pshape[3] = input_slice_size;
    }
    if (m_config.input_trans0213) {
        std::swap(input_pshape[2], input_pshape[1]);
    }
    if (m_config.output_trans0213) {
        // input is [B,L,H,S]
        // output is [B,H,L,S]
        std::swap(input_pshape[2], input_pshape[1]);
    }
    if (m_config.concat_with_past_arg_id > 0) {
        auto past_pshape = get_input_partial_shape(m_config.concat_with_past_arg_id);
        input_pshape[2] += past_pshape[2];
    }
    set_output_type(0, get_input_element_type(0), input_pshape);
}

bool ov::intel_cpu::RoPENode::visit_attributes(ngraph::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(RoPENode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("reshape_H", m_config.reshape_H);
    visitor.on_attribute("slice_start", m_config.slice_start);
    visitor.on_attribute("slice_stop", m_config.slice_stop);
    visitor.on_attribute("input_trans0213", m_config.input_trans0213);
    visitor.on_attribute("is_cos_sin_combined", m_config.is_cos_sin_combined);
    visitor.on_attribute("is_interleaved", m_config.is_interleaved);
    visitor.on_attribute("output_trans0213", m_config.output_trans0213);
    visitor.on_attribute("ndims", m_config.ndims);
    visitor.on_attribute("gather_position_arg_id", m_config.gather_position_arg_id);
    visitor.on_attribute("concat_with_past_arg_id", m_config.concat_with_past_arg_id);
    visitor.finish_structure();
    return true;
}
