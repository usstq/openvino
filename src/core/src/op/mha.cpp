// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mha.hpp"

#include "itt.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {

MultiHeadAttention::MultiHeadAttention(const ov::OutputVector &args, Config cfg) : Op({args}), m_config(cfg) {
    constructor_validate_and_infer_types();
}

void MultiHeadAttention::validate_and_infer_types() {
    // [B,L,H*S] / [B,L,H*3*S]
    auto qkv_pshape = get_input_partial_shape(5);

    // output is always [B, L, n_hidden]
    ov::PartialShape output_pshape{qkv_pshape[0], qkv_pshape[1], m_config.n_hidden};

    set_output_type(0, get_input_element_type(5), output_pshape);
}

std::shared_ptr<ov::Node> MultiHeadAttention::clone_with_new_inputs(const ov::OutputVector &new_args) const {
    return std::make_shared<MultiHeadAttention>(new_args, m_config);
}

bool MultiHeadAttention::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("rotary_dims", m_config.rotary_dims);
    visitor.on_attribute("layer_id", m_config.layer_id);    
    visitor.on_attribute("n_hidden", m_config.n_hidden);
    visitor.on_attribute("n_head", m_config.n_head);
    visitor.on_attribute("num_kv_heads", m_config.num_kv_heads);
    visitor.on_attribute("rope_type", m_config.rope_type);
    visitor.on_attribute("multi_query_is_planar", m_config.multi_query_is_planar);
    return true;
}

bool MultiHeadAttention::has_evaluate() const { return false; }

}  // namespace internal
}  // namespace op
}  // namespace ov
