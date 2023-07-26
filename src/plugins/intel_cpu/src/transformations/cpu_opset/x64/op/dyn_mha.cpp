// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dyn_mha.hpp"

#include "openvino/core/graph_util.hpp"
#include "transformations/itt.hpp"
#include "utils/debug_capabilities.h"
namespace ov {
namespace intel_cpu {

MHADynamic::MHADynamic(const OutputVector& arguments, const Config& cfg) : Op(arguments), m_cfg(cfg) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> MHADynamic::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(MHADynamic_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<MHADynamic>(new_args, m_cfg);
}

void MHADynamic::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(MHADynamic_validate_and_infer_types);

    auto query_type = get_input_element_type(0);

    auto q_shape = static_cast<std::vector<Dimension>>(get_input_partial_shape(0));
    for (auto& layout : m_cfg.relayout_query) {
        q_shape = layout.apply(q_shape);
    }

    // after gone through a re-layout process, query will be of shape [B, H, qL, S]
    auto& batch_size = q_shape[0];
    auto& query_len = q_shape[2];

    std::vector<ov::Dimension> result_shape;

    if (m_cfg.result_is_3d) {
        result_shape = {batch_size * m_cfg.head_count, query_len, m_cfg.head_size};
    } else {
        if (m_cfg.trans_result_to_BLHS) {
            result_shape = {batch_size, query_len, m_cfg.head_count, m_cfg.head_size};
        } else {
            result_shape = {batch_size, m_cfg.head_count, query_len, m_cfg.head_size};
        }
    }

    set_output_type(0, query_type, result_shape);
}

bool MHADynamic::visit_attributes(ngraph::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(MHADynamic_visit_attributes);
    visitor.on_attribute("head_count", m_cfg.head_count);
    visitor.on_attribute("head_size", m_cfg.head_size);
    visitor.on_attribute("with_qk_scale", m_cfg.with_qk_scale);
    visitor.on_attribute("with_alibi_mask", m_cfg.with_alibi_mask);
    visitor.on_attribute("with_causal_mask", m_cfg.with_causal_mask);
    visitor.on_attribute("with_causal_mask_slice1", m_cfg.with_causal_mask_slice1);
    visitor.on_attribute("with_causal_mask_slice2", m_cfg.with_causal_mask_slice2);
    visitor.on_attribute("with_causal_mask_stridedslice1", m_cfg.with_causal_mask_stridedslice1);
    visitor.on_attribute("with_causal_mask_stridedslice2", m_cfg.with_causal_mask_stridedslice2);
    visitor.on_attribute("select_nfltmax_at_0", m_cfg.select_nfltmax_at_0);
    visitor.on_attribute("with_attn_mask", m_cfg.with_attn_mask);
    visitor.on_attribute("with_attn_mask_reshapes", m_cfg.with_attn_mask_reshapes);
    visitor.on_attribute("trans_result_to_BLHS", m_cfg.trans_result_to_BLHS);
    visitor.on_attribute("q_reshape_to_3d", m_cfg.q_reshape_to_3d);
    visitor.on_attribute("k_reshape_to_3d", m_cfg.k_reshape_to_3d);
    visitor.on_attribute("v_reshape_to_3d", m_cfg.v_reshape_to_3d);
    visitor.on_attribute("result_is_3d", m_cfg.result_is_3d);

    visitor.start_structure("relayout_query");
    for (auto& layout : m_cfg.relayout_query) {
        visitor.on_attribute("do_reshape", layout.do_reshape);
        visitor.on_attribute("do_transpose", layout.do_transpose);
        visitor.on_attribute("do_gather", layout.do_gather);
        visitor.on_attribute("param", layout.param);
    }
    visitor.finish_structure();
    visitor.start_structure("relayout_key");
    for (auto& layout : m_cfg.relayout_key) {
        visitor.on_attribute("do_reshape", layout.do_reshape);
        visitor.on_attribute("do_transpose", layout.do_transpose);
        visitor.on_attribute("do_gather", layout.do_gather);
        visitor.on_attribute("param", layout.param);
    }
    visitor.finish_structure();
    visitor.start_structure("relayout_value");
    for (auto& layout : m_cfg.relayout_value) {
        visitor.on_attribute("do_reshape", layout.do_reshape);
        visitor.on_attribute("do_transpose", layout.do_transpose);
        visitor.on_attribute("do_gather", layout.do_gather);
        visitor.on_attribute("param", layout.param);
    }
    visitor.finish_structure();

    return true;
}

}  // namespace intel_cpu
}  // namespace ov
