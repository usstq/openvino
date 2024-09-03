// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <openvino/op/op.hpp>

namespace ov {
namespace op {
namespace v15 {

class OPENVINO_API MultiHeadAttention : public ov::op::Op {
public:
    OPENVINO_OP("MultiHeadAttention", "opset15");

    MultiHeadAttention() = default;

    struct Config {
        int rotary_dims = 0;                  //
        int layer_id = 0;                     //
        int n_hidden = 0;
        int n_head = 0;
        int num_kv_heads = 0;
        int rope_type = 0;                    // 0: gptj style, 1: gptneox style
        int multi_query_is_planar = 0;        // chatglm2 is true, others are false
    };
    MultiHeadAttention(const ov::OutputVector &args, Config cfg);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector &new_args) const override;
    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    bool has_evaluate() const override;

    const Config& get_config() const {
        return m_config;
    }

private:
    Config m_config;
};
}  // namespace internal
}  // namespace op
}  // namespace ov
