// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "eliminate_futile_bcast.hpp"
#include <openvino/opsets/opset10.hpp>
#include "cpu_types.h"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "utils/env.hpp"
#include "utils/pattern_node.hpp"

#include "itt.hpp"

ov::intel_cpu::EnvInt OPT_ELIMINATE_BCAST("OPT_ELIMINATE_BCAST", 1);

ov::intel_cpu::EliminateFutileBcasts::EliminateFutileBcasts() {
    MATCHER_SCOPE(EliminateFutileBcasts);
    PatternNode B(ov::Rank(1));
    PatternNode M(ov::Rank(1));
    PatternNode N(ov::Rank(1));
    PatternNode Ka(ov::Rank(1));
    PatternNode Kb(ov::Rank(1));
    PatternNode a4d(ov::Rank(4));
    PatternNode b4d(ov::Rank(4));
    PatternNode bcast_axes_mapping;

    auto shape_a = PatternNode::Concat({B, M, Ka}, 0);
    auto shape_b = PatternNode::Concat({B, N, Kb}, 0);

    auto a3d = PatternNode::Reshape(a4d, shape_a);
    auto b3d = PatternNode::Reshape(b4d, shape_b);

    // this topology automatically implies Ka == Kb
    auto c3d = PatternNode::MatMul(a3d, b3d, false, true);

    auto shape_c = PatternNode::ShapeOf(c3d);

    auto shape_c_bcast = PatternNode::Concat({B, M, N}, 0);

    auto max_shape0 = PatternNode::Maximum(shape_c_bcast, PatternNode(std::vector<int>{1, 1, 1}));
    auto max_shape = PatternNode::Maximum(max_shape0, shape_c);

    auto cbcast = PatternNode::Broadcast(c3d, max_shape, bcast_axes_mapping);

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto out_matmul = pattern_to_output.at(c3d.node);
        // rmeove bcast
        std::cout << "<OPT_ELIMINATE_BCAST>: " << m.get_match_value() << std::endl;
        ngraph::replace_node(m.get_match_root(), {out_matmul});
        return true;
    };

    if (OPT_ELIMINATE_BCAST) {
        auto m = std::make_shared<ov::pass::pattern::Matcher>(cbcast.node, matcher_name);
        this->register_matcher(m, callback);
    }
}