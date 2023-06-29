// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eliminate_futile_bcast.hpp"

#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "cpu_types.h"
#include "itt.hpp"
#include "utils/env.hpp"
#include "utils/pattern_node.hpp"

ov::intel_cpu::EnvInt OPT_ELIMINATE_BCAST("OPT_ELIMINATE_BCAST", 1);

ov::intel_cpu::EliminateFutileBcasts::EliminateFutileBcasts() {
    MATCHER_SCOPE(EliminateFutileBcasts);
    auto M = GenInput("i32[1]");
    auto N = GenInput("i32[1]");
    auto B = GenInput("i32[1]");
    auto Ka = GenInput("i32[1]");
    auto Kb = GenInput("i32[1]");
    auto mA4d = GenInput("f32[?,?,?,?]");
    auto mB4d = GenInput("f32[?,?,?,?]");

    auto shapeA = GenPattern<opset1::Concat>({B, M, Kb}, "i32[3]", {{"axis", 0}});
    auto mA = GenPattern<opset1::Reshape>({mA4d, shapeA}, "", {{"special_zero", 1}});

    auto shapeB = GenPattern<opset1::Concat>({B, N, Ka}, "i32[3]", {{"axis", 0}});
    auto mB = GenPattern<opset1::Reshape>({mB4d, shapeB}, "", {{"special_zero", 1}});

    auto mC = GenPattern<opset1::MatMul>({mA, mB}, "", {{"transpose_a", 0}, {"transpose_b", 1}});
    auto shapeC = GenPattern<opset1::ShapeOf>({mC}, "i32[3]", {});
    auto bcst_Shape = GenPattern<opset1::Concat>({B, M, N}, "i32[3]", {{"axis", 0}});
    auto maxShape1 = GenPattern<opset1::Maximum>({{1, 1, 1}, bcst_Shape}, "i32[3]", {{"auto_broadcast", "numpy"}});
    auto maxShape2 = GenPattern<opset1::Maximum>({shapeC, maxShape1}, "i32[3]", {{"auto_broadcast", "numpy"}});
    auto zero = GenConst({0});
    auto bcst_Result = GenPattern<opset1::Broadcast>({mC, maxShape2, zero}, "f32[?,?,?]", {{"mode", "numpy"}});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto out_matmul = pattern_to_output.at(mC);
        // rmeove bcast
        std::cout << "<OPT_ELIMINATE_BCAST>: " << m.get_match_value() << std::endl;
        ngraph::replace_node(m.get_match_root(), {out_matmul});
        return true;
    };

    if (OPT_ELIMINATE_BCAST) {
        auto m = std::make_shared<ov::pass::pattern::Matcher>(bcst_Result, matcher_name);
        this->register_matcher(m, callback);
    }
}