// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gelu_fusion.hpp"

#include <cstdint>
#include <iostream>
#include <limits>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::gen_pattern;

ov::intel_cpu::GELUFusion::GELUFusion() {
    MATCHER_SCOPE(GELUFusion);

    auto input = makePattern("[?,?]");
    auto ListConstruct_Concat_1 = makePattern("i32[3]");

    auto mlp_c_fcview_Reshape_1 = makePattern<opset1::Reshape>({input, ListConstruct_Concat_1}, {{"special_zero", false}});   //  tensor_array<f32[?,?,4096]>

    //auto Constant_0_5 = makePattern<opset1::Constant>({}, {}, "f32[1,1,1]"); //makeConst(element::f32, ov::Shape({1,1,1,}), {0.500000f});
    auto mlp_actmul_Multiply = makePattern<opset1::Multiply>({mlp_c_fcview_Reshape_1, 0.5f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,?,4096]>
    //auto Constant_3_0 = makePattern<opset1::Constant>({}, {}, "f32[1,1,1]"); // makeConst(element::f32, ov::Shape({1,1,1,}), {3.000000f});
    auto mlp_actpow_Power = makePattern<opset1::Power>({mlp_c_fcview_Reshape_1, 3.0f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,?,4096]>
    auto Constant_0_044708 = makePattern<opset1::Constant>({}, {}, "f32[1,1,1]"); //makeConst(element::f32, ov::Shape({1,1,1,}), {0.044708f});
    auto mlp_actmul_Multiply_1 = makePattern<opset1::Multiply>({mlp_actpow_Power, Constant_0_044708}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,?,4096]>
    auto mlp_actadd_Add = makePattern<opset1::Add>({mlp_c_fcview_Reshape_1, mlp_actmul_Multiply_1}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,?,4096]>
    auto Constant_0_797852 = makePattern<opset1::Constant>({}, {}, "f32[1,1,1]"); //makeConst(element::f32, ov::Shape({1,1,1,}), {0.797852f});
    auto mlp_actmul_Multiply_2 = makePattern<opset1::Multiply>({mlp_actadd_Add, Constant_0_797852}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,?,4096]>
    auto mlp_acttanh_Tanh = makePattern<opset1::Tanh>({mlp_actmul_Multiply_2});   //  tensor_array<f32[?,?,4096]>
    // auto Constant_1_0 = makePattern<opset1::Constant>({}, {}, "f32[1,1,1]"); // makeConst(element::f32, ov::Shape({1,1,1,}), {1.000000f});
    auto mlp_actadd_Add_1 = makePattern<opset1::Add>({mlp_acttanh_Tanh, 1.0f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,?,4096]>
    auto mlp_actmul_Multiply_3 = makePattern<opset1::Multiply>({mlp_actmul_Multiply, mlp_actadd_Add_1}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,?,4096]>

    auto result = mlp_actmul_Multiply_3;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto const1 = ov::as_type_ptr<opset1::Constant>(pattern_map.at(Constant_0_044708).get_node_shared_ptr());
        auto const2 = ov::as_type_ptr<opset1::Constant>(pattern_map.at(Constant_0_797852).get_node_shared_ptr());
        auto const_0_044708 = const1->cast_vector<float>()[0];
        auto const_0_797852 = const2->cast_vector<float>()[0];

        auto old_node = root;
        auto new_node = std::make_shared<opset7::Gelu>(pattern_map.at(input), op::GeluApproximationMode::TANH);
        new_node->set_friendly_name(old_node->get_friendly_name());

        auto new_reshape = std::make_shared<opset1::Reshape>(new_node, pattern_map.at(ListConstruct_Concat_1), false);

        ov::replace_node(old_node, new_reshape);

        std::cout << "GELUFusion: " << root->get_friendly_name() << "," << const_0_044708 << "," << const_0_797852 << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}