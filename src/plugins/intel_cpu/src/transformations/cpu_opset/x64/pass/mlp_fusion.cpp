// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlp_fusion.hpp"

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
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::gen_pattern;

ov::intel_cpu::MLPFusion::MLPFusion() {
    MATCHER_SCOPE(MLPFusion);

    auto input = makePattern("[?,?,?]");

    auto gate_proj_weight_compressed = makePattern<opset1::Constant>({});  // [up_size, down_size]
    auto gate_proj_weight = makePattern<opset1::Convert>({gate_proj_weight_compressed}, {{"destination_type", "f32"}});
    auto up_proj_weight_compressed = makePattern<opset1::Constant>({});  // [up_size, down_size]
    auto up_proj_weight = makePattern<opset1::Convert>({up_proj_weight_compressed}, {{"destination_type", "f32"}});
    auto down_proj_weight_compressed = makePattern<opset1::Constant>({});  // [down_size, up_size]
    auto down_proj_weight = makePattern<opset1::Convert>({down_proj_weight_compressed}, {{"destination_type", "f32"}});
    auto mlp_gate_proj = makePattern<opset1::MatMul>({input, gate_proj_weight | gate_proj_weight_compressed},
                                                     {{"transpose_a", false}, {"transpose_b", true}});  // [?,?,up_size]
    auto mlp_silu_gate = makePattern<opset4::Swish>({mlp_gate_proj});
    auto mlp_gelu_gate = makePattern<opset7::Gelu>({mlp_gate_proj});
    auto mlp_up_proj = makePattern<opset1::MatMul>({input, up_proj_weight | up_proj_weight_compressed},
                                                   {{"transpose_a", false}, {"transpose_b", true}});
    auto mlp_gated_up = makePattern<opset1::Multiply>({mlp_silu_gate | mlp_gelu_gate, mlp_up_proj}, {{"auto_broadcast", "numpy"}});
    auto down_proj = makePattern<opset1::MatMul>({mlp_gated_up, down_proj_weight | down_proj_weight_compressed},
                                                 {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,down_size]

    auto result = down_proj;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto gate_proj_w = pattern_map.at(gate_proj_weight_compressed);
        auto up_proj_w = pattern_map.at(up_proj_weight_compressed);
        auto down_proj_w = pattern_map.at(down_proj_weight_compressed);

        auto gate_proj_w_pshape = gate_proj_w.get_partial_shape();
        auto up_proj_w_pshape = up_proj_w.get_partial_shape();
        auto down_proj_w_pshape = down_proj_w.get_partial_shape();

        // make sure that:
        //  - shape of gate/up's weight is [down_size, up_size]
        //  - shape of down's weight is [up_size, down_size]
        if (!gate_proj_w_pshape.is_static())
            return false;
        if (!up_proj_w_pshape.is_static())
            return false;
        if (!down_proj_w_pshape.is_static())
            return false;

        auto up_shape = up_proj_w_pshape.get_shape();
        auto down_shape = down_proj_w_pshape.get_shape();

        if (gate_proj_w_pshape.get_shape() != up_shape)
            return false;
        if (up_shape.size() != 2)
            return false;
        if (down_shape.size() != 2)
            return false;

        auto up_size = up_shape[0];
        auto down_size = up_shape[1];
        if (down_shape[0] != down_size)
            return false;
        if (down_shape[1] != up_size)
            return false;

        LLMMLPNode::Config config;
        OutputVector new_args;
        std::shared_ptr<Node> gate_act;
        if (pattern_map.count(mlp_silu_gate) > 0) {
            config.act = LLMMLPNode::ACT_FN::SILU;
            gate_act = mlp_silu_gate;
        } else if (pattern_map.count(mlp_gelu_gate) > 0) {
            config.act = LLMMLPNode::ACT_FN::GELU;
            gate_act = mlp_gelu_gate;
        } else {
            return false;
        }

        new_args.push_back(pattern_map.at(input));
        new_args.push_back(gate_proj_w);
        new_args.push_back(up_proj_w);
        new_args.push_back(down_proj_w);

        auto old_node = root;
        auto new_node = std::make_shared<LLMMLPNode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(mlp_gate_proj).get_node_shared_ptr(),
                               pattern_map.at(gate_act).get_node_shared_ptr(),
                               pattern_map.at(mlp_up_proj).get_node_shared_ptr(),
                               pattern_map.at(down_proj).get_node_shared_ptr()},
                              new_node);

        // callback is for plugin implementation to check if it can be supported
        if (!transformation_callback(new_node)) {
            return false;
        }

        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::GPT2MLPFusion::GPT2MLPFusion() {
    MATCHER_SCOPE(MLPFusion);

    auto input = makePattern("[?,?,1024]"); // tensor_array<f32[?,?,1024]>
    auto mlp_c_fcview_Reshape = makePattern<opset1::Reshape>({input, {-1,1024}}, {{"special_zero", false}});   //  tensor_array<f32[?,1024]>
    auto bias_c_fc = makePattern<opset1::Constant>({}, {}, "f32[1,4096]");
    auto weight_c_fc = makePattern<opset1::Constant>({}, {}, "f16[4096,1024]");
    auto weight_c_fc_f32 = makePattern<opset1::Convert>({weight_c_fc}, {{"destination_type", "f32"}});   //  tensor_array<f32[4096,1024]>
    auto mlp_c_fcaddmm_MatMul = makePattern<opset1::MatMul>({mlp_c_fcview_Reshape, weight_c_fc_f32}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,4096]>
    auto mlp_c_fcaddmm_Add = makePattern<opset1::Add>({bias_c_fc, mlp_c_fcaddmm_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,4096]>

    auto mlp_c_fcsize_ShapeOf_1 = makePattern<opset1::ShapeOf>({input});   //  tensor_array<i32[3]>
    auto Gather_72470 = makePattern<opset8::Gather>({mlp_c_fcsize_ShapeOf_1, {0,1}, 0}, {{"batch_dims", 0}});   //  tensor_array<i32[2]>
    auto ListConstruct_Concat_1 = makePattern<opset1::Concat>({Gather_72470, {4096}}, {{"axis", 0}});   //  tensor_array<i32[3]>
    auto mlp_c_fcview_Reshape_1 = makePattern<opset1::Reshape>({mlp_c_fcaddmm_Add, ListConstruct_Concat_1}, {{"special_zero", false}});   //  tensor_array<f32[?,?,4096]>
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
    auto mlp_c_projview_Reshape = makePattern<opset1::Reshape>({mlp_actmul_Multiply_3, {-1,4096}}, {{"special_zero", false}});   //  tensor_array<f32[?,4096]>
    auto weight_c_proj = makePattern<opset1::Constant>({}, {}, "f16[1024,4096]"); //makeConst(element::f16, ov::Shape({1024,4096,}), {...});
    auto weight_c_proj_f32 = makePattern<opset1::Convert>({weight_c_proj}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,4096]>
    auto mlp_c_projaddmm_MatMul = makePattern<opset1::MatMul>({mlp_c_projview_Reshape, weight_c_proj_f32}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,1024]>
    auto bias_c_proj = makePattern<opset1::Constant>({}, {}, "f32[1,1024]");
    auto mlp_c_projaddmm_Add = makePattern<opset1::Add>({bias_c_proj, mlp_c_projaddmm_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,1024]>
    auto mlp_c_projsize_ShapeOf_1 = makePattern<opset1::ShapeOf>({mlp_actmul_Multiply_3});   //  tensor_array<i32[3]>
    auto Gather_72475 = makePattern<opset8::Gather>({mlp_c_projsize_ShapeOf_1, {0,1}, 0}, {{"batch_dims", 0}});   //  tensor_array<i32[2]>
    auto ListConstruct_Concat_2 = makePattern<opset1::Concat>({Gather_72475, {1024}}, {{"axis", 0}});   //  tensor_array<i32[3]>
    auto mlp_c_projview_Reshape_1 = makePattern<opset1::Reshape>({mlp_c_projaddmm_Add, ListConstruct_Concat_2}, {{"special_zero", false}});   //  tensor_array<f32[?,?,1024]>

    auto result = mlp_c_projview_Reshape_1;

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

        LLMMLPNode::Config config;
        OutputVector new_args;
        config.act = LLMMLPNode::ACT_FN::GPT2_GELU_NEW;

        new_args.push_back(pattern_map.at(input));
        new_args.push_back(pattern_map.at(weight_c_fc));
        new_args.push_back(pattern_map.at(bias_c_fc));
        new_args.push_back(pattern_map.at(weight_c_proj));
        new_args.push_back(pattern_map.at(bias_c_proj));


        auto old_node = root;
        auto new_node = std::make_shared<LLMMLPNode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);

        std::cout << "GPT2MLPFusion: " << root->get_friendly_name() << "," << const_0_044708 << "," << const_0_797852 << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}