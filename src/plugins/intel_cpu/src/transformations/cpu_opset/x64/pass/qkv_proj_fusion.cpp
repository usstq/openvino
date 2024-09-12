// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qkv_proj_fusion.hpp"

#include <cstdint>
#include <iostream>
#include <limits>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::gen_pattern;

ov::intel_cpu::QKVProjFusion::QKVProjFusion(bool allow_dynamic_quantization) {
    MATCHER_SCOPE(QKVProjFusion);

    auto input = makePattern("[?,?,?]");

    auto q_proj_weight_const = makePattern<opset1::Constant>({});

    auto q_proj_weight_const_i8 =
        makeConst(ov::element::i8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto q_proj_weight_f32 = makePattern<opset1::Convert>({q_proj_weight_const_i8}, {{"destination_type", "f32"}});
    auto q_proj_weight_scales_per_OC = makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), 1}), nullptr);
    auto q_proj_weight_deq =
        makePattern<opset1::Multiply>({q_proj_weight_f32, q_proj_weight_scales_per_OC}, {{"auto_broadcast", "numpy"}});

    auto q_proj_weight_cvt =
        makePattern<opset1::Convert>({q_proj_weight_const}, {{"destination_type", "f32"}});  //  [4096,4096]
    auto q_proj = makePattern<opset1::MatMul>({input, q_proj_weight_cvt | q_proj_weight_const | q_proj_weight_deq},
                                              {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,4096]
    auto result = q_proj;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto src = pattern_map.at(input);

        auto&& children = src.get_target_inputs();

        if (children.size() < 3) {
            return false;
        }

        if (!src.get_element_type().is_real()) {
            return false;
        }

        bool is_quantized_int8 = pattern_map.count(q_proj_weight_const_i8);

        if (!allow_dynamic_quantization && is_quantized_int8)
            return false;

        OutputVector args = {src};
        OutputVector deq_scales;
        OutputVector outputs;
        size_t hidden_size = 0;
        std::vector<int> proj_size;
        for (auto& child : children) {
            auto mm = dynamic_cast<opset1::MatMul*>(child.get_node());
            if (!mm) {
                // maybe a ShapeOf
                continue;
            }
            if (mm->get_transpose_a() != false || mm->get_transpose_b() != true) {
                return false;
            }

            auto mm_input1 = mm->input_value(1).get_node_shared_ptr();

            std::shared_ptr<opset1::Constant> constw;
            std::shared_ptr<opset1::Constant> deq_scale;

            if (is_quantized_int8) {
                auto deq_mul = ov::as_type_ptr<opset1::Multiply>(mm_input1);
                if (!deq_mul)
                    return false;

                auto deq_mul_in0 = deq_mul->input_value(0).get_node_shared_ptr();
                auto deq_mul_in1 = deq_mul->input_value(1).get_node_shared_ptr();

                auto cvt = ov::as_type_ptr<opset1::Convert>(deq_mul_in0);
                if (!cvt)
                    return false;

                constw = ov::as_type_ptr<opset1::Constant>(cvt->input_value(0).get_node_shared_ptr());
                if (!constw || constw->get_element_type() != ov::element::i8)
                    return false;

                deq_scale = ov::as_type_ptr<opset1::Constant>(deq_mul_in1);
                if (!deq_scale || deq_scale->get_element_type() != ov::element::f32)
                    return false;
            } else {
                constw = ov::as_type_ptr<opset1::Constant>(mm_input1);
                if (!constw) {
                    if (auto cvt = ov::as_type_ptr<opset1::Convert>(mm_input1)) {
                        constw = ov::as_type_ptr<opset1::Constant>(cvt->input_value(0).get_node_shared_ptr());
                    } else {
                        return false;
                    }
                }
                if (!constw)
                    return false;
            }

            // input feature size should be the same
            const auto& wshape = constw->get_shape();
            if (hidden_size == 0) {
                hidden_size = wshape[1];
            } else if (hidden_size != wshape[1]) {
                return false;
            }

            proj_size.push_back(wshape[0]);
            args.push_back(constw);
            deq_scales.push_back(deq_scale);
            outputs.push_back(mm->get_default_output());
        }

        // make sure just 3 projections are found
        if (outputs.size() != 3) {
            return false;
        }
        if (args.size() != 4) {
            return false;
        }
        // append dequantize scales at the end
        if (is_quantized_int8) {
            for (auto& d : deq_scales)
                args.push_back(d);
        }

        QKVProjectionNode::Config config;
        config.quantized = is_quantized_int8;
        config.hidden_size = hidden_size;

        auto old_node = root;
        auto new_node = std::make_shared<QKVProjectionNode>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({old_node}, new_node);

        // callback is for plugin implementation to check if it can be supported
        if (!transformation_callback(new_node)) {
            return false;
        }

        for (size_t i = 0; i < outputs.size(); i++) {
            auto target = outputs[i].get_node_shared_ptr();
            outputs[i].replace(new_node->output(i));
            new_node->add_node_control_dependents(target);
            new_node->add_node_control_dependencies(target);
            target->clear_control_dependents();
        }
        std::cout << "QKVProjFusion: " << __LINE__ << "  " << root->get_friendly_name()
                  << "  is_quantized_int8=" << is_quantized_int8 << std::endl;

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
