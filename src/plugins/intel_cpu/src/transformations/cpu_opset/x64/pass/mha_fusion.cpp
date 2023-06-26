// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_fusion.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "simplify_fakequantize.hpp"
#include "transformations/cpu_opset/common/op/dimof.hpp"
#include "transformations/cpu_opset/common/op/vnode.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "utils/env.hpp"
#include "utils/pattern_node.hpp"

#include "itt.hpp"

// TODO: draw pattern
ov::intel_cpu::MHAFloatFusion::MHAFloatFusion() {
    MATCHER_SCOPE(MHAFloatFusion);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(transpose1, in2);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, mul);
    auto add = std::make_shared<ngraph::opset4::Add>(matmul0, in3);
    auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, in6, true);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softmax, in7, true);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2);
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(matmul1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto mul_in1 = pattern_to_output.at(in2);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        std::vector<float> mul_scales;
        if (auto mul_node = ngraph::as_type_ptr<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = ngraph::as_type_ptr<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = ngraph::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        auto matmul0_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto reshape0_node = ngraph::as_type_ptr<ngraph::opset1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node)
            return false;

        if (auto reshape_pattern = ngraph::as_type_ptr<ngraph::opset4::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0).size() != 4) {
                return false;
            }

            std::vector<int64_t> reshapeConstData = {static_cast<int64_t>(reshape0_node->get_input_shape(0)[0] *
                                                                          reshape0_node->get_input_shape(0)[1] *
                                                                          reshape0_node->get_input_shape(0)[2]),
                                                     -1};

            if (reshape_pattern->cast_vector<int64_t>() != reshapeConstData) {
                return false;
            }
        } else {
            return false;
        }

        if (auto reshape1_node = ngraph::as_type_ptr<ngraph::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0) != reshape1_node->get_output_shape(0)) {
                return false;
            }
        } else {
            return false;
        }

        auto softmax_node = ngraph::as_type_ptr<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 1)
            return false;

        auto matmul1_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape0).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::MHAFloatFusion2::MHAFloatFusion2() {
    MATCHER_SCOPE(MHAFloatFusion2);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1);
    auto add = std::make_shared<ngraph::opset4::Add>(matmul0, in3);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, transpose2);
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(matmul1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto softmax_node = ngraph::as_type_ptr<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        auto matmul1_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, std::vector<float>(), false,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
ov::intel_cpu::MHAQuantFusion::MHAQuantFusion() {
    MATCHER_SCOPE(MHAQuantFusion);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1);
    auto fakeQuantize0 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul0,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto add = std::make_shared<ngraph::opset4::Add>(fakeQuantize0, in3);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(add, in2);
    auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(mul, in6, true);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softmax, in7, true);
    auto fakeQuantize1 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({reshape1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(fakeQuantize1, transpose2);
    auto fakeQuantize2 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(fakeQuantize2, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = ngraph::as_type_ptr<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = ngraph::as_type_ptr<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = ngraph::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        std::vector<float> fq0_scale;
        auto fq0_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr());
        if (fq0_node) {
            fq0_scale = simplifyToScale(fq0_node);
            if (!fq0_scale.size())
                return false;
        }

        auto reshape0_node = ngraph::as_type_ptr<ngraph::opset1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node)
            return false;

        if (auto reshape_pattern = ngraph::as_type_ptr<ngraph::opset4::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0).size() != 4) {
                return false;
            }

            std::vector<int64_t> reshapeConstData = {static_cast<int64_t>(reshape0_node->get_input_shape(0)[0] *
                                                                          reshape0_node->get_input_shape(0)[1] *
                                                                          reshape0_node->get_input_shape(0)[2]),
                                                     -1};

            if (reshape_pattern->cast_vector<int64_t>() != reshapeConstData) {
                return false;
            }
        } else {
            return false;
        }

        if (auto reshape1_node = ngraph::as_type_ptr<ngraph::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0) != reshape1_node->get_output_shape(0)) {
                return false;
            }
        } else {
            return false;
        }

        auto softmax_node = ngraph::as_type_ptr<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 1)
            return false;

        std::vector<float> fq1_scale;
        auto fq1_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr());
        if (fq1_node) {
            fq1_scale = simplifyToScale(fq1_node);
            if (!fq1_scale.size())
                return false;
        } else {
            return false;
        }

        auto matmul1_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        std::vector<float> fq2_scale;
        if (auto fq_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize2).get_node_shared_ptr())) {
            fq2_scale = simplifyToScale(fq_node);
            if (!fq2_scale.size())
                return false;
        }

        bool is_mul_first = false;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            std::vector<float>(), fq0_scale, fq1_scale, fq2_scale,
                                                            ngraph::element::undefined,
                                                            fq0_node ? fq0_node->get_output_element_type(0) : ngraph::element::undefined,
                                                            fq1_node->get_output_element_type(0), transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape0).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize2).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
ov::intel_cpu::MHAQuantFusion2::MHAQuantFusion2() {
    MATCHER_SCOPE(MHAQuantFusion2);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto fakeQuantize0 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({transpose1,
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, fakeQuantize0);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(matmul0, in2);
    auto add = std::make_shared<ngraph::opset4::Add>(mul, in3);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, transpose2);
    auto fakeQuantize1 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(fakeQuantize1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = ngraph::as_type_ptr<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = ngraph::as_type_ptr<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = ngraph::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        std::vector<float> fq0_scale;
        auto fq0_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr());
        if (fq0_node) {
            fq0_scale = simplifyToScale(fq0_node);
            if (!fq0_scale.size())
                return false;
        } else {
            return false;
        }

        auto softmax_node = ngraph::as_type_ptr<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        std::vector<float> fq1_scale;
        if (auto fq_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr())) {
            fq1_scale = simplifyToScale(fq_node);
            if (!fq1_scale.size())
                return false;
        }

        auto matmul1_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            fq0_scale, std::vector<float>(), std::vector<float>(), fq1_scale,
                                                            fq0_node->get_output_element_type(0), ngraph::element::undefined, ngraph::element::undefined,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

namespace ov {
namespace intel_cpu {

#if 0

struct PatternRoPE {
    // position_ids [bs,1,seq_len,1] : position id of each token in inputs
    // when postion ids are continous, the RoPE can be performed very efficiently
    std::shared_ptr<ov::Node> input_ids, past_key_values_key;

    // cos/sin table
    std::shared_ptr<ov::Node> rope_cos_tab, rope_sin_tab;

    // the query_key_value projection from current embedding of input tokens
    std::shared_ptr<ov::Node> rope_Reshape;
    // the past keys from past_key_values
    std::shared_ptr<ov::Node> past_key;

    // result
    std::shared_ptr<ov::Node> present_key, rope_query;

    PatternRoPE() {
        input_ids = GenInput("i32[?,?]");
        rope_cos_tab = GenInput("f32[1,1,2048,16]");
        rope_sin_tab = GenInput("f32[1,1,2048,16]");
        past_key = GenInput("f32[?,8,?,64]");
        rope_Reshape = GenInput("f32[?,?,8,192]");
        past_key_values_key = GenInput("f32[?,8,?,64]");

        range_start = GenInput("i32[]");
        range_stop = GenInput("i32[]");

        auto past_key_length1 =
                GenPattern<DimOfNode>({past_key_values_key}, "i32[]", {{"axis", -2}, {"output_scalar", 1}});
        auto input_ids_seq_len0 = GenPattern<DimOfNode>({input_ids}, "i32[]", {{"axis", 1}, {"output_scalar", 1}});
        auto gpt_neox_Add =
            GenPattern<opset1::Add>({input_ids_seq_len0, past_key_length1}, "i32[]", {{"auto_broadcast", "numpy"}});

        auto gpt_neox_Range =
            GenPattern<opset4::Range>({range_start, range_stop, 1}, "i32[?]", {{"output_type", "i32"}});
        auto gpt_neox_Unsqueeze =
            GenPattern<opset1::Reshape>({gpt_neox_Range, {1, -1}}, "i32[1,?]", {{"special_zero", 0}});
        auto input_ids_seq_len = GenPattern<DimOfNode>({input_ids}, "i32[1]", {{"axis", 1}, {"output_scalar", 0}});
        auto gpt_neox_Concat = GenPattern<opset1::Concat>({{-1}, input_ids_seq_len}, "i32[2]", {{"axis", 0}});
        auto gpt_neox_Reshape =
            GenPattern<opset1::Reshape>({gpt_neox_Unsqueeze, gpt_neox_Concat}, "i32[?,?]", {{"special_zero", 1}});
        auto rope_Unsqueeze_3 = GenPattern<opset1::Unsqueeze>({gpt_neox_Reshape, {1, 3}}, "i32[?,1,?,1]");

        // cur key projection
        auto rope_Slice_1 = GenPattern<opset8::Slice>({rope_Reshape, {64}, {128}, {1}, {3}}, "f32[?,?,8,64]");
        auto rope_key_Transpose_1 = GenPattern<opset1::Transpose>({rope_Slice_1, {0, 2, 1, 3}}, "f32[?,8,?,64]");
        auto rope_Slice_5 = GenPattern<opset8::Slice>({rope_key_Transpose_1, {0}, {16}, {1}, {3}}, "f32[?,8,?,16]");

        auto rope_key_length =
            GenPattern<DimOfNode>({rope_key_Transpose_1}, "i32[1]", {{"axis", 2}, {"output_scalar", 0}});
        auto past_key_length = GenPattern<DimOfNode>({past_key}, "i32[1]", {{"axis", 2}, {"output_scalar", 0}});
        auto rope_Add =
            GenPattern<opset1::Add>({rope_key_length, past_key_length}, "i32[1]", {{"auto_broadcast", "numpy"}});
        auto rope_cos_tab_slice =
            GenPattern<opset8::Slice>({rope_cos_tab, {0}, rope_Add, {1}, {0}}, "f32[..1,1,2048,16]");

        auto Constant_46597 = GenConst({1}, "i32[1,1,1,1]");
        auto rope_Expand = GenPattern<opset1::Multiply>({rope_Unsqueeze_3, Constant_46597},
                                                        "i32[?,1,?,1]",
                                                        {{"auto_broadcast", "numpy"}});
        auto rope_Tile = GenPattern<opset1::Tile>({rope_Expand, {1, 1, 1, 16}}, "i32[?,1,?,16]");
        auto rope_Tile_dimof0 = GenPattern<DimOfNode>({rope_Tile}, "i32[1]", {{"axis", 0}, {"output_scalar", 0}});

        auto rope_Concat_4 = GenPattern<opset1::Concat>({rope_Tile_dimof0, {1}, {1}, {1}}, "i32[4]", {{"axis", 0}});
        auto rope_cos = GenPattern<opset1::Tile>({rope_cos_tab_slice, rope_Concat_4}, "f32[?,1,2048,16]");
        auto rope_GatherCos = GenPattern<opset6::GatherElements>({rope_cos, rope_Tile}, "f32[?,1,?,16]", {{"axis", 2}});

        auto rope_sin_tab_slice =
            GenPattern<opset8::Slice>({rope_sin_tab, {0}, rope_Add, {1}, {0}}, "f32[..1,1,2048,16]");
        auto rope_Concat_6 = GenPattern<opset1::Concat>({rope_Tile_dimof0, {1}, {1}, {1}}, "i32[4]", {{"axis", 0}});
        auto rope_sin = GenPattern<opset1::Tile>({rope_sin_tab_slice, rope_Concat_6}, "f32[?,1,2048,16]");
        auto rope_GatherSin = GenPattern<opset6::GatherElements>({rope_sin, rope_Tile}, "f32[?,1,?,16]", {{"axis", 2}});

        auto rope_Mul_2 = GenPattern<opset1::Multiply>({rope_Slice_5, rope_GatherCos},
                                                       "f32[?,8,?,16]",
                                                       {{"auto_broadcast", "numpy"}});

        // rotate_half
        auto rope_Slice_10 = GenPattern<opset8::Slice>({rope_Slice_5, {8}, {2147483647}, {1}, {3}}, "f32[?,8,?,8]");
        auto rope_Neg_1 =
            GenPattern<PowerStaticNode>({rope_Slice_10},
                                        "f32[?,8,?,8]",
                                        {{"scale", -1.0}, {"power", 1.0}, {"shift", 0.0}, {"out-type", "f32"}});
        auto rope_Slice_9 = GenPattern<opset8::Slice>({rope_Slice_5, {0}, {8}, {1}, {3}}, "f32[?,8,?,8]");
        auto rope_Concat_8 = GenPattern<opset1::Concat>({rope_Neg_1, rope_Slice_9}, "f32[?,8,?,16]", {{"axis", -1}});

        //
        auto rope_Mul_3 = GenPattern<opset1::Multiply>({rope_Concat_8, rope_GatherSin},
                                                       "f32[?,8,?,16]",
                                                       {{"auto_broadcast", "numpy"}});
        auto rope_Add_2 =
            GenPattern<opset1::Add>({rope_Mul_2, rope_Mul_3}, "f32[?,8,?,16]", {{"auto_broadcast", "numpy"}});
        auto rope_Slice_6 =
            GenPattern<opset8::Slice>({rope_key_Transpose_1, {16}, {2147483647}, {1}, {3}}, "f32[?,8,?,48]");
        auto rope_Concat_10 = GenPattern<opset1::Concat>({rope_Add_2, rope_Slice_6}, "f32[?,8,?,64]", {{"axis", -1}});

        present_key = GenPattern<opset1::Concat>({past_key, rope_Concat_10}, "f32[?,8,?,64]", {{"axis", -2}});

        // query: rope_Reshape[:, :, :, 0:64]
        auto rope_Slice = GenPattern<opset8::Slice>({rope_Reshape, {0}, {64}, {1}, {3}}, "f32[?,?,8,64]");
        auto rope_Transpose = GenPattern<opset1::Transpose>({rope_Slice, {0, 2, 1, 3}}, "f32[?,8,?,64]");
        auto rope_Slice_3 = GenPattern<opset8::Slice>({rope_Transpose, {0}, {16}, {1}, {3}}, "f32[?,8,?,16]");
        auto rope_Mul = GenPattern<opset1::Multiply>({rope_Slice_3, rope_GatherCos},
                                                     "f32[?,8,?,16]",
                                                     {{"auto_broadcast", "numpy"}});

        auto rope_Slice_8 = GenPattern<opset8::Slice>({rope_Slice_3, {8}, {2147483647}, {1}, {3}}, "f32[?,8,?,8]");
        auto rope_Neg =
            GenPattern<PowerStaticNode>({rope_Slice_8},
                                        "f32[?,8,?,8]",
                                        {{"scale", -1.0}, {"power", 1.0}, {"shift", 0.0}, {"out-type", "f32"}});
        auto rope_Slice_7 = GenPattern<opset8::Slice>({rope_Slice_3, {0}, {8}, {1}, {3}}, "f32[?,8,?,8]");
        auto rope_Concat_7 = GenPattern<opset1::Concat>({rope_Neg, rope_Slice_7}, "f32[?,8,?,16]", {{"axis", -1}});

        auto rope_Mul_1 = GenPattern<opset1::Multiply>({rope_Concat_7, rope_GatherSin},
                                                       "f32[?,8,?,16]",
                                                       {{"auto_broadcast", "numpy"}});
        auto rope_Add_1 =
            GenPattern<opset1::Add>({rope_Mul, rope_Mul_1}, "f32[?,8,?,16]", {{"auto_broadcast", "numpy"}});
        auto rope_Slice_4 = GenPattern<opset8::Slice>({rope_Transpose, {16}, {2147483647}, {1}, {3}}, "f32[?,8,?,48]");
        rope_query = GenPattern<opset1::Concat>({rope_Add_1, rope_Slice_4}, "f32[?,8,?,64]", {{"axis", -1}});
    }
};


ov::intel_cpu::RoPEFusionQuery::RoPEFusionQuery() {
    MATCHER_SCOPE(RoPEFusionQuery);
    PatternRoPE rope;
    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        // auto node_src = pattern_to_output.at(select.node).get_node_shared_ptr();
        auto root_value = m.get_match_value();
        std::cout << "RoPEFusionQuery::callback " << root_value << std::endl;
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(rope.rope_query, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::RoPEFusionKey::RoPEFusionKey() {
    MATCHER_SCOPE(RoPEFusionKey);
    PatternRoPE rope;
    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        // auto node_src = pattern_to_output.at(select.node).get_node_shared_ptr();
        auto root_value = m.get_match_value();
        std::cout << "RoPEFusionKey::callback " << root_value << std::endl;
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(rope.present_key, matcher_name);
    this->register_matcher(m, callback);
}

#endif

class VNodeIn : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("VNodeIn", "0");

    int get_all_labels(const Output<Node>& value, OutputVector& all_labels) {
        auto node = value.get_node_shared_ptr();
        if (dynamic_cast<ov::pass::pattern::op::Label*>(node.get())) {
            all_labels.push_back(node);
            return 1;
        }
        int ret = 0;
        for (int i = 0; i < node->get_input_size(); i++) {
            ret += get_all_labels(node->input_value(i), all_labels);
        }
        return ret;
    }

    template <typename F>
    VNodeIn(const char* vtype, F func) {
        MATCHER_SCOPE(VNodeIn);
        OutputVector fake_inputs;
        for (int i = 0; i < 32; i++) {
            fake_inputs.push_back(GenInput());
        }
        auto pattern_value = func(fake_inputs);
        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            // auto node_src = pattern_to_output.at(select.node).get_node_shared_ptr();
            auto root_value = m.get_match_value();
            std::cout << "VNodeIn::callback " << root_value << std::endl;

            OutputVector real_inputs;
            for (auto& in : fake_inputs) {
                auto it = pattern_to_output.find(in.get_node_shared_ptr());
                if (it == pattern_to_output.end())
                    break;
                real_inputs.push_back(it->second);
            }

            auto vnode = std::make_shared<VNode>(real_inputs, root_value, vtype);
            ngraph::replace_node(root_value.get_node_shared_ptr(), vnode);
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(pattern_value, matcher_name);
        this->register_matcher(m, callback);
    }
};

MHADynamicVNodeIn::MHADynamicVNodeIn() {
    auto gptneox_rotate_half = [](const OutputVector& inputs) {
        auto Slice_5 = inputs[0];   // f32[?,8,?,16]
        auto Slice_10 = GenPattern<opset8::Slice>({Slice_5, {8}, {2147483647}, {1}, {3}}, "f32[?,8,?,8]");
        auto Neg_1 =
            GenPattern<PowerStaticNode>({Slice_10},
                                        "f32[?,8,?,8]",
                                        {{"scale", -1.0}, {"power", 1.0}, {"shift", 0.0}, {"out-type", "f32"}});
        auto Slice_9 = GenPattern<opset8::Slice>({Slice_5, {0}, {8}, {1}, {3}}, "f32[?,8,?,8]");
        auto Concat_8 = GenPattern<opset1::Concat>({Neg_1, Slice_9}, "f32[?,8,?,16]", {{"axis", -1}});
        return Concat_8;
    };

    //add_matcher<VNodeIn>("rotate_half", gptneox_rotate_half);

    auto gptneox_rope_cos = [](const OutputVector& inputs) {
        auto input_ids = inputs[0];
        auto past_key_values_key = inputs[1];   //
        auto Slice_1 = inputs[2];               // from qkv projection
        auto rotary_emb_Constant = inputs[3];   // const table "f32[1,1,2048,16]"

        auto Transpose_1 = GenPattern<opset1::Transpose>({Slice_1, {0, 2, 1, 3}}, "f32[?,8,?,64]");

        auto key_length = GenPattern<DimOfNode>({Transpose_1}, "i32[1]", {{"axis", 2}, {"output_scalar", 0}});
        auto past_key_length1 = GenPattern<DimOfNode>({past_key_values_key}, "i32[1]", {{"axis", 2}, {"output_scalar", 0}});
        auto Add = GenPattern<opset1::Add>({key_length, past_key_length1}, "i32[1]", {{"auto_broadcast", "numpy"}});

        auto past_key_length0 = GenPattern<DimOfNode>({past_key_values_key}, "i32[]", {{"axis", -2}, {"output_scalar", 1}});
        auto input_ids_seq_len = GenPattern<DimOfNode>({input_ids}, "i32[]", {{"axis", 1}, {"output_scalar", 1}});
        auto gpt_neox_Add = GenPattern<opset1::Add>({input_ids_seq_len, past_key_length0}, "i32[]", {{"auto_broadcast", "numpy"}});

        auto gpt_neox_Range = GenPattern<opset4::Range>({past_key_length0, gpt_neox_Add, 1}, "i32[?]", {{"output_type", "i32"}});
        auto gpt_neox_Unsqueeze = GenPattern<opset1::Reshape>({gpt_neox_Range, {1, -1}}, "i32[1,?]", {{"special_zero", 0}});
        auto input_ids_seq_len1 = GenPattern<DimOfNode>({input_ids}, "i32[1]", {{"axis", 1}, {"output_scalar", 0}});
        auto gpt_neox_Concat = GenPattern<opset1::Concat>({{-1}, input_ids_seq_len1}, "i32[2]", {{"axis", 0}});
        auto gpt_neox_Reshape = GenPattern<opset1::Reshape>({gpt_neox_Unsqueeze, gpt_neox_Concat}, "i32[?,?]", {{"special_zero", 1}});
        auto Unsqueeze_3 = GenPattern<opset1::Unsqueeze>({gpt_neox_Reshape, {1, 3}}, "i32[?,1,?,1]");
        auto Constant_46597 = GenConst({1}, "i32[1,1,1,1]");
        auto Expand = GenPattern<opset1::Multiply>({Unsqueeze_3, Constant_46597}, "i32[?,1,?,1]", {{"auto_broadcast", "numpy"}});
        auto Tile = GenPattern<opset1::Tile>({Expand, {1, 1, 1, 16}}, "i32[?,1,?,16]");
        auto Tile_size0 = GenPattern<DimOfNode>({Tile}, "i32[1]", {{"axis", 0}, {"output_scalar", 0}});
        auto Concat_4 = GenPattern<opset1::Concat>({Tile_size0, {1}, {1}, {1}}, "i32[4]", {{"axis", 0}});

        auto rotary_emb_Slice = GenPattern<opset8::Slice>({rotary_emb_Constant, {0}, Add, {1}, {0}}, "f32[..1,1,2048,16]");
        auto Tile_1 = GenPattern<opset1::Tile>({rotary_emb_Slice, Concat_4}, "f32[?,1,2048,16]");

        auto GatherElements = GenPattern<opset6::GatherElements>({Tile_1, Tile}, "f32[?,1,?,16]", {{"axis", 2}});
        return GatherElements;
    };


    auto gptneox_rope_cos_sin = [](const OutputVector& inputs) {
        auto past_key_length0 = inputs[0];    // "i32[]"    range_start
        auto gpt_neox_Add = inputs[1];        // "i32[]"    range_stop
        auto input_ids = inputs[2];  // must == (range_stop - range_start)
        auto Add = inputs[3];                 // i32[1]   must be some value > 0 at runtime
        auto rotary_emb_Constant = inputs[4]; // const table "f32[1,1,2048,16]"

        //auto Transpose_1 = GenPattern<opset1::Transpose>({Slice_1, {0, 2, 1, 3}}, "f32[?,8,?,64]");

        //auto key_length = GenPattern<DimOfNode>({Transpose_1}, "i32[1]", {{"axis", 2}, {"output_scalar", 0}});
        //auto past_key_length1 = GenPattern<DimOfNode>({past_key_values_key}, "i32[1]", {{"axis", 2}, {"output_scalar", 0}});
        //auto Add = GenPattern<opset1::Add>({key_length, past_key_length1}, "i32[1]", {{"auto_broadcast", "numpy"}});

        //auto past_key_length0 = GenPattern<DimOfNode>({past_key_values_key}, "i32[]", {{"axis", -2}, {"output_scalar", 1}});
        //auto input_ids_seq_len = GenPattern<DimOfNode>({input_ids}, "i32[]", {{"axis", 1}, {"output_scalar", 1}});
        //auto gpt_neox_Add = GenPattern<opset1::Add>({input_ids_seq_len, past_key_length0}, "i32[]", {{"auto_broadcast", "numpy"}});

        auto gpt_neox_Range = GenPattern<opset4::Range>({past_key_length0, gpt_neox_Add, 1}, "i32[?]", {{"output_type", "i32"}});
        auto gpt_neox_Unsqueeze = GenPattern<opset1::Reshape>({gpt_neox_Range, {1, -1}}, "i32[1,?]", {{"special_zero", 0}});
        auto input_ids_seq_len1 = GenPattern<DimOfNode>({input_ids}, "i32[1]", {{"axis", 1}, {"output_scalar", 0}});
        auto gpt_neox_Concat = GenPattern<opset1::Concat>({{-1}, input_ids_seq_len1}, "i32[2]", {{"axis", 0}});
        auto gpt_neox_Reshape = GenPattern<opset1::Reshape>({gpt_neox_Unsqueeze, gpt_neox_Concat}, "i32[?,?]", {{"special_zero", 1}});
        auto Unsqueeze_3 = GenPattern<opset1::Unsqueeze>({gpt_neox_Reshape, {1, 3}}, "i32[?,1,?,1]");
        auto Constant_46597 = GenConst({1}, "i32[1,1,1,1]");
        auto Expand = GenPattern<opset1::Multiply>({Unsqueeze_3, Constant_46597}, "i32[?,1,?,1]", {{"auto_broadcast", "numpy"}});
        auto Tile = GenPattern<opset1::Tile>({Expand, {1, 1, 1, 16}}, "i32[?,1,?,16]");
        auto Tile_size0 = GenPattern<DimOfNode>({Tile}, "i32[1]", {{"axis", 0}, {"output_scalar", 0}});
        auto Concat_4 = GenPattern<opset1::Concat>({Tile_size0, {1}, {1}, {1}}, "i32[4]", {{"axis", 0}});

        auto rotary_emb_Slice = GenPattern<opset8::Slice>({rotary_emb_Constant, {0}, Add, {1}, {0}}, "f32[..1,1,2048,16]");
        auto Tile_1 = GenPattern<opset1::Tile>({rotary_emb_Slice, Concat_4}, "f32[?,1,2048,16]");

        auto GatherElements = GenPattern<opset6::GatherElements>({Tile_1, Tile}, "f32[?,1,?,16]", {{"axis", 2}});
        return GatherElements;
    };

    //add_matcher<VNodeIn>("rope_cos_sin", gptneox_rope_cos_sin);

    auto gptneox_rope_neox = [=](const OutputVector& inputs) {
        auto Transpose_1 = inputs[0];   // "f32[?,8,?,64]"

        auto past_key_length0 = inputs[1];    // "i32[]"    range_start
        auto gpt_neox_Add = inputs[2];        // "i32[]"    range_stop
        auto input_ids = inputs[3];
        auto Add = inputs[4];                 // i32[1]   must be some value > 0 at runtime
        auto rotary_emb_Cos = inputs[5]; // const table "f32[1,1,2048,16]"
        auto rotary_emb_Sin = inputs[6]; // const table "f32[1,1,2048,16]"

        auto cos = gptneox_rope_cos_sin({past_key_length0, gpt_neox_Add, input_ids, Add, rotary_emb_Cos});
        auto sin = gptneox_rope_cos_sin({past_key_length0, gpt_neox_Add, input_ids, Add, rotary_emb_Sin});

        auto Slice_5 = GenPattern<opset8::Slice>({Transpose_1, {0}, {16}, {1}, {3}}, "f32[?,8,?,16]");
        auto Mul_2 = GenPattern<opset1::Multiply>({Slice_5, cos}, "f32[?,8,?,16]", {{"auto_broadcast", "numpy"}});
        //auto VNode_59309 = GenPattern<VNode>({Slice_5}, "f32[?,8,?,16]", {{"vtype", "rotate_half"}});
        auto VNode_59309 = gptneox_rotate_half({Slice_5});
        auto Mul_3 = GenPattern<opset1::Multiply>({VNode_59309, sin}, "f32[?,8,?,16]", {{"auto_broadcast", "numpy"}});
        auto Add_2 = GenPattern<opset1::Add>({Mul_2, Mul_3}, "f32[?,8,?,16]", {{"auto_broadcast", "numpy"}});

        auto Slice_6 = GenPattern<opset8::Slice>({Transpose_1, {16}, {2147483647}, {1}, {3}}, "f32[?,8,?,48]");
        auto Concat_10 = GenPattern<opset1::Concat>({Add_2, Slice_6}, "f32[?,8,?,64]", {{"axis", -1}});
        return Concat_10;
    };

    add_matcher<VNodeIn>("rope_neox", gptneox_rope_neox);
}

MHADynamicVNodeOut::MHADynamicVNodeOut() {
    MATCHER_SCOPE(MHADynamicVNodeOut);
    auto vnode = ov::pass::pattern::wrap_type<VNode>();

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto root_value = m.get_match_value();
        std::cout << "MHADynamicVNodeOut::callback " << root_value << std::endl;
        auto vnode = std::dynamic_pointer_cast<VNode>(root_value.get_node_shared_ptr());

        // all nodes inside vnode may contain vnodes
        ov::NodeVector nv;
        vnode->get_internal_vnodes(nv, vnode->get_org());
        for (auto & n : nv) {
            register_new_node(n);
        }
        ngraph::replace_node(root_value.get_node_shared_ptr(), {vnode->get_org()});
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(vnode, matcher_name);
    this->register_matcher(m, callback);
}

}   // namespace intel_cpu
}   // namespace ov

ov::intel_cpu::CausalMaskFusion::CausalMaskFusion() {
    MATCHER_SCOPE(CausalMaskFusion);
    /*
    auto B = ov::Dimension::dynamic();  // batch
    auto H = ov::Dimension::dynamic();  // number of heads
    auto qL = ov::Dimension::dynamic();
    auto kL = ov::Dimension::dynamic();
    auto S = ov::Dimension::dynamic();  // head size
    PatternNode attn_scores_4d(ov::Rank(4), element::f32);  // from paramter
    PatternNode present_key(ov::PartialShape{B, H, kL, S}, element::f32);  // conact from past_key & cur
    PatternNode query(ov::PartialShape{B, H, qL, S}, element::f32);        // query

    auto query_length = query.DimOf(2, false);
    auto key_length = present_key.DimOf(-2, false);

    auto gptneox_version = [&]() {
        auto tril_bias = PatternNode::tril<element::Type_t::u8>();
        auto kL_minus_qL = key_length + (query_length * PatternNode(-1));
        auto causal_mask = tril_bias.Slice(kL_minus_qL, key_length, 1, 2).Slice(0, key_length, 1, 3);
        auto select = PatternNode::Select(causal_mask, attn_scores_4d, std::numeric_limits<float>::lowest());
        return select;
    };

    auto gptneox_causal_mask = gptneox_version();

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        //auto node_src = pattern_to_output.at(select.node).get_node_shared_ptr();
        std::cout << "CausalMaskFusion::callback " << m.get_match_value() << std::endl;
        return false;
    };
    */
    std::cout << "CausalMaskFusion=====================\n";
    auto present_key = GenInput("f32[?,?,?,?]");
    auto query = GenInput("f32[?,?,?,?]");
    auto attn_scores = GenInput("f32[?,?,?,?]");

    auto attention_bias = GenPattern<opset1::Constant>({}, "u8[1,1,?,?]");
    auto key_length =
        GenPattern<ov::intel_cpu::DimOfNode>({present_key}, "", {{"axis", -2}, {"output_scalar", 0}});
    auto query_length = GenPattern<ov::intel_cpu::DimOfNode>({query}, "", {{"axis", 2}, {"output_scalar", 0}});
    auto neg_query_length = GenPattern<opset1::Multiply>({query_length, -1}, "", {{"auto_broadcast", "numpy"}});
    auto kL_sub_qL = GenPattern<opset1::Add>({key_length, neg_query_length}, "", {{"auto_broadcast", "numpy"}});
    auto gSlice1 =
        GenPattern<opset8::Slice>({attention_bias, kL_sub_qL, key_length, {1}, {2}}, "", {});
    auto gSlice2 = GenPattern<opset8::Slice>({gSlice1, {0}, key_length, {1}, {3}}, "", {});
    auto where = GenPattern<opset1::Select>({gSlice2, attn_scores, std::numeric_limits<float>::lowest()},
                                            "f32[?,?,?,?]",
                                            {{"auto_broadcast", "numpy"}});

    auto input_ids = GenPattern<opset1::Parameter>({}, "i32[?,?]", {{"element_type", "i32"}});
    auto attention_mask = GenPattern<opset1::Parameter>({}, "i32[?,?]", {{"element_type", "i32"}});
    auto input_ids_shape0 =
        GenPattern<ov::intel_cpu::DimOfNode>({input_ids}, "i32[1]", {{"axis", 0}, {"output_scalar", 0}});
    auto gpt_neox_Concat_1 = GenPattern<opset1::Concat>({input_ids_shape0, {-1}}, "i32[2]", {{"axis", 0}});
    auto gpt_neox_Reshape_1 =
        GenPattern<opset1::Reshape>({attention_mask, gpt_neox_Concat_1}, "i32[?,?]", {{"special_zero", 1}});
    auto gpt_neox_Unsqueeze_4 = GenPattern<opset1::Unsqueeze>({gpt_neox_Reshape_1, {1, 2}}, "i32[?,1,1,?]", {});
    auto gpt_neox_Cast_2 =
        GenPattern<opset1::Convert>({gpt_neox_Unsqueeze_4}, "f32[?,1,1,?]", {{"destination_type", "f32"}});
    auto Multiply_42419 = GenPattern<ov::intel_cpu::PowerStaticNode>(
        {gpt_neox_Cast_2},
        "f32[?,1,1,?]",
        {{"scale", std::numeric_limits<float>::max()}, {"power", 1.000000}, {"shift", 0.000000}, {"out-type", "f32"}});
    auto gpt_neox_Mul = GenPattern<ov::intel_cpu::PowerStaticNode>({Multiply_42419},
                                                                  "f32[?,1,1,?]",
                                                                  {{"scale", 1.000000},
                                                                   {"power", 1.000000},
                                                                   {"shift", std::numeric_limits<float>::lowest()},
                                                                   {"out-type", "f32"}});
    auto atts_scores = GenPattern<opset1::Add>({where, gpt_neox_Mul}, "f32[?,8,?,?]", {{"auto_broadcast", "numpy"}});
    auto Softmax = GenPattern<opset1::Softmax>({atts_scores}, "f32[?,8,?,?]", {{"axis", 3}});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        // auto node_src = pattern_to_output.at(select.node).get_node_shared_ptr();
        auto root_value = m.get_match_value();
        std::cout << "CausalMaskFusion::callback " << root_value << std::endl;
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(Softmax, matcher_name);
    this->register_matcher(m, callback);
}

#if 0
ov::intel_cpu::MHADynamicFloatFusion::MHADynamicFloatFusion() {
    MATCHER_SCOPE(MHADynamicFloatFusion);
    auto B = ov::Dimension::dynamic();  // batch
    auto H = ov::Dimension::dynamic();  // number of heads
    auto qL = ov::Dimension::dynamic();
    auto kL = ov::Dimension::dynamic();
    auto S = ov::Dimension::dynamic();  // head size
    auto float_max = std::numeric_limits<float>::max();
    auto float_lowest = std::numeric_limits<float>::lowest();

    //PatternNode score(ov::PartialShape{B, H, qL, kL});                     //
    PatternNode att_mask_4d(ov::PartialShape{B, 1, 1, kL}, element::i32);  // from paramter
    PatternNode present_key(ov::PartialShape{B, 8, kL, S}, element::f32);  // conact from past_key & cur
    PatternNode query(ov::PartialShape{B, H, qL, S}, element::f32);        // query
    auto BH = query.DimOf(0, false) * 8;
    auto query_length = query.DimOf(2, false);
    auto key_length = present_key.DimOf(-2, false);

    auto tril_bias = PatternNode::tril<element::Type_t::u8>();
    auto kL_minus_qL = (key_length.DimOf(0) + (query.ShapeOf() * -1)[2]).Reshape(1);
    auto causal_mask = tril_bias.Slice(kL_minus_qL, key_length, 1, 2).Slice(0, key_length, 1, 3);


    auto key_3d = (present_key * 0.125f)._view({BH, key_length, PatternNode(64)});
    auto query_3d = query._view({BH, query_length, PatternNode(64)});

    auto attn_scores_raw = PatternNode::MatMul(query_3d, key_3d, false, true);

    query.DimOf(2, false)

    auto max_shape0 = PatternNode::Maximum(PatternNode::ConstVector({1, 1, 1}), PatternNode::Concat({BH, query_length, key_length}));
    auto max_shape1 = PatternNode::Maximum(max_shape0, PatternNode::ShapeOf(attn_scores_raw));

    auto attn_scores_3d = PatternNode::Broadcast(attn_scores_raw, max_shape1, 0)

    // apply causal & attention mask to score
    auto masked_score = PatternNode::Select(causal_mask, attn_scores_4d, float_lowest) +
                        (att_mask_4d.Convert(element::f32) * float_max + float_lowest);

    auto softmax = PatternNode::Softmax(masked_score, 3);

    std::cout << "MHADynamicFloatFusion" << std::endl;
    softmax.node->dump_all(std::cout);

    matcher_pass_callback callback = [=](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node_src = pattern_to_output.at(score.node).get_node_shared_ptr();
        auto node_softmax = pattern_to_output.at(softmax.node).get_node_shared_ptr();

        std::cout << "MHADynamicFloatFusion::callback" << std::endl;
        std::cout << "\t" << node_src << std::endl;
        std::cout << "\t" << node_softmax << std::endl;
        return false;
    };

    auto m = std::make_shared<Matcher>(softmax.node, matcher_name);
    this->register_matcher(m, callback);
}
#endif