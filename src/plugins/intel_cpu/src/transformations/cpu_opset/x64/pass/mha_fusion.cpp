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

#include <memory>
#include <openvino/opsets/opset8.hpp>
#include "ngraph/pattern/op/or.hpp"
using namespace ov::pass::pattern;
using namespace ov;

struct PatternNode;

PatternNode operator*(PatternNode lhs, PatternNode rhs);
PatternNode operator+(PatternNode lhs, PatternNode rhs);

struct PatternNode {
    std::shared_ptr<Node> node;

    operator std::shared_ptr<Node> () {
        return node;
    }

    template <class... Args>
    static std::shared_ptr<Node> node_type(const OutputVector& inputs) {
        return wrap_type<Args...>(inputs);
    }

    template <class... Args>
    static std::shared_ptr<Node> node_type(const OutputVector& inputs, const std::function<bool(const Output<Node>&)>& pred) {
        return wrap_type<Args...>(inputs, [pred](const Output<Node>& output) {
            auto ret = pred(output);
            if (!ret)
                std::cout << "   match failed at : " << output << std::endl;
            return ret;
        });
    }

    // any input with given partial shape
    PatternNode(const ov::PartialShape& pshape, const element::Type & dtype = element::undefined) {
        node = any_input([pshape, dtype](const Output<Node>& value) {
            auto ret = pshape.compatible(value.get_partial_shape());
            if (ret && dtype != element::undefined)
                ret = (value.get_element_type() == dtype);
            if (!ret)
                std::cout << "   match failed at input : " << value << std::endl;
            return ret;
        });
    }

    // implict conversion from std::shared_ptr<Node>
    PatternNode(const std::shared_ptr<Node>& node) : node(node) {}

    // constant scalar pattern with v as value, of shape [], [1], [1,1], ...
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    static std::shared_ptr<Node> ConstScalar(const T v) {
        return node_type<opset1::Constant>({}, [v](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
            auto shape = s1->get_output_shape(0);
            if (shape_size(shape) > 1)
                return false;
            std::vector<T> output_vector = s1->cast_vector<T>();
            return (v == output_vector[0]);
        });
    }

    PatternNode(const int v) {
        node = ConstScalar(v);
    }

    PatternNode(const float v) {
        node = ConstScalar(v);
    }

    PatternNode shape() const {
        return node_type<opset1::ShapeOf>({node});
    }
    // 1D indexing mapped to Gather from axis 0, node is producer of shape vector
    PatternNode operator[](int i) const {
        return node_type<opset8::Gather>({node, PatternNode(i).node, PatternNode(0).node});
    }

    PatternNode Convert(const element::Type& dst_type) const {
        return node_type<opset1::Convert>({node->output(0)}, [dst_type](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::Convert>(value.get_node_shared_ptr());
            return s1->get_convert_element_type() == dst_type;
        });
    }

    PatternNode Softmax(int axis) const {
        return node_type<opset1::Softmax, opset8::Softmax>({node->output(0)}, [axis](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::Softmax>(value.get_node_shared_ptr());
            if (s1)
                return s1->get_axis() == axis;
            auto s8 = as_type_ptr<opset8::Softmax>(value.get_node_shared_ptr());
            if (s8)
                return s8->get_axis() == axis;
            return false;
        });
    }

    PatternNode Slice(const PatternNode& start,
                      const PatternNode& stop,
                      const PatternNode& step,
                      const PatternNode& axis) {
        return node_type<opset8::Slice>({start.node, stop.node, step.node, axis.node});
    }

    PatternNode Select(const PatternNode& n_then, const PatternNode& n_else) {
        return node_type<opset1::Select>({node, n_then.node, n_else.node});
    }

    // lower triangular part of a matrix
    template <element::Type_t ET>
    static PatternNode tril() {
        using T = typename element_type_traits<ET>::value_type;
        return node_type<opset1::Constant>({}, [](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
            if (s1->get_output_element_type(0) != element::u8)
                return false;
            auto shape = s1->get_output_shape(0);
            if (shape.size() != 4)
                return false;
            if (shape[0] != 1 || shape[1] != 1 || shape[2] != shape[3])
                return false;
            // NxN const matrix
            auto max_positions = shape[2];
            std::vector<T> output_vector = s1->cast_vector<T>();
            // check if it's unit lower triangular matrix
            for (int i = 0; i < max_positions; i++) {
                for (int j = 0; j < max_positions; j++) {
                    if (static_cast<bool>(output_vector[i * max_positions + j]) != static_cast<bool>(j <= i))
                        return false;
                }
            }
            return true;
        });
    }
};

PatternNode operator*(PatternNode lhs, PatternNode rhs) {
    auto exchangable = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{PatternNode::node_type<opset1::Multiply>({lhs.node, rhs.node})->output(0),
                     PatternNode::node_type<opset1::Multiply>({rhs.node, lhs.node})->output(0)});
    return std::dynamic_pointer_cast<Node>(exchangable);
}

PatternNode operator+(PatternNode lhs, PatternNode rhs) {
    auto exchangable = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{PatternNode::node_type<opset1::Add>({lhs.node, rhs.node})->output(0),
                     PatternNode::node_type<opset1::Add>({rhs.node, lhs.node})->output(0)});
    return std::dynamic_pointer_cast<Node>(exchangable);
}

ov::intel_cpu::MHADynamicFloatFusion::MHADynamicFloatFusion() {
    MATCHER_SCOPE(MHADynamicFloatFusion);
    auto dyn = ov::Dimension::dynamic();
    auto B = ov::Dimension::dynamic();
    auto H = ov::Dimension::dynamic();
    auto qL = ov::Dimension::dynamic();
    auto kL = ov::Dimension::dynamic();

    PatternNode src(ov::PartialShape{B, H, qL, kL});
    PatternNode att_mask_in(ov::PartialShape{B, 1, 1, kL}, element::i32);

    auto float_max = std::numeric_limits<float>::max();
    auto float_lowest = std::numeric_limits<float>::lowest();

    PatternNode key_length(ov::PartialShape{1});
    PatternNode start(ov::PartialShape{1});

    auto tril_bias = PatternNode::tril<element::Type_t::u8>();
    auto causal_mask = tril_bias.Slice(start, key_length, 1, 2).Slice(0, key_length, 1, 3);

    auto add_att_mask =
        causal_mask.Select(src, float_lowest) + (att_mask_in.Convert(element::f32) * float_max + float_lowest);

    // auto softmax = node_pattern::Softmax(add_att_mask, 3);
    auto softmax = add_att_mask.Softmax(3);

    std::cout << "MHADynamicFloatFusion" << std::endl;
    matcher_pass_callback callback = [=](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node_src = pattern_to_output.at(src.node).get_node_shared_ptr();
        auto node_softmax = pattern_to_output.at(softmax.node).get_node_shared_ptr();

        std::cout << "MHADynamicFloatFusion::callback" << std::endl;
        std::cout << "\t" << node_src << std::endl;
        std::cout << "\t" << node_softmax << std::endl;
        return false;
    };

    auto m = std::make_shared<Matcher>(softmax.node, matcher_name);
    this->register_matcher(m, callback);
}