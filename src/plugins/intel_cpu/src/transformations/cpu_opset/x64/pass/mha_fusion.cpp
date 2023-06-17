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
#include "ngraph/pattern/op/pattern.hpp"
using namespace ov::pass::pattern;
using namespace ov;
using ov::pass::pattern::op::ValuePredicate;
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

    // 1d const tensor or scalar
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    static std::shared_ptr<Node> ConstVector(const std::vector<T>& vec, bool must_be_scalar = false) {
        auto pred = [vec, must_be_scalar](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
            auto shape = s1->get_output_shape(0);
            if (must_be_scalar && shape.size() != 0)
                return false;
            if (shape.size() > 1)
                return false;
            if (shape_size(shape) != vec.size())
                return false;
            std::vector<T> actual = s1->cast_vector<T>();
            return actual == vec;
        };
        return node_type<opset1::Constant>({}, pred);
    }

    // constant scalar pattern with v as value, of shape [], [1], [1,1], ...
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    static std::shared_ptr<Node> ConstScalar(const T v) {
        return ConstVector(std::vector<T>{v}, true);
    }

    PatternNode(const int v) {
        node = ConstScalar(v);
    }

    PatternNode(const float v) {
        node = ConstScalar(v);
    }

    PatternNode ShapeOf() const {
        return node_type<opset1::ShapeOf>({node});
    }

    PatternNode DimOf(int axis, bool output_scalar) const {
        return node_type<ov::intel_cpu::DimOfNode>({node}, [axis, output_scalar](const Output<Node>& value) {
            auto s1 = as_type_ptr<ov::intel_cpu::DimOfNode>(value.get_node_shared_ptr());
            return s1->get_axis() == axis && s1->get_output_scalar() == output_scalar;
        });
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
    return PatternNode::node_type<opset1::Multiply>({lhs.node, rhs.node});
}

PatternNode operator+(PatternNode lhs, PatternNode rhs) {
    return PatternNode::node_type<opset1::Add>({lhs.node, rhs.node});
}

/*
	f32_[?,8,?,64]  present.0.key = opset1::Concat(past_key_values.0.key,/gpt_neox/layers.0/attention/Concat_10) 	 attrs: axis=-2
	f32_[?,8,?,64]  query = opset1::Concat(/gpt_neox/layers.0/attention/Add_1,/gpt_neox/layers.0/attention/Slice_4) 	 attrs: axis=-1

==============================================================================
    i32_[]  query_batch_size = cpu_plugin_opset::DimOf(query) 	 attrs: axis=0 output_scalar=1
	i32_[]  batch_size_heads = opset1::Multiply(query_batch_size,i32(8)) 	 attrs: auto_broadcast=numpy
	i32_[1]  batch_size_heads_1d = opset1::Reshape(batch_size_heads,i32([1])) 	 attrs: special_zero=0
	i32_[1]  query_length = cpu_plugin_opset::DimOf(query) 	 attrs: axis=2 output_scalar=0
	i32_[3]  query_shape3d = opset1::Concat(batch_size_heads_1d,query_length,i32([64])) 	 attrs: axis=0
	f32_[?,?,?]  query3d = opset1::Reshape(query,query_shape3d) 	 attrs: special_zero=1
	f32_[?,8,?,64]  key_div_sqrtd = cpu_plugin_opset::PowerStatic(present.0.key) 	 attrs: scale=0.125000 power=1.000000 shift=0.000000 out-type=f32
	i32_[3]  key_shape3d = opset1::Concat(batch_size_heads_1d,key_length,i32([64])) 	 attrs: axis=0
	f32_[?,?,?]  key3d = opset1::Reshape(key_div_sqrtd,key_shape3d) 	 attrs: special_zero=1
==============================================================================
	f32_[?,?,?]  attn_scores2 = opset1::MatMul(query3d,key3d) 	 attrs: transpose_a=0 transpose_b=1
==============================================================================
	i32_[3]  ShapeOf_24954 = opset1::ShapeOf(attn_scores2) 	 attrs: type_relax=1 input_data_types=[] output_data_types=[i32]
	i32_[1]  query_length = cpu_plugin_opset::DimOf(query) 	 attrs: axis=2 output_scalar=0
	i32_[3]  attn_scores_shape3d = opset1::Concat(batch_size_heads_1d,query_length,key_length) 	 attrs: axis=0
	i32_[3]  Maximum_24821 = opset1::Maximum(i32([1,1,1]),attn_scores_shape3d) 	 attrs: auto_broadcast=numpy
	i32_[3]  Maximum_24955 = opset1::Maximum(ShapeOf_24954,Maximum_24821) 	 attrs: auto_broadcast=numpy
	f32_[?,?,?]  attn_scores1 = opset1::Broadcast(attn_scores2,Maximum_24955,u8(0)) 	 attrs: mode=numpy

	i32_[1]  query_batch_size = cpu_plugin_opset::DimOf(query) 	 attrs: axis=0 output_scalar=0
	i32_[1]  query_length = cpu_plugin_opset::DimOf(query) 	 attrs: axis=2 output_scalar=0
	i32_[4]  attn_scores_shape4d = opset1::Concat(  query_batch_size,
                                                    i32([8]),
                                                    query_length,
                                                    key_length) 	 attrs: axis=0
	f32_[?,?,?,?]  attn_scores0 = opset1::Reshape(  attn_scores1,
                                                    attn_scores_shape4d) 	 attrs: special_zero=1

==============================================================================
	i32_[1]  key_length = cpu_plugin_opset::DimOf(present.0.key) 	 attrs: axis=-2 output_scalar=0
    i32_[]  key_length_scalar = opset1::Squeeze(key_length,i32([0])) 	 attrs:
	i32_[]  query_length = cpu_plugin_opset::DimOf(query) 	 attrs: axis=2 output_scalar=1
	i32_[]  Multiply_59211 = opset1::Multiply(query_length,i32(-1)) 	 attrs: auto_broadcast=numpy
	i32_[]  /gpt_neox/layers.0/attention/Sub = opset1::Add(key_length_scalar,Multiply_59211) 	 attrs: auto_broadcast=numpy
	i32_[1]  kL_sub_qL = opset1::Reshape(/gpt_neox/layers.0/attention/Sub,i32([1])) 	 attrs: special_zero=0
	i32_[1]  key_length = cpu_plugin_opset::DimOf(present.0.key) 	 attrs: axis=-2 output_scalar=0
	u8_[1,1,..2048,2048]  bias_slice1 = opset8::Slice(
                                                gpt_neox.layers.1.attention.bias,
                                                kL_sub_qL,
                                                key_length,
                                                i32([1]),
                                                i32([2])) 	 attrs:
	u8_[1,1,..2048,..2048]  causal_mask = opset8::Slice(
                                                bias_slice1,
                                                i32([0]),
                                                key_length,
                                                i32([1]),
                                                i32([3])) 	 attrs:
==============================================================================
	f32_[?,?,?,?]  after_causal_mask = opset1::Select(  causal_mask,
                                                        attn_scores0,
                                                        f32(-3.40282e+38)) 	 attrs: type_relax=1 input_data_types=[boolean] output_data_types=[] auto_broadcast=numpy


==============================================================================
	i32_[1]  batch_size = cpu_plugin_opset::DimOf(input_ids) 	 attrs: axis=0 output_scalar=0
	i32_[2]  /gpt_neox/Concat_1 = opset1::Concat(batch_size,i32([-1])) 	 attrs: axis=0
	i32_[?,?]  /gpt_neox/Reshape_1 = opset1::Reshape(attention_mask,/gpt_neox/Concat_1) 	 attrs: special_zero=1
	i32_[?,1,1,?]  /gpt_neox/Unsqueeze_4 = opset1::Unsqueeze(/gpt_neox/Reshape_1,i32([1,2])) 	 attrs:
	f32_[?,1,1,?]  /gpt_neox/Cast_2 = opset1::Convert(/gpt_neox/Unsqueeze_4) 	 attrs: destination_type=f32
	f32_[?,1,1,?]  Multiply_42419 = cpu_plugin_opset::PowerStatic(/gpt_neox/Cast_2) 	 attrs: scale=340282346638528859811704183484516925440.000000 power=1.000000 shift=0.000000 out-type=f32
	f32_[?,1,1,?]  /gpt_neox/Mul = cpu_plugin_opset::PowerStatic(Multiply_42419) 	 attrs: scale=1.000000 power=1.000000 shift=-340282346638528859811704183484516925440.000000 out-type=f32
==============================================================================

	f32_[?,?,?,?]  after_attn_mask = opset1::Add(
                                        after_causal_mask,
                                        /gpt_neox/Mul) 	 attrs: auto_broadcast=numpy
	f32_[?,?,?,?]  /gpt_neox/layers.0/attention/Softmax = opset1::Softmax(after_attn_mask) 	 attrs: axis=3
==============================================================================
*/
ov::intel_cpu::MHADynamicFloatFusion::MHADynamicFloatFusion() {
    MATCHER_SCOPE(MHADynamicFloatFusion);
#if 0
    auto B = ov::Dimension::dynamic();  // batch
    auto H = ov::Dimension::dynamic();  // number of heads
    auto qL = ov::Dimension::dynamic();
    auto kL = ov::Dimension::dynamic();
    auto S = ov::Dimension::dynamic();  // head size
    auto float_max = std::numeric_limits<float>::max();
    auto float_lowest = std::numeric_limits<float>::lowest();

    //PatternNode score(ov::PartialShape{B, H, qL, kL});                     //
    PatternNode att_mask_in(ov::PartialShape{B, 1, 1, kL}, element::i32);  // from paramter
    PatternNode present_key(ov::PartialShape{B, 8, kL, S}, element::f32);  // conact from past_key & cur
    PatternNode query(ov::PartialShape{B, H, qL, S}, element::f32);        // query

    auto tril_bias = PatternNode::tril<element::Type_t::u8>();
    auto key_length = present_key.DimOf(-2, false);
    auto kL_minus_qL = (key_length.DimOf(0) + (query.ShapeOf() * -1)[2]).Reshape(1);
    auto causal_mask = tril_bias.Slice(kL_minus_qL, key_length, 1, 2).Slice(0, key_length, 1, 3);

    //
    // auto query_reshaped = query.Reshape();

    /*
    f32_[?,8,?,64]  Multiply_52958 = opset1::Multiply(present.5.key,Constant_52957) 	 attrs: auto_broadcast=numpy

    // rotery embedding
        f32_[?,8,?,64]  /gpt_neox/layers.5/attention/Concat_9 = opset1::Concat(
                /gpt_neox/layers.5/attention/Add_1,
                /gpt_neox/layers.5/attention/Slice_4) 	 attrs: axis=-1

    //query_shape
    i32_[4]  /gpt_neox/layers.5/attention/Shape_14 = opset1::ShapeOf(/gpt_neox/layers.5/attention/Concat_9) 	 attrs:
    type_relax=1 input_data_types=[] output_data_types=[i32]

        i32_[1]  /gpt_neox/layers.5/attention/Gather_9 = opset8::Gather(
                    /gpt_neox/layers.5/attention/Shape_14, 0, 0) 	 attrs: batch_dims=0
        i32_[]  /gpt_neox/layers.5/attention/Gather_10 = opset1::Constant(8) 	 attrs: element_type=i32 shape=[]
    value=?

        i32_[]  /gpt_neox/layers.5/attention/Mul_4 = opset1::Multiply(
                /gpt_neox/layers.5/attention/Gather_9, batch_size
                /gpt_neox/layers.5/attention/Gather_10 num_attention_heads
                ) 	 attrs: auto_broadcast=numpy

        i32_[1]  /gpt_neox/layers.5/attention/Unsqueeze_20 = opset1::Reshape(
                /gpt_neox/layers.5/attention/Mul_4,
                [1]) 	 attrs: special_zero=0
    i32_[3]  /gpt_neox/layers.5/attention/Concat_14 = opset1::Concat(
                    /gpt_neox/layers.5/attention/Unsqueeze_20, batch_size * num_attention_heads
                    /gpt_neox/layers.5/attention/Slice_11,     key_length
                    [64]) 	 attrs: axis=0

         f32_[?,?,64]  /gpt_neox/layers.5/attention/Reshape_2 = opset1::Reshape(
                                                              Multiply_52958,
                                          /gpt_neox/layers.5/attention/Concat_14) 	 attrs: special_zero=1    //


        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)



    i32_[3]  /gpt_neox/layers.5/attention/Concat_13 = opset1::Concat(
                    /gpt_neox/layers.5/attention/Unsqueeze_17,
                    /gpt_neox/layers.5/attention/Gather_11,
                    64) 	 attrs: axis=0
        f32_[?,?,64]  /gpt_neox/layers.5/attention/Reshape_1 = opset1::Reshape(
                    /gpt_neox/layers.5/attention/Concat_9,
                    /gpt_neox/layers.5/attention/Concat_13) 	 attrs: special_zero=1
    // Q*K'
        f32_[?,?,?]  /gpt_neox/layers.5/attention/Mul_5 = opset1::MatMul(
                    /gpt_neox/layers.5/attention/Reshape_1,
                    /gpt_neox/layers.5/attention/Reshape_2) 	 attrs: transpose_a=0 transpose_b=1
    */
    //
    auto query_shape = query.ShapeOf();
    auto query_length = query_shape[2];

    auto BH = (query_shape[0] * 8).Reshape(1); // batch_size * num_head

    auto key_3d = (present_key * 0.125f)._view({BH, key_length, PatternNode(64)});
    auto query_3d = query._view({BH, query_length, PatternNode(64)});

    auto score = PatternNode::MatMul(query_3d, key_3d, false, true);

    //attn_scores.ShapeOf()

    // apply causal & attention mask to score
    auto masked_score = PatternNode::Select(causal_mask, score, float_lowest) +
                        (att_mask_in.Convert(element::f32) * float_max + float_lowest);
    auto softmax = masked_score.Softmax(3);

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
#endif
}