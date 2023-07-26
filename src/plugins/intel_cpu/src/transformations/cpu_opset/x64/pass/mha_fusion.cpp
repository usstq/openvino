// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_fusion.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "simplify_fakequantize.hpp"
#include "transformations/cpu_opset/common/op/vnode.hpp"
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
    auto in11 = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{ matmul1, fakeQuantize1 });
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(in11, in10);

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
        const bool fakeQuantize1Exists = pattern_to_output.find(fakeQuantize1) != pattern_to_output.end();
        if (fakeQuantize1Exists) {
            if (auto fq_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr())) {
                fq1_scale = simplifyToScale(fq_node);
                if (!fq1_scale.size())
                    return false;
            }
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
        std::vector<std::shared_ptr<Node>> merged = {
            pattern_to_output.at(transpose0).get_node_shared_ptr(),
            pattern_to_output.at(transpose1).get_node_shared_ptr(),
            pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
            pattern_to_output.at(matmul0).get_node_shared_ptr(),
            pattern_to_output.at(mul).get_node_shared_ptr(),
            pattern_to_output.at(add).get_node_shared_ptr(),
            pattern_to_output.at(softmax).get_node_shared_ptr(),
            pattern_to_output.at(transpose2).get_node_shared_ptr(),
            pattern_to_output.at(matmul1).get_node_shared_ptr(),
            pattern_to_output.at(transpose3).get_node_shared_ptr(),
        };

        if (fakeQuantize1Exists) {
            merged.push_back(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr());
        }
        ngraph::copy_runtime_info(merged, mha);

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
        for (size_t i = 0; i < node->get_input_size(); i++) {
            ret += get_all_labels(node->input_value(i), all_labels);
        }
        return ret;
    }

    template <typename F>
    VNodeIn(const char* vtype, F func, std::function<bool(OutputVector&)> pred = {}) {
        MATCHER_SCOPE(VNodeIn);

        if (auto * wlist = std::getenv("VNODE_WLIST")) {
            std::string vnode_whitelist(wlist);
            if (vnode_whitelist.find(std::string(vtype) + ",") == std::string::npos) {
                return;
            }
        }

        OutputVector fake_inputs;
        for (int i = 0; i < 32; i++) {
            fake_inputs.push_back(GenInput());
        }
        auto pattern_values = func(fake_inputs);

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            auto& pvmap = m.get_pattern_value_map();
            // auto node_src = pvmap.at(select.node).get_node_shared_ptr();
            auto root_value = m.get_match_value();
            std::map<std::string, double> symbol_name2value;
            std::cout << "VNodeIn::callback " << root_value << std::endl;
            if (!validate_matched_symbols(m, symbol_name2value)) {
                std::cout << "VNodeIn: symbol validation failed!" << std::endl;
                return false;
            }

            OutputVector real_inputs;
            for (auto& in : fake_inputs) {
                auto it = pvmap.find(in.get_node_shared_ptr());
                if (it == pvmap.end())
                    break;
                real_inputs.push_back(it->second);
            }
            OutputVector real_outputs;
            for (size_t i = 0; i < pattern_values.size(); i++) {
                real_outputs.push_back(pvmap[pattern_values[i].get_node_shared_ptr()]);
            }

            if (pred && !pred(real_inputs))
                return false;

            auto vnode = std::make_shared<VNode>(real_inputs, real_outputs, vtype);

            vnode->get_rt_info()["symbol_name2value"] = symbol_name2value;

            for (size_t i = 0; i < pattern_values.size(); i++) {
                auto out = pvmap[pattern_values[i].get_node_shared_ptr()];
                ngraph::replace_node(out.get_node_shared_ptr(), {vnode->output(i)});
                auto name = out.get_node_shared_ptr()->get_friendly_name();
                vnode->output(i).set_names({name});
            }
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(pattern_values[0], matcher_name);
        this->register_matcher(m, callback);
    }
};

#include "vnode_gptneox_attn.txt"
#include "vnode_gpt2.txt"
#include "vnode_opt.txt"
#include "vnode_llama.txt"
#include "vnode_bloom.txt"
#include "vnode_whisper.txt"

MHADynamicVNodeIn::MHADynamicVNodeIn() {
    add_matcher<VNodeIn>("gptneox_attention", vnode_gptneox_attn);
    add_matcher<VNodeIn>("gpt2_attention", vnode_gpt2);
    add_matcher<VNodeIn>("opt_attention", vnode_opt_attn);
    add_matcher<VNodeIn>("open_llama_attention", vnode_llama_attn);
    add_matcher<VNodeIn>("bloom_attention", vnode_bloom_attn);
    add_matcher<VNodeIn>("whisper_enc_attention", vnode_whisper_enc_attention);
    add_matcher<VNodeIn>("whisper_dec_self_attn", vnode_whisper_dec_self_attn);
    add_matcher<VNodeIn>("whisper_dec_enc_attn", vnode_whisper_dec_enc_attn);
    add_matcher<VNodeIn>("whisper_dec2_self_attn", vnode_whisper_dec2_self_attn);
    add_matcher<VNodeIn>("whisper_dec2_enc_attn", vnode_whisper_dec2_enc_attn);
}

MHADynamicVNodeOut::MHADynamicVNodeOut() {
    MATCHER_SCOPE(MHADynamicVNodeOut);
    auto vnode = ov::pass::pattern::wrap_type<VNode>();
    std::string vnode_whitelist = std::getenv("VNODE_WLIST") ? std::getenv("VNODE_WLIST") : "gpt2_attention,";

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto root_value = m.get_match_value();
        auto vnode = std::dynamic_pointer_cast<VNode>(root_value.get_node_shared_ptr());

        std::cout << "MHADynamicVNodeOut found : " << root_value.get_node_shared_ptr()->get_friendly_name() << std::endl;

        if (vnode_whitelist.find(vnode->get_vtype() + ",") != std::string::npos) {
            // leave this VNode since it's in the white-list, clear it's internal references to original subgraph
            vnode->clear_org();
            return false;
        }

        // all nodes inside vnode may contain vnodes
        OutputVector org_outputs = vnode->get_org();
        ov::NodeVector nv;
        vnode->get_internal_vnodes(nv, org_outputs[0]);
        for (auto & n : nv) {
            register_new_node(n);
        }

        ngraph::replace_node(vnode, org_outputs);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(vnode, matcher_name);
    this->register_matcher(m, callback);
}

}   // namespace intel_cpu
}   // namespace ov