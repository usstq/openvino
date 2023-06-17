// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_shapeof_to_dimof.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "transformations/cpu_opset/common/op/dimof.hpp"

ov::intel_cpu::ConvertShapeOfToDimOf1::ConvertShapeOfToDimOf1() {
    MATCHER_SCOPE(ConvertShapeOfToDimOf1);
    auto src = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto shapeof = ngraph::pattern::wrap_type<ngraph::opset1::ShapeOf>({src});

    auto has_one_element = [&](const Output<Node>& value) {
        auto s1 = as_type_ptr<ngraph::opset1::Constant>(value.get_node_shared_ptr());
        auto shape = s1->get_output_shape(0);
        if (shape_size(shape) != 1)
            return false;
        return true;
    };

    auto is_scalar_const = [&](const Output<Node>& value) {
        return value.get_shape().size() == 0;
    };

    // optional steps : [Multiply with scalar]/...
    auto const_scalar = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(is_scalar_const);
    auto mul_scalar = ngraph::pattern::wrap_type<ngraph::opset1::Multiply>({shapeof, const_scalar});
    auto optional_arithmetic = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shapeof, mul_scalar});

    // =>ShapeOf=>[optional]=>Gather=>
    auto indices = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(has_one_element);
    auto axis = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(has_one_element);
    auto dim_gather = ngraph::pattern::wrap_type<ngraph::opset8::Gather>({optional_arithmetic, indices, axis});

    // =>ShapeOf=>Slice=>
    auto start = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(has_one_element);
    auto stop = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(has_one_element);
    auto step = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(has_one_element);
    auto axes = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(has_one_element);
    auto dim_slice = ngraph::pattern::wrap_type<ngraph::opset8::Slice>({optional_arithmetic, start, stop, step, axes});

    auto final_output = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{dim_gather, dim_slice});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto src_out = pattern_map.at(src);

        // additional arithmetic binary op (with scalar) exists between ShapeOf & Gather/Slice,
        // which can be postponed after DimOf
        auto append_arithmetic_ops = [&](std::shared_ptr<ov::intel_cpu::DimOfNode> dimOf) {
            std::shared_ptr<Node> new_node = dimOf;
            auto it_arithmetic = pattern_map.find(mul_scalar);
            if (it_arithmetic != pattern_map.end()) {
                new_node = std::make_shared<ngraph::opset1::Multiply>(dimOf, pattern_map.at(const_scalar));
            }
            return new_node;
        };

        auto do_replace = [&](std::shared_ptr<Node> target, int axis) {
            // further check if target node has Squeeze children, if so, create a scalar version of DimOf
            // to fuse them also into DimOf.
            auto next_inputs = target->get_default_output().get_target_inputs();
            bool need_non_scalar_branch = false;
            std::vector<std::shared_ptr<Node>> squeeze_nodes;
            for (auto& input : next_inputs) {
                auto squeeze = dynamic_cast<ngraph::opset1::Squeeze*>(input.get_node());
                if (squeeze) {
                    squeeze_nodes.push_back(squeeze->shared_from_this());
                } else {
                    // for non_scalar children, just relace the node `target` with DimOf.
                    need_non_scalar_branch = true;
                }
            }

            // scalar version of DimOf which fused Squeeze children
            if (squeeze_nodes.size()) {
                auto DimOf = std::make_shared<ov::intel_cpu::DimOfNode>(src_out, axis, true);
                std::stringstream ss;
                ss << src_out.get_node_shared_ptr()->get_friendly_name() << ".shape[" << axis << "]";
                DimOf->set_friendly_name(ss.str());
                ngraph::copy_runtime_info(target, DimOf);
                auto new_node = append_arithmetic_ops(DimOf);
                for (auto& squeeze : squeeze_nodes) {
                    ngraph::replace_node(squeeze, new_node);
                }
            }

            // non-scalar version of DimOf which directly replace target node
            if (need_non_scalar_branch) {
                bool output_is_scalar = (target->get_output_partial_shape(0).rank().get_length() == 0);
                auto DimOf = std::make_shared<ov::intel_cpu::DimOfNode>(src_out, axis, output_is_scalar);
                std::stringstream ss;
                ss << src_out.get_node_shared_ptr()->get_friendly_name() << ".shape";
                if (output_is_scalar)
                    ss << "[" << axis << "]";  // imply output is scalar
                else
                    ss << "[[" << axis << "]]";  // imply output is 1-element vector
                DimOf->set_friendly_name(ss.str());
                ngraph::copy_runtime_info(target, DimOf);
                auto new_node = append_arithmetic_ops(DimOf);
                ngraph::replace_node(target, new_node);
            }
        };

        auto it = pattern_map.find(dim_gather);
        if (it != pattern_map.end()) {
            // ShapeOf => Gather => [Squeeze]
            auto gather_node = std::dynamic_pointer_cast<ngraph::opset8::Gather>(it->second.get_node_shared_ptr());
            auto indices_node =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(pattern_map.at(indices).get_node_shared_ptr());
            auto axis_node =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(pattern_map.at(axis).get_node_shared_ptr());
            auto axis_val = axis_node->cast_vector<int>()[0];
            if (axis_val != 0)
                return false;

            auto indices_val = indices_node->cast_vector<int>()[0];
            do_replace(gather_node, indices_val);
            return true;
        }

        it = pattern_map.find(dim_slice);
        if (it != pattern_map.end()) {
            // ShapeOf => Slice => [Squeeze]
            auto slice_node = std::dynamic_pointer_cast<ngraph::opset8::Slice>(it->second.get_node_shared_ptr());

            auto start_node =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(pattern_map.at(start).get_node_shared_ptr());
            auto stop_node =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(pattern_map.at(stop).get_node_shared_ptr());

            auto start_val = start_node->cast_vector<int>()[0];
            auto stop_val = stop_node->cast_vector<int>()[0];

            if (stop_val != start_val + 1)
                return false;

            auto axes_node =
                std::dynamic_pointer_cast<ngraph::opset1::Constant>(pattern_map.at(axes).get_node_shared_ptr());
            auto axes_val = axes_node->cast_vector<int>()[0];
            if (axes_val != 0)
                return false;

            do_replace(slice_node, start_val);
            return true;
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(final_output, matcher_name);
    this->register_matcher(m, callback);
}

/*
   DimOf(output_scalar=true) =>
          ... ...            => scalar-arithmetic-subgraph => Reshape(shape=[1]) =>
   DimOf(output_scalar=true) =>

remove the Reshape node at the end , and use 1-element vector from the beginning

   DimOf(output_scalar=false) =>
          ... ...             => scalar-arithmetic-subgraph =>
   DimOf(output_scalar=false) =>

this is OK if scalar arithematic expression subgraph satisfy following conditions:

   1. all inputs are scalar const or DimOf(output_scalar=true)
   2. all nodes are of type +,-,*,/
   3. each nodes in this subgraph has only one children
   4. all nodes in this subgraph outputs scalar

*/

using namespace ov;

bool collect_scalar_subgraph(std::vector<std::shared_ptr<ov::intel_cpu::DimOfNode>>& all_dimof,
                             std::vector<std::shared_ptr<Node>>& all_nodes,
                             const ov::Output<Node>& output,
                             bool is_last) {
    auto node = output.get_node_shared_ptr();

    auto is_output_scalar_i32 = [&]() {
        if (node->get_output_size() > 1)
            return false;
        if (!node->get_output_partial_shape(0).is_static())
            return false;
        return node->get_output_shape(0).size() == 0 && node->get_output_element_type(0) == element::i32;
    };

    if (!is_output_scalar_i32())
        return false;

    if (!is_last) {
        // if not last node, must have only 1 consumer
        // so recusively up-stream search can cover all subgraph
        if (output.get_target_inputs().size() > 1)
            return false;
    }

    // dimof
    auto dimof = std::dynamic_pointer_cast<ov::intel_cpu::DimOfNode>(node);
    if (dimof) {
        all_dimof.push_back(dimof);
        all_nodes.push_back(node);
        return true;
    }

    // +,-,*,/ with scalar const
    if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(node) ||
        std::dynamic_pointer_cast<ngraph::opset1::Multiply>(node) ||
        std::dynamic_pointer_cast<ngraph::opset1::Divide>(node) ||
        std::dynamic_pointer_cast<ngraph::opset1::Add>(node) ||
        std::dynamic_pointer_cast<ngraph::opset1::Subtract>(node)) {
        // recursively check inputs
        auto cnt = node->get_input_size();
        for (int i = 0; i < cnt; i++) {
            if (!collect_scalar_subgraph(all_dimof, all_nodes, node->get_input_source_output(i), false)) {
                return false;
            }
        }
        // all checks passed
        all_nodes.push_back(node);
        return true;
    }

    // not expected type of node
    return false;
}

ov::intel_cpu::RemoveReshapeTailOfDimOfSubgraph::RemoveReshapeTailOfDimOfSubgraph() {
    MATCHER_SCOPE(RemoveReshapeTailOfDimOfSubgraph);

    auto is_scalar = [&](const Output<Node>& value) {
        return value.get_partial_shape().size() == 0;
    };

    auto src = ngraph::pattern::any_input(is_scalar);

    auto is_1ele_shape = [&](const Output<Node>& value) {
        auto constop = std::dynamic_pointer_cast<ngraph::opset1::Constant>(value.get_node_shared_ptr());
        if (!constop)
            return false;
        auto& v_shape = constop->get_output_shape(0);
        if (constop->get_output_element_type(0).is_integral_number() && v_shape.size() == 1 && v_shape[0] == 1) {
            if (constop->cast_vector<int>()[0] == 1)
                return true;
        }
        return false;
    };

    // target shape is [1], so Reshape is doing Unsqueeze from scalar into 1-element vector
    auto shape = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(is_1ele_shape);
    auto reshape = ngraph::pattern::wrap_type<ngraph::opset1::Reshape>({src, shape});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();

        // collect all sibling nodes, make sure they are all
        // all other sibling reshape nodes are also to 1ele
        std::vector<std::shared_ptr<Node>> all_reshapes;
        all_reshapes.push_back(reshape_node);
        for (auto& other_input : reshape_node->input_value(0).get_target_inputs()) {
            auto other_reshape = dynamic_cast<ngraph::opset1::Reshape*>(other_input.get_node());
            if (!other_reshape)
                return false;
            if (!is_1ele_shape(other_reshape->input_value(1)))
                return false;
            all_reshapes.push_back(other_reshape->shared_from_this());
        }

        std::vector<std::shared_ptr<ov::intel_cpu::DimOfNode>> all_dimof;
        std::vector<std::shared_ptr<Node>> all_nodes;

        auto last_output = reshape_node->input_value(0);
        bool found = collect_scalar_subgraph(all_dimof, all_nodes, last_output, true);

        for (auto& dimof : all_dimof) {
            dimof->set_output_scalar(false);
        }

        for (auto& node_in_subgraph : all_nodes) {
            node_in_subgraph->revalidate_and_infer_types();
        }

        for (auto& p_reshape : all_reshapes) {
            ngraph::replace_node(p_reshape, last_output.get_node_shared_ptr());
        }
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape, matcher_name);
    this->register_matcher(m, callback);
}