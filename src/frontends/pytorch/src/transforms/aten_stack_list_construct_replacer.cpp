// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_stack_list_construct_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;
using namespace ov::pass::pattern;

AtenStackListConstructReplacer::AtenStackListConstructReplacer() {
    auto list_construct = wrap_type<ov::op::util::FrameworkNode>();
    auto axis = wrap_type<v0::Constant>();

    // We search for a pattern: ListConstruct -> aten::stack <- Constant
    auto stack = wrap_type<ov::op::util::FrameworkNode>({list_construct, axis});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto stack = cast_fw_node(m.get_match_root(), "aten::stack");
        if (!stack) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        auto input_node = pattern_map.at(list_construct).get_node_shared_ptr();
        auto axis_node = pattern_map.at(axis).get_node_shared_ptr();
        auto axis_const = std::dynamic_pointer_cast<v0::Constant>(axis_node);
        auto axis = axis_const->cast_vector<int64_t>();
        // Check if ListConstruct is an input
        if (auto list_construct_node = cast_fw_node(input_node, "prim::ListConstruct")) {
            const auto& list_inputs = list_construct_node->input_values();
            OutputVector node_vector;
            auto zero = v0::Constant::create(element::i32, Shape{}, {0});
            // Iterate over values in ListConstruct
            for (const auto& list_input : list_inputs) {
                auto node = concat_list_construct(list_input);
                auto unsqueezed_node = std::make_shared<v0::Unsqueeze>(node, axis_const);
                node_vector.push_back(unsqueezed_node);
            }
            // Concat vectors on provided axis
            auto concat = std::make_shared<v0::Concat>(node_vector, axis[0]);

            copy_runtime_info_and_name(stack, {concat}, {input_node});
            replace_node(stack, concat);
            return true;
        }
        add_exception_to_fw_node(stack, "Unsupported case of aten::stack.");
        return false;
    };

    auto m = std::make_shared<Matcher>(stack, "ov::frontend::pytorch::pass::AtenStackListConstructReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
