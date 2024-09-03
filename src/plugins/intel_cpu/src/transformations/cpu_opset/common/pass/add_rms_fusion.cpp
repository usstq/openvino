// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "add_rms_fusion.hpp"
#include <memory>
#include "itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "ov_ops/rms.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/add_rms.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

AddRMSFusion::AddRMSFusion() {
    MATCHER_SCOPE(AddRMSFusion);
    auto data0 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto data1 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto pattern_add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({data0, data1});

    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::internal::RMS>({pattern_add, pass::pattern::any_input()});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto rms = std::dynamic_pointer_cast<ov::op::internal::RMS>(
            pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (rms == nullptr || transformation_callback(rms)) {
            return false;
        }
        auto add = std::dynamic_pointer_cast<op::v1::Add>(pattern_to_output.at(pattern_add).get_node_shared_ptr());
        auto data0 = add->get_input_node_shared_ptr(0);
        auto data1 = add->get_input_node_shared_ptr(1);
        auto gamma = rms->get_input_node_shared_ptr(1);
        auto add_rms = std::make_shared<AddRMSNode>(data0, data1, gamma, static_cast<float>(rms->get_epsilon()));
        add_rms->set_friendly_name(rms->get_friendly_name());
        copy_runtime_info(rms, add_rms);

        std::shared_ptr<ov::Node> parent = add;
        for (size_t i = 0; i < 2; i++) {
            auto present_to = parent->output(0).get_target_inputs();
            for (auto& to : present_to) {
                auto to_node = to.get_node();
                if (to_node == rms.get())
                    continue;
                auto inputs = to_node->input_values();
                inputs[to.get_index()] = add_rms->output(i);
                to_node->set_arguments(inputs);
            }
            parent = rms;
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

}   // namespace intel_cpu
}   // namespace ov
