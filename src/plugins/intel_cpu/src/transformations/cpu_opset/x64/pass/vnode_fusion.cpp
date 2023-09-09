// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnode_fusion.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "ie_common.h"
#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/op/vnode.hpp"
#include "transformations/cpu_opset/x64/op/rope.hpp"
#include "utils/gen_pattern.hpp"

namespace ov {
namespace intel_cpu {

#ifdef CPU_DEBUG_CAPS
#    define VNODE_DEBUG_LOG(...) _verbose_log(__VA_ARGS__)
#else
#    define VNODE_DEBUG_LOG(...)
#endif

class VNodePattern {
public:
    VNodePattern(const char* vtype) : vtype(vtype) {}
    virtual OutputVector get(const OutputVector& inputs) = 0;
    virtual bool predicate(const OutputVector& inputs) {
        return true;
    }

    std::string get_vtype() {
        return vtype;
    }

private:
    std::string vtype;
};

class EliminateMaximum : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateMaximum", "0");
    EliminateMaximum() {
        MATCHER_SCOPE(EliminateMaximum);
        auto in = GenPattern("f32");
        auto maximum = GenPattern<opset1::Maximum>({in, {-FLT_MAX}});
        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            auto root = m.get_match_root();
            return replace_output_update_name(root->output(0), root->input_value(0));
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(maximum, matcher_name);
        this->register_matcher(m, callback);
    }
};

class VNodeMatcher : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("VNodeMatcher", "0");

    int vnode_id = 0;

    VNodeMatcher(std::shared_ptr<VNodePattern> pattern) {
        MATCHER_SCOPE(VNodeMatcher);
#ifdef CPU_DEBUG_CAPS
        if (auto* wlist = std::getenv("VNODE_WLIST")) {
            std::string vnode_whitelist(wlist);
            if (vnode_whitelist != "?") {
                if (vnode_whitelist.find(std::string(pattern->get_vtype()) + ",") == std::string::npos) {
                    return;
                }
            }
        }
#endif
        VNODE_DEBUG_LOG(matcher_name, "vtype =", pattern->get_vtype());

        OutputVector fake_inputs;
        for (int i = 0; i < 32; i++) {
            fake_inputs.push_back(GenInput());
        }
        auto pattern_values = pattern->get(fake_inputs);

        matcher_pass_callback callback = [=, this](ngraph::pattern::Matcher& m) {
            auto& pvmap = m.get_pattern_value_map();
            // auto node_src = pvmap.at(select.node).get_node_shared_ptr();
            auto root_value = m.get_match_value();
            std::map<std::string, double> symbol_name2value;

            if (!validate_matched_symbols(m, symbol_name2value)) {
                VNODE_DEBUG_LOG("VNodeMatcher: symbol validation failed!");
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
                auto pattern_output_node = pattern_values[i].get_node_shared_ptr();
                if (!pvmap.count(pattern_output_node)) {
                    VNODE_DEBUG_LOG("VNodeMatcher: (auxiliary) output",
                                    pattern_output_node->get_friendly_name(),
                                    "not matched!");
                    return false;
                }
                real_outputs.push_back(pvmap[pattern_output_node]);
            }

            if (!pattern->predicate(real_inputs))
                return false;

            auto vnode = std::make_shared<VNode>(real_inputs, real_outputs, pattern->get_vtype());

            vnode->get_rt_info()["symbol_name2value"] = symbol_name2value;

            for (size_t i = 0; i < pattern_values.size(); i++) {
                auto out = pvmap[pattern_values[i].get_node_shared_ptr()];
                ngraph::replace_node(out.get_node_shared_ptr(), {vnode->output(i)});
                auto name = out.get_node_shared_ptr()->get_friendly_name();
                vnode->output(i).set_names({name});
            }

            vnode->set_friendly_name(std::string("VNode_") + std::to_string(this->vnode_id));

            this->vnode_id++;
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(pattern_values[0], matcher_name);
        this->register_matcher(m, callback);
    }
};

static std::string get_op_versioned_type_name(const std::shared_ptr<Node>& op) {
    const auto& type_info = op->get_type_info();
    auto version_info = std::string(type_info.get_version());
    return version_info + "::" + type_info.name;
}

// collect attributes which (togather with vtype) defines
// the expected behaviour of VNode

template <typename T>
class ScalarAttributeCollector : public ngraph::AttributeVisitor {
public:
    std::map<std::string, T> attrs;

    ScalarAttributeCollector() {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<int>>(&adapter)) {
            attrs[name] = static_cast<T>(a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<float>>(&adapter)) {
            attrs[name] = static_cast<T>(a->get());
        } else {
            IE_ASSERT(false) << "ScalarAttributeCollector: unsupported data type for attribute: " << name;
        }
    }
};

class VNodeFromExtension : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("VNodeFromExtension", "0");
    VNodeFromExtension(std::string target_versioned_type_name) {
        MATCHER_SCOPE(VNodeFromExtension);
        auto pnode = std::make_shared<GenericPattern>();
        pnode->set_predicate([=](const Output<Node>& value) {
            auto op = value.get_node_shared_ptr();
            return get_op_versioned_type_name(op) == target_versioned_type_name;
        });

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            auto root = m.get_match_root();
            auto vnode =
                std::make_shared<VNode>(root->input_values(), root->outputs(), get_op_versioned_type_name(root));
            // vnode->get_rt_info()["symbol_name2value"] = symbol_name2value;
            ngraph::replace_node(root, vnode);

            // collect attributes as map and stores it in VNode's rt_info
            {
                ScalarAttributeCollector<double> visitor;
                if (!root->visit_attributes(visitor)) {
                    IE_ASSERT(false) << "visit_attributes failed for node: " << root->get_friendly_name();
                }
                vnode->get_rt_info()["attr_map"] = visitor.attrs;
            }

            // for (size_t i = 0; i < root->get_output_size(); i++) {
            //     auto name = out.get_node_shared_ptr()->get_friendly_name();
            //     vnode->output(i).set_names({name});
            // }
            // std::cout << root << std::endl;
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(pnode, matcher_name);
        this->register_matcher(m, callback);
    }
};

#include "vnode_attn_falcon.txt"
#include "vnode_attn_gptj.txt"
#include "vnode_attn_gptneox.txt"
#include "vnode_attn_llama2.txt"

VNodeFusion::VNodeFusion(const ov::element::Type inferencePrecision) {
    MATCHER_SCOPE(VNodeFusion);

    add_matcher<EliminateMaximum>();
    add_matcher<VNodeFromExtension>("llm::experimental::MultiHeadAttention");

    // enable attention fusion only when proper backend support is ready
    bool enable_attn_fusion = false;
#ifdef OV_CPU_WITH_MLAS
    if (inferencePrecision == ov::element::f32)
        enable_attn_fusion = true;
#endif
#ifdef OV_CPU_WITH_LLMDNN
    if (inferencePrecision == ov::element::bf16)
        enable_attn_fusion = true;
#endif

    if (enable_attn_fusion) {
        add_matcher<VNodeMatcher>(std::make_shared<gptneox_attention>());
        add_matcher<VNodeMatcher>(std::make_shared<gptj_attention>());
        add_matcher<VNodeMatcher>(std::make_shared<falcon_attention>());
        add_matcher<VNodeMatcher>(std::make_shared<llama2_attention>());
    }

    add_matcher<VNodeMatcher>(std::make_shared<llama_RMSNorm>());
}

}  // namespace intel_cpu
}  // namespace ov