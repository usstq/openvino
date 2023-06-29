// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <cfloat>
#include <utility>
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include <openvino/opsets/opset8.hpp>
#include <sstream>
#include <string>
#include <cassert>

#include "transformations/cpu_opset/common/op/dimof.hpp"
//#include "ngraph/pattern/op/pattern.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_cpu {

extern const int _matcher_verbose;

template<typename ... Args>
static inline void verbose_log(Args&& ... args) {
    if (!_matcher_verbose) return;
    std::stringstream ss;
    int dummy[] = {(ss << std::forward<Args>(args) << " ", 0)...};
    (void)(dummy);
    ss << std::endl;
    std::cout << ss.str();
}

inline std::vector<std::string> split_string(const std::string & s, const std::string & delimiter) {
    std::vector<std::string> ret;
    size_t pos = 0, pos_next;
    std::string token;
    while ((pos_next = s.find(delimiter, pos)) != std::string::npos) {
        token = s.substr(pos, pos_next - pos);
        ret.push_back(token);
        pos = pos_next + 1;
    }
    // return whole string if no delimiter if found
    token = s.substr(pos, pos_next);
    ret.push_back(token);
    return ret;
}

struct values_info {
    values_info(const char* pattern_list = nullptr) {
        if (pattern_list == nullptr || pattern_list[0] == 0) {
            all_type_pshape.emplace_back(ov::element::dynamic, ov::PartialShape::dynamic(ov::Dimension::dynamic()));
            return;
        }
        auto pattern_vector = split_string(pattern_list, " ");
        for (auto& pattern : pattern_vector) {
            if (pattern[0] == '[') {
                all_type_pshape.emplace_back(ov::element::dynamic, ov::PartialShape(pattern));
            } else {
                auto sep = pattern.find("[");
                assert(sep != std::string::npos);
                all_type_pshape.emplace_back(ov::element::Type(pattern.substr(0, sep)),
                                             ov::PartialShape(pattern.substr(sep)));
            }
        }
    }

    bool predicate(const ov::Output<ov::Node>& value) const {
        auto index = value.get_index();
        auto& item = all_type_pshape[index];
        if (!item.first.compatible(value.get_element_type()) || !item.second.compatible(value.get_partial_shape())) {
            return false;
        }
        return true;
    }

    size_t get_output_size() {
        return all_type_pshape.size();
    }

    std::vector<std::pair<ov::element::Type, ov::PartialShape>> all_type_pshape;
};

struct attr {
    attr() = default;
    attr(const char* name, const char* v) : name(name) {
        type = 0;
        value.str = v;
    }
    attr(const char* name, int v) : name(name) {
        type = 1;
        value.i32 = v;
    }
    attr(const char* name, float v) : name(name) {
        type = 2;
        value.f32 = v;
    }
    attr(const char* name, double v) : name(name) {
        type = 2;
        value.f32 = v;
    }
    bool predicate(int v) const {
        bool ret = (type == 1 && v == value.i32);
        return ret;
    }
    bool predicate(int64_t v) const {
        bool ret = (type == 1 && v == value.i32);
        return ret;
    }
    bool predicate(float v) const {
        bool ret = (type == 2 && v == value.f32);
        return ret;
    }
    bool predicate(double v) const {
        bool ret = (type == 2 && v == value.f32);
        return ret;
    }
    bool predicate(const std::string& v) const {
        bool ret = (type == 0 && v == value.str);
        return ret;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << name << ":";
        if (type == 0) ss << value.str;
        if (type == 1) ss << value.i32;
        if (type == 2) ss << value.f32;
        return ss.str();
    }
    const char * name;
    union {
        const char* str;
        int i32;
        float f32;
    } value;
    int type;
};

bool attr_compatible(ov::Node& node, const std::vector<attr>& attr);

inline std::shared_ptr<Node> GenInput(values_info vt = nullptr) {
    return ov::pass::pattern::any_input([vt](const Output<Node>& value) {
        if (!vt.predicate(value)) {
            verbose_log("*mismatched GenInput ", value);
            return false;
        }
        verbose_log(" matched GenInput ", value);
        return true;
    });
}

// A glue type which allows more types to be used as input to GenPattern()
struct GenPatternNode {
    std::shared_ptr<Node> node;

    GenPatternNode(values_info vt) {
        node = ov::pass::pattern::any_input([vt](const Output<Node>& value) {
            if (!vt.predicate(value)) {
                verbose_log("*mismatched GenPatternNode ", value);
                return false;
            }
            verbose_log(" matched GenPatternNode ", value);
            return true;
        });
    }
    GenPatternNode(const std::shared_ptr<Node> & node) : node(node) {}
    GenPatternNode(const Output<Node>& out) : node(out.get_node_shared_ptr()) {}
    GenPatternNode(int v) {
        node = ConstVector(std::vector<int>{v}, "i32[]");
    }
    GenPatternNode(float v) {
        node = ConstVector(std::vector<float>{v}, "f32[]");
    }

    GenPatternNode(std::initializer_list<int> v) {
        node = ConstVector(std::vector<int>(v), nullptr);
    }
    GenPatternNode(std::initializer_list<float> v) {
        node = ConstVector(std::vector<float>(v), nullptr);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    GenPatternNode(std::initializer_list<T> v, values_info vt) {
        node = ConstVector(std::vector<T>(v), vt);
    }

    // 1d const tensor or scalar
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    static std::shared_ptr<Node> ConstVector(const std::vector<T>& vec, values_info vt) {
        auto pred = [vec, vt](const Output<Node>& value) {
            if (!vt.predicate(value)) {
                verbose_log("*mismatched ConstVector ", value);
                return false;
            }
            auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
            auto shape = s1->get_output_shape(0);
            if (shape_size(shape) != vec.size()) {
                verbose_log("*mismatched ConstVector ", value);
                return false;
            }
            std::vector<T> actual = s1->cast_vector<T>();
            if (actual != vec) {
                verbose_log("*mismatched ConstVector ", value);
                return false;
            }
            verbose_log(" matched ConstVector ", value);
            return true;
        };
        return ov::pass::pattern::wrap_type<opset1::Constant>({}, pred);
    }
};

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
std::shared_ptr<Node> GenConst(std::initializer_list<T> v, values_info vt = nullptr) {
    GenPatternNode g(v, vt);
    return g.node;
}

template <class... Args>
std::shared_ptr<Node> GenPattern(const std::vector<GenPatternNode>& inputs, values_info vt, const std::vector<attr>& attrs = {}) {
    OutputVector ovs;
    for (auto & i : inputs) {
        ovs.push_back(i.node);
    }

    auto pattern_node = ov::pass::pattern::wrap_type<Args...>(ovs, [vt, attrs](const Output<Node>& value) {
        if (!vt.predicate(value)) {
            verbose_log("*mismatched GenPattern vt ", value);
            return false;
        }

        // match parent node with attribute a0/a1/...
        if (!attr_compatible(*value.get_node_shared_ptr(), attrs)) {
            verbose_log("*mismatched GenPattern attr ", value);
            return false;
        }
        verbose_log(" matched GenPattern ", value);
        return true;
    });

    auto output_size = vt.get_output_size();
    if (output_size > 1)
        pattern_node->set_output_size(output_size);

    return pattern_node;
}

// TODO
// for better performance, pattern matching is done in 2 steps
//  1. match topology based on type of nodes and their inter-connections
//  2. detailed validation on individual node's attributes
//
// but for clear description of the pattern, all information about
// each node is described in the pattern description part.

}  // namespace intel_cpu
}  // namespace ov