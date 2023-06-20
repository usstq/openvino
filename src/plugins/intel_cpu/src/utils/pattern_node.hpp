// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <memory>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include "transformations/cpu_opset/common/op/dimof.hpp"
//#include "ngraph/pattern/op/pattern.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_cpu {

struct PatternNode;

PatternNode operator*(PatternNode lhs, PatternNode rhs);
PatternNode operator+(PatternNode lhs, PatternNode rhs);


template<typename ... Args>
void MatcherVerbose(Args&& ... args) {
    static bool MATCHER_VERBOSE = std::getenv("MATCHER_VERBOSE") ? atoi(std::getenv("MATCHER_VERBOSE")) : false;
    if (!MATCHER_VERBOSE) return;
    std::stringstream ss;
    int dummy[] = {(ss << std::forward<Args>(args) << " ", 0)...};
    (void)(dummy);
    ss << std::endl;
    std::cout << ss.str();
}

struct PatternNode {
    std::shared_ptr<Node> node;

    operator std::shared_ptr<Node>() {
        return node;
    }

    template <class... Args>
    static std::shared_ptr<Node> node_type(const OutputVector& inputs) {
        return ov::pass::pattern::wrap_type<Args...>(inputs, [](const Output<Node>& output) {
            MatcherVerbose("\tmatched : ", output);
            return true;
        });
    }

    template <class... Args>
    static std::shared_ptr<Node> node_type(const OutputVector& inputs,
                                           const std::function<bool(const Output<Node>&)>& pred) {
        return ov::pass::pattern::wrap_type<Args...>(inputs, [pred](const Output<Node>& output) {
            auto ret = pred(output);
            MatcherVerbose(ret ? "\tmatched : " : "******* failed", output);
            return ret;
        });
    }

    PatternNode() {
        node = ov::pass::pattern::any_input();
    }

    // any input with given partial shape
    PatternNode(const ov::PartialShape& pshape, const element::Type& dtype = element::undefined) {
        node = ov::pass::pattern::any_input([pshape, dtype](const Output<Node>& value) {
            auto ret = pshape.compatible(value.get_partial_shape());
            if (ret && dtype != element::undefined)
                ret = (value.get_element_type() == dtype);
            MatcherVerbose(ret ? "\tmatched : " : "******* failed", value);
            return ret;
        });
    }

    PatternNode(const ov::Rank& rank, const element::Type& dtype = element::undefined) {
        node = ov::pass::pattern::any_input([rank, dtype](const Output<Node>& value) {
            auto ret = rank.compatible(value.get_partial_shape().rank());
            if (ret && dtype != element::undefined)
                ret = (value.get_element_type() == dtype);
            MatcherVerbose(ret ? "\tmatched : " : "******* failed", value);
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
        node = ConstVector(std::vector<int>{v});
    }

    PatternNode(const float v) {
        node = ConstVector(std::vector<float>{v});
    }

    PatternNode(const std::vector<int>& vec) {
        node = ConstVector(vec);
    }

    static inline PatternNode ShapeOf(const PatternNode& in) {
        return node_type<opset1::ShapeOf>({in.node});
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

    static inline PatternNode Softmax(const PatternNode& in, int axis) {
        return node_type<opset1::Softmax, opset8::Softmax>({in.node->output(0)}, [axis](const Output<Node>& value) {
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
        return node_type<opset8::Slice>({node, start.node, stop.node, step.node, axis.node});
    }

    static inline PatternNode Select(const PatternNode& cond, const PatternNode& n_then, const PatternNode& n_else) {
        return node_type<opset1::Select>({cond.node, n_then.node, n_else.node});
    }

    static inline PatternNode MatMul(const PatternNode& query_3d, const PatternNode& key_3d, bool transa, bool transb) {
        return node_type<opset1::MatMul>({query_3d.node, key_3d.node}, [transa, transb](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::MatMul>(value.get_node_shared_ptr());
            return s1->get_transpose_a() == transa && s1->get_transpose_b() == transb;
        });
    }

    static inline PatternNode Maximum(const PatternNode& a, const PatternNode& b) {
        return node_type<opset1::Maximum>({a.node, b.node});
    }

    static inline PatternNode Reshape(const PatternNode& in, const PatternNode& shape) {
        return node_type<opset1::Reshape>({in.node, shape.node});
    }

    static inline PatternNode Broadcast(const PatternNode& data,
                                        const PatternNode& taget_shape,
                                        const PatternNode& axes_mapping) {
        return node_type<opset1::Broadcast>({data.node, taget_shape.node, axes_mapping.node});
    }

    static inline PatternNode Concat(std::initializer_list<PatternNode> inputs, int64_t axis) {
        OutputVector outputs;
        for (auto& i : inputs) {
            outputs.push_back(i.node->get_default_output());
        }
        return node_type<opset1::Concat>(outputs, [axis](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::Concat>(value.get_node_shared_ptr());
            return s1->get_axis() == axis;
        });
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

}   // namespace intel_cpu
}   // namespace ov