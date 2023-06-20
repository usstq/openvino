// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pattern_node.hpp"

namespace ov {
namespace intel_cpu {


PatternNode operator*(PatternNode lhs, PatternNode rhs) {
    return PatternNode::node_type<opset1::Multiply>({lhs.node, rhs.node});
}

PatternNode operator+(PatternNode lhs, PatternNode rhs) {
    return PatternNode::node_type<opset1::Add>({lhs.node, rhs.node});
}

}   // namespace intel_cpu
}   // namespace ov