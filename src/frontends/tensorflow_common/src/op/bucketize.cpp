// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_bucketize_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Bucketize"});
    auto input = node.get_input(0);

    // retrieve attribute
    auto boundaries = node.get_attribute<std::vector<float>>("boundaries");

    auto bucketize =
        make_shared<Bucketize>(input,
                               make_shared<Constant>(ov::element::f32, Shape{boundaries.size()}, boundaries),
                               ov::element::i32,
                               false);
    set_node_name(node.get_name(), bucketize);
    return {bucketize};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
