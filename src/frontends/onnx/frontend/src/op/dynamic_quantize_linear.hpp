// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector dynamic_quantize_linear(const Node& node);

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
