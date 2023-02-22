// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/reduce_min.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::ReduceMin;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
