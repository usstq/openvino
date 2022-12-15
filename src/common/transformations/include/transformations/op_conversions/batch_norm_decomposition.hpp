// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

using namespace std;

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BatchNormDecomposition;

}  // namespace pass
}  // namespace ov

class ov::pass::BatchNormDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BatchNormDecomposition", "0");
    BatchNormDecomposition();
};

namespace ngraph {
namespace pass {
using ov::pass::BatchNormDecomposition;
}  // namespace pass
}  // namespace ngraph
