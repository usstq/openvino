// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class MHADynamicFloatFusion4D : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHADynamicFloatFusion4D", "0");
    MHADynamicFloatFusion4D();
};

class MHADynamicFloatFusionWhisper : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHADynamicFloatFusionWhisper", "0");
    MHADynamicFloatFusionWhisper();
};

class MHADynamicVNodeIn : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("MHADynamicVNodeIn", "0");
    MHADynamicVNodeIn();
};

/*
class MHADynamicVNodeOut : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHADynamicVNodeOut", "0");
    MHADynamicVNodeOut();
};
*/

}  // namespace intel_cpu
}  // namespace ov
