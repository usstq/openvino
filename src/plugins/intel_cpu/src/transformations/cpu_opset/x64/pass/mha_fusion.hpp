// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset4.hpp>

namespace ov {
namespace intel_cpu {

class MHAFusionBase : public ngraph::pass::MatcherPass {
protected:
    bool valid_transpose_order(const std::shared_ptr<ngraph::Node>& node, const std::vector<int64_t>& expected_order) {
        if (auto transpose_pattern = ngraph::as_type_ptr<ngraph::opset4::Constant>(node)) {
            if (transpose_pattern->cast_vector<int64_t>() != expected_order) {
                return false;
            }
        } else {
            return false;
        }

        return true;
    }
};

class MHAFloatFusion: public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAFloatFusion", "0");
    MHAFloatFusion();
};

class MHAFloatFusion2: public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAFloatFusion2", "0");
    MHAFloatFusion2();
};

class MHAQuantFusion: public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAQuantFusion", "0");
    MHAQuantFusion();
};

class MHAQuantFusion2: public MHAFusionBase {
public:
    OPENVINO_RTTI("MHAQuantFusion2", "0");
    MHAQuantFusion2();
};

// recognize different CausalMask operations and fuse them into 1
class CausalMaskFusion: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("CausalMaskFusion", "0");
    CausalMaskFusion();
};
/*
class RoPEFusionQuery: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionQuery", "0");
    RoPEFusionQuery();
};

class RoPEFusionKey: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionKey", "0");
    RoPEFusionKey();
};
*/

class MHADynamicVNodeIn: public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("MHADynamicVNodeIn", "0");
    MHADynamicVNodeIn();
};
class MHADynamicVNodeOut: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHADynamicVNodeOut", "0");
    MHADynamicVNodeOut();
};

class MHADynamicFloatFusion: public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("MHADynamicFloatFusion", "0");
    MHADynamicFloatFusion() {
        add_matcher<CausalMaskFusion>();
        //add_matcher<RoPEFusionQuery>();
        //add_matcher<RoPEFusionKey>();
    }
};

class MHAFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("MHAFusion", "0");
    MHAFusion() {
        add_matcher<MHAFloatFusion>();
        add_matcher<MHAFloatFusion2>();
        add_matcher<MHAQuantFusion>();
        add_matcher<MHAQuantFusion2>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
