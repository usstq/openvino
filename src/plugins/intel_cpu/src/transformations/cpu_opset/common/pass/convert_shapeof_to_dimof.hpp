// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ConvertShapeOfToDimOf1: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertShapeOfToDimOf1", "0");
    ConvertShapeOfToDimOf1();
};

class RemoveReshapeTailOfDimOfSubgraph: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveReshapeTailOfDimOfSubgraph", "0");
    RemoveReshapeTailOfDimOfSubgraph();
};

class EliminateDuplicateDimOf: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateDuplicateDimOf", "0");
    EliminateDuplicateDimOf();
};

class ConvertShapeOfToDimOf : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("FullyConnectedBiasFusion", "0");
    ConvertShapeOfToDimOf() {
        add_matcher<ConvertShapeOfToDimOf1>();
        add_matcher<RemoveReshapeTailOfDimOfSubgraph>();
        add_matcher<EliminateDuplicateDimOf>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
