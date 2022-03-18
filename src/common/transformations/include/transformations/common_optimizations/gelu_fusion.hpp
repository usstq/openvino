// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API GeluFusion;
class TRANSFORMATIONS_API GeluFusionWithErfOne;
class TRANSFORMATIONS_API GeluFusionWithErfTwo;
class TRANSFORMATIONS_API GeluFusionWithErfThree;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * (0.5 * x) * (1 + erf(x / sqrt(2))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfOne : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("GeluFusionWithErfOne", "0");
    GeluFusionWithErfOne();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * 0.5 * (x * (1 + erf(x / sqrt(2))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfTwo : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("GeluFusionWithErfTwo", "0");
    GeluFusionWithErfTwo();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph
 * x * (0.5 * (1 + erf(x / sqrt(2))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfThree : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("GeluFusionWithErfThree", "0");
    GeluFusionWithErfThree();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces various sub-graphs with a Gelu op.
 */
class ngraph::pass::GeluFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("GeluFusion", "0");
    GeluFusion() {
        add_matcher<ngraph::pass::GeluFusionWithErfOne>();
        add_matcher<ngraph::pass::GeluFusionWithErfTwo>();
        add_matcher<ngraph::pass::GeluFusionWithErfThree>();
    }
};
