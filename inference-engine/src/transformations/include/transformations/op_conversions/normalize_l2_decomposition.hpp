// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API NormalizeL2Decomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Decomposes NormalizeL2 into subgraph
 */
class ngraph::pass::NormalizeL2Decomposition: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    NormalizeL2Decomposition();
};
