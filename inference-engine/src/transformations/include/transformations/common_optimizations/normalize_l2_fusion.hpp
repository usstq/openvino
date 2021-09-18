// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API NormalizeL2Fusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief NormalizeL2Fusion transformation replaces various sub-graphs with a NormalizeL2 op:
 * x/(max(sqrt(sum(x[j0, ..., jN]**2), eps)) with a NormalizeL2 op.
 * x/(add(sqrt(sum(x[j0, ..., jN]**2), eps)) with a NormalizeL2 op.
 */
class ngraph::pass::NormalizeL2Fusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    NormalizeL2Fusion();
};