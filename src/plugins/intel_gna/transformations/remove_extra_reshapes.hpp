// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Removes reshapes before MaxPool which do nothing. Such reshapes can be a result of conversion from IR10 to IR7.
 */
class RemoveExtraReshapes : public ngraph::pass::MatcherPass {
public:
  OPENVINO_RTTI("RemoveExtraReshapes", "0");
  RemoveExtraReshapes();
};

} // namespace GNAPluginNS
