// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Pooling can be reordered with activation, on GNA there is a strategy to have conv->maxpool->activation
 * it means maxpool receives 4 bytes, and produces 4 bytes
 */
class ReorderActivationAndPooling : public ngraph::pass::MatcherPass {
public:
  OPENVINO_RTTI("ReorderActivationAndPooling", "0");
  ReorderActivationAndPooling();
};

} // namespace GNAPluginNS
