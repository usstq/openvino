// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

// some MHA pattern using torch.baddbmm introduces redundant broadcast
// that can be eliminated by symbolic shape inference
class EliminateFutileBcasts: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateFutileBcasts", "0");
    EliminateFutileBcasts();
};

}  // namespace intel_cpu
}  // namespace ov
