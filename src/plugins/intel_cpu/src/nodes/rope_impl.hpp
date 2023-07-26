// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void rope_impl(const float* input, const float* sin, const float* cos, float* output, size_t rotary_dims);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
