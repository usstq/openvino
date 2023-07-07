// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <vector>
#include <array>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void scale_add_softmax(float* a, float scale, float* mask, float* aliba, size_t len, size_t total_size);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
