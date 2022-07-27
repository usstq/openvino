// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace intel_cpu {

bool flush_to_zero(bool on);
bool denormals_as_zero(bool on);
bool is_denormals_as_zero_set();

}   // namespace intel_cpu
}   // namespace ov
