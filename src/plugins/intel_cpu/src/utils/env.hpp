// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <string>

namespace ov {
namespace intel_cpu {

struct EnvInt{
    int v;
    EnvInt(const char * name, int v_default) {
        auto * pv = std::getenv(name);
        if (!pv)
            v = v_default;
        else
            v = std::stoll(std::string(pv), nullptr, 0);
    }
    operator bool() { return v; }
    operator int() { return v; }
};

}   // namespace intel_cpu
}   // namespace ov
