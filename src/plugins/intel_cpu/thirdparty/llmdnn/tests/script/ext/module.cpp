// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <torch/extension.h>
#include <memory>
#include "module.hpp"
#include "utility_amx.hpp"
#include "mha_gpt.hpp"
#include "test_common.hpp"

PYBIND11_MODULE(libllmdnn_ext, m) {
    static bool initAMX = initXTILE();
    if (!initAMX) {
        std::cout << "init amx failed.\n";
    }
    regclass_mha_gpt(m);
}