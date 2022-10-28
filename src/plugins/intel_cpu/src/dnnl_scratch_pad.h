// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "common/memory.hpp"
#include "cpu_memory.h"
#include "dnnl_extension_utils.h"

namespace ov {
namespace intel_cpu {

class DnnlScratchPad {
    DnnlMemoryMngrPtr mgrPtr;
    dnnl::engine eng;

public:
    DnnlScratchPad(dnnl::engine eng) : eng(eng) {
        auto * p = new MemoryMngrWithReuse();
        mgrPtr = std::make_shared<DnnlMemoryMngr>(std::unique_ptr<MemoryMngrWithReuse>(p));
    }

    MemoryPtr getScratchPadMem(const DnnlMemoryDescPtr& md) {
        if (md->getMaxMemSize() == 0)
            return nullptr;
        auto mem = std::make_shared<Memory>(eng);
        mem->Create(md, mgrPtr);
        return mem;
    }
};

using DnnlScratchPadPtr = std::shared_ptr<DnnlScratchPad>;

}  // namespace intel_cpu
}  // namespace ov
