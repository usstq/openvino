// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling OneDNN attributes & post ops.
 * @file dnnl_postops_composer.h
 */
#pragma once

#include <dnnl_types.h>

#include <string>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/dnnl.h"

namespace ov {
namespace intel_cpu {

class Node;

// so far the API only support per-Tensor or per-OC
class DnnlPostOpsComposer {
    dnnl::primitive_attr& attr;
    dnnl::post_ops& ops;
    std::vector<MemoryPtr>& args;
    const VectorDims outputDims;
    int dimOC;
    VectorDims dimsPerTensor;
    VectorDims dimsPerOC;
    int OC;
    const bool isINT8;  // isINT8 has no output scale
    ov::intel_cpu::Node* node;

public:
    DnnlPostOpsComposer(ov::intel_cpu::Node* node,
                        dnnl::primitive_attr& attr,
                        dnnl::post_ops& ops,
                        std::vector<MemoryPtr>& args,
                        const VectorDims& outputDims,
                        int indexOfOutputChannelDim,
                        bool isINT8);

    void appendBinary(const dnnl::algorithm alg, const std::vector<float>& data);
    void appendEltwise(float scale, const dnnl::algorithm alg, float alpha, float beta);
    void appendRoundHTE();
    bool appendScale(const std::vector<float>& scale, bool allowBinary = true);
    void appendShift(const std::vector<float>& shift);
    bool appendLinear(const std::vector<float>& scale, const std::vector<float>& shift, bool allowBinary = true);
    void appendClip(const std::vector<float>& low, const std::vector<float>& high);
};

}  // namespace intel_cpu
}  // namespace ov
