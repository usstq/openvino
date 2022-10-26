// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_postops_composer.h"

#include <common/primitive_attr.hpp>

#include "node.h"

namespace ov {
namespace intel_cpu {

DnnlPostOpsComposer::DnnlPostOpsComposer(ov::intel_cpu::Node* node,
                                         dnnl::primitive_attr& attr,
                                         dnnl::post_ops& ops,
                                         std::vector<MemoryPtr>& args,
                                         const VectorDims& outputDims,
                                         int indexOfOutputChannelDim,
                                         bool isINT8)
    : node(node),
      attr(attr),
      ops(ops),
      args(args),
      outputDims(outputDims),
      dimOC(indexOfOutputChannelDim),
      isINT8(isINT8) {
    IE_ASSERT(dimOC >= 0 && dimOC < outputDims.size());
    OC = outputDims[dimOC];
    dimsPerOC = dimsPerTensor = VectorDims(outputDims.size(), 1);
    dimsPerOC[dimOC] = OC;
}

void DnnlPostOpsComposer::appendBinary(const dnnl::algorithm alg, const std::vector<float>& data) {
    VectorDims* pdims = &dimsPerTensor;
    if (data.size() > 1) {
        IE_ASSERT(data.size() == OC);
        pdims = &dimsPerOC;
    }
    DnnlBlockedMemoryDesc memoryDesc(InferenceEngine::Precision::FP32, Shape(*pdims));
    ops.append_binary(alg, memoryDesc.getDnnlDesc());

    // copy the data as args
    auto mem = std::make_shared<Memory>(node->getEngine());
    mem->Create(memoryDesc);
    memcpy(mem->GetPtr(), data.data(), data.size() * sizeof(float));
    args.emplace_back(std::move(mem));
}

void DnnlPostOpsComposer::appendEltwise(float scale, const dnnl::algorithm alg, float alpha, float beta) {
    ops.append_eltwise(scale, alg, alpha, beta);
}

void DnnlPostOpsComposer::appendRoundHTE() {
    ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_round_half_to_even, 0, 0);
}

bool DnnlPostOpsComposer::appendScale(const std::vector<float>& scale, bool allowBinary) {
    IE_ASSERT(scale.size() == OC || scale.size() == 1);
    // there are so many possible optimizations can be done, for example:
    //
    // we can switch the existing postOps's order to take
    // advantage of output scale if it's available:
    //    relu(x)*scale = relu(x*scale)
    // or we can fuse it into previous one as long as they are
    // compatible in shape
    //    x*A*s = x*(A*s)
    // or even with add:
    //    (x*A + B)*s = x*(A*s) + (B*s)
    // or we can combine these two tricks:
    //    relu(x*A)*s = relu(x*(A*s))
    //
    // we cannot implement all of them, so we just add the one
    // that we observed in real models.

    // fuse into existing output scale (only when isINT8)
    bool can_fuse_into_oscale = false;
    if (isINT8) {
        if (ops.len() == 0)
            can_fuse_into_oscale = true;

        // relu(x)*s = relu(x*s)
        if (ops.len() == 1) {
            auto& cur_op = ops.get()->entry_[0];
            if (cur_op.kind == dnnl::impl::primitive_kind::eltwise && cur_op.eltwise.alg == dnnl_eltwise_relu) {
                can_fuse_into_oscale = true;
            }
        }

        // (x + dst[:])*s = (x*s + s*dst[:])
        if (scale.size() == 1 && ops.len() == 1) {
            auto& cur_op = ops.get()->entry_.back();
            if (cur_op.kind == dnnl::impl::primitive_kind::sum) {
                cur_op.sum.scale *= scale[0];
                can_fuse_into_oscale = true;
            }
        }
    }

    if (can_fuse_into_oscale) {
        int mask;
        std::vector<float> outScales;
        attr.get_output_scales(mask, outScales);
        if (scale.size() > 1) {
            if (mask == 0)
                outScales.resize(scale.size(), outScales[0]);
            else
                IE_ASSERT(outScales.size() == OC);
            for (int j = 0; j < OC; j++)
                outScales[j] *= scale[j];
        } else {
            for (int j = 0; j < outScales.size(); j++)
                outScales[j] *= scale[0];
        }

        if (outScales.size() == 1)
            mask = 0;
        else
            mask = 1 << 1;  // it works for both Conv/Matmul

        attr.set_output_scales(mask, outScales);
        return true;
    }

    // eltwise(x, scale, alpha, beta)*s = eltwise(x, (scale*s), alpha, beta)
    if (scale.size() == 1 && ops.len() > 0) {
        auto& cur_op = ops.get()->entry_.back();
        if (cur_op.kind == dnnl::impl::primitive_kind::eltwise) {
            cur_op.eltwise.scale *= scale[0];
            return true;
        }
    }

    // final fallback
    if (scale.size() == 1) {
        ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, scale[0], 0);
    } else {
        // this check returns before committing any changes
        if (!allowBinary)
            return false;
        appendBinary(dnnl::algorithm::binary_mul, scale);
    }
    return true;
}

void DnnlPostOpsComposer::appendShift(const std::vector<float>& shift) {
    if (shift.size() == 1) {
        if (shift[0] != 0.0f)
            ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, shift[0]);
    } else {
        appendBinary(dnnl::algorithm::binary_add, shift);
    }
}

bool DnnlPostOpsComposer::appendLinear(const std::vector<float>& scale,
                                       const std::vector<float>& shift,
                                       bool allowBinary) {
    if (scale.size() == 1 && shift.size() == 1) {
        ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, scale[0], shift[0]);
    } else {
        // return before committing any changes
        if (!allowBinary && shift.size() > 1)
            return false;

        if (scale.size() > 0) {
            if (!appendScale(scale, allowBinary))
                return false;
        }
        if (shift.size() > 0)
            appendShift(shift);
    }
    return true;
}

void DnnlPostOpsComposer::appendClip(const std::vector<float>& low, const std::vector<float>& high) {
    if (low.size() == 1 && high.size() == 1) {
        ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip, low[0], high[0]);
    } else if (low.size() == 1) {
        IE_ASSERT(high.size() == OC);
        ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip, low[0], std::numeric_limits<float>::max());
        if (high.size() > 0)
            appendBinary(dnnl::algorithm::binary_min, high);
    } else if (high.size() == 1) {
        IE_ASSERT(low.size() == OC);
        ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip, -std::numeric_limits<float>::max(), high[0]);
        if (low.size() > 0)
            appendBinary(dnnl::algorithm::binary_max, low);
    } else {
        if (low.size() > 0) {
            IE_ASSERT(low.size() == OC);
            appendBinary(dnnl::algorithm::binary_max, low);
        }
        if (high.size() > 0) {
            IE_ASSERT(high.size() == OC);
            appendBinary(dnnl::algorithm::binary_min, high);
        }
    }
}

}  // namespace intel_cpu
}  // namespace ov
