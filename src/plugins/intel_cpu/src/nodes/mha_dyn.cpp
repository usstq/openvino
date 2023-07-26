// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_dyn.h"

#include <ie_ngraph_utils.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "ie_parallel.hpp"
#include "utils/plain_tensor.hpp"
#include "utils/profiler.hpp"
#include "vnode_kernels.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

// MHADynamic still requires primitive cache to share executor among nodes
// but its key is shape-agnostic
struct MHADynamicKey {
    InferenceEngine::Precision::ePrecision runtime_precision;

    size_t hash() const;
    bool operator==(const MHADynamicKey& rhs) const;
};

size_t MHADynamicKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;
    size_t seed = 0;
    seed = hash_combine(seed, runtime_precision);
    return seed;
}

bool MHADynamicKey::operator==(const MHADynamicKey& rhs) const {
    return runtime_precision == rhs.runtime_precision;
}
}  // namespace

bool MHADynamic::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
                                      std::string& errorMessage) noexcept {
    const auto MHADynamic = std::dynamic_pointer_cast<const ov::intel_cpu::MHADynamic>(op);
    if (!MHADynamic) {
        errorMessage = "Only MHADynamic operation is supported";
        return false;
    }

    return true;
}

MHADynamic::MHADynamic(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "MHADynamic layer with name '" + getName() + "'";

    m_mha = std::dynamic_pointer_cast<ov::intel_cpu::MHADynamic>(op);
    m_config = m_mha->get_config();
}

void MHADynamic::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != m_mha->get_input_size())
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().size() < m_mha->get_output_size())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

template <typename T>
PlainTensor<T> relayout(PlainTensor<T> input, const std::vector<ov::intel_cpu::MHADynamic::ReLayout>& relayouts) {
    for (auto& r : relayouts) {
        if (r.do_reshape) {
            auto target_shape = r.param;
            // handling special_zero
            for (size_t i = 0; i < target_shape.size(); i++) {
                if (target_shape[i] == 0)
                    target_shape[i] = input.size(i);
            }
            input = input.reshape(target_shape);
            continue;
        }
        if (r.do_transpose) {
            input = input.permute(r.param);
            continue;
        }
        if (r.do_gather) {
            input = input.slice(r.param[1], r.param[0], r.param[0]);
            continue;
        }
    }
    return input;
}

std::ostream& operator<<(std::ostream& os, const KernelTypes& t) {
    if (t == KT_REF)
        os << "KT_REF";
    if (t == KT_LLMDNN)
        os << "KT_LLMDNN";
    if (t == KT_MLAS)
        os << "KT_MLAS";
    return os;
}

template <KernelTypes KT, typename T>
struct MHADynamicExecutor : public MHADynamic::ExecutorBase {
    MHA_kernel<KT, T> kernel;

    MHADynamicExecutor(Node* node) {
        std::cout << "MHADynamicExecutor<" << KT << "," << data_type_name<T>::value << "> " << node->getName()
                  << std::endl;
    }

    template <typename D>
    D* input_value(MHADynamic* pnode, int parent_edge_id) {
        auto mem = pnode->getParentEdgeAt(parent_edge_id)->getMemoryPtr();
        return reinterpret_cast<D*>(mem->getData());
    }

    void exec(MHADynamic* pnode, ov::intel_cpu::MHADynamic::Config& config) override {
        PlainTensor<T> q_input;
        PlainTensor<T> k_input;
        PlainTensor<T> v_input;
        float qk_scale = 1.0f;
        PlainTensor<float> alibi_mask;
        PlainTensor<uint8_t> causal_mask;
        PlainTensor<float> attn_mask;
        PlainTensor<T> result;
        int idx = 0;

        q_input.reset(pnode->getParentEdgeAt(idx++)->getMemoryPtr());
        k_input.reset(pnode->getParentEdgeAt(idx++)->getMemoryPtr());
        v_input.reset(pnode->getParentEdgeAt(idx++)->getMemoryPtr());

        q_input = relayout(q_input, config.relayout_query);
        k_input = relayout(k_input, config.relayout_key);
        v_input = relayout(v_input, config.relayout_value);

        auto B = v_input.size(0);
        auto H = v_input.size(1);
        auto kL = v_input.size(2);
        auto S = v_input.size(3);
        auto qL = q_input.size(2);

        if (config.q_reshape_to_3d) {
            auto* shape = input_value<int32_t>(pnode, idx++);
            if (shape[0] != B*H || shape[2] != S)
                IE_THROW() << pnode->errorPrefix << " q_reshape_to_3d is not expected!";
        }
        if (config.k_reshape_to_3d) {
            auto* shape = input_value<int32_t>(pnode, idx++);
            if (shape[0] != B*H || shape[2] != S)
                IE_THROW() << pnode->errorPrefix << " k_reshape_to_3d is not expected!";
        }
        if (config.v_reshape_to_3d) {
            auto* shape = input_value<int32_t>(pnode, idx++);
            if (shape[0] != B*H || shape[2] != S)
                IE_THROW() << pnode->errorPrefix << " v_reshape_to_3d is not expected!";
        }

        if (config.with_qk_scale) {
            qk_scale = input_value<float>(pnode, idx++)[0];
        }
        if (config.with_alibi_mask) {
            alibi_mask.reset(pnode->getParentEdgeAt(idx++)->getMemoryPtr());
        }
        if (config.with_causal_mask) {
            causal_mask.reset(pnode->getParentEdgeAt(idx++)->getMemoryPtr());
            if (config.with_causal_mask_slice1) {
                auto start = input_value<int32_t>(pnode, idx++)[0];
                auto stop = input_value<int32_t>(pnode, idx++)[0];
                auto step = input_value<int32_t>(pnode, idx++)[0];
                auto axis = input_value<int32_t>(pnode, idx++)[0];
                if (stop == start)
                    stop++;
                causal_mask = causal_mask.slice(axis, start, stop, step);
            }
            if (config.with_causal_mask_slice2) {
                auto start = input_value<int32_t>(pnode, idx++)[0];
                auto stop = input_value<int32_t>(pnode, idx++)[0];
                auto step = input_value<int32_t>(pnode, idx++)[0];
                auto axis = input_value<int32_t>(pnode, idx++)[0];
                if (stop == start)
                    stop++;
                causal_mask = causal_mask.slice(axis, start, stop, step);
            }
            if (config.with_causal_mask_stridedslice1) {
                auto start = input_value<int32_t>(pnode, idx++)[2];
                auto stop = input_value<int32_t>(pnode, idx++)[2];
                auto step = input_value<int32_t>(pnode, idx++)[2];
                causal_mask = causal_mask.slice(2, start, stop, step);
            }
            if (config.with_causal_mask_stridedslice2) {
                auto start = input_value<int32_t>(pnode, idx++)[3];
                auto stop = input_value<int32_t>(pnode, idx++)[3];
                auto step = input_value<int32_t>(pnode, idx++)[3];
                causal_mask = causal_mask.slice(3, start, stop, step);
            }
        }
        if (config.with_attn_mask) {
            attn_mask.reset(pnode->getParentEdgeAt(idx++)->getMemoryPtr());
            if (config.with_attn_mask_reshapes) {
                auto* shape4d = input_value<int32_t>(pnode, idx++);
                auto* shape3d = input_value<int32_t>(pnode, idx++);
                if (shape4d[0] != B || shape4d[1] != H || shape4d[2] != qL || shape4d[3] != kL)
                    IE_THROW() << pnode->errorPrefix << " attention reshape 4d is not expected!";
                if (shape3d[0] != B*H || shape3d[1] != qL || shape4d[2] != kL)
                    IE_THROW() << pnode->errorPrefix << " attention reshape 4d is not expected!";
            }
        }

        // optimized alibi mask is broadcasted in qL dimension
        // so it's shape maybe [B, H, 1, kL] or [B, H, qL, kL]

        // causal mask is broadcased in head (unless-combined with attn_mask)
        // so it's shape is [B,1,qL,kL] or [B,H,qL,kL]

        // attn_mask is [B, 1, qL, kL] ?

        // inference result shape
        VectorDims result_shape;
        if (config.result_is_3d) {
            result_shape = {B*H, qL, S};
        } else if (config.trans_result_to_BLHS) {
            result_shape = {B, qL, H, S};
        } else {
            result_shape = {B, H, qL, S};
        }

        pnode->redefineOutputMemory({result_shape});

        // result view is always in shape of
        result.reset(pnode->getChildEdgeAt(0)->getMemoryPtr());
        if (config.result_is_3d) {
            // B*H, qL, S
            result = result.reshape({B, H, qL, S}).permute({0, 2, 1, 3});
        } else if (!config.trans_result_to_BLHS) {
            // [B, H, qL, S] => [B, qL, H, S]
            result = result.permute({0, 2, 1, 3});
        }
        result = result.reshape({B, qL, H * S});

        // std::cout << q_input.repr() << std::endl;
        // std::cout << k_input.repr() << std::endl;
        // std::cout << v_input.repr() << std::endl;
        // std::cout << result.repr() << std::endl;

        // [B, 1, qL, kL]
        // std::cout << causal_mask.repr() << std::endl;
        // std::cout << attn_mask.repr() << std::endl;
        // [B, H, 1, kL]
        // std::cout << alibi_mask.repr() << std::endl;
        // asm("int3");
        // combine causal & attn_mask
        // note if only mask is attn_mask

        // call MHA kernel
        kernel.set_causal_mask(causal_mask, config.select_nfltmax_at_0);
        kernel(q_input, k_input, v_input, alibi_mask, attn_mask, result, qk_scale);
    }
};

void MHADynamic::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto builder = [&](const MHADynamicKey& key) -> std::shared_ptr<ExecutorBase> {
        std::shared_ptr<ExecutorBase> executor;
        int use_ref = std::getenv("USE_REF") ? atoi(std::getenv("USE_REF")) : 0;
        if (key.runtime_precision == InferenceEngine::Precision::FP32) {
            if (use_ref)
                executor = std::make_shared<MHADynamicExecutor<KT_REF, float>>(this);
            else
                executor = std::make_shared<MHADynamicExecutor<KT_MLAS, float>>(this);
        } else {
            if (use_ref)
                executor = std::make_shared<MHADynamicExecutor<KT_REF, ov::bfloat16>>(this);
            else
                executor = std::make_shared<MHADynamicExecutor<KT_LLMDNN, ov::bfloat16>>(this);
        }
        return executor;
    };

    // orginal precision at input port 0 as a hint of runtime precisions
    m_runtime_precision = getOriginalInputPrecisionAtPort(0);
    MHADynamicKey key{m_runtime_precision};

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    m_executor = result.first;

    if (!m_executor)
        IE_THROW() << errorPrefix << " Not implemented";

    std::vector<ov::intel_cpu::PortConfigurator> inPortConfigs;
    std::vector<ov::intel_cpu::PortConfigurator> outPortConfigs;

    inPortConfigs.emplace_back(LayoutType::ncsp, m_runtime_precision);
    inPortConfigs.emplace_back(LayoutType::ncsp, m_runtime_precision);
    inPortConfigs.emplace_back(LayoutType::ncsp, m_runtime_precision);

    // the rest inputs keep original precision
    for (size_t i = 3; i < m_mha->get_input_size(); i++) {
        inPortConfigs.emplace_back(LayoutType::ncsp, details::convertPrecision(m_mha->get_input_element_type(i)));
    }
    for (size_t i = 0; i < m_mha->get_output_size(); i++) {
        outPortConfigs.emplace_back(LayoutType::ncsp, m_runtime_precision);
    }
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void MHADynamic::execute(dnnl::stream strm) {
    m_executor->exec(this, m_config);
}

void MHADynamic::executeDynamicImpl(dnnl::stream strm) {
    m_executor->exec(this, m_config);
}

bool MHADynamic::created() const {
    return getType() == Type::MHADynamic;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov