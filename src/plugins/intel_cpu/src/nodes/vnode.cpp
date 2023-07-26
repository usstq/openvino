// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnode.h"

#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "ie_parallel.hpp"
#include "transformations/cpu_opset/common/op/vnode.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

using vnode_executor_map = std::map<std::string, std::function<std::shared_ptr<vnode_executor>()>>;

template <typename executor>
void register_executor(vnode_executor_map& vem, std::string signature) {
    std::function<std::shared_ptr<vnode_executor>()> creator = []() {
        return std::make_shared<executor>();
    };
    vem[signature] = creator;
}

static vnode_executor_map register_all() {
    vnode_executor_map vem;
    // gpt2
    register_executor<gpt2_attention_executor<KT_REF, float>>(vem, "gpt2_attention,REF,FP32");
    register_executor<gpt2_attention_executor<KT_REF, ov::bfloat16>>(vem, "gpt2_attention,REF,BF16");
    // gptneox
    register_executor<gptneox_attention_executor<KT_REF, float>>(vem, "gptneox_attention,REF,FP32");
    register_executor<gptneox_attention_executor<KT_REF, ov::bfloat16>>(vem, "gptneox_attention,REF,BF16");
    // open_llama
    register_executor<open_llama_attention_executor<KT_REF, float>>(vem, "open_llama_attention,REF,FP32");
    register_executor<open_llama_attention_executor<KT_REF, ov::bfloat16>>(vem, "open_llama_attention,REF,BF16");
    // bloom
    register_executor<bloom_attention_executor<KT_REF, float>>(vem, "bloom_attention,REF,FP32");
    register_executor<bloom_attention_executor<KT_REF, ov::bfloat16>>(vem, "bloom_attention,REF,BF16");

    register_executor<opt_attention_executor<KT_REF, float>>(vem, "opt_attention,REF,FP32");
    register_executor<opt_attention_executor<KT_REF, ov::bfloat16>>(vem, "opt_attention,REF,BF16");

    register_executor<whisper_enc_attention_executor<KT_REF, float>>(vem, "whisper_enc_attention,REF,FP32");
    register_executor<whisper_enc_attention_executor<KT_REF, ov::bfloat16>>(vem, "whisper_enc_attention,REF,BF16");

    register_executor<whisper_dec_self_attn_executor<KT_REF, float>>(vem, "whisper_dec_self_attn,REF,FP32");
    register_executor<whisper_dec_self_attn_executor<KT_REF, ov::bfloat16>>(vem, "whisper_dec_self_attn,REF,BF16");

    register_executor<whisper_dec_enc_attn_executor<KT_REF, float>>(vem, "whisper_dec_enc_attn,REF,FP32");
    register_executor<whisper_dec_enc_attn_executor<KT_REF, ov::bfloat16>>(vem, "whisper_dec_enc_attn,REF,BF16");

    register_executor<whisper_dec2_self_attn_executor<KT_REF, float>>(vem, "whisper_dec2_self_attn,REF,FP32");
    register_executor<whisper_dec2_self_attn_executor<KT_REF, ov::bfloat16>>(vem, "whisper_dec2_self_attn,REF,BF16");

    register_executor<whisper_dec2_enc_attn_executor<KT_REF, float>>(vem, "whisper_dec2_enc_attn,REF,FP32");
    register_executor<whisper_dec2_enc_attn_executor<KT_REF, ov::bfloat16>>(vem, "whisper_dec2_enc_attn,REF,BF16");

    #ifdef OV_CPU_WITH_MLAS
    register_executor<gpt2_attention_executor<KT_MLAS, float>>(vem, "gpt2_attention,MLAS,FP32");
    register_executor<gptneox_attention_executor<KT_MLAS, float>>(vem, "gptneox_attention,MLAS,FP32");
    register_executor<open_llama_attention_executor<KT_MLAS, float>>(vem, "open_llama_attention,MLAS,FP32");
    register_executor<bloom_attention_executor<KT_MLAS, float>>(vem, "bloom_attention,MLAS,FP32");
    register_executor<opt_attention_executor<KT_MLAS, float>>(vem, "opt_attention,MLAS,FP32");
    register_executor<whisper_enc_attention_executor<KT_MLAS, float>>(vem, "whisper_enc_attention,MLAS,FP32");
    register_executor<whisper_dec_self_attn_executor<KT_MLAS, float>>(vem, "whisper_dec_self_attn,MLAS,FP32");
    register_executor<whisper_dec_enc_attn_executor<KT_MLAS, float>>(vem, "whisper_dec_enc_attn,MLAS,FP32");
    register_executor<whisper_dec2_self_attn_executor<KT_MLAS, float>>(vem, "whisper_dec2_self_attn,MLAS,FP32");
    register_executor<whisper_dec2_enc_attn_executor<KT_MLAS, float>>(vem, "whisper_dec2_enc_attn,MLAS,FP32");
    #endif
    #ifdef OV_CPU_WITH_LLM
    register_executor<gpt2_attention_executor<KT_LLMDNN, ov::bfloat16>>(vem, "gpt2_attention,LLMDNN,BF16");
    register_executor<gptneox_attention_executor<KT_LLMDNN, ov::bfloat16>>(vem, "gptneox_attention,LLMDNN,BF16");
    register_executor<open_llama_attention_executor<KT_LLMDNN, ov::bfloat16>>(vem, "open_llama_attention,LLMDNN,BF16");
    register_executor<bloom_attention_executor<KT_LLMDNN, ov::bfloat16>>(vem, "bloom_attention,LLMDNN,BF16");
    register_executor<opt_attention_executor<KT_LLMDNN, ov::bfloat16>>(vem, "opt_attention,LLMDNN,BF16");
    #endif
    return vem;
}

static std::shared_ptr<vnode_executor> vnode_executor_create(std::string vtype, InferenceEngine::Precision prec) {
    static vnode_executor_map registered_executors = register_all();
    static int use_ref = std::getenv("USE_REF") ? atoi(std::getenv("USE_REF")) : 0;
    std::string signature = vtype;
    if (use_ref) {
        signature += ",REF";
    } else if (prec == InferenceEngine::Precision::FP32) {
        signature += ",MLAS";
    } else if (prec == InferenceEngine::Precision::BF16) {
        signature += ",LLMDNN";
    }
    signature = signature + "," + prec.name();

    auto it = registered_executors.find(signature);
    if (it != registered_executors.end()) {
        auto exec = it->second();
        exec->signature = signature;
        return exec;
    }

    // fallback to reference and retry
    signature = vtype + ",REF," + prec.name();
    it = registered_executors.find(signature);
    if (it != registered_executors.end()) {
        auto exec = it->second();
        exec->signature = signature;
        return exec;
    }

    std::cout << " vnode_executor_create failed for " << signature << std::endl;
    return nullptr;
}

bool VNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    const auto vnode = std::dynamic_pointer_cast<const ov::intel_cpu::VNode>(op);
    if (!vnode) {
        errorMessage = "Only VNode operation is supported";
        return false;
    }

    return true;
}

VNode::VNode(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "VNode layer with name '" + getName() + "'";

    m_vnode = std::dynamic_pointer_cast<ov::intel_cpu::VNode>(op);
    m_vtype = m_vnode->get_vtype();
    m_symbol_name2value = op->get_rt_info()["symbol_name2value"].as<decltype(m_symbol_name2value)>();
}

void VNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != m_vnode->get_input_size())
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().size() < m_vnode->get_output_size())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

void VNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // orginal precision at input port 0 as a hint of runtime precisions
    auto runtime_precision = getOriginalInputPrecisionAtPort(0);

    m_executor = vnode_executor_create(m_vtype, runtime_precision);
    if (!m_executor) {
        IE_THROW() << errorPrefix << " unsupported vnode " << m_vtype << " with " << runtime_precision;
    }

    std::cout << getName() << " created executor: " << m_executor->signature << std::endl;

    std::vector<ov::intel_cpu::PortConfigurator> inPortConfigs;
    std::vector<ov::intel_cpu::PortConfigurator> outPortConfigs;
    for (auto* p : m_executor->inputs) {
        inPortConfigs.emplace_back(LayoutType::ncsp, p->get_precision());
    }
    for (auto* p : m_executor->outputs) {
        outPortConfigs.emplace_back(LayoutType::ncsp, p->get_precision());
    }
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void VNode::execute(dnnl::stream strm) {
    if (m_executor) {
        m_executor->exec(this, strm, m_symbol_name2value);
    } else {
        IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    }
}

void VNode::executeDynamicImpl(dnnl::stream strm) {
    if (m_executor) {
        m_executor->exec(this, strm, m_symbol_name2value);
    } else {
        IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    }
}

bool VNode::created() const {
    return getType() == Type::VNode;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov