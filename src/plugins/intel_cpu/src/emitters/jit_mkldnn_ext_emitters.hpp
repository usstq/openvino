// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/opsets/opset5.hpp"
#include "jit_mkldnn_emitters.hpp"

namespace ov {
namespace intel_cpu {

class jit_relu_emitter : public jit_mkldnn_emitter {
public:
    jit_relu_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_relu;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_sigmoid_emitter : public jit_mkldnn_emitter {
public:
    jit_sigmoid_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_logistic;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_tanh_emitter : public jit_mkldnn_emitter {
public:
    jit_tanh_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_tanh;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_elu_emitter : public jit_mkldnn_emitter {
public:
    jit_elu_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_elu;
            alpha = ngraph::as_type_ptr<ngraph::opset1::Elu>(n)->get_alpha();
            beta = 0.f;

            set_injector();
        }
};

class jit_exp_emitter : public jit_mkldnn_emitter {
public:
    jit_exp_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_exp;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_abs_emitter : public jit_mkldnn_emitter {
public:
    jit_abs_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_abs;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_clamp_emitter : public jit_mkldnn_emitter {
public:
    jit_clamp_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = dnnl_eltwise_clip;
            auto op = ngraph::as_type_ptr<ngraph::opset1::Clamp>(n);
            alpha = op->get_min();
            beta = op->get_max();

            set_injector();
        }
};

class jit_hswish_emitter : public jit_mkldnn_emitter {
public:
    jit_hswish_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
            : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_hardswish;
        alpha = 0.f;
        beta = 0.f;

        set_injector();
    }
};
class jit_gelu_v0_emitter : public jit_mkldnn_emitter {
public:
    jit_gelu_v0_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
            : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_gelu_erf;

        set_injector();
    }
};

class jit_gelu_v7_emitter : public jit_mkldnn_emitter {
public:
    jit_gelu_v7_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
            : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
        auto gelu = getNgraphOpAs<ngraph::op::v7::Gelu>(n);
        ngraph::op::GeluApproximationMode approximationMode = gelu->get_approximation_mode();
        if (approximationMode == ngraph::op::GeluApproximationMode::ERF)
            kind = dnnl_eltwise_gelu_erf;
        else if (approximationMode == ngraph::op::GeluApproximationMode::TANH)
            kind = dnnl_eltwise_gelu_tanh;
        else
            IE_THROW(NotImplemented) << "Subgraph node doesn't support ngraph operation Gelu with approximation mode: " << approximationMode;

        set_injector();
    }
};

class jit_round_emitter : public jit_mkldnn_emitter {
public:
    jit_round_emitter(
        mkldnn::impl::cpu::x64::jit_generator *host,
        mkldnn::impl::cpu::x64::cpu_isa_t host_isa,
        const std::shared_ptr<ngraph::Node>& n,
        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32) : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
        const auto round = getNgraphOpAs<ngraph::op::v5::Round>(n);
        const auto mode = round->get_mode();
        if ((mode != ngraph::opset5::Round::RoundMode::HALF_AWAY_FROM_ZERO) &&
            (mode != ngraph::opset5::Round::RoundMode::HALF_TO_EVEN)) {
            IE_THROW(NotImplemented) << "Round emitter doesn't support ngraph operation Round with mode: " << static_cast<int>(mode);
        }

        kind = mode == ngraph::opset5::Round::RoundMode::HALF_AWAY_FROM_ZERO ?
            dnnl_eltwise_round_half_away_from_zero :
            dnnl_eltwise_round_half_to_even;
        set_injector();
    }
};

}   // namespace intel_cpu
}   // namespace ov
