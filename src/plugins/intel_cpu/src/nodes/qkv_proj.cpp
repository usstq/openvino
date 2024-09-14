// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qkv_proj.h"

#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#include "kernels/x64/mlp_utils.hpp"
#endif

#include "openvino/core/parallel.hpp"

#include "nodes/linux_perf.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

#if defined(OPENVINO_ARCH_X86_64)
static std::vector<int> allocate_workers(const std::vector<int>& grouped_works, int n_workers) {
    auto n_groups = grouped_works.size();
    // allocate 1 worker for each group
    std::vector<int> g_workers(n_groups, 1);
    auto left_workers = n_workers - n_groups;
    while (left_workers > 0) {
        // which group is working hardest?
        float hardest_works = 0;
        size_t hardest_group = 0;
        for (size_t g = 0; g < n_groups; g++) {
            auto works = static_cast<float>(grouped_works[g]) / g_workers[g];
            if (hardest_works < works) {
                hardest_works = works;
                hardest_group = g;
            }
        }
        g_workers[hardest_group]++;
        left_workers--;
    }

    return g_workers;
}

struct QKVProjection::Impl {
    std::vector<Work> works;
    QKVProjection * m_node;
    DnnlScratchPadPtr m_scrachPad;
    MemoryPtr m_scratchMem;
    uint8_t* m_scratch_base = nullptr;
    int m_M = 0;

    MatrixDynQuantPerRow m_quant_act;

    WeightBuffer wbuffer;

    Impl(QKVProjection * pnode, DnnlScratchPadPtr scrachPad) : m_node(pnode), m_scrachPad(scrachPad) {
        PlainTensor w0(pnode->getSrcMemoryAtPort(1));
        PlainTensor w1(pnode->getSrcMemoryAtPort(2));
        PlainTensor w2(pnode->getSrcMemoryAtPort(3));

        // in quantized mode, weights are already quantized in per-OC mode into INT8
        // and activations will be dynamically per-token quantized and using AMX-INT8 to get the result
        bool quantized_int8 = m_node->m_config.quantized;

        auto cache_blk_k_size = quantized_int8 ? CACHE_BLK_K_SIZE : CACHE_BLK_K_SIZE;
        auto weight_element_size = quantized_int8 ? sizeof(int8_t) : sizeof(ov::float16);

        auto K = w0.size(1);
        OPENVINO_ASSERT((K % cache_blk_k_size) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_K = K / cache_blk_k_size;
        int stride_in_bytes = K * weight_element_size;

        works.resize(nthr);

        int cur_work_id = 0;
        auto create_works = [&](void* pw, int output_id, int N, int valid_nthr) {
            // split task on more cores is better on TBB
            OPENVINO_ASSERT((N % REG_BLK_N_SIZE) == 0);
            auto num_blk_N = N / REG_BLK_N_SIZE;
            auto blkN_per_thread = (num_blk_N) / valid_nthr;
            auto blkN_leftover = num_blk_N - (blkN_per_thread * valid_nthr);
            auto start_blkN = 0;

            for (int ithr = 0; ithr < valid_nthr; ithr++) {
                auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
                if (blkN_leftover > 0) {
                    blkN_leftover--;
                    blkN++;
                }
                if (blkN) {
                    auto& work = works[cur_work_id++];
                    work.blk_K_size = cache_blk_k_size;
                    work.n0 = (start_blkN) * REG_BLK_N_SIZE;
                    work.n1 = (start_blkN + blkN) * REG_BLK_N_SIZE;
                    work.BN = blkN * REG_BLK_N_SIZE;
                    work.k0 = 0;
                    work.k1 = cache_blk_k_size * num_blk_K;
                    work.output_id = output_id;
                    work.p_raw_weights = pw;
                    work.quant_i8 = quantized_int8;
                }
                start_blkN += blkN;
            }
        };
        auto proj_size0 = static_cast<int>(w0.size(0));
        auto proj_size1 = static_cast<int>(w1.size(0));
        auto proj_size2 = static_cast<int>(w2.size(0));
        auto n_group_workers = allocate_workers({proj_size0, proj_size1, proj_size2}, nthr);

        create_works(w0.ptr_v(), 0, proj_size0, n_group_workers[0]);
        create_works(w1.ptr_v(), 1, proj_size1, n_group_workers[1]);
        create_works(w2.ptr_v(), 2, proj_size2, n_group_workers[2]);

        DEBUG_LOG("QKVProj hidden_size=", K, " proj_sizes=",
                    proj_size0, ",", proj_size1, ",", proj_size2,
                    " used_nthr=", cur_work_id);

        wbuffer.alloc(works, weight_element_size);

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                if (quantized_int8)
                    work.setup(wbuffer.get<int8_t>(ithr), reinterpret_cast<int8_t*>(work.p_raw_weights), stride_in_bytes, true);
                else
                    work.setup(wbuffer.get<ov::bfloat16>(ithr), reinterpret_cast<ov::float16*>(work.p_raw_weights), stride_in_bytes);
            }
        });
    }

    void setM(int M) {
        uint8_t* cur_scratch_base = nullptr;
        if (m_scratchMem)
            cur_scratch_base = m_scratchMem->getDataAs<uint8_t>();
        // new M larger than previous or the scratch pointer is changed after the following allocation
        if (m_M < M || cur_scratch_base != m_scratch_base) {
            ScratchBuffAllocator allocator;
            for (auto& work : works) {
                if (work) {
                    auto C_size = work.set_C(M, reinterpret_cast<float*>(cur_scratch_base));
                    allocator.register_allocation(C_size, [&](void* ptr){
                        work.set_C(M, reinterpret_cast<float*>(ptr));
                    });
                }
            }

            if (m_node->m_config.quantized) {
                m_quant_act.M = M;
                m_quant_act.K = m_node->m_config.hidden_size;
                allocator.register_allocation(m_quant_act.size(), [&](void* ptr){
                    m_quant_act.setup(ptr);
                });
            }

            // make sure scrach is big enough
            auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{allocator.size()});
            m_scratchMem = m_scrachPad->createScratchPadMem(newMemDesc);
            m_scratch_base = m_scratchMem->getDataAs<uint8_t>();

            allocator.finalize(m_scratch_base);
            m_M = M;
        }
    }

    void execute() {
        static ReduceAdd2bh jit_2bh(false);
        auto input = m_node->getSrcMemoryAtPort(0);
        const auto& ishape = input->getStaticDims();
        uint8_t* pInput = input->getDataAs<uint8_t>();
        int M = shape_size(ishape) / ishape[ishape.size() - 1];
        auto* dst0 = m_node->getDstMemoryAtPort(0)->getDataAs<ov::bfloat16>();
        auto* dst1 = m_node->getDstMemoryAtPort(1)->getDataAs<ov::bfloat16>();
        auto* dst2 = m_node->getDstMemoryAtPort(2)->getDataAs<ov::bfloat16>();

        float* w_scale[3];

        if (m_node->m_config.quantized) {
            w_scale[0] = m_node->getSrcMemoryAtPort(4)->getDataAs<float>();
            w_scale[1] = m_node->getSrcMemoryAtPort(5)->getDataAs<float>();
            w_scale[2] = m_node->getSrcMemoryAtPort(6)->getDataAs<float>();
        }

        const auto& srcStrides = input->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& dstStrides0 = m_node->getDstMemoryAtPort(0)->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& dstStrides1 = m_node->getDstMemoryAtPort(1)->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& dstStrides2 = m_node->getDstMemoryAtPort(2)->getDescWithType<BlockedMemoryDesc>()->getStrides();

        int strideA = srcStrides[1] * sizeof(ov::bfloat16);
        auto stride_0 = dstStrides0[1];
        auto stride_1 = dstStrides1[1];
        auto stride_2 = dstStrides2[1];

        auto prof = LinuxPerf::Profile("QKV", 0, M);

        auto asym = true;
        for (int m = 0; m < M;) {
            auto prof = LinuxPerf::Profile(0.1f, "loop", M);
            int BM = std::min(M - m, CACHE_BLK_M_SIZE);

            setM(BM);

            // dynamic quantize input tensor A[m0:m1, :] into scratch buffer
            // because it's being shared by all kernels
            uint8_t* pA = pInput;
            if (m_node->m_config.quantized) {
                // quantize pInput into m_quantized_act buffer
                // per-token asym
                m_quant_act.quantize(BM, reinterpret_cast<ov::bfloat16*>(pInput), srcStrides[1]);
                pA = reinterpret_cast<uint8_t*>(m_quant_act.data);
                strideA = m_quant_act.K;
            }

            ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
                auto& work = works[ithr];
                if (work) {
                    {
                        auto prof = LinuxPerf::Profile("work", ithr, work.output_id, work.BN);
                        work.run(BM, pA, strideA);
                    }

                    // determine destination buffer
                    ov::bfloat16* dst = nullptr;
                    int stride_dst = 0;

                    if (work.output_id == 0) {
                        dst = dst0 + work.n0;
                        stride_dst = stride_0;
                    }
                    if (work.output_id == 1) {
                        dst = dst1 + work.n0;
                        stride_dst = stride_1;
                    }
                    if (work.output_id == 2) {
                        dst = dst2 + work.n0;
                        stride_dst = stride_2;
                    }

                    auto* src = work.m_C.ptr<float>();
                    auto stride_src = work.m_C.stride(0);
                    if (m_node->m_config.quantized) {
                        // dequantize output & convert to f32 in-place
                        auto prof = LinuxPerf::Profile("dequant", ithr);
                        ov::Extensions::Cpu::XARCH::llm_mlp_dequantize_i32_f32(
                            BM,
                            work.BN,
                            reinterpret_cast<int32_t*>(src),
                            stride_src,
                            src,
                            stride_src,
                            m_quant_act.scale,
                            m_quant_act.zp,
                            work.w_sum_per_oc.ptr<float>(),
                            w_scale[work.output_id] + work.n0,
                            asym);
                    }
                    // compress accumulation result into target
                    for (int mi = 0; mi < BM; mi++, src += stride_src, dst += stride_dst) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (mi + 2 < BM) ? (dst + 2 * stride_dst) : (dst);
                        jit_2bh(src, dst, prefetch_dst, work.BN);
                    }
                }
            });
            m += BM;
            pInput += BM * strideA;
            dst0 += BM * stride_0;
            dst1 += BM * stride_1;
            dst2 += BM * stride_2;
        }
    }
};
#else
struct QKVProjection::Impl {
    Impl(QKVProjection * pnode, DnnlScratchPadPtr scrachPad) {}
    void execute() {}
};
#endif

void QKVProjection::prepareParams() {
    if (!m_pimpl) {
        m_pimpl = std::make_shared<Impl>(this, context->getScratchPad());
    }
}

void QKVProjection::execute(dnnl::stream strm) {
    MAYBE_UNUSED(strm);
    m_pimpl->execute();
}

QKVProjection::QKVProjection(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
    const auto node = std::dynamic_pointer_cast<const QKVProjectionNode>(op);
    m_config = node->get_config();
}

void QKVProjection::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inPortConfigs;
    std::vector<PortConfigurator> outPortConfigs;

    if (m_config.quantized) {
        auto rtPrecision = ov::element::bf16;
        auto weightPrecision = ov::element::i8;
        auto wScalePrecision = ov::element::f32;

        inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // q_proj
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // k_proj
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(3), false, -1);  // v_proj
        inPortConfigs.emplace_back(LayoutType::ncsp, wScalePrecision, getInputShapeAtPort(4), false, -1);  // q_proj deq-scale per-OC
        inPortConfigs.emplace_back(LayoutType::ncsp, wScalePrecision, getInputShapeAtPort(5), false, -1);  // k_proj deq-scale per-OC
        inPortConfigs.emplace_back(LayoutType::ncsp, wScalePrecision, getInputShapeAtPort(6), false, -1);  // v_proj deq-scale per-OC

        // initialize output port
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(1), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(2), false, -1);
    } else {
        auto rtPrecision = ov::element::bf16;
        auto weightPrecision = ov::element::f16;

        // initialize input ports
        inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // q_proj
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // k_proj
        inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(3), false, -1);  // v_proj

        // initialize output port
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(1), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(2), false, -1);
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

bool QKVProjection::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage, int concurrency) noexcept {
#if defined(OPENVINO_ARCH_X86_64)
    try {
        const auto node_qkv = std::dynamic_pointer_cast<const QKVProjectionNode>(op);
        if (node_qkv) {
            if (concurrency > 0) {
                if (concurrency < 3) {
                    errorMessage = "QKVProjection needs at least 3 cores to work";
                    return false;
                }
                float unbalance_ratio = static_cast<float>(concurrency % 3)/static_cast<float>(concurrency / 3);
                if (unbalance_ratio > 0.2f) {
                    errorMessage = "QKVProjection needs number of cores to be nearly multiple of 3";
                    return false;
                }
            }
            auto proj_pshape1 = op->input_value(1).get_shape();
            auto proj_pshape2 = op->input_value(2).get_shape();
            auto proj_pshape3 = op->input_value(3).get_shape();
            if ((proj_pshape1[1] % CACHE_BLK_K_SIZE) != 0) {
                errorMessage = "QKVProjection input channel size is not multiple of cache blocking size";
                return false;
            }

            auto reg_blk_k_size = node_qkv->get_config().quantized ? REG_BLK_K_SIZE_I8 : REG_BLK_K_SIZE;
            if ((proj_pshape1[0] % reg_blk_k_size) != 0) {
                errorMessage = "QKVProjection 1st proj output channel size is not multiple of register blocking size";
                return false;
            }
            if ((proj_pshape2[0] % reg_blk_k_size) != 0) {
                errorMessage = "QKVProjection 2nd proj output channel size is not multiple of register blocking size";
                return false;
            }
            if ((proj_pshape3[0] % reg_blk_k_size) != 0) {
                errorMessage = "QKVProjection 3rd proj output channel size is not multiple of register blocking size";
                return false;
            }
        } else {
            errorMessage = "Only QKVProjection operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
#else
    return false;
#endif
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
