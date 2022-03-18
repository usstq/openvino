// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_emitter.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

#define SNIPPETS_MAX_SNIPPETS_DIMS 7
#define SNIPPETS_MAX_HARNESS_DIMS 5
#define SNIPPETS_MAX_TILE_RANK 2
#define GET_OFF(field) offsetof(jit_snippets_call_args, field)
struct jit_snippets_call_args {
    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
};

struct jit_snippets_compile_args {
    int64_t scheduler_dims[SNIPPETS_MAX_TILE_RANK] = {};
    int64_t scheduler_offsets[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    int64_t data_offsets[SNIPPETS_MAX_SNIPPETS_DIMS * SNIPPETS_MAX_HARNESS_DIMS] = {};
    std::vector<size_t> output_dims = {};
};
///
/// \brief    Kernel is the only entry point to Codogen Jit compilation. Kernel calculates appropriate data offsets,
/// and invokes enclosed outer Tiles. Only 2d Tiles are currently supported, so the emitters should
/// be organized in the following way:
/// KernelEmitter {          /* entry point */
///     TileEmitter {        /* outer tile */
///         TileEmitter {    /* inner vector tile */
///             ...          /* All the necessary Load/Strore/elementwise emitters */
///         }
///         TileEmitter {    /* inner scalar tile for tail processing */
///             ...          /* All the necessary Load/Strore/elementwise emitters */
///         }
///     }
/// }
/// Note that Kernel params are passed directly to the emit_code(). The vector of inputs should contain 2 arguments, the
/// output vector should be empty. Input parameters
///
/// \param      in[0]       The number of the node inputs
/// \param      in[1]      The number of the node outputs
///
// Todo: Scheduler dims and offsets are currently calculated in Subgraph node and passed to the KernelEmitter.
//  However, it seems more natural to calculate all the offsets right in the Kernel op, because the calculation is
//  not device-specific. It is based only on input/output dims (which we already know) and harness num dims
//  (which we should pass from the plugin). It seems also better to wrap the enclosed emitters in tiles in the Kernel op
//  and avoid creating empty tiles.
class KernelEmitter : public jit_emitter {
public:
    KernelEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa,
    const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n) {
        const auto kernel = ov::as_type_ptr<ngraph::snippets::op::Kernel>(n);
        if (!kernel)
            IE_THROW() << "KernelEmitter invoked with invalid op argument";
        if (!kernel->compile_params)
            IE_THROW() << "KernelEmitter invoked without compile_params";
        code = kernel->region;
        jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    }

    size_t get_inputs_num() const override {return 0;}

    void emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
              const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        validate_arguments(in, out, pool, gpr);
        emit_impl(in, out, pool, gpr, nullptr);
    }

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out,
                            const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        if (in.size() != 2)
            IE_THROW() << "KernelEmitter got invalid number of inputs. Expected 2, got " << in.size();
        if (out.size() != 0)
            IE_THROW() << "KernelEmitter got unexpected output arguments.";
        const size_t num_params = in[0] + in[1];
        if (num_params > SNIPPETS_MAX_SNIPPETS_DIMS)
            IE_THROW() << "KernelEmitter supports only up to " << SNIPPETS_MAX_SNIPPETS_DIMS <<
                       " parameters, got " << num_params;
        const int64_t harness_num_dims = jcp.output_dims.size() - 1;
        if (harness_num_dims > SNIPPETS_MAX_HARNESS_DIMS)
            IE_THROW() << "KernelEmitter supports harness with up to " << SNIPPETS_MAX_HARNESS_DIMS <<
                       " dims, got " << harness_num_dims;
    }

    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override {
        const size_t num_inputs = in[0];
        const size_t num_outputs = in[1];
        const size_t num_params = num_inputs + num_outputs;
        int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
        const int64_t harness_num_dims = jcp.output_dims.size() - 1;

        Reg64 reg_indexes   { dnnl::impl::cpu::x64::abi_param1 };
        Reg64 reg_const_params { dnnl::impl::cpu::x64::abi_param2 };
        Xbyak::Reg64 reg_tmp_64 { dnnl::impl::cpu::x64::abi_not_param1};

        h->preamble();

        std::vector<Reg64> regs(num_params);
        auto init_ptrs_with_offsets = [&](Reg64 pointer, const int64_t *offsets) {
            for (int j = 0; j < harness_num_dims; j++) {
                if (jcp.output_dims[j] != 1 && offsets[j] != 0) {
                    h->mov(reg_tmp_64, offsets[j]);
                    h->imul(reg_tmp_64, h->ptr[reg_indexes + j * sizeof(size_t)]);
                    h->add(pointer, reg_tmp_64);
                }
            }
        };
        for (auto i = 0; i < num_params; i++) {
            regs[i] = Reg64(reg64_tmp_start + i);
            if (i < num_inputs)
                h->mov(regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
            else
                h->mov(regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
            init_ptrs_with_offsets(regs[i], &jcp.data_offsets[i * harness_num_dims]);
        }

        for (auto& c : code) {
            c.first->emit_code(c.second.first, c.second.second, pool, gpr);
        }

        h->postamble();
    }

    jit_snippets_compile_args jcp;
    std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>> code;
};
///
/// \brief    Tile is designed to organize loop over the input and output data. It is essentially a for(...) loop:
/// it calculates the total number of iterations, performs operations specified by enclosed emitters, advances iteration counters
/// and breaks when necessary.
///
/// \param      in[0]    The number of input entities (or scheduler counts) processed during one iteration of the tile.
/// It is expected to be 1 for outer or scalar tiles and vlen for vector tiles.
/// \param      in[1]    Increment of the previous Tile in current dimension. Must be 0 if this is the first Tile.
/// So previous_inc is zero for outer and vector tiles (the are the first in dim) and vlen for scalar tiles (they usually go after vector Tiles).
/// \param      in[2]    sum number inputs and number of outputs of the node.
/// \param      in[3]    dimension of the tile. Note that only 2d Tile are currently supported, so dim is 0 for outer tiles, 1 for inner tiles.
///
// Todo: Inner and outer tiles have different semantics. For example, outer tile always has the increment == 1, and it can contain only
//  tile emitters (one outer or two inner). So it seems better to create different classes for inner and outer tiles.
// Todo: Currently data pointers incremented after each read/write in Load/Store emitters, so we have to decrement them here
//  if the same data needs to be read twice. Better to move all the pointer increments to TileEmitter and avoid the increments if necessary.
class TileEmitter : public jit_emitter {
public:
    TileEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa,
    const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n) {
        const auto tile = ov::as_type_ptr<ngraph::snippets::op::Tile>(n);
        if (!tile)
            IE_THROW() << "TileEmitter invoked with invalid op argument";
        if (!tile->compile_params)
            IE_THROW() << "TileEmitter invoked without compile_params";
        code = tile->region;
        jcp = *reinterpret_cast<const jit_snippets_compile_args*>(tile->compile_params);
    }

    size_t get_inputs_num() const override {return 0;}

    void emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
              const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        validate_arguments(in, out, pool, gpr);
        emit_impl(in, out, pool, gpr, nullptr);
    }

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out,
                            const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        if (in.size() != 4)
            IE_THROW() << "TileEmitter got invalid number of inputs. Expected 4, got " << in.size();
        if (out.size() != 0)
            IE_THROW() << "TileEmitter got unexpected output arguments.";
        const size_t num_params = in[2];
        if (num_params > SNIPPETS_MAX_SNIPPETS_DIMS)
            IE_THROW() << "TileEmitter supports only up to " << SNIPPETS_MAX_SNIPPETS_DIMS <<
                       " parameters, got " << num_params;
        const size_t dim = in[3];
        if (dim >= SNIPPETS_MAX_TILE_RANK)
            IE_THROW() << "TileEmitter supports tile ranks up to " << SNIPPETS_MAX_TILE_RANK <<
                       " got " << dim;
    }

    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override {
        const size_t inc = in[0];
        const size_t previous_inc = in[1]; // increment of a previous tile in the same dim (0 if the first tile in the dim)
        const size_t num_params = in[2];
        const size_t dim = in[3]; // tile dimension: 0 - outer, 1 - inner
        const int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
        Reg64 amount = Reg64(reg64_tmp_start + num_params); // amount
        std::array<Label, 2> for_body;

        // If R15 is not used, reserve it for use in scalar to avoid redundant push-pop's.
        // todo: Do we need explicitly check that code contains ScalarEmitter?
        std::vector<size_t> local_gpr = reg64_tmp_start + num_params < 15 ? std::vector<size_t>{15} : std::vector<size_t>{};
        std::vector<Reg64> regs(num_params);
        for (auto i = 0; dim == 0 && i < num_params; i++)
            regs[i] = Reg64(reg64_tmp_start + i);
        // Loop processing could be simplified in some cases
        if (inc > jcp.scheduler_dims[dim]) {
            return;
        } else if (inc == jcp.scheduler_dims[dim]) {
            for (auto& c : code) {
                c.first->emit_code(c.second.first, c.second.second, pool, local_gpr);
            }
        } else {
            // The previous tile has done nothing, all the work is ours
            if (previous_inc == 0 || previous_inc > jcp.scheduler_dims[dim]) {
                h->mov(amount, jcp.scheduler_dims[dim]);
            // The previous tile has done all the work
            } else if (jcp.scheduler_dims[dim] % previous_inc == 0) {
                return;
            }// else: the previous tile has already set a proper work amount
            h->cmp(amount, inc);
            h->jl(for_body[0], CodeGenerator::T_NEAR);

            h->L(for_body[1]);
            {
                h->push(amount);
                for (auto& c : code) {
                    c.first->emit_code(c.second.first, c.second.second, pool, local_gpr);
                }
                h->pop(amount);
                // Todo: Load and Store emitters are currently implemented so they ALWAYS increment appropriate pointers
                //   after reading/writing. This might be a problem if we need to read the same data multiple times (broadcasting shapes).
                //   To overcome this limitation, we add appropriate negative offsets if necessary.
                for (auto i = 0; dim == 0 && i < num_params; i++) {
                    if (jcp.scheduler_offsets[i] != 0) {
                        h->add(regs[i], jcp.scheduler_offsets[i]);
                    }
                }
                    h->sub(amount, inc);
                    h->cmp(amount, inc);
                    h->jge(for_body[1], CodeGenerator::T_NEAR);
            }

            h->L(for_body[0]);
        }
    }

    // A = <42, 17>
    // B = < 1, 17>
    // for (auto k = 0; k < dom_0; k++) { // 42
    //   for (auto n = 0; n < dom_1; n++) { // 17
    //     auto a = *ptr0; ptr0 += vlan; // vector/scalar load
    //     auto b = *ptr1; ptr1 += vlan; // vector/scalar load
    //   }
    //   ptr0 -= 0*dom_1;
    //   ptr1 -= 1*dom_1;
    // }

    // broadcast by MVD is extra case
    // A = <42, 17>
    // B = <42,  1>
    // for (auto k = 0; k < dom_0; k++) { // 42
    //   for (auto n = 0; n < dom_1; n++) { // 17
    //     auto a = *ptr0; ptr0 += vlan; // vector/scalar load
    //     auto b = *ptr1;  // broadcast load
    //   }
    //   ptr0 -= 0*dom_1;
    //   ptr1 += sizeof(ptr1[0]); //ptr1 -= -sizeof(ptr1[0]);
    // }

    // A = <42, 17, 31>
    // B = < 1, 17, 31>
    // for (auto k = 0; k < dom_0; k++) { // 42
    //   for (auto n = 0; n < dom_1; n++) { // 17
    //     for (auto m = 0; m < dom_2; m++) { // 31
    //       auto a = *ptr0; ptr0 += vlan; // vector/scalar load
    //       auto b = *ptr1; ptr1 += vlan; // vector/scalar load
    //     }
    //   }
    //   ptr0 -= 0*dom_1*dom2;
    //   ptr1 -= 1*dom_1*dom2;
    // }
    jit_snippets_compile_args jcp;
    std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>> code;
};

class NopEmitter : public jit_emitter {
public:
    NopEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override {
    }
};

class FakeBroadcastEmitter : public jit_emitter {
public:
    FakeBroadcastEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n) {
        if (n->get_input_shape(0).empty())
            use_broadcast = true;
        else if (*n->get_input_shape(0).rbegin() != *n->get_output_shape(0).rbegin())
            use_broadcast = true;
        else
            use_broadcast = false;
    }
    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                    Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
        Vmm vmm_src0 = Vmm(in[0]);
        Vmm vmm_dst  = Vmm(out[0]);

        if (use_broadcast) {
            h->uni_vbroadcastss(vmm_dst, Xmm(in[0]));
        } else {
            h->uni_vmovups(vmm_dst, vmm_src0);
        }
    }

private:
    bool use_broadcast;
};

class ScalarEmitter : public jit_emitter {
public:
    ScalarEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n) {
        value = mkldnn::impl::cpu::x64::float2int(ov::as_type_ptr<ngraph::snippets::op::Scalar>(n)->cast_vector<float>()[0]);
        push_arg_entry_of("scalar", value, true);
        prepare_table();
    }

    size_t get_inputs_num() const override {return 0;}

protected:
    size_t aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                    Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
        Vmm vmm_dst  = Vmm(out[0]);
        h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
    }

private:
    int32_t value;
};

///
/// Memory emitters:
///
/// *Note*: post increment is embedded into Load/Store operation which means that
/// it's illigal to load/store to the same address multiple times
/// Typical application can be if Load and BroadcastLoad are performed from the same pointer.
/// If Load goes before BroadcastLoad topologicaly the resilt will be incorrect
/// For scalar loads we can use different tiles. Tiling indeed can be arbitrary and post increment should be somehow coded into ISA.
/// Blocked parameter to tell if input is actually blocked. Broadcast means broadcast by W in other cases no need to substitute load.
class MemoryEmitter : public jit_emitter  {
public:
    MemoryEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n), ea(getEA(n)) {
    }

    size_t get_inputs_num() const override {return 1;}

protected:
    static auto getEA(const std::shared_ptr<ov::Node>& n) -> size_t {
        auto& rt = n->get_rt_info();
        size_t ea = 0;
        auto it = rt.find("effectiveAddress");
        if (it != rt.end()) {
            ea = it->second.as<int64_t>();
        } else {
            throw ov::Exception("effective address for Load generation cannot be determined");
        }
        return ea;
    }

    size_t ea;
};

class StoreEmitter : public MemoryEmitter  {
public:
    StoreEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                    Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
        Reg64 out_reg(ea);
        Vmm vmm_src0 = Vmm(in[0]);
        h->uni_vmovups(h->ptr[out_reg], vmm_src0);
        h->add(out_reg, mkldnn::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    }
};

class ScalarStoreEmitter : public MemoryEmitter {
public:
    ScalarStoreEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                        Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
        Reg64 out_reg(ea);
        Xmm vmm_src0 = Xmm(in[0]);
        h->uni_vmovss(h->ptr[out_reg], vmm_src0);
        h->add(out_reg, sizeof(float));
    }
};

class LoadEmitter : public MemoryEmitter {
public:
    LoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
    }

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
        Reg64 in_reg(ea);
        Vmm vmm_src0 = Vmm(out[0]);
        h->uni_vmovups(vmm_src0, h->ptr[in_reg]);

        if (shouldPostIncrement) {
            h->add(in_reg, mkldnn::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
        }
    }

private:
    bool shouldPostIncrement;
};

class BroadcastLoadEmitter : public MemoryEmitter {
public:
    BroadcastLoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }
    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
        Reg64 in_reg(ea);
        Vmm vmm_src0 = Vmm(out[0]);

        // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
        // key point here is not to add post-increment, it might be fixed by some other approach in future
        h->uni_vbroadcastss(vmm_src0, h->ptr[in_reg]);
    }
};

class ScalarLoadEmitter : public MemoryEmitter {
public:
    ScalarLoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
    }
    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
        Reg64 in_reg(ea);
        Xmm vmm_src0 = Xmm(out[0]);
        h->uni_vmovss(vmm_src0, h->ptr[in_reg]);

        // Doesn't work if the same pointer comes with multiple load operations
        if (shouldPostIncrement) {
            h->add(in_reg, sizeof(float));
        }
    }

private:
    bool shouldPostIncrement;
};

}   // namespace intel_cpu
}   // namespace ov
