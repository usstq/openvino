// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_def_conv_node.h"
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include "ie_parallel.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_def_conv_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_def_conv_kernel_f32 : public jit_uni_def_conv_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_def_conv_kernel_f32)

    constexpr static int sampledPointsPerPixel = MKLDNNDeformableConvolutionNode::sampledPointsPerPixel;

    explicit jit_uni_def_conv_kernel_f32(jit_def_conv_params jcp) : jit_uni_def_conv_kernel(jcp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    };

    void generate() override {
        this->preamble();

        mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
        mov(reg_sampled_wei, ptr[this->param1 + GET_OFF(sampledWei)]);
        mov(reg_sampled_offs, ptr[this->param1 + GET_OFF(sampledCoords)]);

        mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
        if (jcp_.with_bias)
            mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
        mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
        mov(reg_input_buffer_temp, ptr[this->param1 + GET_OFF(buf)]);
        mov(oh_pos_temp, ptr[param1 + GET_OFF(oh_pos)]);

        // need to save temporary to prevent using of %rdi during GET_OFF(...)
        mov(reg_oh_pos, oh_pos_temp);
        // prevents mismatching param1 == %rcx (on windows) and reg_input_buffer
        mov(reg_input_buffer, reg_input_buffer_temp);

        ow_loop();

        this->postamble();

        prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    using Ymm = const Xbyak::Ymm;
    using Xmm = const Xbyak::Xmm;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg8_t = const Xbyak::Reg8;

    reg64_t reg_input = r8;
    reg64_t reg_sampled_wei = r9;
    reg64_t reg_kernel = r10;
    reg64_t reg_bias = r11;
    reg64_t reg_output = r12;
    reg64_t reg_oh_pos = rdi;
    reg64_t aux_reg_bias = rsi;
    reg64_t reg_ow_pos = rdx;
    reg64_t aux_reg_output = reg_ow_pos;
    reg64_t reg_dg_iter = reg_output;
    reg64_t aux_reg_input = rax;
    reg64_t aux2_reg_input = reg_kernel;
    reg64_t reg_ic_iter = rbx;
    reg64_t reg_oc_work = reg_ic_iter;
    reg64_t aux_reg_sampled_wei = reg_bias;
    reg64_t reg_input_buffer = rcx;
    reg64_t aux_reg_input_buffer = r14;
    reg32_t reg_tmp_32 = r15d;
    reg64_t reg_tmp_64 = r15;
    reg64_t reg_table = rbp;
    reg64_t aux_reg_kernel = reg_table;
    reg64_t aux2_reg_kernel = r15;
    reg64_t oh_pos_temp = aux2_reg_kernel;
    reg64_t aux2_reg_input_buffer = aux_reg_bias;
    reg64_t reg_sampled_offs = aux2_reg_input_buffer;
    reg64_t aux3_reg_input_buffer = reg_input;
    reg64_t aux_reg_sampled_offs = r13;
    reg64_t reg_input_buffer_temp = aux_reg_sampled_offs;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    inline Xbyak::Address table_val(int index)
    { return ptr[reg_table + index * vlen]; }

    inline Vmm get_vmm_ker(int idx) { return Vmm(idx + 0); }
    inline Vmm get_vmm_src(int idx) { return Vmm(idx + 1); }
    inline Vmm get_vmm_acc(int idx) { return Vmm(idx + jcp_.ur_w + 1); }
    inline Ymm get_ymm_acc(int idx) { return Ymm(idx + jcp_.ur_w + 1); }
    inline Xmm get_xmm_acc(int idx) { return Xmm(idx + jcp_.ur_w + 1); }

    Xbyak::Label l_table;

    void ow_loop() {
        Label ow_loop_main;
        Label ow_tail;

        mov(reg_ow_pos, 0);

        L(ow_loop_main); {
            cmp(reg_ow_pos, jcp_.ow - jcp_.ur_w);
            jg(ow_tail, T_NEAR);

            oc_loop(jcp_.ur_w);
            add(reg_input, jcp_.ur_w * jcp_.stride_w * jcp_.ic * jcp_.typesize_in);
            add(reg_sampled_wei, jcp_.ur_w * jcp_.kh * jcp_.kw * sampledPointsPerPixel * jcp_.typesize_sampled_wei);  // type = float
            add(reg_sampled_offs, jcp_.ur_w * jcp_.kh * jcp_.kw * sampledPointsPerPixel * jcp_.typesize_sampled_offsets);  // type = int

            add(reg_output, jcp_.ur_w * jcp_.oc * jcp_.typesize_out);

            add(reg_ow_pos, jcp_.ur_w);
            jmp(ow_loop_main, T_NEAR);
        }

        L(ow_tail); {
            if (jcp_.ow % jcp_.ur_w != 0)
                oc_loop(jcp_.ow % jcp_.ur_w);
        }
    }

    void prepare_table() {
        align(64);
        L(l_table);
        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(0);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(cpu::x64::float2int(static_cast<float>(jcp_.ih)));
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(cpu::x64::float2int(static_cast<float>(jcp_.iw)));
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(jcp_.ih - 1);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(jcp_.iw - 1);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(1);
        }
    }

    void apply_filter(int ow_step, int oc_blocks_step, int oc_step, int ic_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;

        for (int kh = 0; kh < jcp_.kh; kh++) {
            for (int kw = 0; kw < jcp_.kw; kw++) {
                for (int ic = 0; ic < ic_step; ic++) {
                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_src = get_vmm_src(ow);
                        size_t inp_off = (size_t) ow * jcp_.kh * jcp_.kw * jcp_.ic + kh * jcp_.kw * jcp_.ic + kw * jcp_.ic + ic;

                        uni_vbroadcastss(vmm_src, ptr[aux2_reg_input_buffer + inp_off * jcp_.typesize_in]);
                    }

                    for (int r = 0; r < repeats; r++) {
                        for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                            Vmm vmm_ker = get_vmm_ker(0);
                            size_t ker_off = (size_t) ocb * jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block +
                                             kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block +
                                             kw * jcp_.ic_block * jcp_.oc_block +
                                             ic * jcp_.oc_block + r * jcp_.oc_block / 2;

                            uni_vmovups(vmm_ker, ptr[aux2_reg_kernel + ker_off * jcp_.typesize_in]);
                            for (int ow = 0; ow < ow_step; ow++) {
                                Vmm vmm_src = get_vmm_src(ow);
                                Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);

                                if (isa == cpu::x64::sse41 && ow > 0) {
                                    uni_vmovups(vmm_ker, ptr[aux2_reg_kernel + ker_off * jcp_.typesize_in]);
                                }
                                uni_vfmadd231ps(vmm_acc, vmm_ker, vmm_src);
                            }
                        }
                    }
                }
            }
        }
    }

    void init_accums(int ow_step, int oc_blocks_step, int oc_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;
        for (int r = 0; r < repeats; r++) {
            for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);
                    uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
                }
            }
        }
    }

    void ic_loop(int ow_step, int oc_blocks_step, int oc_step) {
        Label ic_main_loop;
        Label ic_tail;
        Label exit;

        push(reg_oc_work);
        push(aux_reg_bias);
        push(reg_sampled_offs);

        mov(aux2_reg_kernel, aux_reg_kernel);
        mov(aux2_reg_input_buffer, reg_input_buffer);
        mov(reg_ic_iter, jcp_.ic);

        init_accums(ow_step, oc_blocks_step, oc_step);

        L(ic_main_loop); {
            cmp(reg_ic_iter, jcp_.ic_block);
            jl(ic_tail, T_NEAR);

            apply_filter(ow_step, oc_blocks_step, oc_step, jcp_.ic_block);
            add(aux2_reg_input_buffer, jcp_.ic_block * jcp_.typesize_in);
            add(aux2_reg_kernel, jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block * jcp_.typesize_in);
            sub(reg_ic_iter, jcp_.ic_block);
            jmp(ic_main_loop, T_NEAR);
        }

        L(ic_tail); {
            if (jcp_.ic % jcp_.ic_block != 0) {
                apply_filter(ow_step, oc_blocks_step, oc_step, jcp_.ic % jcp_.ic_block);
            }
        }

        pop(reg_sampled_offs);
        pop(aux_reg_bias);
        pop(reg_oc_work);
    }

    void prepare_buffer(int ow_step) {
        Label dg_loop;
        Label dg_loop_end;

        mov(reg_table, l_table);
        mov(aux_reg_sampled_wei, reg_sampled_wei);
        mov(aux_reg_sampled_offs, reg_sampled_offs);
        mov(aux_reg_input, reg_input);
        push(reg_sampled_offs);
        mov(aux2_reg_input_buffer, aux_reg_input_buffer);
        xor_(reg_dg_iter, reg_dg_iter);

        const int ic_per_def_group = jcp_.ic / jcp_.dg;
        L(dg_loop); {
            cmp(reg_dg_iter, jcp_.dg);
            jge(dg_loop_end, T_NEAR);

            for (int ow = 0; ow < ow_step; ow++) {
                for (int kh = 0; kh < jcp_.kh; kh++) {
                    for (int kw = 0; kw < jcp_.kw; kw++) {
                        Label ic_loop_main;
                        Label ic_loop_tail;
                        Label loop_end;

                        mov(aux2_reg_input, aux_reg_input);
                        add(aux2_reg_input, (ow * jcp_.stride_w * jcp_.ic) * jcp_.typesize_in);

                        mov(aux3_reg_input_buffer, aux2_reg_input_buffer);
                        add(aux3_reg_input_buffer, (ow * jcp_.kh * jcp_.kw * jcp_.ic) * jcp_.typesize_in);

                        Xmm xmm_v1_off = Xmm(9);
                        Xmm xmm_v2_off = Xmm(10);
                        Xmm xmm_v3_off = Xmm(11);
                        Xmm xmm_v4_off = Xmm(12);

                        Xmm xmm_w1 = Xmm(4);
                        Xmm xmm_w2 = Xmm(1);
                        Xmm xmm_w3 = Xmm(8);
                        Xmm xmm_w4 = Xmm(5);

                        Xmm xmm_v1 = Xmm(2);
                        Xmm xmm_v2 = Xmm(3);;
                        Xmm xmm_v3 = Xmm(6);
                        Xmm xmm_v4 = Xmm(7);

                        Vmm vmm_w1 = Vmm(xmm_w1.getIdx());
                        Vmm vmm_w2 = Vmm(xmm_w2.getIdx());
                        Vmm vmm_w3 = Vmm(xmm_w3.getIdx());
                        Vmm vmm_w4 = Vmm(xmm_w4.getIdx());

                        Vmm vmm_v1 = Vmm(xmm_v1.getIdx());
                        Vmm vmm_v2 = Vmm(xmm_v2.getIdx());
                        Vmm vmm_v3 = Vmm(xmm_v3.getIdx());
                        Vmm vmm_v4 = Vmm(xmm_v4.getIdx());

                        // offsets computation
                        size_t ind_off_hh = sampledPointsPerPixel * (((size_t) kh * jcp_.kw + kw) + ow * (jcp_.kh * jcp_.kw));
                        size_t ind_off_hl = ind_off_hh + 1;
                        size_t ind_off_lh = ind_off_hl + 1;
                        size_t ind_off_ll = ind_off_lh + 1;

                        movq(xmm_v1_off, qword[aux_reg_sampled_offs + ind_off_ll * jcp_.typesize_sampled_offsets]);
                        movq(xmm_v2_off, qword[aux_reg_sampled_offs + ind_off_hl * jcp_.typesize_sampled_offsets]);
                        movq(xmm_v3_off, qword[aux_reg_sampled_offs + ind_off_lh * jcp_.typesize_sampled_offsets]);
                        movq(xmm_v4_off, qword[aux_reg_sampled_offs + ind_off_hh * jcp_.typesize_sampled_offsets]);

                        // w's computation
                        uni_vbroadcastss(vmm_w1, dword[aux_reg_sampled_wei + ind_off_ll * jcp_.typesize_sampled_wei]);
                        uni_vbroadcastss(vmm_w2, dword[aux_reg_sampled_wei + ind_off_hl * jcp_.typesize_sampled_wei]);
                        uni_vbroadcastss(vmm_w3, dword[aux_reg_sampled_wei + ind_off_lh * jcp_.typesize_sampled_wei]);
                        uni_vbroadcastss(vmm_w4, dword[aux_reg_sampled_wei + ind_off_hh * jcp_.typesize_sampled_wei]);

                        int simd_w = vlen / jcp_.typesize_in;
                        mov(reg_ic_iter, ic_per_def_group);

                        L(ic_loop_main);
                        {
                            cmp(reg_ic_iter, simd_w);
                            jl(ic_loop_tail, T_NEAR);

                            size_t input_buffer_off = (size_t) kh * jcp_.kw * jcp_.ic + kw * jcp_.ic;

                            pmovsxdq(xmm_v1_off, xmm_v1_off);
                            movq(reg_tmp_64, xmm_v1_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            uni_vmovups(vmm_v1, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v1, vmm_v1, vmm_w1);

                            pmovsxdq(xmm_v2_off, xmm_v2_off);
                            movq(reg_tmp_64, xmm_v2_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            uni_vmovups(vmm_v2, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v2, vmm_v2, vmm_w2);

                            pmovsxdq(xmm_v3_off, xmm_v3_off);
                            movq(reg_tmp_64, xmm_v3_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            uni_vmovups(vmm_v3, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v3, vmm_v3, vmm_w3);

                            pmovsxdq(xmm_v4_off, xmm_v4_off);
                            movq(reg_tmp_64, xmm_v4_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            uni_vmovups(vmm_v4, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v4, vmm_v4, vmm_w4);

                            uni_vaddps(vmm_v1, vmm_v1, vmm_v2);
                            uni_vaddps(vmm_v1, vmm_v1, vmm_v3);
                            uni_vaddps(vmm_v1, vmm_v1, vmm_v4);
                            uni_vmovups(ptr[aux3_reg_input_buffer + input_buffer_off * jcp_.typesize_in], vmm_v1);

                            add(aux2_reg_input, simd_w * jcp_.typesize_in);
                            add(aux3_reg_input_buffer, simd_w * jcp_.typesize_in);
                            sub(reg_ic_iter, simd_w);
                            jmp(ic_loop_main, T_NEAR);
                        }

                        L(ic_loop_tail);
                        {
                            cmp(reg_ic_iter, 1);
                            jl(loop_end, T_NEAR);

                            size_t input_buffer_off = (size_t) kh * jcp_.kw * jcp_.ic + kw * jcp_.ic;
                            pmovsxdq(xmm_v1_off, xmm_v1_off);
                            movq(reg_tmp_64, xmm_v1_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            movss(xmm_v1, ptr[reg_tmp_64]);
                            mulss(xmm_v1, xmm_w1);

                            pmovsxdq(xmm_v2_off, xmm_v2_off);
                            movq(reg_tmp_64, xmm_v2_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            movss(xmm_v2, ptr[reg_tmp_64]);
                            mulss(xmm_v2, xmm_w2);

                            pmovsxdq(xmm_v3_off, xmm_v3_off);
                            movq(reg_tmp_64, xmm_v3_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            movss(xmm_v3, ptr[reg_tmp_64]);
                            mulss(xmm_v3, xmm_w3);

                            pmovsxdq(xmm_v4_off, xmm_v4_off);
                            movq(reg_tmp_64, xmm_v4_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            movss(xmm_v4, ptr[reg_tmp_64]);
                            mulss(xmm_v4, xmm_w4);

                            addss(xmm_v1, xmm_v2);
                            addss(xmm_v1, xmm_v3);
                            addss(xmm_v1, xmm_v4);
                            movss(ptr[aux3_reg_input_buffer + input_buffer_off * jcp_.typesize_in], xmm_v1);

                            add(aux2_reg_input, jcp_.typesize_in);
                            add(aux3_reg_input_buffer, jcp_.typesize_in);
                            sub(reg_ic_iter, 1);
                            jmp(ic_loop_tail, T_NEAR);
                        }
                        jmp(loop_end, T_NEAR);
                        L(loop_end);
                    }
                }
            }

            add(aux_reg_sampled_wei, sampledPointsPerPixel * jcp_.kh * jcp_.kw * jcp_.oh * jcp_.ow * jcp_.typesize_sampled_wei);
            add(aux_reg_sampled_offs, sampledPointsPerPixel * jcp_.kh * jcp_.kw * jcp_.oh * jcp_.ow * jcp_.typesize_sampled_offsets);
            add(aux_reg_input, ic_per_def_group * jcp_.typesize_in);
            add(aux2_reg_input_buffer, ic_per_def_group * jcp_.typesize_in);
            inc(reg_dg_iter);
            jmp(dg_loop, T_NEAR);
        }

        L(dg_loop_end);
        pop(reg_sampled_offs);
    }

    void store_output(int ow_step, int oc_blocks_step, int oc_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;

        if (jcp_.with_bias) {
            for (int r = 0; r < repeats; r++) {
                for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                    size_t bias_off = (size_t) ocb * jcp_.oc_block + r * jcp_.oc_block / 2;
                    uni_vmovups(Vmm(0), ptr[aux_reg_bias + bias_off * jcp_.typesize_bia]);

                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);
                        uni_vaddps(vmm_acc, vmm_acc, Vmm(0));
                    }
                }
            }
        }

        if (isa == avx512_common && oc_step != jcp_.oc_block) {
            int mask = (1 << oc_step) - 1;
            mov(reg_tmp_32, mask);
            kmovw(ktail_mask, reg_tmp_32);
        }

        for (int r = 0; r < repeats; r++) {
            int tail_size = isa == cpu::x64::sse41 ? std::min(jcp_.oc_block / 2, oc_step - r * jcp_.oc_block / 2) : oc_step;
            bool is_scalar_store = isa == cpu::x64::sse41 ? tail_size < jcp_.oc_block / 2 : tail_size < jcp_.oc_block;
            if (is_scalar_store) {
                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_dst = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ow);
                    Xmm xmm_dst = get_xmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ow);

                    if (isa == avx512_common) {
                        size_t out_off = (size_t) ow * jcp_.oc;
                        uni_vmovups(ptr[aux_reg_output + out_off * jcp_.typesize_out], vmm_dst | ktail_mask);
                    } else {
                        for (int oc = 0; oc < tail_size; oc++) {
                            size_t out_off = (size_t) ow * jcp_.oc + oc + r * (jcp_.oc_block / 2);
                            movq(reg_tmp_64, xmm_dst);
                            mov(ptr[aux_reg_output + out_off * jcp_.typesize_out], reg_tmp_32);

                            if (isa == cpu::x64::sse41) {
                                psrldq(vmm_dst, jcp_.typesize_out);
                            } else {
                                Ymm ymm_dst = get_ymm_acc(ow);
                                Vmm vmm_tmp = Vmm(0);
                                Ymm ymm_tmp = Ymm(0);

                                vperm2i128(ymm_tmp, ymm_dst, ymm_dst, 0x01);
                                vpalignr(ymm_dst, vmm_tmp, ymm_dst, jcp_.typesize_out);
                            }
                        }
                    }
                }
            } else {
                for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);
                        size_t out_off = (size_t) ow * jcp_.oc * jcp_.ngroups + ocb * jcp_.oc_block + r * (jcp_.oc_block / 2);
                        uni_vmovups(ptr[aux_reg_output + out_off * jcp_.typesize_out], vmm_acc);
                    }
                }
            }
        }
    }

    void oc_loop(int ow_step) {
        Label oc_unrolled_loop;
        Label oc_main_loop;
        Label oc_tail;

        mov(aux_reg_input_buffer, reg_input_buffer);

        push(reg_output);
        push(reg_bias);
        push(reg_input);
        push(reg_kernel);

        prepare_buffer(ow_step);

        pop(reg_kernel);
        pop(reg_input);
        pop(reg_bias);
        pop(reg_output);

        push(reg_sampled_offs);
        push(reg_ow_pos);
        push(aux2_reg_kernel);

        mov(aux_reg_kernel, reg_kernel);
        mov(aux_reg_output, reg_output);
        mov(aux_reg_bias, reg_bias);
        mov(reg_oc_work, jcp_.oc);

        L(oc_unrolled_loop); {
            cmp(reg_oc_work, jcp_.nb_oc_blocking * jcp_.oc_block);
            jl(oc_main_loop, T_NEAR);

            ic_loop(ow_step, jcp_.nb_oc_blocking, jcp_.oc_block);
            store_output(ow_step, jcp_.nb_oc_blocking, jcp_.oc_block);

            add(aux_reg_kernel, jcp_.nb_oc_blocking * jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block * jcp_.typesize_in);
            add(aux_reg_output, jcp_.nb_oc_blocking * jcp_.oc_block * jcp_.typesize_out);
            add(aux_reg_bias, jcp_.nb_oc_blocking * jcp_.oc_block * jcp_.typesize_bia);
            sub(reg_oc_work, jcp_.nb_oc_blocking * jcp_.oc_block);

            jmp(oc_unrolled_loop, T_NEAR);
        }

        L(oc_main_loop); {
            cmp(reg_oc_work, jcp_.oc_block);
            jl(oc_tail, T_NEAR);

            ic_loop(ow_step, 1, jcp_.oc_block);
            store_output(ow_step, 1, jcp_.oc_block);

            add(aux_reg_kernel, jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block * jcp_.typesize_in);
            add(aux_reg_output, jcp_.oc_block * jcp_.typesize_out);
            add(aux_reg_bias, jcp_.oc_block * jcp_.typesize_bia);
            sub(reg_oc_work, jcp_.oc_block);

            jmp(oc_main_loop, T_NEAR);
        }

        L(oc_tail); {
            if (jcp_.oc % jcp_.oc_block != 0) {
                ic_loop(ow_step, 1, jcp_.oc % jcp_.oc_block);
                store_output(ow_step, 1, jcp_.oc % jcp_.oc_block);
            }
        }

        pop(aux2_reg_kernel);
        pop(reg_ow_pos);
        pop(reg_sampled_offs);
    }
};

bool MKLDNNDeformableConvolutionNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        if (!one_of(op->get_type_info(),
                ngraph::op::v1::DeformableConvolution::type_info,
                ngraph::op::v8::DeformableConvolution::type_info)) {
            errorMessage = "Node is not an instance of DeformableConvolution form the operation set v1 or v8.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNDeformableConvolutionNode::MKLDNNDeformableConvolutionNode(const std::shared_ptr<ngraph::Node>& op,
        const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    auto defConvNodeBase = std::dynamic_pointer_cast<ngraph::op::util::DeformableConvolutionBase>(op);
    if (defConvNodeBase == nullptr)
        IE_THROW() << "Operation with name '" << op->get_friendly_name() <<
            "' is not an instance of DeformableConvolutionBase.";

    group = defConvNodeBase->get_group();
    deformable_group = defConvNodeBase->get_deformable_group();
    auto& strides = defConvNodeBase->get_strides();
    for (int i = 0; i < strides.size(); i++) {
        stride.push_back(strides[i]);
    }

    auto& dilations = defConvNodeBase->get_dilations();
    for (int i = 1; i <= dilations.size(); i++) {
        dilation.push_back(dilations[dilations.size() - i] - 1);
    }

    paddingL = defConvNodeBase->get_pads_begin();

    if (op->get_type_info() == ngraph::op::v8::DeformableConvolution::type_info) {
        auto defConvNode = std::dynamic_pointer_cast<ngraph::op::v8::DeformableConvolution>(op);
        if (defConvNode == nullptr)
            IE_THROW() << "Operation with name '" << op->get_friendly_name() <<
                "' is not an instance of DeformableConvolution from opset8.";
        with_bilinear_pad = defConvNode->get_bilinear_interpolation_pad();
    } else {
        with_bilinear_pad = false;
    }
}

void MKLDNNDeformableConvolutionNode::getSupportedDescriptors() {
    std::string errorPrefix = "DeformableConvolution layer with name '" + getName() + "' ";
    if (getParentEdges().size() != 3 && getParentEdges().size() != 4)
        IE_THROW() << errorPrefix << "has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges";
    if (getInputShapeAtPort(0).getRank() != 4) {
        IE_THROW() << "Deformable convolution layer. Unsupported mode. Only 4D blobs are supported as input.";
    }
    if (getInputShapeAtPort(1).getRank() != 4) {
        IE_THROW() << errorPrefix << "doesn't support 1st input with rank: " << getInputShapeAtPort(1).getRank();
    }
    if (getInputShapeAtPort(2).getRank() != 4) {
        IE_THROW() << errorPrefix << "doesn't support 2nd input with rank: " << getInputShapeAtPort(2).getRank();
    }
    if (getOutputShapeAtPort(0).getRank() != 4) {
        IE_THROW() << errorPrefix << "doesn't support output with rank: " << getOutputShapeAtPort(0).getRank();
    }
}

void MKLDNNDeformableConvolutionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    size_t inputsNumber = getOriginalInputsNumber();
    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(inputsNumber);
    config.inConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.inConfs[1].constant = false;
    config.inConfs[1].inPlace = -1;
    config.inConfs[2].constant = false;
    config.inConfs[2].inPlace = -1;
    if (inputsNumber > 3) {
        config.inConfs[3].constant = false;
        config.inConfs[3].inPlace = -1;
    }

    config.outConfs.resize(1);
    config.outConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;

    impl_desc_type impl_type;
    const int simd_w = mayiuse(cpu::x64::avx512_common) ? 16 : 8;
    if (group != 1 || (((getInputShapeAtPort(0).getStaticDims()[1] / group) % simd_w != 0)
    || ((getOutputShapeAtPort(0).getStaticDims()[1] / group) % simd_w != 0))) {
        enforceRef = true;
    }

    if (enforceRef) {
        impl_type = impl_desc_type::ref;
    } else if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    if (!enforceRef && mayiuse(cpu::x64::sse41)) {
        // optimized implementation
        auto dataFormat = memory::format_tag::nhwc;
        auto offFormat = memory::format_tag::nchw;
        auto weiFormat = mayiuse(avx512_common) ? memory::format_tag::OIhw16i16o : memory::format_tag::OIhw8i8o;
        config.inConfs[0].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(0),
                                                                              memory::data_type::f32, dataFormat);
        config.inConfs[1].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(1),
                                                                              memory::data_type::f32, offFormat);
        auto& wDims = getInputShapeAtPort(2).getStaticDims();
        if (group > 1 && wDims.size() != 5) {
            auto new_dims = InferenceEngine::SizeVector({group, div_up(wDims[0], group)});
            for (int i = 1; i < wDims.size(); i++) {
                new_dims.push_back(wDims[i]);
            }
            config.inConfs[2].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(2),
                                                                                 memory::data_type::f32, weiFormat);
        } else {
            config.inConfs[2].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(2),
                                                                                 memory::data_type::f32, weiFormat);
        }

        if (inputsNumber > 3) {
            config.inConfs[3].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(3),
                                                                                 memory::data_type::f32, memory::format_tag::nchw);
        }
        config.outConfs[0].desc = std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(0),
                                                                              memory::data_type::f32, dataFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_type});
    } else {
        // reference implementation
        config.inConfs[0].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(0), memory::data_type::f32,
                                                               memory::format_tag::nchw);
        config.inConfs[1].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(1), memory::data_type::f32,
                                                               memory::format_tag::nchw);
        config.inConfs[2].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(2), memory::data_type::f32,
                                                               memory::format_tag::oihw);
        if (inputsNumber > 3) {
            config.inConfs[3].desc = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(3), memory::data_type::f32,
                                                                                 memory::format_tag::nchw);
        }
        config.outConfs[0].desc = std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(0), memory::data_type::f32,
                                                                memory::format_tag::nchw);
        supportedPrimitiveDescriptors.push_back({config, impl_type});
    }
}

void MKLDNNDeformableConvolutionNode::prepareSamplingWeights(
        const std::vector<size_t>& src_strides, const float* offsets, const std::vector<size_t>& off_strides,
        const float* modulation, const std::vector<size_t>& modulation_strides) {
    const int MB = jcp.mb;
    const int OH = jcp.oh;
    const int OW = jcp.ow;

    const int KH = jcp.kh;
    const int KW = jcp.kw;
    const int ker_size = KH * KW;

    const int DG = jcp.dg;

    const int IH = jcp.ih;
    const int IW = jcp.iw;

    const int KSH = jcp.stride_h;
    const int KSW = jcp.stride_w;

    const int KDH = jcp.dilate_h;
    const int KDW = jcp.dilate_w;

    const int padT = jcp.t_pad;
    const int padL = jcp.l_pad;

    const bool with_bi_pad = jcp.with_bi_pad;

    // prepare weights and indices
    sampledCoordsVector.resize(MB * DG * KH * KW * OH * OW * sampledPointsPerPixel);
    interpWeightsVector.resize(MB * DG * KH * KW * OH * OW * sampledPointsPerPixel);
    auto precompKer = [&](int mb, int dg, int oh, int ow) {
        int sampledCoordIndex = (mb * DG * OH * OW + dg * OH * OW + oh * OW + ow) * KH * KW * sampledPointsPerPixel;
        const int h_in = oh * KSH - padT;
        const int w_in = ow * KSW - padL;

        const int waOffsetH = (enforceRef ? 0 : h_in);
        const int waOffsetW = (enforceRef ? 0 : w_in);

        const float *data_offset_ptr = offsets + mb * off_strides[0] + (dg * 2 * KH * KW) * off_strides[1];
        const float *modulation_offset_ptr = nullptr;
        if (modulation != nullptr) {
            modulation_offset_ptr = modulation + mb * modulation_strides[0] + (dg * ker_size) * modulation_strides[1];
        }

        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                const size_t data_offset_h_index = 2 * ((size_t) kh * KW + kw) * off_strides[1] + oh * off_strides[2] + ow * off_strides[3];
                const size_t data_offset_w_index = (2 * ((size_t) kh * KW + kw) + 1) * off_strides[1] + oh * off_strides[2] + ow * off_strides[3];
                const float offset_h = data_offset_ptr[data_offset_h_index];
                const float offset_w = data_offset_ptr[data_offset_w_index];
                float map_h = h_in + kh * (KDH + 1) + offset_h;
                float map_w = w_in + kw * (KDW + 1) + offset_w;
                bool skip_compute;
                if (with_bilinear_pad) {
                    skip_compute = !(static_cast<int>(map_w) > -1 &&
                                     static_cast<int>(map_w) < IW &&
                                     static_cast<int>(map_h) > -1 &&
                                     static_cast<int>(map_h) < IH);
                } else {
                    skip_compute = !(map_w >= 0 && map_w < IW &&
                                     map_h >= 0 && map_h < IH);
                }
                if (!skip_compute) {
                    // modulations precomp.
                    float modulation_scalar = 1.0f;

                    if (modulation_offset_ptr != nullptr) {
                        size_t modulation_index = (kh * KW + kw) * modulation_strides[1] + oh * modulation_strides[2] + ow * modulation_strides[3];
                        modulation_scalar = modulation_offset_ptr[modulation_index];
                    }
                    // interpolation precomp.
                    const int cur_h_end = IH;
                    const int cur_w_end = IW;
                    int h_low = with_bi_pad ? static_cast<int>(floorf(map_h)) :
                                std::max(static_cast<int>(floorf(map_h)), 0);
                    int w_low = with_bi_pad ? static_cast<int>(floorf(map_w)) :
                                std::max(static_cast<int>(floorf(map_w)), 0);
                    int h_high = with_bi_pad ? h_low + 1 : std::min(static_cast<int>(ceilf(map_h)), cur_h_end - 1);
                    int w_high = with_bi_pad ? w_low + 1 : std::min(static_cast<int>(ceilf(map_w)), cur_w_end - 1);

                    float lh = map_h - h_low;
                    float lw = map_w - w_low;
                    float hh = 1 - lh, hw = 1 - lw;

                    int h_ind_low = std::max(h_low, 0) - waOffsetH;
                    int h_ind_high = std::min(h_high, cur_h_end - 1) - waOffsetH;
                    int w_ind_low = std::max(w_low, 0) - waOffsetW;
                    int w_ind_high = std::min(w_high, cur_w_end - 1) - waOffsetW;

                    hh = (h_low >= 0 ? hh : 0);
                    hw = (w_low >= 0 ? hw : 0);
                    lh = (h_high < cur_h_end ? lh : 0);
                    lw = (w_high < cur_w_end ? lw : 0);

                    const int h_off_low = h_ind_low * src_strides[2] / src_strides[3];
                    const int h_off_high = h_ind_high * src_strides[2] / src_strides[3];
                    const int w_off_low  = w_ind_low;
                    const int w_off_high = w_ind_high;
                    sampledCoordsVector[sampledCoordIndex] = h_off_high + w_off_high;
                    sampledCoordsVector[sampledCoordIndex + 1] = h_off_high + w_off_low;
                    sampledCoordsVector[sampledCoordIndex + 2] = h_off_low + w_off_high;
                    sampledCoordsVector[sampledCoordIndex + 3] = h_off_low + w_off_low;

                    float w22 = hh * hw * modulation_scalar, w21 = hh * lw * modulation_scalar,
                            w12 = lh * hw * modulation_scalar, w11 = lh * lw * modulation_scalar;

                    interpWeightsVector[sampledCoordIndex] = w11;
                    interpWeightsVector[sampledCoordIndex + 1] = w12;
                    interpWeightsVector[sampledCoordIndex + 2] = w21;
                    interpWeightsVector[sampledCoordIndex + 3] = w22;
                } else {
                    sampledCoordsVector[sampledCoordIndex] = 0;
                    interpWeightsVector[sampledCoordIndex] = 0;
                    interpWeightsVector[sampledCoordIndex + 1] = 0;
                    interpWeightsVector[sampledCoordIndex + 2] = 0;
                    interpWeightsVector[sampledCoordIndex + 3] = 0;
                }
                sampledCoordIndex += sampledPointsPerPixel;
            }
        }
    };

    parallel_nd(MB, DG, OH, OW, [&](int mb, int dg, int oh, int ow)  {
        precompKer(mb, dg, oh, ow);
    });
}

void MKLDNNDeformableConvolutionNode::createPrimitive() {
    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        IE_THROW() << "CPU deformable convolution with name '" << getName() << "' doesn't have primitive descriptors.";
    auto config = selectedPrimitiveDescriptor->getConfig();

    auto srcDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    auto weiDims = getParentEdgeAt(2)->getMemory().getStaticDims();
    auto dstDims = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();

    jcp.dg = deformable_group;

    jcp.ngroups = group;

    jcp.mb = srcDims[0];

    jcp.oc = dstDims[1] / jcp.ngroups;
    jcp.ic = srcDims[1] / jcp.ngroups;

    jcp.ih = srcDims[2];
    jcp.iw = srcDims[3];
    jcp.oh = dstDims[2];
    jcp.ow = dstDims[3];

    jcp.kh = weiDims[2];
    jcp.kw = weiDims[3];

    jcp.t_pad = paddingL[0];
    jcp.l_pad = paddingL[1];

    jcp.stride_h = stride[0];
    jcp.stride_w = stride[1];

    jcp.dilate_h = dilation[0];
    jcp.dilate_w = dilation[1];

    jcp.with_bias = false;
    jcp.with_bi_pad = with_bilinear_pad;
    jcp.with_modulation = getParentEdges().size() > 3;

    const int simd_w = mayiuse(cpu::x64::avx512_common) ? 16 : 8;
    jcp.ic_block = simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.oc_padded = rnd_up(jcp.oc, jcp.oc_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.typesize_in = sizeof(float);
    jcp.typesize_off = sizeof(float);
    jcp.typesize_sampled_wei = sizeof(float);
    jcp.typesize_sampled_offsets = sizeof(int);
    jcp.typesize_out = sizeof(float);

    jcp.ur_w = mayiuse(cpu::x64::avx512_common) ? 6 : 3;
    jcp.nb_oc_blocking = !mayiuse(cpu::x64::avx2) ? 2 : 4;

    jcp.nthr = dnnl_get_max_threads();

    if (enforceRef) {
        return;
    } else if (mayiuse(cpu::x64::avx512_common)) {
        def_conv_kernel.reset(new jit_uni_def_conv_kernel_f32<cpu::x64::avx512_common>(jcp));
    } else if (mayiuse(cpu::x64::avx2)) {
        def_conv_kernel.reset(new jit_uni_def_conv_kernel_f32<cpu::x64::avx2>(jcp));
    } else if (mayiuse(cpu::x64::sse41)) {
        def_conv_kernel.reset(new jit_uni_def_conv_kernel_f32<cpu::x64::sse41>(jcp));
    }

    if (def_conv_kernel)
        def_conv_kernel->create_ker();
}

void MKLDNNDeformableConvolutionNode::executeReference(const float* src, const float* weights, float* dst, const std::vector<size_t>& src_strides,
                                                       const std::vector<size_t>& wei_strides, const std::vector<size_t>& dst_strides) {
    const int G = jcp.ngroups;
    const int MB = jcp.mb;
    const int OH = jcp.oh;
    const int OW = jcp.ow;

    const int OC = jcp.oc;
    const int IC = jcp.ic;
    const int KH = jcp.kh;
    const int KW = jcp.kw;
    const int ker_size = KH * KW;

    const int DG = jcp.dg;

    const int DGHW = DG * OH * OW;
    const int HW = OH * OW;

    const int channel_per_deformable_group = (IC * G) / DG;
    const size_t group_wei_stride = wei_strides[0] * OC;
    auto compKer = [=](int g, int mb, int oc, int oh, int ow) {
        float d = 0;
        for (int ic = 0; ic < IC; ic++) {
            const float *data_im_ptr = src + mb * src_strides[0] + (g * IC + ic) * src_strides[1];
            const int deformable_group_index = (IC * g + ic) / channel_per_deformable_group;
            int sampledCoordIndex = (mb * DGHW + deformable_group_index * HW + oh * OW + ow) * ker_size * sampledPointsPerPixel;
            size_t weiIndex = (size_t) g * group_wei_stride + oc * wei_strides[0] + ic * wei_strides[1];
            for (int kh_off = 0; kh_off < KH * wei_strides[2]; kh_off += wei_strides[2]) {
                for (int kw_off = 0; kw_off < KW * wei_strides[3]; kw_off += wei_strides[3]) {
                    // check if current addendum marked as equal zero
                    if (sampledCoordsVector[sampledCoordIndex] != -1) {
                        const int v11 = sampledCoordsVector[sampledCoordIndex];
                        const int v12 = sampledCoordsVector[sampledCoordIndex + 1];
                        const int v21  = sampledCoordsVector[sampledCoordIndex + 2];
                        const int v22 = sampledCoordsVector[sampledCoordIndex + 3];
                        float val = interpWeightsVector[sampledCoordIndex++] * data_im_ptr[v11];  // v11
                        val += interpWeightsVector[sampledCoordIndex++] * data_im_ptr[v12];  // v12
                        val += interpWeightsVector[sampledCoordIndex++] * data_im_ptr[v21];  // v21
                        val += interpWeightsVector[sampledCoordIndex++] * data_im_ptr[v22];  // v22
                        d += val * weights[weiIndex + kh_off + kw_off];
                    } else {
                        sampledCoordIndex += sampledPointsPerPixel;
                    }
                }
            }
        }
        return d;
    };

    parallel_nd(G, MB, OC, OH, OW,
                [&](int g, int mb, int oc, int oh, int ow)  {
                    dst[mb * dst_strides[0] + (g * OC + oc) * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]] = compKer(g, mb, oc, oh, ow);
                });
}


void MKLDNNDeformableConvolutionNode::executeOptimized(const float* src, const float* weights, float* dst,
                                                       const std::vector<size_t>& src_strides,
                                                       const std::vector<size_t>& dst_strides) {
    size_t buffer_size = (size_t)jcp.nthr * jcp.ur_w * jcp.kh * jcp.kw * jcp.ic * jcp.typesize_in;
    std::vector<float> input_buffer(buffer_size, 0);
    float* input_buffer_ptr = input_buffer.data();

    parallel_for3d(jcp.mb, jcp.ngroups, jcp.oh, [&](size_t n, size_t g, size_t oh) {
        auto ithr = parallel_get_thread_num();

        auto par_conv = jit_def_conv_call_args();

        const size_t _oc = g * jcp.nb_oc;
        const size_t _ic = g * jcp.nb_ic;

        par_conv.src = &src[n * src_strides[0] + _ic*jcp.ic_block * src_strides[1] +
                            (oh * jcp.stride_h - jcp.t_pad) * src_strides[2] - jcp.l_pad * src_strides[3]];
        par_conv.sampledWei = &interpWeightsVector[(n * jcp.dg * jcp.oh + oh) * jcp.kh * jcp.kw * jcp.ow * sampledPointsPerPixel];
        par_conv.sampledCoords = &sampledCoordsVector[(n * jcp.dg * jcp.oh + oh) * jcp.kh * jcp.kw * jcp.ow * sampledPointsPerPixel];
        par_conv.filt = &weights[g * jcp.nb_oc * jcp.nb_ic * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block];
        par_conv.dst = &dst[n * dst_strides[0] + _oc * jcp.oc_block * dst_strides[1] + oh * dst_strides[2]];
        par_conv.buf = input_buffer_ptr + ithr * jcp.ur_w * jcp.kh * jcp.kw * jcp.ic;

        par_conv.oh_pos = oh;

        (*def_conv_kernel)(&par_conv);
    });
}

void MKLDNNDeformableConvolutionNode::execute(mkldnn::stream strm) {
    const size_t inputsNumber = getOriginalInputsNumber();

    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &srcMemory2 = getParentEdgeAt(2)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();

    const auto *src = reinterpret_cast<const float *>(srcMemory0.GetPtr());
    const auto *offsets = reinterpret_cast<const float *>(srcMemory1.GetPtr());
    const auto *weights = reinterpret_cast<const float *>(srcMemory2.GetPtr());
    float* modulation = nullptr;
    if (inputsNumber > 3) {
        modulation = reinterpret_cast<float *>(getParentEdgeAt(3)->getMemory().GetPtr());
    }

    float *dst = reinterpret_cast<float *>(dstMemory.GetPtr());

    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        IE_THROW() << "CPU deformable convolution with name '" << getName() << "' doesn't have primitive descriptors.";
    auto config = selectedPrimitiveDescriptor->getConfig();

    auto src_block_desc = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    std::vector<size_t> src_strides(src_block_desc->getStrides().size());
    for (int i = 0; i < src_strides.size(); i++) {
        src_strides[src_block_desc->getOrder()[i]] = src_block_desc->getStrides()[i];
    }

    auto dst_block_desc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    std::vector<size_t> dst_strides(dst_block_desc->getStrides().size());
    for (int i = 0; i < dst_strides.size(); i++) {
        dst_strides[dst_block_desc->getOrder()[i]] = dst_block_desc->getStrides()[i];
    }

    auto off_strides =  getParentEdgeAt(1)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    auto wei_strides =  getParentEdgeAt(2)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    InferenceEngine::SizeVector modulation_strides;
    if (inputsNumber > 3) {
        modulation_strides = getParentEdgeAt(3)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    }

    prepareSamplingWeights(src_strides, offsets, off_strides, modulation, modulation_strides);

    if (def_conv_kernel) {
        executeOptimized(src, weights, dst, src_strides, dst_strides);
    } else {
        executeReference(src, weights, dst, src_strides, wei_strides, dst_strides);
    }
}

bool MKLDNNDeformableConvolutionNode::created() const {
    return getType() == DeformableConvolution;
}

InferenceEngine::Precision MKLDNNDeformableConvolutionNode::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

REG_MKLDNN_PRIM_FOR(MKLDNNDeformableConvolutionNode, DeformableConvolution);