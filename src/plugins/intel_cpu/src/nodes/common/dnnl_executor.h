// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu_memory.h>
#include <primitive.h>
#include <onednn/iml_type_mapper.h>

#include "graph_context.h"
namespace ov {
namespace intel_cpu {

class DnnlExecutor {
    protected:
        class IntermReorder {
            public:
                IntermReorder(const dnnl::memory::desc& descSrc, const dnnl::memory::desc& descDst, const dnnl::engine& engine);
                void exec(dnnl::memory& memSrc, dnnl::memory& memDst, dnnl::stream strm);
                const dnnl::memory::desc& getSrcDesc() const { return m_descSrc; }
                const dnnl::memory::desc& getDstDesc() const { return m_descDst; }

            private:
                dnnl::reorder m_reorder;
                dnnl::memory::desc m_descSrc;
                dnnl::memory::desc m_descDst;
        };

    public:
        void exec(std::unordered_map<int, dnnl::memory> primArgs, dnnl::stream strm);
        bool needReordering() const;
        virtual ~DnnlExecutor() = default;
        Primitive getExecPrim() const;
        const_dnnl_primitive_desc_t getPrimitiveDesc() const;
        dnnl::memory::desc getSrcDesc() const;
        dnnl::memory::desc getWeightDesc() const;
        dnnl::memory::desc getDstDesc() const;
        impl_desc_type getImplementationType() const;

    protected:
        DnnlExecutor() = default;
        Primitive execPrim;
        // key is the port number for the primitive that needs memory reordering
        std::unordered_map<int, IntermReorder> inputReorders;
        std::unordered_map<int, IntermReorder> outputReorders;
};

class DnnlExecutor2 {
    protected:
        using PrimArgs = std::unordered_map<int, dnnl::memory>;
        struct IntermReorder {
            dnnl::reorder prim;
            PrimArgs args;
            IntermReorder() = default;
            IntermReorder(IntermReorder&& s) : prim(std::move(s.prim)), args(std::move(s.args)) {}
            IntermReorder(dnnl::reorder prim, PrimArgs args)
                : prim(std::move(prim)),
                args(std::move(args)) {}
        };

    public:
        DnnlExecutor2(const GraphContext::CPtr context, const std::string& name)
            : context(context),
              name(name) {}

        void reset(dnnl::primitive p, dnnl::primitive_desc_base prim_desc) {
            prim = p;
            pd = prim_desc;
            inputReorders.clear();
            outputReorders.clear();
            constFoldings.clear();
            updateHandles.clear();
            constFolded = false;
            // privateWeightCache was kept to always reference weights of different format
            setScratchPad();
        }

        operator bool() {
            return static_cast<bool>(prim);
        }

        void exec(dnnl::stream strm);
        bool needReordering() const;
        virtual ~DnnlExecutor2() = default;
        const dnnl::primitive & getPrimitive() const {
            return prim;
        }
        const dnnl::primitive_desc_base & getPrimitiveDesc() const {
            return pd;
        }
        dnnl::memory::desc queryMD(const dnnl::query& what, int idx);
        impl_desc_type getImplementationType() const;

        // Arg type specification.
        enum class arg_mem_type {
            normal,
            constant,
            reinterpret
        };

        void setArg(int arg_id, dnnl::memory arg_mem, arg_mem_type atype = arg_mem_type::normal);
        void setDynamicBatch(int newBatch);
        void setArgDynamicBatch(int arg_id, int newBatch);

    protected:
        const GraphContext::CPtr context;
        std::string name;

        dnnl::primitive prim;
        dnnl::primitive_desc_base pd;
        PrimArgs args;

        // holding scratch pad memory object with callbacks responsible for change memory handle
        // when underlying scratch pad memory is resized.
        MemoryPtr scratchpadMem;

        // will set handle of src_mem into dst_mem before each execution
        struct UpdateHandle {
            dnnl::memory src_mem;
            void* handle;
            dnnl::memory dst_mem;
            UpdateHandle(dnnl::memory src_mem, dnnl::memory dst_mem)
                : src_mem(src_mem),
                  dst_mem(dst_mem),
                  handle(nullptr) {}
        };

        struct ConstFolding {
            dnnl::memory src_mem;
            // before const folding dst_mem is referencing a memory with expected desc but no handle(pointer)
            // and all primitives need this memory as argument have been setup with reference exactly to this memory.
            // after const folding, it's handle will be set to valid pointer with data filled.
            dnnl::memory dst_mem;
            std::string key;
        };

        bool constFolded;
        std::vector<ConstFolding> constFoldings;
        std::vector<UpdateHandle> updateHandles;
        std::vector<IntermReorder> inputReorders;
        std::vector<IntermReorder> outputReorders;
        dnnl::memory addConstFolding(dnnl::memory src, std::string privateSrcKey, DnnlMemoryDescPtr expectedWeightDesc);
        void doConstFolding(ConstFolding& cf);
        void setScratchPad();

        // when weightCache is not enabled (such as stream=1), brgconv weights may change due to
        // different shapes. Weights will be cached in privateWeightCache.
        // When weightCache is enabled, it holds weight ptr reference since weightCache does not hold the
        // reference
        std::unordered_map<std::string, MemoryPtr> privateWeightCache;
};

}   // namespace intel_cpu
}   // namespace ov
