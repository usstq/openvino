// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>
#include <cassert>
#include <cstdint>

namespace ov {
namespace intel_cpu {

template<typename T>
inline void assert_dt(dnnl::memory::data_type dt) {
    IE_ASSERT(false);
}

template<>
inline void assert_dt<float>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::f32);
}

template<>
inline void assert_dt<ov::bfloat16>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::bf16);
}

template<>
inline void assert_dt<uint8_t>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::u8);
}

template<>
inline void assert_dt<int8_t>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::s8);
}

template<>
inline void assert_dt<int32_t>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::s32);
}

template<typename DT>
struct PlainTensor {
    VectorDims m_dims;
    std::shared_ptr<void> m_ptr;
    size_t m_capacity = 0;

    uint8_t* batched_ptr_buff[8];
    std::vector<uint8_t*> batched_ptr_backup;

    const VectorDims& get_dims() {
        return m_dims;
    }

    PlainTensor(MemoryPtr mem) {
        assert_dt<DT>(mem->GetDataType());
        resize<DT>(mem->getStaticDims(), reinterpret_cast<DT*>(mem->GetPtr()));
    }

    PlainTensor() = default;

    template<typename T = DT>
    void resize(const VectorDims& new_dims, T* data = nullptr) {
        m_dims = new_dims;
        if (!data) {
            auto capacity_new = shape_size(m_dims) * sizeof(T);
            if (capacity_new > m_capacity) {
                m_ptr = std::shared_ptr<void>(
                                    aligned_alloc(64, capacity_new),
                                    [](void * p) { ::free(p); });
                m_capacity = capacity_new;
            }
        } else {
            // m_capacity is zero to indicate that we don't own the memory
            m_capacity = 0;
            m_ptr = std::shared_ptr<void>(reinterpret_cast<void*>(data), [](void*){});
        }
    }

    template <typename T = DT>
    T* data() {
        return reinterpret_cast<T*>(m_ptr.get());
    }

    template <typename T = DT>
    T& at(const std::initializer_list<size_t>& index) {
        size_t off = 0;
        auto it = index.begin();
        for (size_t i = 0; i < m_dims.size(); i++) {
            off = off * m_dims[i];
            if (i < index.size())
                off += *it++;
        }
        return reinterpret_cast<T*>(m_ptr.get())[off];
    }

    void assert_dims(const std::initializer_list<size_t>& expect_dims) {
        IE_ASSERT(m_dims.size() == expect_dims.size());
        IE_ASSERT(std::equal(m_dims.begin(), m_dims.end(), expect_dims.begin()));
    }

    template<typename T = DT>
    uint8_t** get_batched_ptrs() {
        uint8_t** ret_ptrs = batched_ptr_buff;
        auto batch_size = m_dims[0];
        if (batch_size > sizeof(batched_ptr_buff)/sizeof(batched_ptr_buff[0])) {
            batched_ptr_backup.resize(batch_size);
            ret_ptrs = &batched_ptr_backup[0];
        }
        for (size_t b = 0; b < batch_size; b++) {
            ret_ptrs[b] = reinterpret_cast<uint8_t*>(&at<T>({b}));
        }
        return ret_ptrs;
    }
};

}   // namespace intel_cpu
}   // namespace ov