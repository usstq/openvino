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
#include <iostream>

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

template<typename T>
struct data_type_name {
    static constexpr char * value = "?";
};

template<>
struct data_type_name<float> {
    static constexpr char * value = "float";
};

template<>
struct data_type_name<bfloat16> {
    static constexpr char * value = "bfloat16";
};

template<>
struct data_type_name<uint8_t> {
    static constexpr char * value = "uint8_t";
};


template<typename T>
struct precision_of {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::UNSPECIFIED;
};

template<>
struct precision_of<float> {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::FP32;
};

template<>
struct precision_of<int32_t> {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::I32;
};

template<>
struct precision_of<bfloat16> {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::BF16;
};

template<>
struct precision_of<uint8_t> {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::U8;
};

struct PlainTensorBase {
    VectorDims m_strides;
    VectorDims m_dims;
    std::shared_ptr<void> m_ptr;
    size_t m_capacity = 0;

    uint8_t* batched_ptr_buff[8];
    std::vector<uint8_t*> batched_ptr_backup;

    const VectorDims& get_dims() {
        return m_dims;
    }

    virtual InferenceEngine::Precision::ePrecision get_precision(void) = 0;
    virtual void reset(MemoryPtr mem) = 0;
};

template<typename DT>
struct PlainTensor : public PlainTensorBase {
    PlainTensor(MemoryPtr mem) {
        assert_dt<DT>(mem->GetDataType());
        resize<DT>(mem->getStaticDims(), reinterpret_cast<DT*>(mem->GetPtr()));
    }

    PlainTensor() = default;

    void reset(MemoryPtr mem) override {
        assert_dt<DT>(mem->GetDataType());
        resize<DT>(mem->getStaticDims(), reinterpret_cast<DT*>(mem->GetPtr()));
    }

    InferenceEngine::Precision::ePrecision get_precision(void) override {
        return precision_of<DT>::value;
    }

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
    T* data() const {
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

    template <typename T = DT>
    T& operator()(const std::initializer_list<size_t>& index) {
        return at<T>(index);
    }

    void assert_dims(const std::initializer_list<size_t>& expect_dims) {
        IE_ASSERT(m_dims.size() == expect_dims.size());
        if (!std::equal(m_dims.begin(), m_dims.end(), expect_dims.begin())) {
            std::stringstream ss;
            ss << "m_dims=[";
            for (auto& i : m_dims)
                ss << i << ",";
            ss << "] expect_dims=[";
            for (auto& i : expect_dims)
                ss << i << ",";
            ss << "]";
            IE_ASSERT(false) << ss.str();
        }
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

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const PlainTensor<U>& dt);
};

template<typename DT>
std::ostream& operator<<(std::ostream& os, const PlainTensor<DT>& dt) {
    DT * p = dt.data();
    auto rank = dt.m_dims.size();
    const char * sep = "";
    os << data_type_name<DT>::value << "[";
    sep = "";
    for (auto& d : dt.m_dims) {
        os << sep << d;
        sep = ",";
    }
    os << "] {";

    if (rank > 1) os << "\n\t";

    auto last_dim_size = dt.m_dims[dt.m_dims.size() - 1];

    auto sz = shape_size(dt.m_dims);
    std::stringstream ss;
    int lines = 0;
    for (size_t i = 0; i < sz; i++) {
        if (ss.tellp() < 256)
            ss << p[i] << ",";

        if ((i % last_dim_size) == (last_dim_size - 1)) {
            os << lines << " : " <<  ss.str() << "...\n\t";
            ss.str("");
            lines++;
            if (lines > 16) {
                os << "... ... ... ... \n\t";
                break;
            }
        }
    }
    os << "}";
    return os;
}

}   // namespace intel_cpu
}   // namespace ov