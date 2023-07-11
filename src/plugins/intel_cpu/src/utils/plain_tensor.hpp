// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <dnnl_extension_utils.h>
#include <node.h>

#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {

template <typename T>
inline void assert_dt(dnnl::memory::data_type dt) {
    IE_ASSERT(false);
}

template <>
inline void assert_dt<float>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::f32);
}

template <>
inline void assert_dt<ov::bfloat16>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::bf16);
}

template <>
inline void assert_dt<uint8_t>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::u8);
}

template <>
inline void assert_dt<int8_t>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::s8);
}

template <>
inline void assert_dt<int32_t>(dnnl::memory::data_type dt) {
    IE_ASSERT(dt == dnnl::memory::data_type::s32);
}

template <typename T>
struct data_type_name {
    static constexpr const char* value = "?";
};

template <>
struct data_type_name<float> {
    static constexpr const char* value = "float";
};

template <>
struct data_type_name<bfloat16> {
    static constexpr const char* value = "bfloat16";
};

template <>
struct data_type_name<uint8_t> {
    static constexpr const char* value = "uint8_t";
};

template <typename T>
struct precision_of {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::UNSPECIFIED;
};

template <>
struct precision_of<float> {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::FP32;
};

template <>
struct precision_of<int32_t> {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::I32;
};

template <>
struct precision_of<bfloat16> {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::BF16;
};

template <>
struct precision_of<uint8_t> {
    static constexpr InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::ePrecision::U8;
};

#define PLAINTENSOR_RANK_MAX 8
struct PlainTensorBase {
    size_t m_strides[PLAINTENSOR_RANK_MAX];
    size_t m_dims[PLAINTENSOR_RANK_MAX];
    size_t m_rank;

    std::shared_ptr<void> m_ptr;
    size_t m_capacity = 0;

    uint8_t* batched_ptr_buff[8];
    std::vector<uint8_t*> batched_ptr_backup;

    operator bool() const {
        return static_cast<bool>(m_ptr);
    }

    size_t size(int i) const {
        assert(i < m_rank);
        return m_dims[i];
    }
    size_t stride(int i) const {
        assert(i < m_rank);
        return m_strides[i];
    }
    virtual InferenceEngine::Precision::ePrecision get_precision(void) = 0;
    virtual void reset(MemoryPtr mem) = 0;
};

template <typename DT>
struct PlainTensor : public PlainTensorBase {
    PlainTensor(MemoryPtr mem) {
        reset(mem);
    }

    PlainTensor() = default;

    void reset(MemoryPtr mem) override {
        assert_dt<DT>(mem->GetDataType());
        // this reshape_to() can do reshape w/o additional cost
        resize(mem->getStaticDims(), reinterpret_cast<DT*>(mem->GetPtr()));
    }

    InferenceEngine::Precision::ePrecision get_precision(void) override {
        return precision_of<DT>::value;
    }

    // gives names
    // define("BLKHS")
    // index("K", 0)
    // permute("BHLS")

    struct tensor_index {
        int start;
        int end;
        int step;
        int count;
        // select all
        tensor_index() {
            start = 0;
            end = INT_MAX;
            step = 1;
        }
        bool slice_with_squeeze() {
            return end == INT_MIN;
        }
        // tensor_index(start)            : select 1 element (with squeeze)
        // tensor_index(start, end, step) : select a range w/o squeeze
        tensor_index(int start, int end = INT_MIN, int step = 1) : start(start), end(end), step(step) {}

        void regularize(int size) {
            if (start < 0)
                start += size;
            assert(start >= 0 && start < size);
            if (end != INT_MIN) {
                if (end < 0)
                    end += size;
                if (end > size)
                    end = size;
                assert(end >= 0 && end <= size);
                count = (end - start + step - 1) / step;
            } else {
                count = 1;
            }
        }
    };

    PlainTensor<DT> index(const std::initializer_list<tensor_index>& indices) {
        PlainTensor<DT> sub_tensor;
        assert(indices.size() <= m_rank);
        int i_src = 0;
        int i_dst = 0;
        sub_tensor.m_capacity = 0;
        size_t off = 0;
        for (auto idx : indices) {
            auto src_dim = m_dims[i_src];
            auto src_stride = m_strides[i_src];
            idx.regularize(src_dim);
            off += idx.start * src_stride;
            if (idx.slice_with_squeeze()) {
                // no output dimension
                i_src++;
                continue;
            }
            sub_tensor.m_dims[i_dst] = idx.count;
            sub_tensor.m_strides[i_dst] = src_stride;
            i_dst++;
            i_src++;
        }
        sub_tensor.m_rank = i_dst;  // index may imply squeeze
        sub_tensor.m_ptr = std::shared_ptr<void>(reinterpret_cast<DT*>(m_ptr.get()) + off, [](void*) {});
        return sub_tensor;
    }

    // slice: return a sub-view (w/o ownership/refcount to original data)
    PlainTensor<DT> slice(int axis, int start, int end) const {
        PlainTensor<DT> sub_tensor;
        assert(axis < m_rank);

        sub_tensor.m_capacity = 0;
        sub_tensor.m_rank = m_rank;  // slice dosen't change rank & strides
        for (size_t i = 0; i < m_rank; i++) {
            sub_tensor.m_strides[i] = m_strides[i];
            sub_tensor.m_dims[i] = m_dims[i];
        }
        sub_tensor.m_dims[axis] = end - start;

        auto off = start * m_strides[axis];
        auto* data = reinterpret_cast<DT*>(m_ptr.get()) + off;
        sub_tensor.m_ptr = std::shared_ptr<void>(reinterpret_cast<void*>(data), [](void*) {});

        return sub_tensor;
    }

    bool is_dense() const {
        // check if it's dense tensor
        size_t stride = 1;
        for (int i = m_rank - 1; i >= 0; i--) {
            if (m_strides[i] != stride)
                return false;
            stride *= m_dims[i];
        }
        return true;
    }

    /*
       suppose current shape is [a0,a1,...,am]
       and target shape is [b0,b1,...,bn]
       reshape is only valid when (a0*a1*...*am) == (b0*b1*...*bn) <======= (A)

       uniform a tensor's shape into groups from last to first, the dimension is merged
       into current group if the subtensor in the group is still dense after merge.
       otherwise a new group is formed.

       then reshape is performed on group basis, the check (A) is performed on group bases.
       which means any reshape inside the group is OK, but not across the group boundary.

       this can be done in one-loop, while group is forming, and checks are performed.

       simplified form is when whole tensor is dense
    */
    PlainTensor<DT> reshape(const std::initializer_list<size_t>& target_shape) const {
        // only valid for dense memory
        PlainTensor<DT> new_tensor_view;
        assert(is_dense());
        assert(shape_size(target_shape) == shape_size(m_dims));
        new_tensor_view.resize(VectorDims(target_shape), reinterpret_cast<DT*>(m_ptr.get()));
        return new_tensor_view;
    }

    PlainTensor<DT> permute(const std::initializer_list<size_t>& order) const {
        PlainTensor<DT> new_tensor_view;
        assert(order.size() == m_rank);
        new_tensor_view.m_capacity = 0;
        new_tensor_view.m_ptr = m_ptr;
        new_tensor_view.m_rank = m_rank;
        auto it_order = order.begin();
        // also should check order has no repeat element
        for (size_t i = 0; i < m_rank; i++) {
            auto j = *it_order++;
            assert(j >= 0 && j < m_rank);
            new_tensor_view.m_dims[i] = m_dims[j];
            new_tensor_view.m_strides[i] = m_strides[j];
        }
        return new_tensor_view;
    }

    void resize(const VectorDims& new_dims, DT* data = nullptr) {
        // initialize strides for compact/dense tensor
        m_rank = new_dims.size();
        assert(m_rank <= PLAINTENSOR_RANK_MAX);
        size_t stride = 1;
        for (int i = m_rank - 1; i >= 0; i--) {
            m_dims[i] = new_dims[i];
            m_strides[i] = stride;
            stride *= new_dims[i];
        }

        if (!data) {
            auto capacity_new = m_strides[0] * m_dims[0] * sizeof(DT);
            if (capacity_new > m_capacity) {
                m_ptr = std::shared_ptr<void>(aligned_alloc(64, capacity_new), [](void* p) {
                    ::free(p);
                });
                m_capacity = capacity_new;
            }
        } else {
            // m_capacity is zero to indicate that we don't own the memory
            m_capacity = 0;
            m_ptr = std::shared_ptr<void>(reinterpret_cast<void*>(data), [](void*) {});
        }
    }

    DT* data() const {
        return reinterpret_cast<DT*>(m_ptr.get());
    }

    DT& at(const std::initializer_list<size_t>& index) const {
        size_t off = 0;
        auto it = index.begin();
        for (auto& stride : m_strides) {
            auto coordinate = (it != index.end()) ? (*it++) : 0;
            off += stride * coordinate;
        }
        return reinterpret_cast<DT*>(m_ptr.get())[off];
    }

    DT& operator()(const std::initializer_list<size_t>& index) const {
        return at(index);
    }

    void assert_dims(const std::initializer_list<size_t>& expect_dims) const {
        if (m_rank != expect_dims.size()) {
            asm("int3");
            IE_ASSERT(false);
        }
        if (!std::equal(expect_dims.begin(), expect_dims.end(), m_dims)) {
            std::stringstream ss;
            ss << " m_dims=[";
            for (size_t i = 0; i < m_rank; i++)
                ss << m_dims[i] << ",";
            ss << "] expect_dims=[";
            for (auto& i : expect_dims)
                ss << i << ",";
            ss << "]";
            asm("int3");
            IE_ASSERT(false) << ss.str();
        }
    }

    uint8_t** get_batched_ptrs() {
        uint8_t** ret_ptrs = batched_ptr_buff;
        auto batch_size = m_dims[0];
        if (batch_size > sizeof(batched_ptr_buff) / sizeof(batched_ptr_buff[0])) {
            batched_ptr_backup.resize(batch_size);
            ret_ptrs = &batched_ptr_backup[0];
        }
        for (size_t b = 0; b < batch_size; b++) {
            ret_ptrs[b] = reinterpret_cast<uint8_t*>(&at({b}));
        }
        return ret_ptrs;
    }

    int max_repr_len = 256;

    std::string repr(int max_total_lines = 16, int lines_per_row = 1) const {
        std::stringstream ss;
        ss << data_type_name<DT>::value << " shape=[";
        const char* sep = "";
        size_t sz = 1;
        for (size_t i = 0; i < m_rank; i++) {
            ss << sep << m_dims[i];
            sz *= m_dims[i];
            sep = ",";
        }
        ss << "] strides=[";
        sep = "";
        for (size_t i = 0; i < m_rank; i++) {
            ss << sep << m_strides[i];
            sep = ",";
        }
        ss << "] {";
        if (m_rank > 1)
            ss << "\n";
        auto last_dim_size = m_dims[m_rank - 1];
        int row_id = 0;
        int cur_row_lines_left = lines_per_row;
        int cur_line_elecnt = 0;
        int cur_row_elecnt = 0;
        size_t i;
        auto* p = reinterpret_cast<DT*>(m_ptr.get());
        for (i = 0; i < sz && max_total_lines > 0; i++) {
            if ((i % last_dim_size) == 0) {
                ss << row_id << ":\t\t";
                row_id++;
                cur_row_lines_left = lines_per_row;
            }

            // display current element if we still have buget
            if (cur_row_lines_left > 0) {
                ss << p[i] << ",";
                cur_line_elecnt++;
                cur_row_elecnt++;
                if ((cur_line_elecnt % 16) == 15 || (cur_row_elecnt == last_dim_size)) {
                    max_total_lines--;
                    cur_row_lines_left--;
                    if (cur_row_lines_left == 0) {
                        if (cur_row_elecnt == last_dim_size)
                            ss << ",\n";
                        else
                            ss << "...\n";
                        cur_row_elecnt = 0;
                    } else {
                        ss << "\n\t\t";
                    }
                    cur_line_elecnt = 0;
                }
            }
        }
        if (i < sz) {
            ss << "... ... ... ... \n";
        }
        ss << "}";
        return ss.str();
    }

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const PlainTensor<U>& dt);
};

template <typename U>
std::ostream& operator<<(std::ostream& os, const PlainTensor<U>& dt) {
    os << dt.repr();
    return os;
}

}  // namespace intel_cpu
}  // namespace ov