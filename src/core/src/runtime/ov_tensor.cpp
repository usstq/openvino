// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shape_util.hpp"

namespace ov {

#define OV_TENSOR_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized."); \
    try {                                                             \
        __VA_ARGS__;                                                  \
    } catch (const std::exception& ex) {                              \
        OPENVINO_THROW(ex.what());                                    \
    } catch (...) {                                                   \
        OPENVINO_ASSERT(false, "Unexpected exception");               \
    }

void Tensor::type_check(const Tensor&) {}

Tensor::~Tensor() {
    _impl = {};
}

Tensor::Tensor(const Tensor& tensor, const std::shared_ptr<void>& so) : _impl{tensor._impl}, _so{tensor._so} {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
    if (!_so)
        _so = so;
}

Tensor::Tensor(const std::shared_ptr<ITensor>& impl, const std::shared_ptr<void>& so) : _impl{impl}, _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
}

Tensor::Tensor(const element::Type& element_type, const Shape& shape, const Allocator& allocator)
    : _impl{make_tensor(element_type, shape, allocator)} {}

Tensor::Tensor(const element::Type& element_type, const Shape& shape, void* host_ptr, const Strides& byte_strides)
    : _impl{make_tensor(element_type, shape, host_ptr, byte_strides)} {}

Tensor::Tensor(const Tensor& owner, const Coordinate& begin, const Coordinate& end)
    : _impl{make_tensor(owner._impl, begin, end)},
      _so{owner._so} {}

Tensor::Tensor(const ov::Output<const ov::Node>& port, const Allocator& allocator)
    : Tensor(port.get_element_type(),
             port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_shape(),
             allocator) {}

Tensor::Tensor(const ov::Output<const ov::Node>& port, void* host_ptr, const Strides& byte_strides)
    : Tensor(port.get_element_type(),
             port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_shape(),
             host_ptr,
             byte_strides) {}

const element::Type& Tensor::get_element_type() const {
    OV_TENSOR_STATEMENT(return _impl->get_element_type());
}

void Tensor::set_shape(const ov::Shape& shape) {
    OV_TENSOR_STATEMENT(_impl->set_shape(shape));
}

const Shape& Tensor::get_shape() const {
    OV_TENSOR_STATEMENT(return _impl->get_shape());
}

void Tensor::copy_to(ov::Tensor& dst) const {
    const auto& is_scalar = [](const ov::Shape& shape) {
        return shape.empty() || (shape.size() == 1 && shape[0] == 1);
    };
    const auto shapes_equal = [is_scalar](const ov::Shape& src, const ov::Shape& dst) {
        // WA for scalar tensors to copy {1} to {} or otherwise
        return src == dst || (is_scalar(src) && is_scalar(dst));
    };
    OV_TENSOR_STATEMENT({
        OPENVINO_ASSERT(dst, "Destination tensor was not initialized.");
        OPENVINO_ASSERT(!is<ov::RemoteTensor>(), "Default copy to doesn't support copy from remote tensor.");
        OPENVINO_ASSERT(!dst.is<ov::RemoteTensor>(), "Default copy to doesn't support copy to remote tensor.");
        OPENVINO_ASSERT(dst.get_element_type() == get_element_type(),
                        "Tensor element types are not equal. (src: ",
                        get_element_type(),
                        " != dst: ",
                        dst.get_element_type(),
                        ")");
        if (dst.get_shape() == ov::Shape{0})
            dst.set_shape(get_shape());
        OPENVINO_ASSERT(shapes_equal(get_shape(), dst.get_shape()),
                        "Tensor shapes are not equal. (src: ",
                        get_shape(),
                        " != dst: ",
                        dst.get_shape(),
                        ")");
        const auto& shape = get_shape();
        auto* src_data = static_cast<const uint8_t*>(data());
        auto* dst_data = static_cast<uint8_t*>(dst.data());
        ov::Strides src_strides{get_byte_size()};
        ov::Strides dst_strides{dst.get_byte_size()};
        ov::Shape cur_pos{0};
        ov::Shape max_pos{1};

        if (get_element_type().bitwidth() < 8 || (get_strides() == dst.get_strides() && is_continuous()) ||
            (is_scalar(get_shape()) && is_scalar(dst.get_shape()))) {
            // OpenVINO doesn't support strides for LP types
            // or both tensors have default strides
            // Strides and positions already initialized
        } else {
            // Tensors have default strides
            const auto& type = get_element_type();
            std::vector<size_t> strides(shape.size());
            if (!shape.empty()) {
                strides[shape.size() - 1] = 1;
            }
            auto size = shape.size();
            for (size_t i = 1; i < size; i++) {
                strides[size - i - 1] = strides[size - i] * shape[size - i];
            }

            ov::Strides default_strides(strides.size());
            for (size_t i = 0; i < strides.size(); ++i)
                default_strides[i] = strides[i] * type.size();

            src_strides = get_strides();
            dst_strides = dst.get_strides();

            ov::Strides src_str, dst_str;

            // Calculate src and dst shapes
            bool found_step = false;
            for (size_t i = 0; i < shape.size(); i++) {
                size_t inverted_idx = shape.size() - i - 1;
                if (!found_step) {
                    if (default_strides[inverted_idx] == src_strides[inverted_idx] &&
                        src_strides[inverted_idx] == dst_strides[inverted_idx]) {
                        continue;
                    } else {
                        found_step = true;
                        size_t strides_size = inverted_idx + 1;
                        // Set right size
                        src_str.resize(strides_size + 1);
                        dst_str.resize(strides_size + 1);
                        max_pos.resize(strides_size + 1);
                        cur_pos.resize(strides_size + 1);
                        // In case of default continuous strides we can copy several elements
                        // In other case only one element
                        size_t dim = 1;
                        size_t strides = type.size();

                        if (strides_size < default_strides.size()) {
                            strides = default_strides[strides_size];
                            dim = get_shape()[strides_size];
                        }
                        src_str[strides_size] = strides;
                        dst_str[strides_size] = strides;
                        max_pos[strides_size] = dim;
                        cur_pos[strides_size] = 0;
                    }
                }
                src_str[inverted_idx] = src_strides[inverted_idx];
                dst_str[inverted_idx] = dst_strides[inverted_idx];
                max_pos[inverted_idx] = shape[inverted_idx];
                cur_pos[inverted_idx] = 0;
            }
            src_strides = src_str;
            dst_strides = dst_str;
        }

        const auto update_index = [](const ov::Shape& pos, const ov::Shape& shape, const ov::Strides& strides) {
            size_t offset = 0;

            for (size_t i = 0; i < pos.size(); i++) {
                offset += pos[i] * strides[i];
            }
            return offset;
        };

        bool finish = false;
        for (size_t dst_idx = 0, src_idx = 0; !finish;) {
            memcpy(dst_data + dst_idx, src_data + src_idx, src_strides[src_strides.size() - 1]);
            // update indexes
            for (size_t i = 0; i < cur_pos.size(); i++) {
                size_t inverted_idx = cur_pos.size() - i - 1;
                cur_pos[inverted_idx]++;
                if (cur_pos[inverted_idx] != max_pos[inverted_idx]) {
                    break;
                }
                if (inverted_idx)
                    cur_pos[inverted_idx] = 0;
                else
                    finish = true;
            }
            src_idx = update_index(cur_pos, max_pos, src_strides);
            dst_idx = update_index(cur_pos, max_pos, dst_strides);
        }
    });
}

Strides Tensor::get_strides() const {
    OV_TENSOR_STATEMENT(return _impl->get_strides(););
}

size_t Tensor::get_size() const {
    OV_TENSOR_STATEMENT(return _impl->get_size());
}

size_t Tensor::get_byte_size() const {
    OV_TENSOR_STATEMENT(return _impl->get_byte_size(););
}

void* Tensor::data(const element::Type& element_type) const {
    OV_TENSOR_STATEMENT(return _impl->data(element_type));
}

bool Tensor::operator!() const noexcept {
    return !_impl;
}

Tensor::operator bool() const noexcept {
    return (!!_impl);
}

bool Tensor::is_continuous() const {
    OV_TENSOR_STATEMENT(return _impl->is_continuous());
}

}  // namespace ov
