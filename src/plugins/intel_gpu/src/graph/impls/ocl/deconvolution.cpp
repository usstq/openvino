// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "deconvolution/deconvolution_kernel_selector.h"
#include "deconvolution/deconvolution_kernel_base.h"
#include <algorithm>

namespace cldnn {
namespace ocl {

struct deconvolution_impl : typed_primitive_impl_ocl<deconvolution> {
    using parent = typed_primitive_impl_ocl<deconvolution>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::deconvolution_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::deconvolution_params, kernel_selector::deconvolution_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deconvolution_impl>(*this);
    }

    deconvolution_impl() : parent() {}

    explicit deconvolution_impl(const deconvolution_impl& other) : parent(other),
        _split(other._split),
        _groups(other._groups) {}

    deconvolution_impl(const deconvolution_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
        set_node_params(arg);
        this->can_reuse_memory = kd.can_reuse_memory;
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<deconvolution>());
        const auto& node = arg.as<deconvolution>();
        _split = node.get_split();
        _groups = node.get_groups();
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _split;
        ob << _groups;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _split;
        ib >> _groups;
    }

protected:
    // TODO: share it with convolution and fully connected
    bool validate_impl(const typed_primitive_inst<deconvolution>& instance) const override {
        bool res = true;

        CLDNN_ERROR_NOT_EQUAL(_node_id,
                              "deconvolution filling value",
                              instance.node->get_output_layout().data_padding.filling_value(),
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in deconvolution.");

        return res;
    }

    kernel_arguments_data get_arguments(const typed_primitive_inst<deconvolution>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = instance.weights_memory(split);
        args.bias = instance.bias_term() ? instance.bias_memory(split) : nullptr;

        return args;
    }

    int32_t get_split() const override { return _split; }

    uint32_t get_groups() const override { return _groups; }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<deconvolution>();
        const auto& split = primitive->split();
        const auto& stride = primitive->stride;
        const ov::Strides dilation(impl_param.get_output_layout().get_spatial_rank(), 1);
        const auto actual_split = split;

        const auto& pad = primitive->pad;
        const auto& groups = primitive->groups;

        auto params = get_weights_bias_default_params<kernel_selector::deconvolution_params>(
            impl_param,
            (groups > 1) ? 1 : actual_split,
            1,
            primitive->grouped_weights_shape);
        auto optional_params = get_default_weights_bias_optional_params<kernel_selector::deconvolution_optional_params>(impl_param.get_program());

        params.split = split;
        params.groups = groups;

        const auto weights_idx = 1 + 0;
        const auto& weights_layout = impl_param.input_layouts[weights_idx].convert_to_weights_layout(primitive->grouped_weights_shape);
        uint32_t kx = weights_layout.spatial(0);
        uint32_t ky = weights_layout.spatial(1);
        uint32_t kz = weights_layout.spatial(2);

        params.filterSize = { kx, ky, kz };

        uint32_t pad_z = std::max<std::ptrdiff_t>(pad.size() >= 3 ? pad[pad.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pad.size() >= 2 ? pad[pad.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pad.size() >= 1 ? pad[pad.size() - 1] : 0, 0);
        params.padding = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
        uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
        uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
        params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;
        params.dilation = {dilation_x, dilation_y, dilation_z};

        return {params, optional_params};
    }

private:
    int32_t _split;
    uint32_t _groups;
};

namespace detail {

attach_deconvolution_impl::attach_deconvolution_impl() {
    static auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8};
    static auto formats = {
        format::bfyx,
        format::byxf,
        format::yxfb,

        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
    };

    implementation_map<deconvolution>::add(impl_types::ocl, typed_primitive_impl_ocl<deconvolution>::create<deconvolution_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::deconvolution_impl, cldnn::object_type::DECONVOLUTION_IMPL)