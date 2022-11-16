// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reorder/reorder_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct reorder_impl : typed_primitive_impl_ocl<reorder> {
    using parent = typed_primitive_impl_ocl<reorder>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reorder_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::reorder_params, kernel_selector::reorder_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reorder_impl>(*this);
    }

    reorder_impl() : parent() {}

    explicit reorder_impl(const reorder_impl& other) : parent(other),
        _can_be_optimized(other._can_be_optimized),
        _has_mean(other._has_mean) {}

    reorder_impl(const reorder_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
        set_node_params(arg);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<reorder>());
        const auto& node = arg.as<reorder>();
        _can_be_optimized = node.can_be_optimized();
        _has_mean = node.has_mean();
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _can_be_optimized;
        ob << _has_mean;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _can_be_optimized;
        ib >> _has_mean;
    }

protected:
    bool optimized_out(reorder_inst& instance) const override {
        return parent::optimized_out(instance) || _can_be_optimized;
    }

    kernel_arguments_data get_arguments(const reorder_inst& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        auto input = &instance.input_memory();
        auto input_layout = input->get_layout();
        if (_has_mean) {
            if (input_layout.format == cldnn::format::nv12) {
                args.bias = instance.mean_nv12_memory();
            } else {
                args.bias = instance.mean_memory();
            }
        }
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<reorder>();
        auto&& output_layout = impl_param.get_output_layout();
        auto params = get_default_params<kernel_selector::reorder_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::reorder_optional_params>(impl_param.get_program());

        auto inputs_count = primitive->input.size();
        bool has_mean = !primitive->mean.empty();
        for (size_t i = 1; i < inputs_count; i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        if (impl_param.get_output_layout().data_padding) {
            params.has_padded_output = true;
        }

        if (has_mean) {
            if (impl_param.get_input_layout(0).format == cldnn::format::nv12) {
                const auto& mean_layout = impl_param.get_input_layout(2);
                params.mean = convert_data_tensor(mean_layout);
                params.mode = kernel_selector::mean_subtruct_mode::IN_BUFFER;
            } else {
                const auto mean_idx = 1;
                const auto& mean_layout = impl_param.get_input_layout(mean_idx);
                params.mean = convert_data_tensor(mean_layout);
                params.mode = kernel_selector::mean_subtruct_mode::IN_BUFFER;
            }
        } else if (primitive->subtract_per_feature.empty() == false) {
            params.mode = kernel_selector::mean_subtruct_mode::INSIDE_PARAMS;
            params.meanValues = primitive->subtract_per_feature;
        } else {
            params.mode = kernel_selector::mean_subtruct_mode::NONE;
        }

        if (params.mode != kernel_selector::mean_subtruct_mode::NONE) {
            switch (primitive->mean_mode) {
                case reorder_mean_mode::none:
                    params.mean_op = kernel_selector::mean_op::NONE;
                    break;
                case reorder_mean_mode::mul:
                    params.mean_op = kernel_selector::mean_op::MUL;
                    break;
                case reorder_mean_mode::subtract:
                    params.mean_op = kernel_selector::mean_op::SUB;
                    break;
                case reorder_mean_mode::div:
                    params.mean_op = kernel_selector::mean_op::DIV;
                    break;
                default: OPENVINO_ASSERT(false, "[GPU] Unsupported mean_mode value in primitive ", primitive->id);
            }
        }

        if (output_layout.format == format::winograd_2x3_s1_data) {
            params.winograd_input_offset_x = 0;
            params.winograd_input_offset_y = 0;
            params.winograd_nr_tiles_x = ceil_div(output_layout.spatial(0), 4);
        }

        params.winograd = impl_param.input_layouts[0].format.is_winograd() || output_layout.format.is_winograd();
        return {params, optional_params};
    }

private:
    bool _can_be_optimized;
    bool _has_mean;
};

namespace detail {

attach_reorder_impl::attach_reorder_impl() {
    implementation_map<reorder>::add(impl_types::ocl, typed_primitive_impl_ocl<reorder>::create<reorder_impl>, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reorder_impl, cldnn::object_type::REORDER_IMPL)
