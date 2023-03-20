// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "eltwise_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <vector>
#include <algorithm>

#include "openvino/op/add.hpp"
#include "utils.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(eltwise)

layout eltwise_inst::calc_output_layout(eltwise_node const& node, kernel_impl_params const& impl_param) {
    size_t primary_input_idx = 0;
    if (node.input(primary_input_idx).is_constant()) {
        for (size_t i = 1; i < node.get_dependencies().size(); i++) {
            if (!node.input(i).is_constant()) {
                primary_input_idx = i;
                break;
            }
        }
    }
    auto input_node_layout = impl_param.get_non_padded_input_layout(primary_input_idx);
    auto desc = impl_param.typed_desc<eltwise>();
    auto output_type = desc->output_data_types[0].value_or(input_node_layout.data_type);

    auto size = input_node_layout.get_tensor();
    auto format = input_node_layout.format;
    for (size_t i = 0; i < desc->input_size(); i++) {
        if (i == primary_input_idx)
            continue;

        auto l = impl_param.get_non_padded_input_layout(i);
        size = tensor::max(size, l.get_tensor());
        if (l.format == format::b_fs_zyx_fsv16)  // use optimized 5D
            format = format::b_fs_zyx_fsv16;
        else if (l.format == format::bs_fs_zyx_bsv16_fsv16)
            format = format::bs_fs_zyx_bsv16_fsv16;
    }
    auto output_layout = layout(output_type, format, size);

    auto mode = desc->mode;
    // list of operations supported for integer types
    if (input_node_layout.data_type == data_types::i8 || input_node_layout.data_type == data_types::u8 ||
        input_node_layout.data_type == data_types::i32 || input_node_layout.data_type == data_types::i64) {
        std::vector<eltwise_mode> eltwise_int_modes = {eltwise_mode::sum,
                                                       eltwise_mode::sub,
                                                       eltwise_mode::prod,
                                                       eltwise_mode::div,
                                                       eltwise_mode::min,
                                                       eltwise_mode::max,
                                                       eltwise_mode::mod,
                                                       eltwise_mode::eq,
                                                       eltwise_mode::ne,
                                                       eltwise_mode::lt,
                                                       eltwise_mode::le,
                                                       eltwise_mode::gt,
                                                       eltwise_mode::ge,
                                                       eltwise_mode::squared_diff,
                                                       eltwise_mode::floor_mod,
                                                       eltwise_mode::logic_and,
                                                       eltwise_mode::logic_or,
                                                       eltwise_mode::logic_xor};
        if (std::find(eltwise_int_modes.begin(), eltwise_int_modes.end(), mode) == eltwise_int_modes.end())
            CLDNN_ERROR_MESSAGE(desc->id, "Requested eltwise mode is not supported for integer types.");
    }

    // Logic and comparison operations should return i8 for any inputs
    std::vector<eltwise_mode> eltwise_bool_modes = {eltwise_mode::eq,
                                                    eltwise_mode::ne,
                                                    eltwise_mode::lt,
                                                    eltwise_mode::le,
                                                    eltwise_mode::gt,
                                                    eltwise_mode::ge,
                                                    eltwise_mode::logic_and,
                                                    eltwise_mode::logic_or,
                                                    eltwise_mode::logic_xor,
                                                    eltwise_mode::is_finite,
                                                    eltwise_mode::is_inf,
                                                    eltwise_mode::is_nan};
    if (std::find(eltwise_bool_modes.begin(), eltwise_bool_modes.end(), mode) != eltwise_bool_modes.end()) {
        output_layout.data_type = data_types::i8;
    }

    if (desc->output_data_types[0]) {
        output_layout.data_type = *desc->output_data_types[0];
    }

    if (node.has_fused_primitives()) {
        output_layout.data_type = node.get_fused_output_layout().data_type;
    }

    if (!desc->stride.empty()) {
        auto new_size = input_node_layout.get_tensor();
        // we can safely use only first stride, since we're using first input, and input / stride should give exact same
        // value for every input
        new_size.spatial[0] = (input_node_layout.spatial(0) - 1) / desc->stride[0].spatial[0] + 1;
        new_size.spatial[1] = (input_node_layout.spatial(1) - 1) / desc->stride[0].spatial[1] + 1;
        new_size.spatial[2] = (input_node_layout.spatial(2) - 1) / desc->stride[0].spatial[2] + 1;
        input_node_layout.set_tensor(new_size);
        return input_node_layout;
    }
    return output_layout;
}

template<typename ShapeType>
std::vector<layout> eltwise_inst::calc_output_layouts(eltwise_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<eltwise>();
    auto input_layout = impl_param.get_non_padded_input_layout(impl_param.primary_input_idx);
    auto out_data_type = desc->output_data_types[0].value_or(input_layout.data_type);

    auto get_output_layout = [&]() {
        auto out_pshape = input_layout.get<ShapeType>();
        cldnn::format out_format = input_layout.format;

        // We create dummy Add op as shape infer is exactly the same for any eltwise op type, so there is no need to have correct op type
        ov::op::v1::Add op;
        op.set_autob(desc->broadcast_spec);

        std::vector<ShapeType> output_shapes = {ShapeType()};
        std::vector<ShapeType> input_shapes;
        for (size_t i = 0; i < desc->input_size(); i++) {
            input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
        }
        eltwise_shape_infer(&op, input_shapes, output_shapes);

        if (input_layout.format == format::b_fs_zyx_fsv16)  // use optimized 5D
            out_format = format::b_fs_zyx_fsv16;
        else if (input_layout.format == format::bs_fs_zyx_bsv16_fsv16)
            out_format = format::bs_fs_zyx_bsv16_fsv16;

        for (size_t i = 0; i < desc->input_size(); i++) {
            if (impl_param.primary_input_idx == i)
                continue;

            auto l = impl_param.get_non_padded_input_layout(i);
            if (l.format == format::b_fs_zyx_fsv16)  // use optimized 5D
                out_format = format::b_fs_zyx_fsv16;
            else if (l.format == format::bs_fs_zyx_bsv16_fsv16)
                out_format = format::bs_fs_zyx_bsv16_fsv16;
        }

        return layout(output_shapes[0], out_data_type, out_format);
    };

    auto output_layout = get_output_layout();
    auto mode = desc->mode;
    // list of operations supported for integer types
    if (input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8 ||
        input_layout.data_type == data_types::i32 || input_layout.data_type == data_types::i64) {
        std::vector<eltwise_mode> eltwise_int_modes = {eltwise_mode::sum,
                                                       eltwise_mode::sub,
                                                       eltwise_mode::prod,
                                                       eltwise_mode::div,
                                                       eltwise_mode::min,
                                                       eltwise_mode::max,
                                                       eltwise_mode::mod,
                                                       eltwise_mode::eq,
                                                       eltwise_mode::ne,
                                                       eltwise_mode::lt,
                                                       eltwise_mode::le,
                                                       eltwise_mode::gt,
                                                       eltwise_mode::ge,
                                                       eltwise_mode::squared_diff,
                                                       eltwise_mode::floor_mod,
                                                       eltwise_mode::logic_and,
                                                       eltwise_mode::logic_or,
                                                       eltwise_mode::logic_xor};

        OPENVINO_ASSERT((std::find(eltwise_int_modes.begin(), eltwise_int_modes.end(), mode) != eltwise_int_modes.end()),
                            desc->id + "Requested eltwise mode is not supported for integer types.");
    }

    // Logic and comparison operations should return i8 for any inputs
    std::vector<eltwise_mode> eltwise_bool_modes = {eltwise_mode::eq,
                                                    eltwise_mode::ne,
                                                    eltwise_mode::lt,
                                                    eltwise_mode::le,
                                                    eltwise_mode::gt,
                                                    eltwise_mode::ge,
                                                    eltwise_mode::logic_and,
                                                    eltwise_mode::logic_or,
                                                    eltwise_mode::logic_xor,
                                                    eltwise_mode::is_finite,
                                                    eltwise_mode::is_inf,
                                                    eltwise_mode::is_nan};
    if (std::find(eltwise_bool_modes.begin(), eltwise_bool_modes.end(), mode) != eltwise_bool_modes.end()) {
        output_layout.data_type = data_types::i8;
    }

    output_layout.data_type = desc->output_data_types[0].value_or(output_layout.data_type);

    if (impl_param.has_fused_primitives()) {
        output_layout.data_type = impl_param.get_fused_output_layout().data_type;
    }

    if (!desc->stride.empty()) {
        auto input_pshape = input_layout.get<ShapeType>();
        if (input_pshape.is_static()) {
            // we can safely use only first stride, since we're using first input, and input / stride should give exact same
            // value for every input
            auto in_shape = input_pshape.get_shape();
            for (size_t i = 0; i < desc->stride[0].spatial.size(); i++) {
                const size_t idx = in_shape.size() - 1 - i;
                if (idx < 0)
                    break;
                in_shape[idx] = (in_shape[idx] - 1) / desc->stride[0].spatial[i] + 1;
            }
            input_layout.set_partial_shape({in_shape});
        }
        return { input_layout };
    }
    return { output_layout };
}

template std::vector<layout> eltwise_inst::calc_output_layouts<ov::PartialShape>(eltwise_node const& node, const kernel_impl_params& impl_param);

static inline std::string stringify_vector(const std::vector<float>& v) {
    std::stringstream s;

    s << "{ ";

    for (size_t i = 0; i < v.size(); ++i) {
        s << v.at(i);
        if (i + 1 < v.size())
            s << ", ";
    }

    s << " }";

    return s.str();
}

std::string eltwise_inst::to_string(eltwise_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;
    std::string str_mode;

    switch (desc->mode) {
        case eltwise_mode::sum:
            str_mode = "sum";
            break;
        case eltwise_mode::sub:
            str_mode = "subtract";
            break;
        case eltwise_mode::max:
            str_mode = "max";
            break;
        case eltwise_mode::prod:
            str_mode = "product";
            break;
        case eltwise_mode::div:
            str_mode = "div";
            break;
        case eltwise_mode::min:
            str_mode = "min";
            break;
        case eltwise_mode::pow:
            str_mode = "pow";
            break;
        case eltwise_mode::squared_diff:
            str_mode = "squared_diff";
            break;
        case eltwise_mode::mod:
            str_mode = "mod";
            break;
        case eltwise_mode::eq:
            str_mode = "equal";
            break;
        case eltwise_mode::ne:
            str_mode = "not equal";
            break;
        case eltwise_mode::lt:
            str_mode = "less";
            break;
        case eltwise_mode::le:
            str_mode = "less-or-equal";
            break;
        case eltwise_mode::gt:
            str_mode = "greater";
            break;
        case eltwise_mode::ge:
            str_mode = "greater-or-equal";
            break;
        case eltwise_mode::logic_and:
            str_mode = "and";
            break;
        case eltwise_mode::logic_or:
            str_mode = "or";
            break;
        case eltwise_mode::logic_xor:
            str_mode = "xor";
            break;
        case eltwise_mode::floor_mod:
            str_mode = "floor_mod";
            break;
        case eltwise_mode::is_finite:
            str_mode = "is_finite";
            break;
        case eltwise_mode::is_inf:
            str_mode = "is_inf";
            break;
        case eltwise_mode::is_nan:
            str_mode = "is_nan";
            break;
        default:
            str_mode = "not supported mode";
            break;
    }

    json_composite eltwise_info;
    for (size_t i = 0; i < node.inputs_count(); i++) {
        eltwise_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    eltwise_info.add("mode", str_mode);
    if (!desc->coefficients.empty()) {
        eltwise_info.add("coefficients", stringify_vector(desc->coefficients));
    }
    node_info->add("eltwise info", eltwise_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

eltwise_inst::typed_primitive_inst(network& network, eltwise_node const& node) : parent(network, node) {
    check_inputs_count(node);
    // check for stride
    auto prim = node.get_primitive();
    auto inputs_count = node.inputs_count();

    if (is_dynamic())
        return;

    if (!prim->stride.empty()) {
        // number of strides must match number of inputs
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Eltwise inputs count",
                              inputs_count,
                              "Eltwise strides count",
                              prim->stride.size(),
                              "");

        const auto out_x = node.get_output_layout().spatial(0);
        const auto out_y = node.get_output_layout().spatial(1);
        // check if strides are correctly set. I.e INPUT_SIZE_X / STRIDE_X = OUTPUT_SIZE_X, same for Y dimension
        for (size_t i = 0; i < inputs_count; i++) {
            const auto& in_layout = node.input(i).get_output_layout();
            auto stride = prim->stride[i];

            const auto in_x_div_stride_x = (in_layout.spatial(0) - 1) / stride.spatial[0] + 1;
            if (in_x_div_stride_x != out_x && in_x_div_stride_x != 1)
                CLDNN_ERROR_NOT_EQUAL(node.id(),
                                      "Eltwise input_x / stride_x",
                                      in_x_div_stride_x,
                                      "Eltwise output_x",
                                      out_x,
                                      "");

            const auto in_y_div_stride_y = (in_layout.spatial(1) - 1) / stride.spatial[1] + 1;
            if (in_y_div_stride_y != out_y && in_y_div_stride_y != 1)
                CLDNN_ERROR_NOT_EQUAL(node.id(),
                                      "Eltwise inputyx / stride_y",
                                      in_y_div_stride_y,
                                      "Eltwise output_y",
                                      out_y,
                                      "");
        }
    } else {
        std::vector<int32_t> input0_size = node.input().get_output_layout().get_tensor().raw.vector();
        for (size_t i = 1; i < inputs_count; i++) {
            std::vector<int32_t> input_size = node.input(i).get_output_layout().get_tensor().raw.vector();
            for (size_t d = 0; d < input0_size.size(); d++) {
                bool sizes_equal = input0_size[d] == input_size[d];
                bool broadcast =
                    (input0_size[d] == 1 || input_size[d] == 1) && (input0_size[d] != 1 || input_size[d] != 1);
                CLDNN_ERROR_BOOL(node.id(),
                                 "Sizes equal or broadcast is possible",
                                 !(sizes_equal || broadcast),
                                 "Invalid input shapes");
            }
        }
    }
}

void eltwise_inst::check_inputs_count(eltwise_node const& node) {
    const size_t inputs_number = node.get_primitive()->input.size();
    const eltwise_mode mode = node.get_primitive()->mode;

    switch (mode) {
        case eltwise_mode::sum:
        case eltwise_mode::sub:
        case eltwise_mode::div:
        case eltwise_mode::prod:
        case eltwise_mode::max:
        case eltwise_mode::min:
        case eltwise_mode::mod:
        case eltwise_mode::logic_and:
        case eltwise_mode::logic_or:
        case eltwise_mode::logic_xor:
            OPENVINO_ASSERT(inputs_number >= 2,
                            "Node id: ", node.id(), ". Invalid eltwise inputs number (should be equal at least to 2). Actual: ", inputs_number);
            break;
        case eltwise_mode::eq:
        case eltwise_mode::ne:
        case eltwise_mode::lt:
        case eltwise_mode::le:
        case eltwise_mode::gt:
        case eltwise_mode::ge:
        case eltwise_mode::squared_diff:
        case eltwise_mode::pow:
        case eltwise_mode::floor_mod:
            OPENVINO_ASSERT(inputs_number == 2,
                            "Node id: ", node.id(), ". Invalid eltwise inputs number (should be equal to 2). Actual: ", inputs_number);
            break;
        case eltwise_mode::is_finite:
        case eltwise_mode::is_inf:
        case eltwise_mode::is_nan:
            OPENVINO_ASSERT(inputs_number == 1,
                            "Node id: ", node.id(), ". Invalid eltwise inputs number (should be equal to 1). Actual: ", inputs_number);
            break;
    }
}
}  // namespace cldnn
