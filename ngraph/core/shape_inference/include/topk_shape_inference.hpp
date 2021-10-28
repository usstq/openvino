// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/validation_util.hpp>
#include <openvino/op/topk.hpp>

namespace ov {
namespace op {
namespace v1 {

template <typename DimType>
void set_dim(Op* op, DimType& dim, ov::Dimension d) {
    NODE_VALIDATION_CHECK(op, false, "Cannot set Dimension to in-compatible type.");
}

template <>
void set_dim<ov::Dimension>(Op* op, ov::Dimension& dim, ov::Dimension d) {
    dim = d;
}

template <typename T>
void shape_infer(TopK* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 && output_shapes.size() == 2));
    const auto& input_shape = input_shapes[0];
    const auto input_rank = input_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          input_rank.is_dynamic() || input_rank.get_length() > 0,
                          "Input rank must be greater than 0.");

    const auto& k_shape = input_shapes[1];
    NODE_VALIDATION_CHECK(op, k_shape.rank().compatible(0), "The 'K' input must be a scalar.");

    auto output_shape = input_shape;
    if (input_shape.rank().is_static()) {
        ov::PartialShape k_as_shape;
        op->m_normalized_axis = ov::normalize_axis(op, op->m_axis, input_shape.rank());
        auto& dim_axis = output_shape[op->m_normalized_axis];

        if (ov::evaluate_as_partial_shape(op->input_value(1), k_as_shape)) {
            NODE_VALIDATION_CHECK(op,
                                  k_as_shape.size() == 1,
                                  "Only one value (scalar) should be provided as the 'K' input to TopK",
                                  " (got ",
                                  k_as_shape.size(),
                                  " elements).");

            NODE_VALIDATION_CHECK(op,
                                  k_as_shape[0].get_max_length() > 0,
                                  "The value of 'K' must be a positive number.",
                                  " (got ",
                                  k_as_shape[0].get_max_length(),
                                  ").");
            if (k_as_shape.is_static())
                dim_axis = k_as_shape[0].get_length();
            else if (k_as_shape[0].get_interval().size() == 1) {
                // k_as_shape[0] is a static dimension in dynamic-form
                dim_axis = k_as_shape[0].get_min_length();
            } else {
                // in this dynamic branch we are sure of dim_axis's type
                const auto in_min = dim_axis.get_min_length();
                const auto in_max = dim_axis.get_max_length();

                const auto k_min = k_as_shape[0].get_min_length();
                const auto k_max = k_as_shape[0].get_max_length();

                const auto lower = std::min<Dimension::value_type>(in_min, k_min);
                const auto upper =
                    in_max < 0 ? Dimension::dynamic().get_max_length() : std::max<Dimension::value_type>(in_max, k_max);
                set_dim(op, dim_axis, Dimension(lower, upper));
            }
        } else {
            set_dim(op, dim_axis, Dimension(0, dim_axis.get_max_length()));
        }
    }

    output_shapes[0] = output_shape;
    output_shapes[1] = output_shape;
}

}  // namespace v1
}  // namespace op
}  // namespace ov
