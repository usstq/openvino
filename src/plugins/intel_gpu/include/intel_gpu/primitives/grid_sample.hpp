// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/op/grid_sample.hpp"
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

using GridSampleOp = ov::op::v9::GridSample;

/// @brief GridSample-9 primitive.
struct grid_sample : primitive_base<grid_sample> {
    CLDNN_DECLARE_PRIMITIVE(grid_sample)

    /// @brief Constructs grid_sample primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param attributes Structure which contains all GridSample attributes.
    grid_sample(const primitive_id& id,
                const std::vector<input_info>& inputs,
                const GridSampleOp::Attributes& attributes,
                const padding& output_padding = {})
        : primitive_base(id, inputs, {output_padding}),
          attributes(attributes) {}

    GridSampleOp::Attributes attributes;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
