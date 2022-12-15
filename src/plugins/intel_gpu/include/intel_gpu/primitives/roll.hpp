// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Roll-7 primitive.
struct roll : primitive_base<roll> {
    CLDNN_DECLARE_PRIMITIVE(roll)

    /// @brief Constructs roll primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param shift Tensor which specifies the number of places by which the elements are shifted.
    roll(const primitive_id& id,
         const input_info& input,
         const tensor& shift,
         const padding& output_padding = {})
        : primitive_base(id, {input}, {output_padding}),
          shift(shift) {}

    /// @brief Tensor which specifies the number of places by which the elements are shifted.
    tensor shift;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
