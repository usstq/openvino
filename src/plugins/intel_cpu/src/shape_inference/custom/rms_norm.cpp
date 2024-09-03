// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_norm.hpp"
#include "transformations/cpu_opset/common/op/add_rms.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class RMSNormShapeInfer : public ShapeInferEmptyPads {
public:
    RMSNormShapeInfer(bool has_add) : m_has_add(has_add) {}

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& dims = input_shapes.front().get();
        if (m_has_add)
            return {{dims, dims}, ShapeInferStatus::success};
        return {{dims}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
private:
    bool m_has_add;
};

ShapeInferPtr RMSNormShapeInferFactory::makeShapeInfer() const {
    bool has_add = false;
    if (std::dynamic_pointer_cast<const AddRMSNode>(m_op))
        has_add = true;
    return std::make_shared<RMSNormShapeInfer>(has_add);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
