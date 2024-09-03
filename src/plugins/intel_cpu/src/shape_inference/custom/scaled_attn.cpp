// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.hpp"

#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "openvino/op/mha.hpp"
#include "utils.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class SDPAShapeInfer : public ShapeInferEmptyPads {
public:
    SDPAShapeInfer(const ScaledDotProductAttentionWithKVCache::Config& config) : m_config(config) {}

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& query_dims = input_shapes.front().get();
        VectorDims present_kv_dims = input_shapes.back().get();
        const auto& beam_idx_dims = input_shapes.end()[-3].get();
        const auto& permute_axes = m_config.permute_axes;

        if (permute_axes.empty()) {
            // [B, H, L, S]
            present_kv_dims[0] = beam_idx_dims[0];
            present_kv_dims[2] += query_dims[2];
            return {{query_dims, present_kv_dims, present_kv_dims}, ShapeInferStatus::success};
        }

        // permute_axes[0,1,2,3] gives axis indices of B,H,L,S for query & present_kv
        const size_t batch_index = permute_axes[0];
        const size_t length_index = permute_axes[2];
        present_kv_dims[batch_index] = beam_idx_dims[0];
        present_kv_dims[length_index] += query_dims[length_index];

        auto n_dims = query_dims.size();
        VectorDims output_dims(n_dims);
        for (size_t i = 0; i < n_dims; i++) {
            output_dims[i] = query_dims[permute_axes[i]];
        }
        return {{output_dims, present_kv_dims, present_kv_dims}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    ScaledDotProductAttentionWithKVCache::Config m_config;
};

class MultiHeadAttentionShapeInfer : public ShapeInferEmptyPads {
public:
    MultiHeadAttentionShapeInfer(size_t hidden_size) : m_hidden_size(hidden_size) {}

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        auto query_dims = input_shapes[5].get();
        query_dims.back() = m_hidden_size;

        return {{query_dims}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
private:
    size_t m_hidden_size;
};

ShapeInferPtr SDPAShapeInferFactory::makeShapeInfer() const {
    if (auto mha = std::dynamic_pointer_cast<const op::v15::MultiHeadAttention>(m_op)) {
        const auto& config = mha->get_config();
        return std::make_shared<MultiHeadAttentionShapeInfer>(static_cast<size_t>(config.n_hidden));
    } else if (auto sdpa = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(m_op)) {
        const auto& config = sdpa->get_config();
        if (config.output_BLHxS == false)
            return std::make_shared<SDPAShapeInfer>(config);
    }
    // fallback to ngraph shape infer on non-perf-critical case
    return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), EMPTY_PORT_MASK);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
