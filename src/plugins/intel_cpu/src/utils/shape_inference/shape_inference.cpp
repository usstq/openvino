// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/opsets/opset11.hpp>
#include <openvino/opsets/opset12.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset7.hpp>

#include "adaptive_avg_pool_shape_inference.hpp"
#include "adaptive_max_pool_shape_inference.hpp"
#include "assign_shape_inference.hpp"
#include "augru_cell_shape_inference.hpp"
#include "augru_sequence_shape_inference.hpp"
#include "avg_pool_shape_inference.hpp"
#include "batch_to_space_shape_inference.hpp"
#include "binary_convolution_shape_inference.hpp"
#include "broadcast_shape_inference.hpp"
#include "bucketize_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "convolution_backprop_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "ctc_greedy_decoder_seq_len_shape_inference.hpp"
#include "ctc_greedy_decoder_shape_inference.hpp"
#include "ctc_loss_shape_inference.hpp"
#include "deformable_convolution_shape_inference.hpp"
#include "deformable_psroi_pooling_shape_inference.hpp"
#include "depth_to_space_shape_inference.hpp"
#include "detection_output_shape_inference.hpp"
#include "einsum_shape_inference.hpp"
#include "embedding_segments_sum_shape_inference.hpp"
#include "embeddingbag_offsets_shape_inference.hpp"
#include "embeddingbag_packed_shape_inference.hpp"
#include "experimental_detectron_detection_output_shape_inference.hpp"
#include "experimental_detectron_generate_proposals_shape_inference.hpp"
#include "experimental_detectron_prior_grid_generator_shape_inference.hpp"
#include "experimental_detectron_roi_feature_shape_inference.hpp"
#include "experimental_detectron_topkrois_shape_inference.hpp"
#include "extract_image_patches_shape_inference.hpp"
#include "eye_shape_inference.hpp"
#include "fake_quantize.hpp"
#include "fft_base_shape_inference.hpp"
#include "gather_elements_shape_inference.hpp"
#include "gather_nd_shape_inference.hpp"
#include "gather_shape_inference.hpp"
#include "gather_tree_shape_inference.hpp"
#include "grid_sample_shape_inference.hpp"
#include "group_convolution_backprop_shape_inference.hpp"
#include "group_convolution_shape_inference.hpp"
#include "gru_cell_shape_inference.hpp"
#include "gru_sequence_shape_inference.hpp"
#include "interpolate_shape_inference.hpp"
#include "irdft_shape_inference.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "lstm_sequence_shape_inference.hpp"
#include "matmul_shape_inference.hpp"
#include "max_pool_shape_inference.hpp"
#include "one_hot_shape_inference.hpp"
#include "pad_shape_inference.hpp"
#include "prior_box_clustered_shape_inference.hpp"
#include "prior_box_shape_inference.hpp"
#include "proposal_shape_inference.hpp"
#include "psroi_pooling_shape_inference.hpp"
#include "range_shape_inference.hpp"
#include "rdft_shape_inference.hpp"
#include "read_value_shape_inference.hpp"
#include "reduce_shape_inference.hpp"
#include "region_yolo_shape_inference.hpp"
#include "reorg_yolo_shape_inference.hpp"
#include "reverse_sequence_shape_inference.hpp"
#include "reverse_shape_inference.hpp"
#include "rnn_cell_shape_inference.hpp"
#include "rnn_sequence_shape_inference.hpp"
#include "roi_align_shape_inference.hpp"
#include "roi_pooling_shape_inference.hpp"
#include "roll_shape_inference.hpp"
#include "scatter_elements_update_shape_inference.hpp"
#include "scatter_nd_base_shape_inference.hpp"
#include "select_shape_inference.hpp"
#include "shape_inference.hpp"
#include "shape_nodes.hpp"
#include "shuffle_channels_shape_inference.hpp"
#include "slice_shape_inference.hpp"
#include "space_to_batch_shape_inference.hpp"
#include "space_to_depth_shape_inference.hpp"
#include "split_shape_inference.hpp"
#include "squeeze_shape_inference.hpp"
#include "static_shape.hpp"
#include "strided_slice_shape_inference.hpp"
#include "tile_shape_inference.hpp"
#include "topk_shape_inference.hpp"
#include "transpose_shape_inference.hpp"
#include "unsqueeze_shape_inference.hpp"
#include "utils.hpp"
#include "utils/bit_util.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov {
namespace intel_cpu {

class entryBase : public IShapeInferCommon {
public:
    using iface_type = IShapeInferCommon;

    entryBase(std::shared_ptr<ov::Node> node) : node{node} {
        for (size_t i = 0; i < node->get_input_size(); i++) {
            const auto& shape = node->get_input_partial_shape(i);
            if (shape.rank().is_static()) {
                input_ranks.push_back(shape.rank().get_length());
            } else {
                input_ranks.push_back(-1);
            }
        }
    }

    const ov::CoordinateDiff& get_pads_begin() override {
        OPENVINO_ASSERT(false, "entryBase do not support get_pads_begin() by default.");
    }

    const ov::CoordinateDiff& get_pads_end() override {
        OPENVINO_ASSERT(false, "entryBase do not support get_pads_end() by default.");
    }

    const std::vector<int64_t>& get_input_ranks() override {
        return input_ranks;
    }

protected:
    std::vector<int64_t> input_ranks;
    std::shared_ptr<ov::Node> node;
};

template <typename OP>
class entryIO : public entryBase {
public:
    using entryBase::entryBase;

    IShapeInferCommon::Result
    infer(const std::vector<StaticShape>& input_shapes, const std::map<size_t, HostTensorPtr>& constant_data) override {
        std::vector<StaticShape> output_shapes(node->get_output_size());
        shape_infer(static_cast<OP*>(node.get()), input_shapes, output_shapes);
        return {std::move(output_shapes), ShapeInferStatus::success};
    }
};

template <typename OP>
class entryIOC : public entryBase {
public:
    using entryBase::entryBase;

    IShapeInferCommon::Result
    infer(const std::vector<StaticShape>& input_shapes, const std::map<size_t, HostTensorPtr>& constant_data) override {
        auto op = static_cast<OP*>(node.get());
        std::vector<StaticShape> output_shapes(op->get_output_size());
        shape_infer(op, input_shapes, output_shapes, constant_data);
        return {std::move(output_shapes), ShapeInferStatus::success};
    }
};

class entryCopy : public entryBase {
public:
    using entryBase::entryBase;

    IShapeInferCommon::Result
    infer(const std::vector<StaticShape>& input_shapes, const std::map<size_t, HostTensorPtr>& constant_data) override {
        auto op = node.get();
        std::vector<StaticShape> output_shapes(op->get_output_size());
        copy_shape_infer(op, input_shapes, output_shapes);
        return {std::move(output_shapes), ShapeInferStatus::success};
    }
};

class entryFirstPassthrough : public entryBase {
public:
    using entryBase::entryBase;

    IShapeInferCommon::Result
    infer(const std::vector<StaticShape>& input_shapes, const std::map<size_t, HostTensorPtr>& constant_data) override {
        auto op = node.get();
        std::vector<StaticShape> output_shapes(op->get_output_size());
        first_input_passthrough_infer(op, input_shapes, output_shapes);
        return {std::move(output_shapes), ShapeInferStatus::success};
    }
};

class entryEltwise : public entryBase {
public:
    using entryBase::entryBase;

    IShapeInferCommon::Result
    infer(const std::vector<StaticShape>& input_shapes, const std::map<size_t, HostTensorPtr>& constant_data) override {
        auto op = node.get();
        std::vector<StaticShape> output_shapes(op->get_output_size());
        eltwise_shape_infer(op, input_shapes, output_shapes);
        return {std::move(output_shapes), ShapeInferStatus::success};
    }
};

class entryFallback : public entryBase {
public:
    std::shared_ptr<ov::Node> local_op_default;

    entryFallback(std::shared_ptr<ov::Node> node) : entryBase(node) {
        ngraph::OutputVector new_inputs;
        auto op = node.get();
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            if (dynamic_cast<ov::opset1::Constant*>(op->get_input_node_ptr(i))) {
                new_inputs.push_back(op->get_input_node_ptr(i)->clone_with_new_inputs(ov::OutputVector{}));
            } else {
                new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                             op->get_input_partial_shape(i)));
            }
        }

        local_op_default = op->clone_with_new_inputs(new_inputs);
    }

    virtual void post_validate_and_infer_types(const std::shared_ptr<ov::Node>& local_op) {}

    IShapeInferCommon::Result
    infer(const std::vector<StaticShape>& input_shapes, const std::map<size_t, HostTensorPtr>& constant_data) override {
        auto op = node.get();
        std::vector<StaticShape> output_shapes;

        std::shared_ptr<ov::Node> local_op;
        if (!constant_data.empty()) {
            ngraph::OutputVector new_inputs;
            for (size_t i = 0; i < op->get_input_size(); ++i) {
                if (constant_data.count(i)) {
                    new_inputs.push_back(std::make_shared<ov::opset1::Constant>(constant_data.at(i)));
                } else if (dynamic_cast<ov::opset1::Constant*>(op->get_input_node_ptr(i))) {
                    new_inputs.push_back(op->get_input_node_ptr(i)->clone_with_new_inputs(ov::OutputVector{}));
                } else {
                    new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                                 input_shapes[i].to_partial_shape()));
                }
            }
            local_op = op->clone_with_new_inputs(new_inputs);
        } else {
            local_op = local_op_default;
            for (size_t i = 0; i < local_op->get_input_size(); i++) {
                if (auto parameter = dynamic_cast<ov::opset1::Parameter*>(local_op->get_input_node_ptr(i))) {
                    parameter->set_partial_shape(input_shapes[i].to_partial_shape());
                    parameter->validate_and_infer_types();
                }
            }
        }

        local_op->validate_and_infer_types();

        output_shapes.resize(local_op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);

            if (partial_shape.is_dynamic()) {
                return {{}, ShapeInferStatus::skip};
            }

            output_shapes[i] = StaticShape(partial_shape.to_shape());
        }

        post_validate_and_infer_types(local_op);

        return {std::move(output_shapes), ShapeInferStatus::success};
    }
};

template <class TOp>
class ShapeInferWithPadding : public entryBase {
public:
    ShapeInferWithPadding(std::shared_ptr<Node> node) : entryBase{std::move(node)}, m_pads_begin{}, m_pads_end{} {}

    IShapeInferCommon::Result infer(const std::vector<StaticShape>& input_shapes,
                                    const std::map<size_t, ov::HostTensorPtr>& constant_data) override {
        auto op = static_cast<TOp*>(node.get());
        auto out_shapes = shape_infer(op, input_shapes, m_pads_begin, m_pads_end, constant_data);
        return {std::move(out_shapes), ShapeInferStatus::success};
    }

    const ov::CoordinateDiff& get_pads_begin() override {
        return m_pads_begin;
    }

    const ov::CoordinateDiff& get_pads_end() override {
        return m_pads_end;
    }

protected:
    ov::CoordinateDiff m_pads_begin, m_pads_end;
};

/**
 * @brief Base shape inference object implementing the IStaticShapeInfer without padding support.
 */
class ShapeInferBase : public IStaticShapeInfer {
public:
    using iface_type = IStaticShapeInfer;
    virtual ~ShapeInferBase() = default;

    ShapeInferBase(std::shared_ptr<Node> node) : m_input_ranks{}, m_node{node} {
        static_assert(std::is_same<int64_t, Dimension::value_type>::value, "Rank type not match to input_ranks type.");
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            const auto& shape = node->get_input_partial_shape(i);
            const auto& rank_length = shape.rank().is_static() ? shape.rank().get_length() : -1;
            m_input_ranks.push_back(rank_length);
        }
    }

    IShapeInferCommon::Result
    infer(const std::vector<StaticShape>& input_shapes, const std::map<size_t, HostTensorPtr>& constant_data) override {
        // For backward compatibility, create ov tensors and run shape inference.
        return infer(input_shapes, make_tensor_accessor(constant_data));
    }

    IShapeInferCommon::Result infer(const std::vector<StaticShape>& input_shapes, const ov::ITensorAccessor&) override {
        OPENVINO_THROW("Not implemented by base class");
    }

    ov::optional<std::vector<StaticShapeCon>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                    const ov::ITensorAccessor& tensor_accessor) override {
        OPENVINO_THROW("Not implemented by base class");
    }

    const ov::CoordinateDiff& get_pads_begin() override {
        OPENVINO_ASSERT(false, "ShapeInferBase do not support get_pads_begin() by default.");
    }

    const ov::CoordinateDiff& get_pads_end() override {
        OPENVINO_ASSERT(false, "ShapeInferBase do not support get_pads_end() by default.");
    }

    const std::vector<int64_t>& get_input_ranks() override {
        return m_input_ranks;
    }

    port_mask_t get_port_mask() const override {
        return 0;
    }

protected:
    std::vector<int64_t> m_input_ranks;
    std::shared_ptr<ov::Node> m_node;
};

/**
 * @brief Shape inference using tensor accessor to get constant data.
 *
 * @tparam TOp   Type of operator.
 * @tparam MASK  The bit mask where each bit corresponds to an input port number.
 */
template <class TOp, uint32_t MASK>
class ShapeInferTA : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    IShapeInferCommon::Result infer(const std::vector<StaticShape>& input_shapes,
                                    const ov::ITensorAccessor& tensor_accessor) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes, tensor_accessor), ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return MASK;
    }
};

template <class TOp, uint32_t MASK>
class ShapeInferenceTA : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    IShapeInferCommon::Result infer(const std::vector<StaticShape>& input_shapes,
                                    const ov::ITensorAccessor& tensor_accessor) override {
        // Temporary support of StaticShape.
        auto in_shapes = std::vector<StaticShapeRef>();
        in_shapes.reserve(input_shapes.size());
        for (auto& s : input_shapes) {
            in_shapes.emplace_back(reinterpret_cast<const VectorDims&>(s));
        }

        auto out_shapes = infer(in_shapes, tensor_accessor);
        Result result{{}, out_shapes ? ShapeInferStatus::success : ShapeInferStatus::skip};

        if (out_shapes) {
            result.shapes.reserve(out_shapes->size());
            std::transform(out_shapes->begin(),
                           out_shapes->end(),
                           std::back_inserter(result.shapes),
                           [](StaticShapeCon& s) {
                               return std::move(reinterpret_cast<StaticShape&&>(*s));
                           });
        }

        return result;
    }

    ov::optional<std::vector<StaticShapeCon>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                    const ov::ITensorAccessor& tensor_accessor) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes, tensor_accessor)};
    }

    port_mask_t get_port_mask() const override {
        return MASK;
    }
};

/**
 * @brief Shape inference not using tensor accessor
 *
 * The MASK is 0 there is no dependant inputs with data for shape inference.
 *
 * @tparam TOp  Type of operator.
 */
template <class TOp>
class ShapeInferTA<TOp, 0> : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    IShapeInferCommon::Result infer(const std::vector<StaticShape>& input_shapes, const ov::ITensorAccessor&) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes), ShapeInferStatus::success};
    }
};

/** @brief Base shape inference object implementing the IStaticShapeInfer with padding support. */
class ShapeInferPaddingBase : public ShapeInferBase {
public:
    ShapeInferPaddingBase(std::shared_ptr<Node> node) : ShapeInferBase(std::move(node)), m_pads_begin{}, m_pads_end{} {}

    IShapeInferCommon::Result infer(const std::vector<StaticShape>& input_shapes,
                                    const ITensorAccessor& tensor_accessor) override {
        OPENVINO_THROW("Not implemented by base class");
    }

    const ov::CoordinateDiff& get_pads_begin() override {
        return m_pads_begin;
    }

    const ov::CoordinateDiff& get_pads_end() override {
        return m_pads_end;
    }

protected:
    ov::CoordinateDiff m_pads_begin, m_pads_end;
};

/**
 * @brief Shape inference using tensor accessor to get constant data and padding
 *
 * @tparam TOp   Type of operator.
 * @tparam MASK  The bit mask where each bit corresponds to an input port number.
 */
template <class TOp, uint32_t MASK>
class ShapeInferPaddingTA : public ShapeInferPaddingBase {
public:
    using ShapeInferPaddingBase::ShapeInferPaddingBase;

    IShapeInferCommon::Result infer(const std::vector<StaticShape>& input_shapes,
                                    const ov::ITensorAccessor& tensor_accessor) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes, m_pads_begin, m_pads_end, tensor_accessor),
                ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return MASK;
    }
};

/**
 * \brief Shape infer factory
 *
 * \tparam R     Result type of created interface object.
 * \tparam TKey  Type of Maker map key.
 * \tparam Args  TypesInference object ctor args.
 */
template <class TKey, class R, class... Args>
class ShapeInferFactory {
public:
    // Helper type to define specific Makers map values.
    using TValue = std::function<R(Args...)>;

    // Helper type to define specific Makers map type.
    using TRegistry = std::unordered_map<TKey, TValue>;

    /**
     * \brief  Creates the shape inference object.
     *
     * \param key   Key value to get specified shape inference object maker.
     * \param args  Inference object args.
     *
     * \return The shape inference object or R{} if not found in the map.
     */
    static R make(const TKey& key, Args... args) {
        const auto& maker_iter = registry.find(key);
        if (maker_iter != registry.end()) {
            return maker_iter->second(std::forward<Args>(args)...);
        } else {
            return {};
        }
    }

private:
    /** \brief Factory makers registry which can be specialized for key and value. */
    static const TRegistry registry;
};

// Helpers to make shape inference objects (primary template).
template <template <class> class TShapeInfer, class TOp, class... Args>
std::shared_ptr<typename TShapeInfer<TOp>::iface_type> make_infer(Args&&... args) {
    return std::make_shared<TShapeInfer<TOp>>(std::forward<Args>(args)...);
}

template <template <class, IStaticShapeInfer::port_mask_t> class TShapeInfer,
          class TOp,
          IStaticShapeInfer::port_mask_t mask>
std::shared_ptr<typename TShapeInfer<TOp, mask>::iface_type> make_shape_infer(std::shared_ptr<ov::Node> node) {
    return std::make_shared<TShapeInfer<TOp, mask>>(std::move(node));
}

template <template <class> class TShapeInfer, class TOp>
std::shared_ptr<typename TShapeInfer<TOp>::iface_type> make_shape_infer(std::shared_ptr<ov::Node> node) {
    return make_infer<TShapeInfer, TOp>(std::move(node));
}

template <class TShapeInfer>
std::shared_ptr<typename TShapeInfer::iface_type> make_shape_infer(std::shared_ptr<ov::Node> node) {
    return std::make_shared<TShapeInfer>(std::move(node));
}

template <template <class, bool> class TConvInfer, class TOp, bool flag>
std::shared_ptr<typename TConvInfer<TOp, flag>::iface_type> make_shape_infer(std::shared_ptr<ov::Node> node) {
    return std::make_shared<TConvInfer<TOp, flag>>(std::move(node));
}

// Type of key in shape inference Makers maps.
using ShapeInferKey = ov::NodeTypeInfo;

// Default opset used for 'default' in inference map.
using namespace ov::opset10;

// Helper macros to make map entries
#define _OV_OP_SHAPE_INFER_VA_REG(OP, ...) \
    { OP::get_type_info_static(), make_shape_infer<__VA_ARGS__> }
#define _OV_OP_SHAPE_INFER_REG(OP, SHAPE_INFER)              _OV_OP_SHAPE_INFER_VA_REG(OP, SHAPE_INFER, OP)
#define _OV_OP_SHAPE_INFER_MASK_REG(OP, SHAPE_INFER, MASK)   _OV_OP_SHAPE_INFER_VA_REG(OP, SHAPE_INFER, OP, MASK)
#define _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(OP, SHAPE_INFER) _OV_OP_SHAPE_INFER_VA_REG(OP, SHAPE_INFER)

// Helper types for IShapeInferCommon makers map.
using IShapeInferCommonFactory =
    ShapeInferFactory<ShapeInferKey, std::shared_ptr<IShapeInferCommon>, std::shared_ptr<ov::Node>>;

// Initialization map for operators supporting IShapeInferCommon objects.
// First group in map is 'default' opset defined by alias above.
// To use other version of operators, explicitly specify operator with opset version namespace.
// const IShapeInferCommonMapType IShapeInferCommonFactory::Makers::map{};
template <>
const IShapeInferCommonFactory::TRegistry IShapeInferCommonFactory::registry{
    // Default opset
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(BatchNormInference, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(Convert, entryCopy),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(CumSum, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(HardSigmoid, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(LogicalNot, entryCopy),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(LRN, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(MVN, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(NormalizeL2, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(PRelu, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(ScatterUpdate, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(Selu, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(Softmax, entryCopy),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(Swish, entryFirstPassthrough),
    _OV_OP_SHAPE_INFER_REG(AdaptiveAvgPool, entryIOC),
    _OV_OP_SHAPE_INFER_REG(AdaptiveMaxPool, entryIOC),
    _OV_OP_SHAPE_INFER_REG(Assign, entryIO),
    _OV_OP_SHAPE_INFER_REG(AvgPool, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(BatchToSpace, entryIOC),
    _OV_OP_SHAPE_INFER_REG(BinaryConvolution, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(Broadcast, entryIOC),
    _OV_OP_SHAPE_INFER_REG(Bucketize, entryIO),
    _OV_OP_SHAPE_INFER_REG(Concat, entryIO),
    _OV_OP_SHAPE_INFER_REG(Convolution, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(ConvolutionBackpropData, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(CTCGreedyDecoder, entryIO),
    _OV_OP_SHAPE_INFER_REG(CTCGreedyDecoderSeqLen, entryIO),
    _OV_OP_SHAPE_INFER_REG(CTCLoss, entryIO),
    _OV_OP_SHAPE_INFER_REG(DeformableConvolution, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(DeformablePSROIPooling, entryIO),
    _OV_OP_SHAPE_INFER_REG(DepthToSpace, entryIO),
    _OV_OP_SHAPE_INFER_REG(DetectionOutput, entryIO),
    _OV_OP_SHAPE_INFER_REG(DFT, entryIOC),
    _OV_OP_SHAPE_INFER_REG(Einsum, entryIO),
    _OV_OP_SHAPE_INFER_REG(EmbeddingBagOffsetsSum, entryIO),
    _OV_OP_SHAPE_INFER_REG(EmbeddingBagPackedSum, entryIO),
    _OV_OP_SHAPE_INFER_REG(EmbeddingSegmentsSum, entryIOC),
    _OV_OP_SHAPE_INFER_REG(ExperimentalDetectronDetectionOutput, entryIO),
    _OV_OP_SHAPE_INFER_REG(ExperimentalDetectronGenerateProposalsSingleImage, entryIO),
    _OV_OP_SHAPE_INFER_REG(ExperimentalDetectronPriorGridGenerator, entryIO),
    _OV_OP_SHAPE_INFER_REG(ExperimentalDetectronTopKROIs, entryIO),
    _OV_OP_SHAPE_INFER_REG(ExtractImagePatches, entryIO),
    _OV_OP_SHAPE_INFER_REG(Eye, entryIOC),
    _OV_OP_SHAPE_INFER_REG(FakeQuantize, entryIO),
    _OV_OP_SHAPE_INFER_REG(GatherElements, entryIO),
    _OV_OP_SHAPE_INFER_REG(GatherND, entryIO),
    _OV_OP_SHAPE_INFER_REG(GatherTree, entryIO),
    _OV_OP_SHAPE_INFER_REG(GridSample, entryIO),
    _OV_OP_SHAPE_INFER_REG(GroupConvolution, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(GroupConvolutionBackpropData, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(GRUCell, entryIO),
    _OV_OP_SHAPE_INFER_REG(GRUSequence, entryIO),
    _OV_OP_SHAPE_INFER_REG(IDFT, entryIOC),
    _OV_OP_SHAPE_INFER_REG(IRDFT, entryIOC),
    _OV_OP_SHAPE_INFER_REG(LSTMCell, entryIO),
    _OV_OP_SHAPE_INFER_REG(MatMul, entryIO),
    _OV_OP_SHAPE_INFER_REG(MaxPool, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(OneHot, entryIOC),
    _OV_OP_SHAPE_INFER_REG(ov::op::internal::AUGRUCell, entryIO),
    _OV_OP_SHAPE_INFER_REG(ov::op::internal::AUGRUSequence, entryIO),
    _OV_OP_SHAPE_INFER_REG(PriorBox, entryIOC),
    _OV_OP_SHAPE_INFER_REG(PriorBoxClustered, entryIOC),
    _OV_OP_SHAPE_INFER_REG(PSROIPooling, entryIO),
    _OV_OP_SHAPE_INFER_REG(Range, entryIOC),
    _OV_OP_SHAPE_INFER_REG(RDFT, entryIOC),
    _OV_OP_SHAPE_INFER_REG(ReadValue, entryIO),
    _OV_OP_SHAPE_INFER_REG(RegionYolo, entryIO),
    _OV_OP_SHAPE_INFER_REG(ReorgYolo, entryIO),
    _OV_OP_SHAPE_INFER_REG(Reshape, entryIOC),
    _OV_OP_SHAPE_INFER_REG(ReverseSequence, entryIO),
    _OV_OP_SHAPE_INFER_REG(ROIAlign, entryIO),
    _OV_OP_SHAPE_INFER_REG(ROIPooling, entryIO),
    _OV_OP_SHAPE_INFER_REG(Roll, entryIOC),
    _OV_OP_SHAPE_INFER_REG(ScatterElementsUpdate, entryIOC),
    _OV_OP_SHAPE_INFER_REG(ScatterNDUpdate, entryIO),
    _OV_OP_SHAPE_INFER_REG(Select, entryIO),
    _OV_OP_SHAPE_INFER_REG(ShapeOf, entryIO),
    _OV_OP_SHAPE_INFER_REG(ShuffleChannels, entryIO),
    _OV_OP_SHAPE_INFER_REG(Slice, entryIOC),
    _OV_OP_SHAPE_INFER_REG(SpaceToBatch, entryIOC),
    _OV_OP_SHAPE_INFER_REG(SpaceToDepth, entryIO),
    _OV_OP_SHAPE_INFER_REG(Split, entryIOC),
    _OV_OP_SHAPE_INFER_REG(Squeeze, entryIOC),
    _OV_OP_SHAPE_INFER_REG(StridedSlice, entryIOC),
    _OV_OP_SHAPE_INFER_REG(TopK, entryIOC),
    _OV_OP_SHAPE_INFER_REG(Transpose, entryIOC),
    _OV_OP_SHAPE_INFER_REG(Unsqueeze, entryIOC),
    _OV_OP_SHAPE_INFER_REG(VariadicSplit, entryIOC),
    _OV_OP_SHAPE_INFER_VA_REG(Gather, entryIOC, ov::op::util::GatherBase),
    // opset12
    _OV_OP_SHAPE_INFER_REG(opset12::Pad, entryIOC),
    // opset7
    _OV_OP_SHAPE_INFER_VA_REG(opset7::Gather, entryIOC, ov::op::util::GatherBase),
    // opset5
    _OV_OP_SHAPE_INFER_REG(opset5::GatherND, entryIO),
    // opset3
    _OV_OP_SHAPE_INFER_REG(opset3::Assign, entryIO),
    _OV_OP_SHAPE_INFER_REG(opset3::ReadValue, entryIO),
    _OV_OP_SHAPE_INFER_REG(opset3::ROIAlign, entryIO),
    // opset2
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(opset2::MVN, entryCopy),
    // opset1
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(opset1::BatchNormInference, entryFirstPassthrough),
    _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG(opset1::Softmax, entryCopy),
    _OV_OP_SHAPE_INFER_REG(opset1::Broadcast, entryIOC),
    _OV_OP_SHAPE_INFER_REG(opset1::DeformableConvolution, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(opset1::DetectionOutput, entryIO),
    _OV_OP_SHAPE_INFER_REG(opset1::LSTMCell, entryIO),
    _OV_OP_SHAPE_INFER_REG(opset1::MaxPool, ShapeInferWithPadding),
    _OV_OP_SHAPE_INFER_REG(opset1::Pad, entryIOC),
    _OV_OP_SHAPE_INFER_REG(opset1::Range, entryIOC),
    _OV_OP_SHAPE_INFER_REG(opset1::ShapeOf, entryIO),
    _OV_OP_SHAPE_INFER_REG(opset1::TopK, entryIOC),
    _OV_OP_SHAPE_INFER_VA_REG(opset1::Gather, entryIOC, ov::op::util::GatherBase),
};

// Helper types for IStaticShapeInfer makers.
using IStaticShapeInferFactory =
    ShapeInferFactory<ShapeInferKey, std::shared_ptr<IStaticShapeInfer>, std::shared_ptr<ov::Node>>;

// Initialization map for operators supporting IStaticShapeInfer objects.
// First group in map is 'default' opset defined by alias above.
// To use other version of operators, explicitly specify operator with opset version namespace.
template <>
const IStaticShapeInferFactory::TRegistry IStaticShapeInferFactory::registry{
    // Default opset
    _OV_OP_SHAPE_INFER_MASK_REG(ExperimentalDetectronROIFeatureExtractor, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(Proposal, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceL1, ShapeInferTA, op::util::ArithmeticReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceL2, ShapeInferTA, op::util::ArithmeticReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceLogicalAnd, ShapeInferTA, op::util::LogicalReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceLogicalOr, ShapeInferTA, op::util::LogicalReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceMax, ShapeInferTA, op::util::ArithmeticReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceMean, ShapeInferTA, op::util::ArithmeticReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceMin, ShapeInferTA, op::util::ArithmeticReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceProd, ShapeInferTA, op::util::ArithmeticReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_VA_REG(ReduceSum, ShapeInferTA, op::util::ArithmeticReductionKeepDims, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(Tile, ShapeInferenceTA, util::bit::mask(1)),
    // Operators shape inferences for specific opset version should be specified below
    // opset11
    _OV_OP_SHAPE_INFER_MASK_REG(opset11::Interpolate, ShapeInferPaddingTA, util::bit::mask(1, 2, 3)),
    // opset5
    _OV_OP_SHAPE_INFER_MASK_REG(opset5::LSTMSequence, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset5::RNNSequence, ShapeInferTA, util::bit::mask()),
    // opset4
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::Interpolate, ShapeInferPaddingTA, util::bit::mask(1, 2)),
    // opset1
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Interpolate, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::LSTMSequence, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Proposal, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Reverse, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::RNNCell, ShapeInferTA, util::bit::mask()),
};

#undef _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG
#undef _OV_OP_SHAPE_INFER_MASK_REG
#undef _OV_OP_SHAPE_INFER_REG
#undef _OV_OP_SHAPE_INFER_VA_REG
template <>
std::shared_ptr<IShapeInferCommon> make_shape_inference<IShapeInferCommon>(std::shared_ptr<ov::Node> op) {
    if (auto shape_infer = IShapeInferCommonFactory::make(op->get_type_info(), op)) {
        return shape_infer;
    } else if (auto shape_infer = make_shape_inference<IStaticShapeInfer>(op)) {
        return shape_infer;
    } else if (ov::is_type<op::util::UnaryElementwiseArithmetic>(op)) {
        // The unary nad binary elementwise ops can be moved to map but it is easier to handle them by these statements.
        return std::make_shared<entryCopy>(op);
    } else if (ov::is_type<op::util::BinaryElementwiseArithmetic>(op) ||
               ov::is_type<op::util::BinaryElementwiseComparison>(op) ||
               ov::is_type<op::util::BinaryElementwiseLogical>(op)) {
        return std::make_shared<entryEltwise>(op);
    } else {
        return std::make_shared<entryFallback>(op);
    }
}

template <>
std::shared_ptr<IStaticShapeInfer> make_shape_inference<IStaticShapeInfer>(std::shared_ptr<ov::Node> op) {
    if (auto shape_infer = IStaticShapeInferFactory::make(op->get_type_info(), op)) {
        return shape_infer;
    } else {
        // TODO 101252: It should return equivalent of entryFallback which supports new interface.
        return {};
    }
}
}  // namespace intel_cpu
}  // namespace ov
