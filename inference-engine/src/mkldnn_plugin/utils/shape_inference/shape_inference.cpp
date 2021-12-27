// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shape_inference.hpp"

#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset2.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>

#include "assign_shape_inference.hpp"
#include "bucketize_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "ctc_greedy_decoder_seq_len_shape_inference.hpp"
#include "ctc_greedy_decoder_shape_inference.hpp"
#include "ctc_loss_shape_inference.hpp"
#include "einsum_shape_inference.hpp"
#include "embedding_segments_sum_shape_inference.hpp"
#include "embeddingbag_offsets_shape_inference.hpp"
#include "experimental_detectron_detection_output_shape_inference.hpp"
#include "experimental_detectron_generate_proposals_shape_inference.hpp"
#include "experimental_detectron_prior_grid_generator_shape_inference.hpp"
#include "experimental_detectron_roi_feature_shape_inference.hpp"
#include "experimental_detectron_topkrois_shape_inference.hpp"
#include "extract_image_patches_shape_inference.hpp"
#include "fake_quantize.hpp"
#include "fft_base_shape_inference.hpp"
#include "gather_elements_shape_inference.hpp"
#include "gather_shape_inference.hpp"
#include "gather_tree_shape_inference.hpp"
#include "interpolate_shape_inference.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "one_hot_shape_inference.hpp"
#include "pad_shape_inference.hpp"
#include "proposal_shape_inference.hpp"
#include "range_shape_inference.hpp"
#include "read_value_shape_inference.hpp"
#include "reduce_shape_inference.hpp"
#include "region_yolo_shape_inference.hpp"
#include "reorg_yolo_shape_inference.hpp"
#include "reverse_sequence_shape_inference.hpp"
#include "roi_align_shape_inference.hpp"
#include "roll_shape_inference.hpp"
#include "scatter_elements_update_shape_inference.hpp"
#include "scatter_nd_base_shape_inference.hpp"
#include "shape_inference.hpp"
#include "shape_nodes.hpp"
#include "split_shape_inference.hpp"
#include "static_shape.hpp"
#include "strided_slice_shape_inference.hpp"
#include "tile_shape_inference.hpp"
#include "topk_shape_inference.hpp"
#include "utils.hpp"
#include "variadic_split_shape_inference.hpp"

void shape_inference_0(ov::Node* op,
                       const std::vector<ov::StaticShape>& input_shapes,
                       std::vector<ov::StaticShape>& output_shapes,
                       const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data);
void shape_inference_1(ov::Node* op,
                       const std::vector<ov::StaticShape>& input_shapes,
                       std::vector<ov::StaticShape>& output_shapes,
                       const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data);

void shape_inference_0(ov::Node* op,
                       const std::vector<ov::StaticShape>& input_shapes,
                       std::vector<ov::StaticShape>& output_shapes,
                       const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (auto node = ov::as_type<ov::opset8::Convolution>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        bool status = resolve_auto_pad_for_shape(node, pads_begin, pads_end, input_shapes, 2, 2);
        OPENVINO_ASSERT(status,
                        "Convolution shape inference doesn't have enough information to calculate static shapes");
        shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset8::GroupConvolution>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        bool status = resolve_auto_pad_for_shape(node, pads_begin, pads_end, input_shapes, 2, 3);
        OPENVINO_ASSERT(status,
                        "GroupConvolution shape inference doesn't have enough information to calculate static shapes");
        shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset8::ConvolutionBackpropData>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        ov::StaticShape output_shape_input;
        if (node->get_input_size() == 3)
            get_data_as_shape<ov::StaticShape>(2, op, output_shape_input, constant_data);
        bool status =
            resolve_auto_pad_for_shape_back_prop(node, pads_begin, pads_end, input_shapes, output_shape_input, 2, 2);
        OPENVINO_ASSERT(
            status,
            "ConvolutionBackpropData shape inference doesn't have enough information to calculate static shapes");
        shape_infer(node, pads_begin, pads_end, output_shape_input, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset8::GroupConvolutionBackpropData>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        ov::StaticShape output_shape_input;
        if (node->get_input_size() == 3)
            get_data_as_shape<ov::StaticShape>(2, op, output_shape_input, constant_data);
        bool status =
            resolve_auto_pad_for_shape_back_prop(node, pads_begin, pads_end, input_shapes, output_shape_input, 2, 3);
        OPENVINO_ASSERT(status,
                        "GroupConvolutionBackpropData shape inference doesn't have enough information to calculate "
                        "static shapes");
        shape_infer(node, pads_begin, pads_end, output_shape_input, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::op::util::ArithmeticReductionKeepDims>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::op::util::LogicalReductionKeepDims>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(op) || ov::is_type<ov::opset1::Convert>(op) ||
               ov::is_type<ov::opset1::Clamp>(op) || ov::is_type<ov::opset1::GRN>(op) ||
               ov::is_type<ov::opset1::LRN>(op) || ov::is_type<ov::opset1::LogicalNot>(op) ||
               ov::is_type<ov::opset4::Mish>(op) || ov::is_type<ov::opset2::MVN>(op) ||
               ov::is_type<ov::opset6::MVN>(op) || ov::is_type<ov::opset1::PRelu>(op) ||
               ov::is_type<ov::opset1::Relu>(op) || ov::is_type<ov::opset4::Swish>(op) ||
               ov::is_type<ov::opset1::Elu>(op) || ov::is_type<ov::opset1::Softmax>(op) ||
               ov::is_type<ov::opset8::Softmax>(op) || ov::is_type<ov::opset5::Round>(op)) {
        copy_shape_infer(node, input_shapes, output_shapes);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseComparison>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseLogical>(op)) {
        eltwise_shape_infer(op, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::FakeQuantize>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::Reshape>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Squeeze>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Unsqueeze>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::ShapeOf>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ShapeOf>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronDetectionOutput>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::TopK>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset3::Bucketize>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::EmbeddingSegmentsSum>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset3::EmbeddingBagOffsetsSum>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronROIFeatureExtractor>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::Pad>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset4::Range>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Range>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::RegionYolo>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset2::ReorgYolo>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::Split>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::VariadicSplit>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset7::Einsum>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::StridedSlice>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset3::Assign>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::Assign>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronPriorGridGenerator>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::LSTMCell>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::LSTMCell>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ReadValue>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ReadValue>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::Tile>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronTopKROIs>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset4::Interpolate>(op)) {
        std::vector<size_t> pads_begin, pads_end;
        correct_pads_attr(node, pads_begin, pads_end, input_shapes);
        shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Interpolate>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset3::ScatterElementsUpdate>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset4::ScatterNDUpdate>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::GatherElements>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::op::util::GatherBase>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::GatherTree>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::OneHot>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset4::CTCLoss>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset7::DFT>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset7::IDFT>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset6::CTCGreedyDecoderSeqLen>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::CTCGreedyDecoder>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ExtractImagePatches>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::ReverseSequence>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset7::Roll>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronGenerateProposalsSingleImage>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset4::Proposal>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::Proposal>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ROIAlign>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else {
        ngraph::OutputVector new_inputs;
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            if (constant_data.count(i)) {
                new_inputs.push_back(std::make_shared<ov::opset1::Constant>(constant_data.at(i)));
            } else {
                new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                             input_shapes[i].to_partial_shape()));
            }
        }
        const auto local_op = op->clone_with_new_inputs(new_inputs);
        local_op->validate_and_infer_types();

        output_shapes.resize(op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);
            OPENVINO_ASSERT(
                partial_shape.is_static(),
                "On device shape infer shouldn't support default shape infer for nodes with internal dynamism");
            output_shapes[i] = ov::StaticShape(partial_shape.to_shape());
        }
    }
}

template <typename OP>
void shape_infer2(ov::Node* op,
                  const std::vector<ov::StaticShape>& input_shapes,
                  std::vector<ov::StaticShape>& output_shapes,
                  const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {}

template <>
void shape_infer2<ov::opset8::Convolution>(
    ov::Node* op,
    const std::vector<ov::StaticShape>& input_shapes,
    std::vector<ov::StaticShape>& output_shapes,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<ov::opset8::Convolution*>(op);
    ov::CoordinateDiff pads_begin, pads_end;
    bool status = resolve_auto_pad_for_shape(node, pads_begin, pads_end, input_shapes, 2, 2);
    OPENVINO_ASSERT(status, "Convolution shape inference doesn't have enough information to calculate static shapes");
    shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes);
}

template <>
void shape_infer2<ov::opset8::GroupConvolution>(
    ov::Node* op,
    const std::vector<ov::StaticShape>& input_shapes,
    std::vector<ov::StaticShape>& output_shapes,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<ov::opset8::GroupConvolution*>(op);
    ov::CoordinateDiff pads_begin, pads_end;
    bool status = resolve_auto_pad_for_shape(node, pads_begin, pads_end, input_shapes, 2, 3);
    OPENVINO_ASSERT(status,
                    "GroupConvolution shape inference doesn't have enough information to calculate static shapes");
    shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes);
}

template <>
void shape_infer2<ov::opset8::ConvolutionBackpropData>(
    ov::Node* op,
    const std::vector<ov::StaticShape>& input_shapes,
    std::vector<ov::StaticShape>& output_shapes,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<ov::opset8::ConvolutionBackpropData*>(op);
    ov::CoordinateDiff pads_begin, pads_end;
    ov::StaticShape output_shape_input;
    if (node->get_input_size() == 3)
        get_data_as_shape<ov::StaticShape>(2, op, output_shape_input, constant_data);
    bool status =
        resolve_auto_pad_for_shape_back_prop(node, pads_begin, pads_end, input_shapes, output_shape_input, 2, 2);
    OPENVINO_ASSERT(status,
                    "ConvolutionBackpropData shape inference doesn't have enough information to calculate "
                    "static shapes");
    shape_infer(node, pads_begin, pads_end, output_shape_input, input_shapes, output_shapes);
}

template <>
void shape_infer2<ov::opset4::Interpolate>(
    ov::Node* op,
    const std::vector<ov::StaticShape>& input_shapes,
    std::vector<ov::StaticShape>& output_shapes,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<ov::opset4::Interpolate*>(op);
    std::vector<size_t> pads_begin, pads_end;
    correct_pads_attr(node, pads_begin, pads_end, input_shapes);
    shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes, constant_data);
}

template <>
void shape_infer2<ov::opset8::GroupConvolutionBackpropData>(
    ov::Node* op,
    const std::vector<ov::StaticShape>& input_shapes,
    std::vector<ov::StaticShape>& output_shapes,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<ov::opset8::GroupConvolutionBackpropData*>(op);
    ov::CoordinateDiff pads_begin, pads_end;
    ov::StaticShape output_shape_input;
    if (node->get_input_size() == 3)
        get_data_as_shape<ov::StaticShape>(2, op, output_shape_input, constant_data);
    bool status =
        resolve_auto_pad_for_shape_back_prop(node, pads_begin, pads_end, input_shapes, output_shape_input, 2, 3);
    OPENVINO_ASSERT(status,
                    "GroupConvolutionBackpropData shape inference doesn't have enough information to calculate "
                    "static shapes");
    shape_infer(node, pads_begin, pads_end, output_shape_input, input_shapes, output_shapes);
}

template <typename T>
void shape_infer_ioc(ov::Node* op,
                     const std::vector<ov::StaticShape>& input_shapes,
                     std::vector<ov::StaticShape>& output_shapes,
                     const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<T*>(op);
    shape_infer(node, input_shapes, output_shapes, constant_data);
}

template <typename T>
void shape_infer_io(ov::Node* op,
                    const std::vector<ov::StaticShape>& input_shapes,
                    std::vector<ov::StaticShape>& output_shapes,
                    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<T*>(op);
    shape_infer(node, input_shapes, output_shapes);
}

template <typename T>
void shape_infer_copy(ov::Node* op,
                      const std::vector<ov::StaticShape>& input_shapes,
                      std::vector<ov::StaticShape>& output_shapes,
                      const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<T*>(op);
    copy_shape_infer(node, input_shapes, output_shapes);
}

template <typename T>
void shape_infer_eltwise(ov::Node* op,
                         const std::vector<ov::StaticShape>& input_shapes,
                         std::vector<ov::StaticShape>& output_shapes,
                         const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto node = static_cast<T*>(op);
    eltwise_shape_infer(node, input_shapes, output_shapes);
}

void shape_inference_1(ov::Node* op,
                       const std::vector<ov::StaticShape>& input_shapes,
                       std::vector<ov::StaticShape>& output_shapes,
                       const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
#define OP_MAP_ENTRY(opType, func) \
    { opType::get_type_info_static(), func<opType> }

    static std::unordered_map<
        ngraph::DiscreteTypeInfo,
        std::function<void(ov::Node*,
                           const std::vector<ov::StaticShape>&,
                           std::vector<ov::StaticShape>&,
                           const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>&)>>
        op_map = {
            OP_MAP_ENTRY(ov::opset8::Convolution, shape_infer2),
            OP_MAP_ENTRY(ov::opset8::GroupConvolution, shape_infer2),
            OP_MAP_ENTRY(ov::opset8::ConvolutionBackpropData, shape_infer2),
            OP_MAP_ENTRY(ov::opset8::GroupConvolutionBackpropData, shape_infer2),
            OP_MAP_ENTRY(ov::op::util::ArithmeticReductionKeepDims, shape_infer_ioc),
            OP_MAP_ENTRY(ov::op::util::LogicalReductionKeepDims, shape_infer_ioc),
            OP_MAP_ENTRY(ov::op::util::UnaryElementwiseArithmetic, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::Convert, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::Clamp, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::GRN, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::LRN, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::LogicalNot, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset4::Mish, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset2::MVN, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset6::MVN, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::PRelu, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::Relu, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset4::Swish, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::Elu, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset1::Softmax, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset8::Softmax, shape_infer_copy),
            OP_MAP_ENTRY(ov::opset5::Round, shape_infer_copy),

            OP_MAP_ENTRY(ov::op::util::BinaryElementwiseArithmetic, shape_infer_eltwise),
            OP_MAP_ENTRY(ov::op::util::BinaryElementwiseComparison, shape_infer_eltwise),
            OP_MAP_ENTRY(ov::op::util::BinaryElementwiseLogical, shape_infer_eltwise),

            OP_MAP_ENTRY(ov::opset1::FakeQuantize, shape_infer_io),
            OP_MAP_ENTRY(ov::opset1::Reshape, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset1::Squeeze, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset1::Unsqueeze, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset1::ShapeOf, shape_infer_io),
            OP_MAP_ENTRY(ov::opset3::ShapeOf, shape_infer_io),

            OP_MAP_ENTRY(ov::opset6::ExperimentalDetectronDetectionOutput, shape_infer_io),
            OP_MAP_ENTRY(ov::opset3::TopK, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset3::Bucketize, shape_infer_io),
            OP_MAP_ENTRY(ov::opset3::EmbeddingSegmentsSum, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset3::EmbeddingBagOffsetsSum, shape_infer_io),
            OP_MAP_ENTRY(ov::opset6::ExperimentalDetectronROIFeatureExtractor, shape_infer_io),
            OP_MAP_ENTRY(ov::opset1::Pad, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset4::Range, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset1::Range, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset1::RegionYolo, shape_infer_io),
            OP_MAP_ENTRY(ov::opset2::RegionYolo, shape_infer_io),
            OP_MAP_ENTRY(ov::opset1::Split, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset1::VariadicSplit, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset7::Einsum, shape_infer_io),
            OP_MAP_ENTRY(ov::opset1::StridedSlice, shape_infer_ioc),

            OP_MAP_ENTRY(ov::opset3::Assign, shape_infer_io),
            OP_MAP_ENTRY(ov::opset6::Assign, shape_infer_io),
            OP_MAP_ENTRY(ov::opset6::ExperimentalDetectronPriorGridGenerator, shape_infer_io),
            OP_MAP_ENTRY(ov::opset1::LSTMCell, shape_infer_io),
            OP_MAP_ENTRY(ov::opset6::LSTMCell, shape_infer_io),
            OP_MAP_ENTRY(ov::opset3::ReadValue, shape_infer_io),
            OP_MAP_ENTRY(ov::opset6::ReadValue, shape_infer_io),
            OP_MAP_ENTRY(ov::opset6::Tile, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset6::ExperimentalDetectronTopKROIs, shape_infer_io),
            OP_MAP_ENTRY(ov::opset4::Interpolate, shape_infer2),
            OP_MAP_ENTRY(ov::opset1::Interpolate, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset3::ScatterElementsUpdate, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset4::ScatterNDUpdate, shape_infer_io),
            OP_MAP_ENTRY(ov::opset6::GatherElements, shape_infer_io),
            OP_MAP_ENTRY(ov::op::util::GatherBase, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset1::GatherTree, shape_infer_io),
            OP_MAP_ENTRY(ov::opset1::OneHot, shape_infer_ioc),

            OP_MAP_ENTRY(ov::opset4::CTCLoss, shape_infer_io),
            OP_MAP_ENTRY(ov::opset7::DFT, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset7::IDFT, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset6::CTCGreedyDecoderSeqLen, shape_infer_io),
            OP_MAP_ENTRY(ov::opset6::CTCGreedyDecoder, shape_infer_io),
            OP_MAP_ENTRY(ov::opset3::ExtractImagePatches, shape_infer_io),
            OP_MAP_ENTRY(ov::opset1::ReverseSequence, shape_infer_io),
            OP_MAP_ENTRY(ov::opset7::Roll, shape_infer_ioc),
            OP_MAP_ENTRY(ov::opset6::ExperimentalDetectronGenerateProposalsSingleImage, shape_infer_io),
            OP_MAP_ENTRY(ov::opset4::Proposal, shape_infer_io),
            OP_MAP_ENTRY(ov::opset1::Proposal, shape_infer_io),
            OP_MAP_ENTRY(ov::opset3::ROIAlign, shape_infer_io),
        };

    const ngraph::DiscreteTypeInfo* p_typeinfo = &(op->get_type_info());
    auto it = op_map.find(*p_typeinfo);

    while (it == op_map.end() && p_typeinfo->parent) {
        p_typeinfo = p_typeinfo->parent;
        it = op_map.find(*p_typeinfo);
    }
    if (it != op_map.end()) {
        it->second(op, input_shapes, output_shapes, constant_data);
    } else {
        ngraph::OutputVector new_inputs;
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            if (constant_data.count(i)) {
                new_inputs.push_back(std::make_shared<ov::opset1::Constant>(constant_data.at(i)));
            } else {
                new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                             input_shapes[i].to_partial_shape()));
            }
        }
        const auto local_op = op->clone_with_new_inputs(new_inputs);
        local_op->validate_and_infer_types();

        output_shapes.resize(op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);
            OPENVINO_ASSERT(
                partial_shape.is_static(),
                "On device shape infer shouldn't support default shape infer for nodes with internal dynamism");
            output_shapes[i] = ov::StaticShape(partial_shape.to_shape());
        }
    }
}

void shape_inference(ov::Node* op,
                     const std::vector<ov::StaticShape>& input_shapes,
                     std::vector<ov::StaticShape>& output_shapes,
                     const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    static int perf_test_N = std::getenv("PERFN") ? std::atoi(std::getenv("PERFN")) : 1;
    static int perf_test_K = std::getenv("PERFK") ? std::atoi(std::getenv("PERFK")) : 0;

    if (perf_test_K == 0) {
        for (int ixx = 0; ixx < perf_test_N; ixx++)
            shape_inference_0(op, input_shapes, output_shapes, constant_data);
    }

    if (perf_test_K == 1) {
        for (int ixx = 0; ixx < perf_test_N; ixx++)
            shape_inference_1(op, input_shapes, output_shapes, constant_data);
    }

    if (perf_test_K == 2) {
        auto shapeInfer = make_shape_inference(op->shared_from_this());
        for (int ixx = 0; ixx < perf_test_N; ixx++)
            shapeInfer->infer(input_shapes, output_shapes, constant_data);
    }
}

// Template that calls shape_infer(node, input_shapes, output_shapes)
class entryBase : public IShapeInfer {
public:
    entryBase(std::shared_ptr<ov::Node> node) : node(node) {}
    ov::Node* get_op() override {
        return node.get();
    }

protected:
    std::shared_ptr<ov::Node> node;
};

template <typename OP>
class entryIO : public entryBase {
public:
    using entryBase::entryBase;

    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        OP* op = static_cast<OP*>(node.get());
        shape_infer(op, input_shapes, output_shapes);
    }
};

// Template that calls shape_infer(node, input_shapes, output_shapes)
template <typename OP>
class entryIOC : public entryBase {
public:
    using entryBase::entryBase;

    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = static_cast<OP*>(node.get());
        shape_infer(op, input_shapes, output_shapes, constant_data);
    }
};

class entryCOPY : public entryBase {
public:
    using entryBase::entryBase;

    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        copy_shape_infer(node.get(), input_shapes, output_shapes);
    }
};

class entryEltwise : public entryBase {
public:
    using entryBase::entryBase;

    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        eltwise_shape_infer(node.get(), input_shapes, output_shapes);
    }
};

class entryFallback : public entryBase {
public:
    using entryBase::entryBase;

    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        ngraph::OutputVector new_inputs;
        auto op = node.get();
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
        const auto local_op = op->clone_with_new_inputs(new_inputs);
        local_op->validate_and_infer_types();

        output_shapes.resize(op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);
            OPENVINO_ASSERT(
                partial_shape.is_static(),
                "On device shape infer shouldn't support default shape infer for nodes with internal dynamism");
            output_shapes[i] = ov::StaticShape(partial_shape.to_shape());
        }
    }
};

template <typename OP>
class entryPooling : public entryBase {
public:
    using entryBase::entryBase;

    ov::CoordinateDiff pads_begin, pads_end;

    const ov::CoordinateDiff& get_pads_begin() override {
        return pads_begin;
    }
    const ov::CoordinateDiff& get_pads_end() override {
        return pads_end;
    }
    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        // no shape infer routine, fallback to validate_and_infer_types + get_pads_begin/get_pads_end
        auto op = static_cast<OP*>(this->node.get());
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
        const auto local_op = op->clone_with_new_inputs(new_inputs);
        local_op->validate_and_infer_types();

        auto node = dynamic_cast<OP*>(local_op.get());
        OPENVINO_ASSERT(node);

        const auto convertPadding = [](const ov::Shape& newPads) {
            std::vector<ptrdiff_t> pads(newPads.size());
            for (int i = 0; i < newPads.size(); i++) {
                pads[i] = static_cast<ptrdiff_t>(newPads[i]);
            }
            return pads;
        };
        pads_begin = convertPadding(node->get_pads_begin());
        pads_end = convertPadding(node->get_pads_end());

        output_shapes.resize(op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);
            OPENVINO_ASSERT(
                partial_shape.is_static(),
                "On device shape infer shouldn't support default shape infer for nodes with internal dynamism");
            output_shapes[i] = ov::StaticShape(partial_shape.to_shape());
        }
    }
};

template <typename OP>
class entryInterpolate : public entryBase {
public:
    using entryBase::entryBase;

    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        std::vector<size_t> pads_begin, pads_end;
        auto op = static_cast<OP*>(node.get());
        correct_pads_attr(op, pads_begin, pads_end, input_shapes);
        shape_infer(op, pads_begin, pads_end, input_shapes, output_shapes, constant_data);
    }
};

template <typename OP>
class entryConv : public entryBase {
public:
    entryConv(std::shared_ptr<OP> node, bool is_grouped) : entryBase(node), is_grouped(is_grouped) {}
    const ov::CoordinateDiff& get_pads_begin() override {
        return pads_begin;
    }
    const ov::CoordinateDiff& get_pads_end() override {
        return pads_end;
    }
    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = static_cast<OP*>(node.get());
        bool status = resolve_auto_pad_for_shape(op, pads_begin, pads_end, input_shapes, 2, is_grouped ? 3 : 2);
        OPENVINO_ASSERT(status,
                        "Convolution shape inference doesn't have enough information to calculate static shapes");
        shape_infer(op, pads_begin, pads_end, input_shapes, output_shapes);
    }

protected:
    ov::CoordinateDiff pads_begin, pads_end;
    bool is_grouped;
};

template <typename OP>
class entryConvBackprop : public entryBase {
public:
    entryConvBackprop(std::shared_ptr<OP> node, bool is_grouped) : entryBase(node), is_grouped(is_grouped) {}
    const ov::CoordinateDiff& get_pads_begin() override {
        return pads_begin;
    }
    const ov::CoordinateDiff& get_pads_end() override {
        return pads_end;
    }
    void infer(const std::vector<ov::StaticShape>& input_shapes,
               std::vector<ov::StaticShape>& output_shapes,
               const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        ov::StaticShape output_shape_input;
        auto op = static_cast<OP*>(node.get());
        if (op->get_input_size() == 3)
            get_data_as_shape<ov::StaticShape>(2, op, output_shape_input, constant_data);
        bool status = resolve_auto_pad_for_shape_back_prop(op,
                                                           pads_begin,
                                                           pads_end,
                                                           input_shapes,
                                                           output_shape_input,
                                                           2,
                                                           is_grouped ? 3 : 2);
        OPENVINO_ASSERT(
            status,
            "ConvolutionBackpropData shape inference doesn't have enough information to calculate static shapes");
        shape_infer(op, pads_begin, pads_end, output_shape_input, input_shapes, output_shapes);
    }

protected:
    ov::CoordinateDiff pads_begin, pads_end;
    bool is_grouped;
};

const ov::CoordinateDiff& IShapeInfer::get_pads_begin() {
    OPENVINO_ASSERT(false, "IShapeInfer do not support get_pads_begin() by default.");
}

const ov::CoordinateDiff& IShapeInfer::get_pads_end() {
    OPENVINO_ASSERT(false, "IShapeInfer do not support get_pads_end() by default.");
}

template <typename OP>
std::shared_ptr<entryIOC<OP>> make_shared_entryIOC(std::shared_ptr<OP> node) {
    return std::make_shared<entryIOC<OP>>(node);
}

template <typename OP>
std::shared_ptr<entryIO<OP>> make_shared_entryIO(std::shared_ptr<OP> node) {
    return std::make_shared<entryIO<OP>>(node);
}

std::shared_ptr<IShapeInfer> make_shape_inference(const std::shared_ptr<ngraph::Node>& op) {
    if (auto node = ov::as_type_ptr<ov::opset8::Convolution>(op)) {
        return std::make_shared<entryConv<ov::opset8::Convolution>>(node, false);
    } else if (auto node = ov::as_type_ptr<ov::opset8::GroupConvolution>(op)) {
        return std::make_shared<entryConv<ov::opset8::GroupConvolution>>(node, true);
    } else if (auto node = ov::as_type_ptr<ov::opset8::ConvolutionBackpropData>(op)) {
        return std::make_shared<entryConvBackprop<ov::opset8::ConvolutionBackpropData>>(node, false);
    } else if (auto node = ov::as_type_ptr<ov::opset8::GroupConvolutionBackpropData>(op)) {
        return std::make_shared<entryConvBackprop<ov::opset8::GroupConvolutionBackpropData>>(node, true);
    } else if (auto node = ov::as_type_ptr<ov::op::util::ArithmeticReductionKeepDims>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::op::util::LogicalReductionKeepDims>(op)) {
        return make_shared_entryIOC(node);
    } else if (ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(op) || ov::is_type<ov::opset1::Convert>(op) ||
               ov::is_type<ov::opset1::Clamp>(op) || ov::is_type<ov::opset1::GRN>(op) ||
               ov::is_type<ov::opset1::LRN>(op) || ov::is_type<ov::opset1::LogicalNot>(op) ||
               ov::is_type<ov::opset4::Mish>(op) || ov::is_type<ov::opset2::MVN>(op) ||
               ov::is_type<ov::opset6::MVN>(op) || ov::is_type<ov::opset1::PRelu>(op) ||
               ov::is_type<ov::opset1::Relu>(op) || ov::is_type<ov::opset4::Swish>(op) ||
               ov::is_type<ov::opset1::Elu>(op) || ov::is_type<ov::opset1::Softmax>(op) ||
               ov::is_type<ov::opset8::Softmax>(op) || ov::is_type<ov::opset5::Round>(op)) {
        return std::make_shared<entryCOPY>(op);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseComparison>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseLogical>(op)) {
        return std::make_shared<entryEltwise>(op);
    } else if (auto node = ov::as_type_ptr<ov::opset1::FakeQuantize>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Reshape>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Squeeze>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Unsqueeze>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::ShapeOf>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ShapeOf>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronDetectionOutput>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::TopK>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::Bucketize>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::EmbeddingSegmentsSum>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::EmbeddingBagOffsetsSum>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronROIFeatureExtractor>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Pad>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::Range>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Range>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::RegionYolo>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset2::ReorgYolo>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Split>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::VariadicSplit>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset7::Einsum>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::StridedSlice>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::Assign>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::Assign>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronPriorGridGenerator>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::LSTMCell>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::LSTMCell>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ReadValue>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ReadValue>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::Tile>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronTopKROIs>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::Interpolate>(op)) {
        return std::make_shared<entryInterpolate<ov::opset4::Interpolate>>(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Interpolate>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ScatterElementsUpdate>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::ScatterNDUpdate>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::GatherElements>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::op::util::GatherBase>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::GatherTree>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::OneHot>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::CTCLoss>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset7::DFT>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset7::IDFT>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::CTCGreedyDecoderSeqLen>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::CTCGreedyDecoder>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ExtractImagePatches>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::ReverseSequence>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset7::Roll>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronGenerateProposalsSingleImage>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::Proposal>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Proposal>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ROIAlign>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::op::v8::MaxPool>(op)) {
        return std::make_shared<entryPooling<ov::op::v8::MaxPool>>(node);
    } else if (auto node = ov::as_type_ptr<ov::op::v1::MaxPool>(op)) {
        return std::make_shared<entryPooling<ov::op::v1::MaxPool>>(node);
    } else if (auto node = ov::as_type_ptr<ov::op::v1::AvgPool>(op)) {
        return std::make_shared<entryPooling<ov::op::v1::AvgPool>>(node);
    } else {
        return std::make_shared<entryFallback>(op);
    }
}
