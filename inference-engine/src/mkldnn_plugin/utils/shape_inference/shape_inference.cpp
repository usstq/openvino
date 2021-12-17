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
#include <openvino/opsets/opset8.hpp>

#include "assign_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "experimental_detectron_detection_output_shape_inference.hpp"
#include "experimental_detectron_prior_grid_generator_shape_inference.hpp"
#include "experimental_detectron_topkrois_shape_inference.hpp"
#include "fake_quantize.hpp"
#include "gather_elements_shape_inference.hpp"
#include "gather_shape_inference.hpp"
#include "gather_tree_shape_inference.hpp"
#include "interpolate_shape_inference.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "one_hot_shape_inference.hpp"
#include "read_value_shape_inference.hpp"
#include "reduce_shape_inference.hpp"
#include "scatter_elements_update_shape_inference.hpp"
#include "scatter_nd_base_shape_inference.hpp"
#include "shape_inference.hpp"
#include "shape_nodes.hpp"
#include "static_shape.hpp"
#include "tile_shape_inference.hpp"
#include "utils.hpp"

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

void shape_inference(ov::Node* op,
                     const std::vector<ov::StaticShape>& input_shapes,
                     std::vector<ov::StaticShape>& output_shapes,
                     const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    static int perf_test_N = std::getenv("PERFN") ? std::atoi(std::getenv("PERFN")) : 1;

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
        };

    for (int ixx = 0; ixx < perf_test_N; ixx++) {
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
}