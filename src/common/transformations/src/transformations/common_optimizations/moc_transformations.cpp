// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/align_eltwise_input_ranks.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/binarize_weights.hpp>
#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>
#include <transformations/common_optimizations/clamp_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/conv_to_binary_conv.hpp>
#include <transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/dilated_convolution_converter.hpp>
#include <transformations/common_optimizations/disable_random_uniform_constant_folding.hpp>
#include <transformations/common_optimizations/disable_shapeof_constant_folding.hpp>
#include <transformations/common_optimizations/divide_fusion.hpp>
#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>
#include <transformations/common_optimizations/fold_subgraph_empty_inputs.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/common_optimizations/gelu_fusion.hpp>
#include <transformations/common_optimizations/gru_cell_fusion.hpp>
#include <transformations/common_optimizations/hsigmoid_fusion.hpp>
#include <transformations/common_optimizations/hswish_fusion.hpp>
#include <transformations/common_optimizations/leaky_relu_fusion.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/matmul_const_transposes_extraction.hpp>
#include <transformations/common_optimizations/matmul_multiply_fusion.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/mul_conv_fusion.hpp>
#include <transformations/common_optimizations/mul_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/nearest_neighbor_upsampling_fusion.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/common_optimizations/normalize_l2_fusion.hpp>
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/prelu_fusion.hpp>
#include <transformations/common_optimizations/pull_through_reduce.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>
#include <transformations/common_optimizations/random_uniform_fusion.hpp>
#include <transformations/common_optimizations/reduce_reshape_fusion.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/remove_concat_zero_dim_input.hpp>
#include <transformations/common_optimizations/remove_filtering_boxes_by_size.hpp>
#include <transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp>
#include <transformations/common_optimizations/reshape_sequence_fusion.hpp>
#include <transformations/common_optimizations/ric_fusion.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/common_optimizations/softmax_fusion.hpp>
#include <transformations/common_optimizations/softplus_fusion.hpp>
#include <transformations/common_optimizations/softplus_to_mish_fusion.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/split_concat_pair_to_interpolate_fusion.hpp>
#include <transformations/common_optimizations/split_squeeze_concat_fusion.hpp>
#include <transformations/common_optimizations/subtract_fusion.hpp>
#include <transformations/common_optimizations/swish_fusion.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/op_conversions/convert_negative.hpp>
#include <transformations/op_conversions/convert_scatter_elements_to_scatter.hpp>
#include <transformations/smart_reshape/lstm_states_broadcast.hpp>
#include <transformations/smart_reshape/reshape_sinking.hpp>

#include "itt.hpp"

bool ov::pass::MOCTransformations::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(MOCTransformations);
    // To avoid issues with dynamism we make nGraph Function dynamic and after we apply all
    // transformations we restore original shapes to the nGraph Function back
    std::unordered_map<ngraph::op::Parameter*, PartialShape> input_shapes;
    if (!m_use_shapes) {
        for (auto&& param : f->get_parameters()) {
            input_shapes[param.get()] = param->get_partial_shape();
            param->set_partial_shape(PartialShape::dynamic(param->get_partial_shape().rank()));
        }
        f->validate_nodes_and_infer_types();
    }

    ov::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);

    manager.register_pass<ngraph::pass::InitNodeInfo>();
    if (m_low_precision_enabled) {
        manager.register_pass<ov::pass::MarkDequantizationSubgraph>(
            element::TypeVector{ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4});
    }
    if (!m_use_shapes) {
        manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
    }
    // RemoveConcatZeroDimInput and RemoveMultiSubGraphOpDanglingParams
    // should be performed before first ConstantFolding call.
    // The passes can deteach graph branches where zero dimesion is calculated.
    // Zero dimensions in shape causes creation empty tensors, which are incorrect during CF.
    // In particular, if zero dim tensor is consumed in body of MultiSubGraphOp
    // RemoveConcatZeroDimInput and RemoveMultiSubGraphOpDanglingParams should be called together.
    manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    manager.register_pass<ov::pass::Validate>();
    manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParams>();
    manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
    manager.register_pass<ov::pass::DisableRandomUniformConstantFolding>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::Validate>();

    // FusedFilteringBoxesBySize transformation has the complex pattern
    // which can be affected by further transformations. So we have to
    // execute it at the beginning of the pipeline. Also, this pass resolves
    // dynamism, so we have to execute type/shape propagation after.
    manager.register_pass<ov::pass::FuseFilteringBoxesBySize>();
    manager.register_pass<ov::pass::Validate>();

    if (!m_use_shapes) {  // Approved Smart Reshape
        manager.register_pass<ov::pass::LSTMStatesBroadcast>();
        manager.register_pass<ov::pass::Validate>();
        manager.register_pass<ov::pass::ReshapeSinkingMatMul>();
        manager.register_pass<ov::pass::Validate>();
    }

    manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    if (!m_use_shapes) {
        manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
    }
    // workaround until dynamism in NMS is not supported
    manager.register_pass<ov::pass::ConvertNmsGatherPathToUnsigned>();
    manager.register_pass<ov::pass::StridedSliceOptimization>(m_use_shapes);
    manager.register_pass<ov::pass::BroadcastElementwiseFusion>();
    manager.register_pass<ov::pass::PullThroughReduce>();

    auto transpose_sinking = manager.register_pass<ov::pass::GraphRewrite>();
    transpose_sinking->add_matcher<ov::pass::TransposeSinking>();
    // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
    // because it replaces pattern that may contain Transposes which must be optimized before
    // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
    transpose_sinking->add_matcher<ov::pass::SplitSqueezeConcatFusion>();

    auto eliminations = manager.register_pass<ov::pass::GraphRewrite>();
    eliminations->add_matcher<ov::pass::EliminateUnsqueezeGather>();
    eliminations->add_matcher<ov::pass::NopElimination>(m_use_shapes /* do not use shape for elimination */);
    eliminations->set_name("ov::pass::CommonEliminations");

    manager.register_pass<ov::pass::ConstantFolding>();

    auto common_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    common_fusions->add_matcher<ngraph::pass::ConvertScatterElementsToScatter>();
    common_fusions->add_matcher<ov::pass::SoftPlusFusion>();
    common_fusions->add_matcher<ov::pass::SoftPlusToMishFusion>();
    common_fusions->add_matcher<ov::pass::SwishFusion>();
    common_fusions->add_matcher<ov::pass::HSwishFusion>();
    common_fusions->add_matcher<ov::pass::HSigmoidFusion>();
    common_fusions->add_matcher<ov::pass::NormalizeL2Fusion>();
    common_fusions->add_matcher<ov::pass::ClampFusion>();
    common_fusions->add_matcher<ov::pass::PadFusion>();
    common_fusions->add_matcher<ov::pass::SoftmaxFusion>();
    common_fusions->add_matcher<ov::pass::ReduceReshapeFusion>();
    common_fusions->add_matcher<ov::pass::MVNFusion>();
    common_fusions->add_matcher<ov::pass::DilatedConvolutionConverter>();
    common_fusions->add_matcher<ov::pass::GeluFusion>();
    common_fusions->add_matcher<ov::pass::LeakyReluFusion>();
    common_fusions->add_matcher<ov::pass::RandomUniformFusion>();
    common_fusions->add_matcher<ov::pass::SplitConcatPairToInterpolateFusion>(m_use_shapes);
    if (m_use_shapes) {
        common_fusions->add_matcher<ov::pass::NearestNeighborUpsamplingFusion>();
    }
    common_fusions->add_matcher<ov::pass::DivideFusion>();
    common_fusions->add_matcher<ov::pass::SubtractFusion>();
    common_fusions->add_matcher<ov::pass::TransposeToReshape>();
    common_fusions->add_matcher<ov::pass::ReshapeSequenceFusion>(m_use_shapes);
    common_fusions->add_matcher<ov::pass::MatMulConstTransposesExtraction>();
    common_fusions->add_matcher<ov::pass::PReluFusion>();
    common_fusions->add_matcher<ov::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::ShuffleChannelsFusion>(!m_use_shapes);
    common_fusions->set_name("ov::pass::CommonFusions");

    manager.register_pass<ov::pass::BinarizeWeights>();
    manager.register_pass<ov::pass::ConvToBinaryConv>();

    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ngraph::pass::BatchNormDecomposition>();
    decomp->add_matcher<ngraph::pass::ConvertDivideWithConstant>();
    decomp->add_matcher<ngraph::pass::ConvertNegative>();

    manager.register_pass<ov::pass::LinOpSequenceFusion>();

    auto multiply_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    multiply_fusions->add_matcher<ov::pass::ConvolutionMultiplyFusion>();
    multiply_fusions->add_matcher<ov::pass::GroupConvolutionMultiplyFusion>();
    multiply_fusions->add_matcher<ov::pass::ConvolutionBackpropDataMultiplyFusion>();
    multiply_fusions->add_matcher<ov::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    multiply_fusions->add_matcher<ov::pass::MultiplyConvolutionFusion>();
    multiply_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionFusion>();
    multiply_fusions->add_matcher<ov::pass::MultiplyConvolutionBackpropDataFusion>();
    multiply_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    multiply_fusions->add_matcher<ov::pass::MatMulMultiplyFusion>();
    multiply_fusions->set_name("ov::pass::MultiplyFusions");
    manager.register_pass<ov::pass::ConstantFolding>();

    auto fq_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    fq_fusions->add_matcher<ov::pass::FakeQuantizeMulFusion>();
    fq_fusions->add_matcher<ov::pass::FakeQuantizeReshapeFusion>();
    fq_fusions->add_matcher<ov::pass::PullTransposeThroughFQUp>();
    fq_fusions->add_matcher<ov::pass::ReluFakeQuantizeFusion>();
    fq_fusions->add_matcher<ov::pass::AddFakeQuantizeFusion>();
    fq_fusions->add_matcher<ov::pass::MulFakeQuantizeFusion>();
    fq_fusions->set_name("ov::pass::FakeQuantizeFusions");

    manager.register_pass<ov::pass::ReverseInputChannelsFusion>();

    manager.register_pass<ov::pass::AlignEltwiseInputRanks>();

    manager.register_pass<ov::pass::ConstantFolding>();

    manager.run_passes(f);

    if (!m_use_shapes) {
        // Restore original shapes to the nGraph Function
        for (auto&& param : f->get_parameters()) {
            param->set_partial_shape(input_shapes.at(param.get()));
        }
        f->validate_nodes_and_infer_types();
    }

    return false;
}
