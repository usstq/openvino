// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_types.h"

#include <vector>
#include <string>

namespace ov {
namespace intel_cpu {

const InferenceEngine::details::caseless_unordered_map<std::string, Type> type_to_name_tbl = {
        { "Constant", Type::Input },
        { "Parameter", Type::Input },
        { "Result", Type::Output },
        { "Eye", Type::Eye },
        { "Convolution", Type::Convolution },
        { "GroupConvolution", Type::Convolution },
        { "MatMul", Type::MatMul },
        { "FullyConnected", Type::FullyConnected },
        { "MaxPool", Type::Pooling },
        { "AvgPool", Type::Pooling },
        { "AdaptiveMaxPool", Type::AdaptivePooling},
        { "AdaptiveAvgPool", Type::AdaptivePooling},
        { "Add", Type::Eltwise },
        { "IsFinite", Type::Eltwise },
        { "IsInf", Type::Eltwise },
        { "IsNaN", Type::Eltwise },
        { "Subtract", Type::Eltwise },
        { "Multiply", Type::Eltwise },
        { "Divide", Type::Eltwise },
        { "SquaredDifference", Type::Eltwise },
        { "Maximum", Type::Eltwise },
        { "Minimum", Type::Eltwise },
        { "Mod", Type::Eltwise },
        { "FloorMod", Type::Eltwise },
        { "Power", Type::Eltwise },
        { "PowerStatic", Type::Eltwise },
        { "Equal", Type::Eltwise },
        { "NotEqual", Type::Eltwise },
        { "Greater", Type::Eltwise },
        { "GreaterEqual", Type::Eltwise },
        { "Less", Type::Eltwise },
        { "LessEqual", Type::Eltwise },
        { "LogicalAnd", Type::Eltwise },
        { "LogicalOr", Type::Eltwise },
        { "LogicalXor", Type::Eltwise },
        { "LogicalNot", Type::Eltwise },
        { "Relu", Type::Eltwise },
        { "LeakyRelu", Type::Eltwise },
        { "Gelu", Type::Eltwise },
        { "Elu", Type::Eltwise },
        { "Tanh", Type::Eltwise },
        { "Sigmoid", Type::Eltwise },
        { "Abs", Type::Eltwise },
        { "Sqrt", Type::Eltwise },
        { "Clamp", Type::Eltwise },
        { "Exp", Type::Eltwise },
        { "SwishCPU", Type::Eltwise },
        { "HSwish", Type::Eltwise },
        { "Mish", Type::Eltwise },
        { "HSigmoid", Type::Eltwise },
        { "Round", Type::Eltwise },
        { "PRelu", Type::Eltwise },
        { "Erf", Type::Eltwise },
        { "SoftPlus", Type::Eltwise },
        { "SoftSign", Type::Eltwise },
        { "Select", Type::Eltwise},
        { "Log", Type::Eltwise },
        { "Reshape", Type::Reshape },
        { "Squeeze", Type::Reshape },
        { "Unsqueeze", Type::Reshape },
        { "ShapeOf", Type::ShapeOf },
        { "NonZero", Type::NonZero },
        { "Softmax", Type::Softmax },
        { "Reorder", Type::Reorder },
        { "BatchToSpace", Type::BatchToSpace },
        { "SpaceToBatch", Type::SpaceToBatch },
        { "DepthToSpace", Type::DepthToSpace },
        { "SpaceToDepth", Type::SpaceToDepth },
        { "Roll", Type::Roll },
        { "LRN", Type::Lrn },
        { "Split", Type::Split },
        { "VariadicSplit", Type::Split },
        { "Concat", Type::Concatenation },
        { "ConvolutionBackpropData", Type::Deconvolution },
        { "GroupConvolutionBackpropData", Type::Deconvolution },
        { "StridedSlice", Type::StridedSlice },
        { "Slice", Type::StridedSlice },
        { "Tile", Type::Tile },
        { "ROIAlign", Type::ROIAlign },
        { "ROIPooling", Type::ROIPooling },
        { "PSROIPooling", Type::PSROIPooling },
        { "DeformablePSROIPooling", Type::PSROIPooling },
        { "Pad", Type::Pad },
        { "Transpose", Type::Transpose },
        { "LSTMCell", Type::RNNCell },
        { "GRUCell", Type::RNNCell },
        { "AUGRUCell", Type::RNNCell },
        { "RNNCell", Type::RNNCell },
        { "LSTMSequence", Type::RNNSeq },
        { "GRUSequence", Type::RNNSeq },
        { "AUGRUSequence", Type::RNNSeq },
        { "RNNSequence", Type::RNNSeq },
        { "FakeQuantize", Type::FakeQuantize },
        { "BinaryConvolution", Type::BinaryConvolution },
        { "DeformableConvolution", Type::DeformableConvolution },
        { "TensorIterator", Type::TensorIterator },
        { "Loop", Type::TensorIterator },
        { "ReadValue", Type::MemoryInput},  // for construction from name ctor, arbitrary name is used
        { "Assign", Type::MemoryOutput },  // for construction from layer ctor
        { "Convert", Type::Convert },
        { "NV12toRGB", Type::ColorConvert },
        { "NV12toBGR", Type::ColorConvert },
        { "I420toRGB", Type::ColorConvert },
        { "I420toBGR", Type::ColorConvert },
        { "MVN", Type::MVN},
        { "NormalizeL2", Type::NormalizeL2},
        { "ScatterUpdate", Type::ScatterUpdate},
        { "ScatterElementsUpdate", Type::ScatterElementsUpdate},
        { "ScatterNDUpdate", Type::ScatterNDUpdate},
        { "Interpolate", Type::Interpolate},
        { "ReduceL1", Type::Reduce},
        { "ReduceL2", Type::Reduce},
        { "ReduceLogicalAnd", Type::Reduce},
        { "ReduceLogicalOr", Type::Reduce},
        { "ReduceMax", Type::Reduce},
        { "ReduceMean", Type::Reduce},
        { "ReduceMin", Type::Reduce},
        { "ReduceProd", Type::Reduce},
        { "ReduceSum", Type::Reduce},
        { "ReduceLogSum", Type::Reduce},
        { "ReduceLogSumExp", Type::Reduce},
        { "ReduceSumSquare", Type::Reduce},
        { "Broadcast", Type::Broadcast},
        { "EmbeddingSegmentsSum", Type::EmbeddingSegmentsSum},
        { "EmbeddingBagPackedSum", Type::EmbeddingBagPackedSum},
        { "EmbeddingBagOffsetsSum", Type::EmbeddingBagOffsetsSum},
        { "Gather", Type::Gather},
        { "GatherElements", Type::GatherElements},
        { "GatherND", Type::GatherND},
        { "GridSample", Type::GridSample},
        { "OneHot", Type::OneHot},
        { "RegionYolo", Type::RegionYolo},
        { "ShuffleChannels", Type::ShuffleChannels},
        { "DFT", Type::DFT},
        { "IDFT", Type::DFT},
        { "RDFT", Type::RDFT},
        { "IRDFT", Type::RDFT},
        { "Abs", Type::Math},
        { "Acos", Type::Math},
        { "Acosh", Type::Math},
        { "Asin", Type::Math},
        { "Asinh", Type::Math},
        { "Atan", Type::Math},
        { "Atanh", Type::Math},
        { "Ceil", Type::Math},
        { "Ceiling", Type::Math},
        { "Cos", Type::Math},
        { "Cosh", Type::Math},
        { "Floor", Type::Math},
        { "HardSigmoid", Type::Math},
        { "If", Type::If},
        { "Neg", Type::Math},
        { "Reciprocal", Type::Math},
        { "Selu", Type::Math},
        { "Sign", Type::Math},
        { "Sin", Type::Math},
        { "Sinh", Type::Math},
        { "SoftPlus", Type::Math},
        { "Softsign", Type::Math},
        { "Tan", Type::Math},
        { "CTCLoss", Type::CTCLoss},
        { "Bucketize", Type::Bucketize},
        { "CTCGreedyDecoder", Type::CTCGreedyDecoder},
        { "CTCGreedyDecoderSeqLen", Type::CTCGreedyDecoderSeqLen},
        { "CumSum", Type::CumSum},
        { "DetectionOutput", Type::DetectionOutput},
        { "ExperimentalDetectronDetectionOutput", Type::ExperimentalDetectronDetectionOutput},
        { "LogSoftmax", Type::LogSoftmax},
        { "TopK", Type::TopK},
        { "GatherTree", Type::GatherTree},
        { "GRN", Type::GRN},
        { "Range", Type::Range},
        { "Proposal", Type::Proposal},
        { "ReorgYolo", Type::ReorgYolo},
        { "ReverseSequence", Type::ReverseSequence},
        { "ExperimentalDetectronTopKROIs", Type::ExperimentalDetectronTopKROIs},
        { "ExperimentalDetectronROIFeatureExtractor", Type::ExperimentalDetectronROIFeatureExtractor},
        { "ExperimentalDetectronPriorGridGenerator", Type::ExperimentalDetectronPriorGridGenerator},
        { "ExperimentalDetectronGenerateProposalsSingleImage", Type::ExperimentalDetectronGenerateProposalsSingleImage},
        { "GenerateProposals", Type::GenerateProposals},
        { "ExtractImagePatches", Type::ExtractImagePatches},
        { "NonMaxSuppression", Type::NonMaxSuppression},
        { "NonMaxSuppressionIEInternal", Type::NonMaxSuppression},
        { "MatrixNms", Type::MatrixNms},
        { "MulticlassNms", Type::MulticlassNms},
        { "MulticlassNmsIEInternal", Type::MulticlassNms},
        { "Reference", Type::Reference},
        { "Subgraph", Type::Subgraph},
        { "PriorBox", Type::PriorBox},
        { "PriorBoxClustered", Type::PriorBoxClustered},
        { "Interaction", Type::Interaction},
        { "MHA", Type::MHA},
        { "Unique", Type::Unique},
        { "Ngram", Type::Ngram},
        { "DimOf", Type::DimOf},
        { "VNode", Type::VNode}
};

Type TypeFromName(const std::string& type) {
    auto itType = type_to_name_tbl.find(type);
    if (type_to_name_tbl.end() != itType) {
        return itType->second;
    } else {
        return Type::Unknown;
    }
}

std::string NameFromType(const Type type) {
#define CASE(_alg) case Type::_alg: return #_alg;
    switch (type) {
        CASE(Generic);
        CASE(Reorder);
        CASE(Input);
        CASE(Output);
        CASE(Eye);
        CASE(Convolution);
        CASE(Deconvolution);
        CASE(Lrn);
        CASE(Pooling);
        CASE(AdaptivePooling);
        CASE(FullyConnected);
        CASE(MatMul);
        CASE(Softmax);
        CASE(Split);
        CASE(Concatenation);
        CASE(StridedSlice);
        CASE(Reshape);
        CASE(ShapeOf);
        CASE(NonZero);
        CASE(Tile);
        CASE(ROIAlign);
        CASE(ROIPooling);
        CASE(PSROIPooling);
        CASE(DepthToSpace);
        CASE(BatchToSpace);
        CASE(Pad);
        CASE(Transpose);
        CASE(SpaceToDepth);
        CASE(SpaceToBatch);
        CASE(MemoryOutput);
        CASE(MemoryInput);
        CASE(RNNSeq);
        CASE(RNNCell);
        CASE(Eltwise);
        CASE(FakeQuantize);
        CASE(BinaryConvolution);
        CASE(DeformableConvolution);
        CASE(MVN);
        CASE(TensorIterator);
        CASE(Convert);
        CASE(ColorConvert);
        CASE(NormalizeL2);
        CASE(ScatterUpdate);
        CASE(ScatterElementsUpdate);
        CASE(ScatterNDUpdate);
        CASE(Interaction);
        CASE(Interpolate);
        CASE(Reduce);
        CASE(Broadcast);
        CASE(EmbeddingSegmentsSum);
        CASE(EmbeddingBagPackedSum);
        CASE(EmbeddingBagOffsetsSum);
        CASE(Gather);
        CASE(GatherElements);
        CASE(GatherND);
        CASE(GridSample);
        CASE(OneHot);
        CASE(RegionYolo);
        CASE(Roll);
        CASE(ShuffleChannels);
        CASE(DFT);
        CASE(RDFT);
        CASE(Math);
        CASE(CTCLoss);
        CASE(Bucketize);
        CASE(CTCGreedyDecoder);
        CASE(CTCGreedyDecoderSeqLen);
        CASE(CumSum);
        CASE(DetectionOutput);
        CASE(ExperimentalDetectronDetectionOutput);
        CASE(If);
        CASE(LogSoftmax);
        CASE(TopK);
        CASE(GatherTree);
        CASE(GRN);
        CASE(Range);
        CASE(Proposal);
        CASE(ReorgYolo);
        CASE(ReverseSequence);
        CASE(ExperimentalDetectronTopKROIs);
        CASE(ExperimentalDetectronROIFeatureExtractor);
        CASE(ExperimentalDetectronPriorGridGenerator);
        CASE(ExperimentalDetectronGenerateProposalsSingleImage);
        CASE(GenerateProposals);
        CASE(ExtractImagePatches);
        CASE(NonMaxSuppression);
        CASE(MatrixNms);
        CASE(MulticlassNms);
        CASE(Reference);
        CASE(Subgraph);
        CASE(PriorBox);
        CASE(PriorBoxClustered)
        CASE(MHA);
        CASE(Unique);
        CASE(Ngram);
        CASE(DimOf);
        CASE(VNode);
        CASE(Unknown);
    }
#undef CASE
    return "Unknown";
}

std::string algToString(const Algorithm alg) {
#define CASE(_alg) case Algorithm::_alg: return #_alg;
    switch (alg) {
        CASE(Default);
        CASE(PoolingMax);
        CASE(PoolingAvg);
        CASE(AdaptivePoolingMax);
        CASE(AdaptivePoolingAvg);
        CASE(ConvolutionCommon);
        CASE(ConvolutionGrouped);
        CASE(DeconvolutionCommon);
        CASE(DeconvolutionGrouped);
        CASE(EltwiseAdd);
        CASE(EltwiseIsFinite);
        CASE(EltwiseIsInf);
        CASE(EltwiseIsNaN);
        CASE(EltwiseMultiply);
        CASE(EltwiseSubtract);
        CASE(EltwiseDivide);
        CASE(EltwiseFloorMod);
        CASE(EltwiseMod);
        CASE(EltwiseMaximum);
        CASE(EltwiseMinimum);
        CASE(EltwiseSquaredDifference);
        CASE(EltwisePowerDynamic);
        CASE(EltwisePowerStatic);
        CASE(EltwiseMulAdd);
        CASE(EltwiseEqual);
        CASE(EltwiseNotEqual);
        CASE(EltwiseGreater);
        CASE(EltwiseGreaterEqual);
        CASE(EltwiseLess);
        CASE(EltwiseLessEqual);
        CASE(EltwiseLogicalAnd);
        CASE(EltwiseLogicalOr);
        CASE(EltwiseLogicalXor);
        CASE(EltwiseLogicalNot);
        CASE(EltwiseRelu);
        CASE(EltwiseGeluErf);
        CASE(EltwiseGeluTanh);
        CASE(EltwiseElu);
        CASE(EltwiseTanh);
        CASE(EltwiseSelect);
        CASE(EltwiseSigmoid);
        CASE(EltwiseAbs);
        CASE(EltwiseSqrt);
        CASE(EltwiseSoftRelu);
        CASE(EltwiseExp);
        CASE(EltwiseClamp);
        CASE(EltwiseSwish);
        CASE(EltwisePrelu);
        CASE(EltwiseMish);
        CASE(EltwiseHswish);
        CASE(EltwiseHsigmoid);
        CASE(EltwiseRoundHalfToEven);
        CASE(EltwiseRoundHalfAwayFromZero);
        CASE(EltwiseErf);
        CASE(EltwiseSoftSign);
        CASE(EltwiseLog);
        CASE(FQCommon);
        CASE(FQQuantization);
        CASE(FQBinarization);
        CASE(ROIPoolingMax);
        CASE(ROIPoolingBilinear);
        CASE(ROIAlignMax);
        CASE(ROIAlignAvg);
        CASE(PSROIPoolingAverage);
        CASE(PSROIPoolingBilinear);
        CASE(PSROIPoolingBilinearDeformable);
        CASE(ReduceL1);
        CASE(ReduceL2);
        CASE(ReduceAnd);
        CASE(ReduceOr);
        CASE(ReduceMax);
        CASE(ReduceMean);
        CASE(ReduceMin);
        CASE(ReduceProd);
        CASE(ReduceSum);
        CASE(ReduceLogSum);
        CASE(ReduceLogSumExp);
        CASE(ReduceSumSquare);
        CASE(MathAbs);
        CASE(MathAcos);
        CASE(MathAcosh);
        CASE(MathAsin);
        CASE(MathAsinh);
        CASE(MathAtan);
        CASE(MathAtanh);
        CASE(MathCeiling);
        CASE(MathCos);
        CASE(MathCosh);
        CASE(MathErf);
        CASE(MathFloor);
        CASE(MathHardSigmoid);
        CASE(MathNegative);
        CASE(MathReciprocal);
        CASE(MathSelu);
        CASE(MathSign);
        CASE(MathSin);
        CASE(MathSinh);
        CASE(MathSoftPlus);
        CASE(MathSoftsign);
        CASE(MathTan);
        CASE(TensorIteratorCommon);
        CASE(TensorIteratorLoop);
        CASE(ColorConvertNV12toRGB);
        CASE(ColorConvertNV12toBGR);
        CASE(ColorConvertI420toRGB);
        CASE(ColorConvertI420toBGR);
    }
#undef CASE
    return "Undefined";
}

}   // namespace intel_cpu
}   // namespace ov

