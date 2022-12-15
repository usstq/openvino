// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/comparison.hpp"

using namespace LayerTestsDefinitions::ComparisonParams;
using namespace ngraph::helpers;

namespace LayerTestsDefinitions {
std::string ComparisonLayerTest::getTestCaseName(const testing::TestParamInfo<ComparisonTestParams> &obj) {
    InputShapesTuple inputShapes;
    InferenceEngine::Precision ngInputsPrecision;
    ComparisonTypes comparisonOpType;
    InputLayerType secondInputType;
    InferenceEngine::Precision ieInPrecision;
    InferenceEngine::Precision ieOutPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes,
             ngInputsPrecision,
             comparisonOpType,
             secondInputType,
             ieInPrecision,
             ieOutPrecision,
             targetName,
             additional_config) = obj.param;
    std::ostringstream results;

    results << "IS0=" << CommonTestUtils::vec2str(inputShapes.first) << "_";
    results << "IS1=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
    results << "inputsPRC=" << ngInputsPrecision.name() << "_";
    results << "comparisonOpType=" << comparisonOpType << "_";
    results << "secondInputType=" << secondInputType << "_";
    if (ieInPrecision != InferenceEngine::Precision::UNSPECIFIED) {
        results << "IEInPRC=" << ieInPrecision.name() << "_";
    }
    if (ieOutPrecision != InferenceEngine::Precision::UNSPECIFIED) {
        results << "IEOutPRC=" << ieOutPrecision.name() << "_";
    }
    results << "targetDevice=" << targetName;
    return results.str();
}

void ComparisonLayerTest::SetUp() {
    InputShapesTuple inputShapes;
    InferenceEngine::Precision ngInputsPrecision;
    InputLayerType secondInputType;
    InferenceEngine::Precision ieInPrecision;
    InferenceEngine::Precision ieOutPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes,
             ngInputsPrecision,
             comparisonOpType,
             secondInputType,
             ieInPrecision,
             ieOutPrecision,
             targetDevice,
             additional_config) = this->GetParam();

    auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(ngInputsPrecision);
    configuration.insert(additional_config.begin(), additional_config.end());

    inPrc = ieInPrecision;
    outPrc = ieOutPrecision;

    auto inputs = ngraph::builder::makeParams(ngInputsPrc, {inputShapes.first});

    auto secondInput = ngraph::builder::makeInputLayer(ngInputsPrc, secondInputType, inputShapes.second);
    if (secondInputType == InputLayerType::PARAMETER) {
        inputs.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(secondInput));
    }

    auto comparisonNode = ngraph::builder::makeComparison(inputs[0], secondInput, comparisonOpType);
    function = std::make_shared<ov::Model>(comparisonNode, inputs, "Comparison");
}

InferenceEngine::Blob::Ptr ComparisonLayerTest::GenerateInput(const InferenceEngine::InputInfo &inputInfo) const {
    auto blob = LayerTestsUtils::LayerTestsCommon::GenerateInput(inputInfo);

    if (comparisonOpType == ComparisonTypes::IS_FINITE || comparisonOpType == ComparisonTypes::IS_NAN) {
        auto *dataPtr = blob->buffer().as<float*>();
        auto range = blob->size();
        testing::internal::Random random(1);

        if (comparisonOpType == ComparisonTypes::IS_FINITE) {
            for (size_t i = 0; i < range / 2; i++) {
                dataPtr[random.Generate(range)] =
                        i % 3 == 0 ? std::numeric_limits<float>::infinity() : i % 3 == 1 ? -std::numeric_limits<float>::infinity() :
                                                                              std::numeric_limits<double>::quiet_NaN();
            }
        } else {
            for (size_t i = 0; i < range / 2; i++) {
                dataPtr[random.Generate(range)] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    return blob;
}

} // namespace LayerTestsDefinitions
