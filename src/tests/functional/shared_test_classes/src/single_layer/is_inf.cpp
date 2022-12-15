// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/is_inf.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "ie_plugin_config.hpp"

using namespace ov::test::subgraph;

std::string IsInfLayerTest::getTestCaseName(const testing::TestParamInfo<IsInfParams>& obj) {
    std::vector<InputShape> inputShapes;
    ElementType dataPrc;
    bool detectNegative, detectPositive;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(inputShapes, detectNegative, detectPositive, dataPrc, targetName, additionalConfig) = obj.param;
    std::ostringstream result;

    result << "IS=(";
    for (size_t i = 0lu; i < inputShapes.size(); i++) {
        result << CommonTestUtils::partialShape2str({inputShapes[i].first}) << (i < inputShapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < inputShapes.size(); j++) {
            result << CommonTestUtils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << ")_detectNegative=" << (detectNegative ? "True" : "False") << "_";
    result << "detectPositive=" << (detectPositive ? "True" : "False") << "_";
    result << "dataPrc=" << dataPrc << "_";
    result << "trgDev=" << targetName;

    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto &item : additionalConfig) {
            if (item.second == InferenceEngine::PluginConfigParams::YES)
                result << "_" << item.first << "=" << item.second;
        }
    }

    return result.str();
}

void IsInfLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ElementType dataPrc;
    bool detectNegative, detectPositive;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapes, detectNegative, detectPositive, dataPrc, targetDevice, additionalConfig) = this->GetParam();

    init_input_shapes(shapes);
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto parameters = ngraph::builder::makeDynamicParams(dataPrc, inputDynamicShapes);
    parameters[0]->set_friendly_name("Data");
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(parameters));

    ov::op::v10::IsInf::Attributes attributes {detectNegative, detectPositive};
    auto isInf = std::make_shared<ov::op::v10::IsInf>(paramOuts[0], attributes);
    ov::ResultVector results;
    for (int i = 0; i < isInf->get_output_size(); i++) {
        results.push_back(std::make_shared<ov::op::v0::Result>(isInf->output(i)));
    }

    function = std::make_shared<ov::Model>(results, parameters, "IsInf");
}

void IsInfLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    const auto& input = funcInputs[0];

    int32_t range = std::accumulate(targetInputStaticShapes[0].begin(), targetInputStaticShapes[0].end(), 1u, std::multiplies<uint32_t>());
    auto tensor = utils::create_and_fill_tensor(
           input.get_element_type(), targetInputStaticShapes[0], range, -range / 2, 1);

    auto pointer = tensor.data<element_type_traits<ov::element::Type_t::f32>::value_type>();
    testing::internal::Random random(1);

    for (size_t i = 0; i < range / 2; i++) {
        pointer[random.Generate(range)] = i % 2 == 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    }

    inputs.insert({input.get_node_shared_ptr(), tensor});
}
