// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/insert_movebroadcast.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, InsertBroadcastMove) {
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 3});
        auto add = std::make_shared<opset1::Add>(data0, data1);
        function = std::make_shared<Function>(NodeVector{add}, ParameterVector{data0, data1});

        manager.register_pass<snippets::pass::InsertMoveBroadcast>();
    }
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 3});
        auto move0 = std::make_shared<snippets::isa::BroadcastMove>(data0, Shape{1, 2, 3});
        auto move1 = std::make_shared<snippets::isa::BroadcastMove>(data1, Shape{1, 2, 3});
        auto add = std::make_shared<opset1::Add>(move0, move1);
        function_ref = std::make_shared<Function>(NodeVector{add}, ParameterVector{data0, data1});
    }
}
