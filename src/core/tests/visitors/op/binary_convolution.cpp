// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, bin_convolution) {
    NodeBuilder::get_ops().register_factory<op::v1::Convolution>();
    const PartialShape data_batch_shape{1, 1, 5, 5};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 1.0f;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::u1, filters_shape);

    auto conv = make_shared<op::v1::BinaryConvolution>(data_batch,
                                                       filters,
                                                       strides,
                                                       pads_begin,
                                                       pads_end,
                                                       dilations,
                                                       mode,
                                                       pad_value,
                                                       auto_pad);
    NodeBuilder builder(conv, {data_batch, filters});
    auto g_convolution = ov::as_type_ptr<op::v1::BinaryConvolution>(builder.create());

    // attribute count
    const auto expected_attr_count = 7;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), conv->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), conv->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), conv->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), conv->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), conv->get_auto_pad());
    EXPECT_EQ(g_convolution->get_mode(), conv->get_mode());
    EXPECT_EQ(g_convolution->get_pad_value(), conv->get_pad_value());
}
