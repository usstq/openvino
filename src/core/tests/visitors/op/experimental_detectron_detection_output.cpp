// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset6.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

using Attrs = opset6::ExperimentalDetectronDetectionOutput::Attributes;
using ExperimentalDetection = opset6::ExperimentalDetectronDetectionOutput;

TEST(attributes, detectron_detection_output) {
    NodeBuilder::get_ops().register_factory<ExperimentalDetection>();

    Attrs attrs;
    attrs.class_agnostic_box_regression = false;
    attrs.deltas_weights = {10.0f, 10.0f, 5.0f, 5.0f};
    attrs.max_delta_log_wh = 4.135166645050049f;
    attrs.max_detections_per_image = 100;
    attrs.nms_threshold = 0.5f;
    attrs.num_classes = 81;
    attrs.post_nms_count = 2000;
    attrs.score_threshold = 0.05000000074505806f;

    auto rois = std::make_shared<op::Parameter>(element::f32, Shape{1000, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{1000, 324});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{1000, 81});
    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{1, 3});

    auto detection = std::make_shared<ExperimentalDetection>(rois, deltas, scores, im_info, attrs);

    NodeBuilder builder(detection, {rois, deltas, scores, im_info});

    auto g_detection = ov::as_type_ptr<ExperimentalDetection>(builder.create());

    const auto expected_attr_count = 8;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_detection->get_attrs().class_agnostic_box_regression,
              detection->get_attrs().class_agnostic_box_regression);
    EXPECT_EQ(g_detection->get_attrs().deltas_weights, detection->get_attrs().deltas_weights);
    EXPECT_EQ(g_detection->get_attrs().max_delta_log_wh, detection->get_attrs().max_delta_log_wh);
    EXPECT_EQ(g_detection->get_attrs().max_detections_per_image, detection->get_attrs().max_detections_per_image);
    EXPECT_EQ(g_detection->get_attrs().nms_threshold, detection->get_attrs().nms_threshold);
    EXPECT_EQ(g_detection->get_attrs().num_classes, detection->get_attrs().num_classes);
    EXPECT_EQ(g_detection->get_attrs().post_nms_count, detection->get_attrs().post_nms_count);
    EXPECT_EQ(g_detection->get_attrs().score_threshold, detection->get_attrs().score_threshold);
}
