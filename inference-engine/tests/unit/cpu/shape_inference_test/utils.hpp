// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

#pragma once

struct TestTensor {
    std::shared_ptr<ngraph::runtime::HostTensor> tensor;
    ov::StaticShape static_shape;

    template <typename T>
    TestTensor(std::initializer_list<T> values) : TestTensor(ov::StaticShape({values.size()}), values) {}

    template <typename T>
    TestTensor(T scalar) : TestTensor(ov::StaticShape({}), {scalar}) {}

    TestTensor(ov::StaticShape shape) : static_shape(shape) {}

    template <typename T>
    TestTensor(ov::StaticShape shape, std::initializer_list<T> values) {
        static_shape = shape;

        ov::Shape s;
        for (auto dim : shape)
            s.push_back(dim.get_length());

        if (values.size() > 0) {
            tensor = std::make_shared<ngraph::runtime::HostTensor>(ov::element::from<T>(), s);
            T* ptr = tensor->get_data_ptr<T>();
            int i = 0;
            for (auto& v : values)
                ptr[i++] = v;
        }
    }
};

// TestTensor can be constructed from initializer_list<T>/int64_t/Shape/Shape+initializer_list
// so each element of inputs can be:
//      {1,2,3,4}                   tensor of shape [4] and values (1,2,3,4)
//      2                           tensor of scalar with value 2
//      Shape{2,2}                  tensor of shape [2,2] and value unknown
//      {Shape{2,2}, {1,2,3,4}}     tensor of shape [2,2] and values (1,2,3,4)
template <typename OP>
static void check_static_shape(std::shared_ptr<OP> op,
                               std::initializer_list<TestTensor> inputs,
                               std::initializer_list<ov::StaticShape> expect_shapes,
                               bool b_expect_throw = false) {
    std::vector<ov::StaticShape> output_shapes;
    std::vector<ov::StaticShape> input_shapes;
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constData;

    int index = 0;
    std::for_each(inputs.begin(), inputs.end(), [&](TestTensor t) {
        input_shapes.push_back(t.static_shape);
        if (t.tensor)
            constData[index] = t.tensor;
        index++;
    });

    if (b_expect_throw) {
        EXPECT_ANY_THROW(shape_infer(op.get(), input_shapes, output_shapes, constData));
    } else {
        output_shapes.resize(expect_shapes.size(), ov::StaticShape{});

        shape_infer(op.get(), input_shapes, output_shapes, constData);

        EXPECT_EQ(output_shapes.size(), expect_shapes.size());
        int id = 0;
        for (auto& shape : expect_shapes) {
            EXPECT_EQ(output_shapes[id], shape);
            id++;
        }
    }
}

template <typename OP>
static void check_output_shape(std::shared_ptr<OP> op, std::initializer_list<ov::PartialShape> expect_shapes) {
    int id = 0;
    EXPECT_EQ(op->outputs().size(), expect_shapes.size());
    for (auto& shape : expect_shapes) {
        EXPECT_EQ(op->get_output_partial_shape(id), shape);
        id++;
    }
}

template <typename OP>
static void check_partial_shape(std::shared_ptr<OP> op,
                                std::vector<ov::PartialShape> input_shapes,
                                std::initializer_list<ov::PartialShape> expect_shapes,
                                bool b_expect_throw = false) {
    std::vector<ov::PartialShape> output_shapes;

    if (b_expect_throw) {
        EXPECT_ANY_THROW(shape_infer(op.get(), input_shapes, output_shapes));
    } else {
        output_shapes.resize(expect_shapes.size(), ov::PartialShape::dynamic());

        shape_infer(op.get(), input_shapes, output_shapes);

        EXPECT_EQ(output_shapes.size(), expect_shapes.size());
        int id = 0;
        for (auto& shape : expect_shapes) {
            EXPECT_EQ(output_shapes[id], shape);
            id++;
        }
    }
}