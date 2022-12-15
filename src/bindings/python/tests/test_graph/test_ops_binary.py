# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator

import numpy as np
import pytest

import openvino.runtime.opset8 as ov


@pytest.mark.parametrize(
    ("graph_api_helper", "numpy_function"),
    [
        (ov.add, np.add),
        (ov.divide, np.divide),
        (ov.multiply, np.multiply),
        (ov.subtract, np.subtract),
        (ov.minimum, np.minimum),
        (ov.maximum, np.maximum),
        (ov.mod, np.mod),
        (ov.equal, np.equal),
        (ov.not_equal, np.not_equal),
        (ov.greater, np.greater),
        (ov.greater_equal, np.greater_equal),
        (ov.less, np.less),
        (ov.less_equal, np.less_equal),
    ],
)
def test_binary_op(graph_api_helper, numpy_function):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)
    parameter_b = ov.parameter(shape, name="B", dtype=np.float32)

    model = graph_api_helper(parameter_a, parameter_b)

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)


@pytest.mark.parametrize(
    ("graph_api_helper", "numpy_function"),
    [
        (ov.add, np.add),
        (ov.divide, np.divide),
        (ov.multiply, np.multiply),
        (ov.subtract, np.subtract),
        (ov.minimum, np.minimum),
        (ov.maximum, np.maximum),
        (ov.mod, np.mod),
        (ov.equal, np.equal),
        (ov.not_equal, np.not_equal),
        (ov.greater, np.greater),
        (ov.greater_equal, np.greater_equal),
        (ov.less, np.less),
        (ov.less_equal, np.less_equal),
    ],
)
def test_binary_op_with_scalar(graph_api_helper, numpy_function):
    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = graph_api_helper(parameter_a, value_b)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)


@pytest.mark.parametrize(
    ("graph_api_helper", "numpy_function"),
    [(ov.logical_and, np.logical_and), (ov.logical_or, np.logical_or), (ov.logical_xor, np.logical_xor)],
)
def test_binary_logical_op(graph_api_helper, numpy_function):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=bool)
    parameter_b = ov.parameter(shape, name="B", dtype=bool)

    model = graph_api_helper(parameter_a, parameter_b)

    value_a = np.array([[True, False], [False, True]], dtype=bool)
    value_b = np.array([[False, True], [False, True]], dtype=bool)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)


@pytest.mark.parametrize(
    ("graph_api_helper", "numpy_function"),
    [(ov.logical_and, np.logical_and), (ov.logical_or, np.logical_or), (ov.logical_xor, np.logical_xor)],
)
def test_binary_logical_op_with_scalar(graph_api_helper, numpy_function):
    value_a = np.array([[True, False], [False, True]], dtype=bool)
    value_b = np.array([[False, True], [False, True]], dtype=bool)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=bool)

    model = graph_api_helper(parameter_a, value_b)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)


@pytest.mark.parametrize(
    ("operator", "numpy_function"),
    [
        (operator.add, np.add),
        (operator.sub, np.subtract),
        (operator.mul, np.multiply),
        (operator.truediv, np.divide),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.lt, np.less),
        (operator.le, np.less_equal),
    ],
)
def test_binary_operators(operator, numpy_function):
    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[4, 5], [1, 7]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)


@pytest.mark.parametrize(
    ("operator", "numpy_function"),
    [
        (operator.add, np.add),
        (operator.sub, np.subtract),
        (operator.mul, np.multiply),
        (operator.truediv, np.divide),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.lt, np.less),
        (operator.le, np.less_equal),
    ],
)
def test_binary_operators_with_scalar(operator, numpy_function):
    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)


def test_multiply():
    param_a = np.arange(48, dtype=np.int32).reshape((8, 1, 6, 1))
    param_b = np.arange(35, dtype=np.int32).reshape((7, 1, 5))

    expected_shape = np.multiply(param_a, param_b).shape
    node = ov.multiply(param_a, param_b)

    assert node.get_type_name() == "Multiply"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(expected_shape)


def test_power_v1():
    param_a = np.arange(48, dtype=np.float32).reshape((8, 1, 6, 1))
    param_b = np.arange(20, dtype=np.float32).reshape((4, 1, 5))

    expected_shape = np.power(param_a, param_b).shape
    node = ov.power(param_a, param_b)

    assert node.get_type_name() == "Power"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(expected_shape)
