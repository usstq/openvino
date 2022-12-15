# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino.runtime.opset8 as ov


def test_lrn():
    input_image_shape = (2, 3, 2, 1)
    input_image = np.arange(int(np.prod(input_image_shape))).reshape(input_image_shape).astype("f")
    axes = np.array([1], dtype=np.int64)
    model = ov.lrn(ov.constant(input_image), ov.constant(axes), alpha=1.0, beta=2.0, bias=1.0, size=3)
    assert model.get_type_name() == "LRN"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 3, 2, 1]

    # Test LRN default parameter values
    model = ov.lrn(ov.constant(input_image), ov.constant(axes))
    assert model.get_type_name() == "LRN"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 3, 2, 1]


def test_lrn_factory():
    alpha = 0.0002
    beta = 0.5
    bias = 2.0
    nsize = 3
    axis = np.array([1], dtype=np.int32)
    inputs = ov.parameter((1, 2, 3, 4), name="inputs", dtype=np.float32)
    expected_shape = [1, 2, 3, 4]

    node = ov.lrn(inputs, axis, alpha, beta, bias, nsize)
    assert node.get_type_name() == "LRN"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_batch_norm():
    data = ov.parameter((2, 3), name="data", dtype=np.float32)
    gamma = ov.parameter((3,), name="gamma", dtype=np.float32)
    beta = ov.parameter((3,), name="beta", dtype=np.float32)
    mean = ov.parameter((3,), name="mean", dtype=np.float32)
    variance = ov.parameter((3,), name="variance", dtype=np.float32)
    epsilon = 9.99e-06
    expected_shape = [2, 3]

    node = ov.batch_norm_inference(data, gamma, beta, mean, variance, epsilon)
    assert node.get_type_name() == "BatchNormInference"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_mvn_no_variance():
    data = ov.parameter((1, 3, 3, 3), name="data", dtype=np.float32)
    axes = np.array([2, 3], dtype=np.int64)
    epsilon = 1e-9
    normalize_variance = False
    eps_mode = "outside_sqrt"
    expected_shape = [1, 3, 3, 3]

    node = ov.mvn(data, axes, normalize_variance, epsilon, eps_mode)

    assert node.get_type_name() == "MVN"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_mvn():
    data = ov.parameter((1, 3, 3, 3), name="data", dtype=np.float32)
    axes = np.array([2, 3], dtype=np.int64)
    epsilon = 1e-9
    normalize_variance = True
    eps_mode = "outside_sqrt"
    expected_shape = [1, 3, 3, 3]

    node = ov.mvn(data, axes, normalize_variance, epsilon, eps_mode)

    assert node.get_type_name() == "MVN"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
