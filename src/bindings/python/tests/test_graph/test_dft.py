# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime.opset9 as ov
import numpy as np


def build_fft_input_data():
    np.random.seed(202104)
    return np.random.uniform(0, 1, (2, 10, 10, 2)).astype(np.float32)


def test_dft_1d():
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([2], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes)
    np_results = np.fft.fft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                            axis=2).astype(np.complex64)
    expected_shape = list(np.stack((np_results.real, np_results.imag), axis=-1).shape)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == expected_shape


def test_dft_2d():
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([1, 2], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes)
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                             axes=[1, 2]).astype(np.complex64)
    expected_shape = list(np.stack((np_results.real, np_results.imag), axis=-1).shape)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == expected_shape


def test_dft_3d():
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 1, 2], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes)
    np_results = np.fft.fftn(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                             axes=[0, 1, 2]).astype(np.complex64)
    expected_shape = list(np.stack((np_results.real, np_results.imag), axis=-1).shape)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == expected_shape


def test_dft_1d_signal_size():
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([-2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([20], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    np_results = np.fft.fft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), n=20,
                            axis=-2).astype(np.complex64)
    expected_shape = list(np.stack((np_results.real, np_results.imag), axis=-1).shape)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == expected_shape


def test_dft_2d_signal_size_1():
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), s=[4, 5],
                             axes=[0, 2]).astype(np.complex64)
    expected_shape = list(np.stack((np_results.real, np_results.imag), axis=-1).shape)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == expected_shape


def test_dft_2d_signal_size_2():
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([1, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), s=[4, 5],
                             axes=[1, 2]).astype(np.complex64)
    expected_shape = list(np.stack((np_results.real, np_results.imag), axis=-1).shape)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == expected_shape


def test_dft_3d_signal_size():
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 1, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5, 16], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    np_results = np.fft.fftn(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                             s=[4, 5, 16], axes=[0, 1, 2]).astype(np.complex64)
    expected_shape = list(np.stack((np_results.real, np_results.imag), axis=-1).shape)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == expected_shape
