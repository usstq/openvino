# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino
import ngraph as ng
from openvino.inference_engine import IECore, IENetwork
from ngraph.impl import Function, Shape, Type
from ngraph.impl.op import Parameter

from typing import Tuple, Union, List
import numpy as np


def get_test_function():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], "test")
    assert func is not None
    return func


def generate_image(shape: Tuple = (1, 3, 32, 32), dtype: Union[str, np.dtype] = "float32") -> np.array:
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)


def generate_model(input_shape: List[int]) -> openvino.inference_engine.ExecutableNetwork:
    param = ng.parameter(input_shape, np.float32, name="parameter")
    relu = ng.relu(param, name="relu")
    func = Function([relu], [param], "test")
    func.get_ordered_ops()[2].friendly_name = "friendly"

    core = IECore()
    caps = Function.to_capsule(func)
    cnnNetwork = IENetwork(caps)
    return core.load_network(cnnNetwork, "CPU", {})
