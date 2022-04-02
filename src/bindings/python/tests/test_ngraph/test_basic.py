# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

import openvino.runtime.opset8 as ops
import openvino.runtime as ov

from openvino.runtime.exceptions import UserInputError
from openvino.runtime import Model, PartialShape, Shape, Type, layout_helpers
from openvino.runtime import Strides, AxisVector, Coordinate, CoordinateDiff
from openvino.runtime import Tensor, OVAny
from openvino.pyopenvino import DescriptorTensor
from openvino.runtime.op import Parameter
from tests.runtime import get_runtime
from openvino.runtime.utils.types import get_dtype
from tests.test_ngraph.util import run_op_node


def test_ngraph_function_api():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=Type.f32, name="B")
    parameter_c = ops.parameter(shape, dtype=np.float32, name="C")
    model = (parameter_a + parameter_b) * parameter_c

    assert parameter_a.element_type == Type.f32
    assert parameter_b.element_type == Type.f32
    assert parameter_a.partial_shape == PartialShape([2, 2])
    parameter_a.layout = ov.Layout("NC")
    assert parameter_a.layout == ov.Layout("NC")
    function = Model(model, [parameter_a, parameter_b, parameter_c], "TestFunction")

    function.get_parameters()[1].set_partial_shape(PartialShape([3, 4, 5]))

    ordered_ops = function.get_ordered_ops()
    op_types = [op.get_type_name() for op in ordered_ops]
    assert op_types == ["Parameter", "Parameter", "Parameter", "Add", "Multiply", "Result"]
    assert len(function.get_ops()) == 6
    assert function.get_output_size() == 1
    assert ["A", "B", "C"] == [input.get_node().friendly_name for input in function.inputs]
    assert ["Result"] == [output.get_node().get_type_name() for output in function.outputs]
    assert function.input(0).get_node().friendly_name == "A"
    assert function.output(0).get_node().get_type_name() == "Result"
    assert function.input(tensor_name="A").get_node().friendly_name == "A"
    assert function.output().get_node().get_type_name() == "Result"
    assert function.get_output_op(0).get_type_name() == "Result"
    assert function.get_output_element_type(0) == parameter_a.get_element_type()
    assert list(function.get_output_shape(0)) == [2, 2]
    assert (function.get_parameters()[1].get_partial_shape()) == PartialShape([3, 4, 5])
    assert len(function.get_parameters()) == 3
    results = function.get_results()
    assert len(results) == 1
    assert results[0].get_output_element_type(0) == Type.f32
    assert results[0].get_output_partial_shape(0) == PartialShape([2, 2])
    results[0].layout = ov.Layout("NC")
    assert results[0].layout.to_string() == ov.Layout("NC")
    assert function.get_friendly_name() == "TestFunction"


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        Type.f16,
        Type.f32,
        Type.f64,
        Type.i8,
        Type.i16,
        Type.i32,
        Type.i64,
        Type.u8,
        Type.u16,
        Type.u32,
        Type.u64,
    ],
)
def test_simple_computation_on_ndarrays(dtype):
    runtime = get_runtime()

    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=dtype, name="A")
    parameter_b = ops.parameter(shape, dtype=dtype, name="B")
    parameter_c = ops.parameter(shape, dtype=dtype, name="C")
    model = (parameter_a + parameter_b) * parameter_c
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)

    np_dtype = get_dtype(dtype) if isinstance(dtype, Type) else dtype

    value_a = np.array([[1, 2], [3, 4]], dtype=np_dtype)
    value_b = np.array([[5, 6], [7, 8]], dtype=np_dtype)
    value_c = np.array([[2, 3], [4, 5]], dtype=np_dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[12, 24], [40, 60]], dtype=np_dtype))

    value_a = np.array([[9, 10], [11, 12]], dtype=np_dtype)
    value_b = np.array([[13, 14], [15, 16]], dtype=np_dtype)
    value_c = np.array([[5, 4], [3, 2]], dtype=np_dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[110, 96], [78, 56]], dtype=np_dtype))


def test_serialization():
    dtype = np.float32
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=dtype, name="A")
    parameter_b = ops.parameter(shape, dtype=dtype, name="B")
    parameter_c = ops.parameter(shape, dtype=dtype, name="C")
    model = (parameter_a + parameter_b) * parameter_c

    runtime = get_runtime()
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)
    try:
        serialized = computation.serialize(2)
        serial_json = json.loads(serialized)

        assert serial_json[0]["name"] != ""
        assert 10 == len(serial_json[0]["ops"])
    except Exception:
        pass


def test_broadcast_1():
    input_data = np.array([1, 2, 3], dtype=np.int32)

    new_shape = [3, 3]
    expected = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    result = run_op_node([input_data], ops.broadcast, new_shape)
    assert np.allclose(result, expected)


def test_broadcast_2():
    input_data = np.arange(4, dtype=np.int32)
    new_shape = [3, 4, 2, 4]
    expected = np.broadcast_to(input_data, new_shape)
    result = run_op_node([input_data], ops.broadcast, new_shape)
    assert np.allclose(result, expected)


def test_broadcast_3():
    input_data = np.array([1, 2, 3], dtype=np.int32)
    new_shape = [3, 3]
    axis_mapping = [0]
    expected = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

    result = run_op_node([input_data], ops.broadcast, new_shape, axis_mapping, "EXPLICIT")
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "destination_type, input_data",
    [(bool, np.zeros((2, 2), dtype=np.int32)), ("boolean", np.zeros((2, 2), dtype=np.int32))],
)
def test_convert_to_bool(destination_type, input_data):
    expected = np.array(input_data, dtype=bool)
    result = run_op_node([input_data], ops.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == bool


@pytest.mark.parametrize(
    "destination_type, rand_range, in_dtype, expected_type",
    [
        pytest.param(np.float32, (-8, 8), np.int32, np.float32),
        pytest.param(np.float64, (-16383, 16383), np.int64, np.float64),
        pytest.param("f32", (-8, 8), np.int32, np.float32),
        pytest.param("f64", (-16383, 16383), np.int64, np.float64),
    ],
)
def test_convert_to_float(destination_type, rand_range, in_dtype, expected_type):
    np.random.seed(133391)
    input_data = np.random.randint(*rand_range, size=(2, 2), dtype=in_dtype)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ops.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


@pytest.mark.parametrize(
    "destination_type, expected_type",
    [
        (np.int8, np.int8),
        (np.int16, np.int16),
        (np.int32, np.int32),
        (np.int64, np.int64),
        ("i8", np.int8),
        ("i16", np.int16),
        ("i32", np.int32),
        ("i64", np.int64),
    ],
)
def test_convert_to_int(destination_type, expected_type):
    np.random.seed(133391)
    input_data = (np.ceil(-8 + np.random.rand(2, 3, 4) * 16)).astype(expected_type)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ops.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


@pytest.mark.parametrize(
    "destination_type, expected_type",
    [
        (np.uint8, np.uint8),
        (np.uint16, np.uint16),
        (np.uint32, np.uint32),
        (np.uint64, np.uint64),
        ("u8", np.uint8),
        ("u16", np.uint16),
        ("u32", np.uint32),
        ("u64", np.uint64),
    ],
)
def test_convert_to_uint(destination_type, expected_type):
    np.random.seed(133391)
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16).astype(expected_type)
    expected = np.array(input_data, dtype=expected_type)
    result = run_op_node([input_data], ops.convert, destination_type)
    assert np.allclose(result, expected)
    assert np.array(result).dtype == expected_type


def test_bad_data_shape():
    A = ops.parameter(shape=[2, 2], name="A", dtype=np.float32)
    B = ops.parameter(shape=[2, 2], name="B")
    model = A + B
    runtime = get_runtime()
    computation = runtime.computation(model, A, B)

    value_a = np.array([[1, 2]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    with pytest.raises(UserInputError):
        computation(value_a, value_b)


def test_constant_get_data_bool():
    input_data = np.array([True, False, False, True])
    node = ops.constant(input_data, dtype=np.bool)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_constant_get_data_floating_point(data_type):
    np.random.seed(133391)
    input_data = np.random.randn(2, 3, 4).astype(data_type)
    min_value = -1.0e20
    max_value = 1.0e20
    input_data = min_value + input_data * max_value * data_type(2)
    node = ops.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.int64, np.int32, np.int16, np.int8])
def test_constant_get_data_signed_integer(data_type):
    np.random.seed(133391)
    input_data = np.random.randint(
        np.iinfo(data_type).min, np.iinfo(data_type).max, size=[2, 3, 4], dtype=data_type
    )
    node = ops.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


@pytest.mark.parametrize("data_type", [np.uint64, np.uint32, np.uint16, np.uint8])
def test_constant_get_data_unsigned_integer(data_type):
    np.random.seed(133391)
    input_data = np.random.randn(2, 3, 4).astype(data_type)
    input_data = (
        np.iinfo(data_type).min + input_data * np.iinfo(data_type).max + input_data * np.iinfo(data_type).max
    )
    node = ops.constant(input_data, dtype=data_type)
    retrieved_data = node.get_data()
    assert np.allclose(input_data, retrieved_data)


def test_set_argument():
    runtime = get_runtime()

    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])
    data3 = np.array([7, 8, 9])

    node1 = ops.constant(data1, dtype=np.float32)
    node2 = ops.constant(data2, dtype=np.float32)
    node3 = ops.constant(data3, dtype=np.float32)
    node_add = ops.add(node1, node2)

    # Original arguments
    computation = runtime.computation(node_add)
    output = computation()
    assert np.allclose(data1 + data2, output)

    # Arguments changed by set_argument
    node_add.set_argument(1, node3.output(0))
    output = computation()
    assert np.allclose(data1 + data3, output)

    # Arguments changed by set_argument
    node_add.set_argument(0, node3.output(0))
    output = computation()
    assert np.allclose(data3 + data3, output)

    # Arguments changed by set_argument(OutputVector)
    node_add.set_arguments([node2.output(0), node3.output(0)])
    output = computation()
    assert np.allclose(data2 + data3, output)

    # Arguments changed by set_arguments(NodeVector)
    node_add.set_arguments([node1, node2])
    output = computation()
    assert np.allclose(data1 + data2, output)


def test_clone_model():
    # Create an original model
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")
    model_original = ov.Model(parameter_a + parameter_b, [parameter_a, parameter_b])

    # Make copies of it
    model_copy1 = ov.utils.clone_model(model_original)
    model_copy2 = model_original.clone()

    # Make changes to the copied models' inputs
    model_copy1.reshape({"A": [3, 3], "B": [3, 3]})
    model_copy2.reshape({"A": [3, 3], "B": [3, 3]})

    original_model_shapes = [single_input.get_shape() for single_input in model_original.inputs]
    model_copy1_shapes = [single_input.get_shape() for single_input in model_copy1.inputs]
    model_copy2_shapes = [single_input.get_shape() for single_input in model_copy2.inputs]

    assert original_model_shapes != model_copy1_shapes
    assert original_model_shapes != model_copy2_shapes
    assert model_copy1_shapes == model_copy2_shapes


def test_result():
    node = np.array([[11, 10], [1, 8], [3, 4]], dtype=np.float32)
    result = run_op_node([node], ops.result)
    assert np.allclose(result, node)


def test_node_friendly_name():
    dummy_node = ops.parameter(shape=[1], name="dummy_name")

    assert(dummy_node.friendly_name == "dummy_name")

    dummy_node.set_friendly_name("changed_name")

    assert(dummy_node.get_friendly_name() == "changed_name")

    dummy_node.friendly_name = "new_name"

    assert(dummy_node.get_friendly_name() == "new_name")


def test_node_output():
    input_array = np.array([0, 1, 2, 3, 4, 5])
    splits = 3
    expected_shape = len(input_array) // splits

    input_tensor = ops.constant(input_array, dtype=np.int32)
    axis = ops.constant(0, dtype=np.int64)
    split_node = ops.split(input_tensor, axis, splits)

    split_node_outputs = split_node.outputs()

    assert len(split_node_outputs) == splits
    assert [output_node.get_index() for output_node in split_node_outputs] == [0, 1, 2]
    assert np.equal(
        [output_node.get_element_type() for output_node in split_node_outputs],
        input_tensor.get_element_type(),
    ).all()
    assert np.equal(
        [output_node.get_shape() for output_node in split_node_outputs],
        Shape([expected_shape]),
    ).all()
    assert np.equal(
        [output_node.get_partial_shape() for output_node in split_node_outputs],
        PartialShape([expected_shape]),
    ).all()

    output0 = split_node.output(0)
    output1 = split_node.output(1)
    output2 = split_node.output(2)

    assert [output0.get_index(), output1.get_index(), output2.get_index()] == [0, 1, 2]


def test_node_input_size():
    node = ops.add([1], [2])
    assert node.get_input_size() == 2


def test_node_input_values():
    shapes = [Shape([3]), Shape([3])]
    data1 = np.array([1, 2, 3])
    data2 = np.array([3, 2, 1])

    node = ops.add(data1, data2)

    assert node.get_input_size() == 2

    assert np.equal(
        [input_node.get_shape() for input_node in node.input_values()],
        shapes
    ).all()

    assert np.equal(
        [node.input_value(i).get_shape() for i in range(node.get_input_size())],
        shapes
    ).all()

    assert np.allclose(
        [input_node.get_node().get_vector() for input_node in node.input_values()],
        [data1, data2]
    )

    assert np.allclose(
        [node.input_value(i).get_node().get_vector() for i in range(node.get_input_size())],
        [data1, data2]
    )


def test_node_input_tensor():
    data1 = np.array([[1, 2, 3], [1, 2, 3]])
    data2 = np.array([3, 2, 1])

    node = ops.add(data1, data2)

    inputTensor1 = node.get_input_tensor(0)
    inputTensor2 = node.get_input_tensor(1)

    assert(isinstance(inputTensor1, DescriptorTensor))
    assert(isinstance(inputTensor2, DescriptorTensor))
    assert np.equal(inputTensor1.get_shape(), data1.shape).all()
    assert np.equal(inputTensor2.get_shape(), data2.shape).all()


def test_node_evaluate():
    data1 = np.array([3, 2, 3])
    data2 = np.array([4, 2, 3])
    expected_result = data1 + data2

    data1 = np.ascontiguousarray(data1)
    data2 = np.ascontiguousarray(data2)

    output = np.array([0, 0, 0])
    output = np.ascontiguousarray(output)

    node = ops.add(data1, data2)

    inputTensor1 = Tensor(array=data1, shared_memory=True)
    inputTensor2 = Tensor(array=data2, shared_memory=True)
    inputsTensorVector = [inputTensor1, inputTensor2]

    outputTensorVector = [Tensor(array=output, shared_memory=True)]
    assert node.evaluate(outputTensorVector, inputsTensorVector) is True
    assert np.equal(outputTensorVector[0].data, expected_result).all()


def test_node_input():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")

    model = parameter_a + parameter_b

    model_inputs = model.inputs()

    assert len(model_inputs) == 2
    assert [input_node.get_index() for input_node in model_inputs] == [0, 1]
    assert np.equal(
        [input_node.get_element_type() for input_node in model_inputs],
        model.get_element_type(),
    ).all()
    assert np.equal(
        [input_node.get_shape() for input_node in model_inputs], Shape(shape)
    ).all()
    assert np.equal(
        [input_node.get_partial_shape() for input_node in model_inputs],
        PartialShape(shape),
    ).all()

    input0 = model.input(0)
    input1 = model.input(1)

    assert [input0.get_index(), input1.get_index()] == [0, 1]


def test_node_target_inputs_soruce_output():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")

    model = parameter_a + parameter_b

    out_a = list(parameter_a.output(0).get_target_inputs())[0]
    out_b = list(parameter_b.output(0).get_target_inputs())[0]

    assert out_a.get_node().name == model.name
    assert out_b.get_node().name == model.name
    assert np.equal([out_a.get_shape()], [model.get_output_shape(0)]).all()
    assert np.equal([out_b.get_shape()], [model.get_output_shape(0)]).all()

    in_model0 = model.input(0).get_source_output()
    in_model1 = model.input(1).get_source_output()

    assert in_model0.get_node().name == parameter_a.name
    assert in_model1.get_node().name == parameter_b.name
    assert np.equal([in_model0.get_shape()], [model.get_output_shape(0)]).all()
    assert np.equal([in_model1.get_shape()], [model.get_output_shape(0)]).all()


def test_any():
    any_int = OVAny(32)
    any_str = OVAny("test_text")

    assert any_int.get() == 32
    assert any_str.get() == "test_text"

    any_int.set(777)
    any_str.set("another_text")

    assert any_int.get() == 777
    assert any_str.get() == "another_text"


def test_runtime_info():
    test_shape = PartialShape([1, 1, 1, 1])
    test_type = Type.f32
    test_param = Parameter(test_type, test_shape)
    relu_node = ops.relu(test_param)
    runtime_info = relu_node.get_rt_info()
    runtime_info["affinity"] = "test_affinity"
    relu_node.set_friendly_name("testReLU")
    runtime_info_after = relu_node.get_rt_info()
    assert runtime_info_after["affinity"] == "test_affinity"


def test_multiple_outputs():
    input_shape = [4, 4]
    input_data = np.arange(-8, 8).reshape(input_shape).astype(np.float32)

    expected_output = np.split(input_data, 2, axis=1)[0]
    expected_output[expected_output < 0] = 0

    test_param = ops.parameter(input_shape, dtype=np.float32, name="A")
    split = ops.split(test_param, axis=1, num_splits=2)
    split_first_output = split.output(0)
    relu = ops.relu(split_first_output)

    runtime = get_runtime()
    computation = runtime.computation(relu, test_param)
    output = computation(input_data)

    assert np.equal(output, expected_output).all()


def test_sink_function_ctor():
    input_data = ops.parameter([2, 2], name="input_data", dtype=np.float32)
    rv = ops.read_value(input_data, "var_id_667")
    add = ops.add(rv, input_data, name="MemoryAdd")
    node = ops.assign(add, "var_id_667")
    res = ops.result(add, "res")
    function = Model(results=[res], sinks=[node], parameters=[input_data], name="TestFunction")

    ordered_ops = function.get_ordered_ops()
    op_types = [op.get_type_name() for op in ordered_ops]
    assert op_types == ["Parameter", "ReadValue", "Add", "Assign", "Result"]
    assert len(function.get_ops()) == 5
    assert function.get_output_size() == 1
    assert function.get_output_op(0).get_type_name() == "Result"
    assert function.get_output_element_type(0) == input_data.get_element_type()
    assert list(function.get_output_shape(0)) == [2, 2]
    assert (function.get_parameters()[0].get_partial_shape()) == PartialShape([2, 2])
    assert len(function.get_parameters()) == 1
    assert len(function.get_results()) == 1
    assert function.get_friendly_name() == "TestFunction"


def test_node_version():
    node = ops.add([1], [2])

    assert node.get_version() == 1
    assert node.version == 1


def test_strides_iteration_methods():
    data = np.array([1, 2, 3])
    strides = Strides(data)

    assert len(strides) == data.size
    assert np.equal(strides, data).all()
    assert np.equal([strides[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        strides[i] = data2[i]

    assert np.equal(strides, data2).all()


def test_axis_vector_iteration_methods():
    data = np.array([1, 2, 3])
    axisVector = AxisVector(data)

    assert len(axisVector) == data.size
    assert np.equal(axisVector, data).all()
    assert np.equal([axisVector[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        axisVector[i] = data2[i]

    assert np.equal(axisVector, data2).all()


def test_coordinate_iteration_methods():
    data = np.array([1, 2, 3])
    coordinate = Coordinate(data)

    assert len(coordinate) == data.size
    assert np.equal(coordinate, data).all()
    assert np.equal([coordinate[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        coordinate[i] = data2[i]

    assert np.equal(coordinate, data2).all()


def test_coordinate_diff_iteration_methods():
    data = np.array([1, 2, 3])
    coordinateDiff = CoordinateDiff(data)

    assert len(coordinateDiff) == data.size
    assert np.equal(coordinateDiff, data).all()
    assert np.equal([coordinateDiff[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        coordinateDiff[i] = data2[i]

    assert np.equal(coordinateDiff, data2).all()


def test_get_and_set_layout():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ops.parameter(shape, dtype=np.float32, name="B")

    model = Model(parameter_a + parameter_b, [parameter_a, parameter_b])

    assert layout_helpers.get_layout(model.input(0)) == ov.Layout()
    assert layout_helpers.get_layout(model.input(1)) == ov.Layout()

    layout_helpers.set_layout(model.input(0), ov.Layout("CH"))
    layout_helpers.set_layout(model.input(1), ov.Layout("HW"))

    assert layout_helpers.get_layout(model.input(0)) == ov.Layout("CH")
    assert layout_helpers.get_layout(model.input(1)) == ov.Layout("HW")


def test_layout():
    layout = ov.Layout("NCWH")
    layout2 = ov.Layout("NCWH")
    scalar = ov.Layout.scalar()
    scalar2 = ov.Layout.scalar()

    assert layout == layout2
    assert layout != scalar
    assert scalar == scalar2
    assert scalar2 != layout2

    assert str(scalar) == str(scalar2)
    assert not(scalar.has_name("N"))
    assert not(scalar.has_name("C"))
    assert not(scalar.has_name("W"))
    assert not(scalar.has_name("H"))
    assert not(scalar.has_name("D"))

    assert layout.to_string() == layout2.to_string()
    assert layout.has_name("N")
    assert layout.has_name("C")
    assert layout.has_name("W")
    assert layout.has_name("H")
    assert not(layout.has_name("D"))
    assert layout.get_index_by_name("N") == 0
    assert layout.get_index_by_name("C") == 1
    assert layout.get_index_by_name("W") == 2
    assert layout.get_index_by_name("H") == 3

    layout = ov.Layout("NC?")
    layout2 = ov.Layout("N")
    assert layout != layout2
    assert str(layout) != str(layout2)
    assert layout.has_name("N")
    assert layout.has_name("C")
    assert not(layout.has_name("W"))
    assert not(layout.has_name("H"))
    assert not(layout.has_name("D"))
    assert layout.get_index_by_name("N") == 0
    assert layout.get_index_by_name("C") == 1

    layout = ov.Layout("N...C")
    assert layout.has_name("N")
    assert not(layout.has_name("W"))
    assert not(layout.has_name("H"))
    assert not(layout.has_name("D"))
    assert layout.has_name("C")
    assert layout.get_index_by_name("C") == -1

    layout = ov.Layout()
    assert not(layout.has_name("W"))
    assert not(layout.has_name("H"))
    assert not(layout.has_name("D"))
    assert not(layout.has_name("C"))

    layout = ov.Layout("N...C")
    assert layout == "N...C"
    assert layout != "NC?"


def test_layout_helpers():
    layout = ov.Layout("NCHWD")
    assert(layout_helpers.has_batch(layout))
    assert(layout_helpers.has_channels(layout))
    assert(layout_helpers.has_depth(layout))
    assert(layout_helpers.has_height(layout))
    assert(layout_helpers.has_width(layout))

    assert layout_helpers.batch_idx(layout) == 0
    assert layout_helpers.channels_idx(layout) == 1
    assert layout_helpers.height_idx(layout) == 2
    assert layout_helpers.width_idx(layout) == 3
    assert layout_helpers.depth_idx(layout) == 4

    layout = ov.Layout("N...C")
    assert(layout_helpers.has_batch(layout))
    assert(layout_helpers.has_channels(layout))
    assert not(layout_helpers.has_depth(layout))
    assert not(layout_helpers.has_height(layout))
    assert not (layout_helpers.has_width(layout))

    assert layout_helpers.batch_idx(layout) == 0
    assert layout_helpers.channels_idx(layout) == -1

    with pytest.raises(RuntimeError):
        layout_helpers.height_idx(layout)

    with pytest.raises(RuntimeError):
        layout_helpers.width_idx(layout)

    with pytest.raises(RuntimeError):
        layout_helpers.depth_idx(layout)

    layout = ov.Layout("NC?")
    assert(layout_helpers.has_batch(layout))
    assert(layout_helpers.has_channels(layout))
    assert not(layout_helpers.has_depth(layout))
    assert not(layout_helpers.has_height(layout))
    assert not (layout_helpers.has_width(layout))

    assert layout_helpers.batch_idx(layout) == 0
    assert layout_helpers.channels_idx(layout) == 1

    with pytest.raises(RuntimeError):
        layout_helpers.height_idx(layout)

    with pytest.raises(RuntimeError):
        layout_helpers.width_idx(layout)

    with pytest.raises(RuntimeError):
        layout_helpers.depth_idx(layout)
