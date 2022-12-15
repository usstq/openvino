// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/model.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"  // ov::Model
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"  // ov::op::v0::Parameter
#include "openvino/op/sink.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/tensor.hpp"
#include "pyopenvino/graph/ops/result.hpp"
#include "pyopenvino/graph/ops/util/variable.hpp"
#include "pyopenvino/graph/rt_map.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

using PyRTMap = ov::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

static void set_tensor_names(const ov::ParameterVector& parameters) {
    for (const auto& param : parameters) {
        ov::Output<ov::Node> p = param;
        if (p.get_node()->output(0).get_names().empty()) {
            std::unordered_set<std::string> p_names({p.get_node()->get_friendly_name()});
            p.get_node()->output(0).set_names(p_names);
        }
    }
}

static ov::SinkVector cast_to_sink_vector(const std::vector<std::shared_ptr<ov::Node>>& nodes) {
    ov::SinkVector sinks;
    for (const auto& node : nodes) {
        auto sink = std::dynamic_pointer_cast<ov::op::Sink>(node);
        NGRAPH_CHECK(sink != nullptr, "Node {} is not instance of Sink");
        sinks.push_back(sink);
    }
    return sinks;
}

void regclass_graph_Model(py::module m) {
    py::class_<ov::Model, std::shared_ptr<ov::Model>> model(m, "Model", py::module_local());
    model.doc() = "openvino.runtime.Model wraps ov::Model";

    model.def(py::init([](const ov::ResultVector& res,
                          const std::vector<std::shared_ptr<ov::Node>>& nodes,
                          const ov::ParameterVector& params,
                          const std::string& name) {
                  set_tensor_names(params);
                  const auto sinks = cast_to_sink_vector(nodes);
                  return std::make_shared<ov::Model>(res, sinks, params, name);
              }),
              py::arg("results"),
              py::arg("sinks"),
              py::arg("parameters"),
              py::arg("name"),
              R"(
                    Create user-defined Model which is a representation of a model.

                    :param results: List of results.
                    :type results: List[op.Result]
                    :param sinks: List of Nodes to be used as Sinks (e.g. Assign ops).
                    :type sinks: List[openvino.runtime.Node]
                    :param parameters: List of parameters.
                    :type parameters: List[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
                 )");

    model.def(py::init([](const std::vector<std::shared_ptr<ov::Node>>& results,
                          const ov::ParameterVector& parameters,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  return std::make_shared<ov::Model>(results, parameters, name);
              }),
              py::arg("results"),
              py::arg("parameters"),
              py::arg("name") = "",
              R"(
                    Create user-defined Model which is a representation of a model.

                    :param results: List of Nodes to be used as results.
                    :type results: List[openvino.runtime.Node]
                    :param parameters: List of parameters.
                    :type parameters:  List[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
                 )");

    model.def(py::init([](const std::shared_ptr<ov::Node>& result,
                          const ov::ParameterVector& parameters,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  return std::make_shared<ov::Model>(result, parameters, name);
              }),
              py::arg("result"),
              py::arg("parameters"),
              py::arg("name") = "",
              R"(
                    Create user-defined Model which is a representation of a model.

                    :param result: Node to be used as result.
                    :type result: openvino.runtime.Node
                    :param parameters: List of parameters.
                    :type parameters: List[op.Parameter]
                    :param name: String to set as model's friendly name.
                    :type name: str
                 )");

    model.def(
        py::init([](const ov::OutputVector& results, const ov::ParameterVector& parameters, const std::string& name) {
            set_tensor_names(parameters);
            return std::make_shared<ov::Model>(results, parameters, name);
        }),
        py::arg("results"),
        py::arg("parameters"),
        py::arg("name") = "",
        R"(
            Create user-defined Model which is a representation of a model

            :param results: List of outputs.
            :type results: List[openvino.runtime.Output]
            :param parameters: List of parameters.
            :type parameters: List[op.Parameter]
            :param name: String to set as model's friendly name.
            :type name: str
        )");

    model.def(py::init([](const ov::OutputVector& results,
                          const std::vector<std::shared_ptr<ov::Node>>& nodes,
                          const ov::ParameterVector& parameters,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  const auto sinks = cast_to_sink_vector(nodes);
                  return std::make_shared<ov::Model>(results, sinks, parameters, name);
              }),
              py::arg("results"),
              py::arg("sinks"),
              py::arg("parameters"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of outputs.
            :type results: List[openvino.runtime.Output]
            :param sinks: List of Nodes to be used as Sinks (e.g. Assign ops).
            :type sinks: List[openvino.runtime.Node]
            :param name: String to set as model's friendly name.
            :type name: str
            )");
    model.def(py::init([](const ov::ResultVector& results,
                          const std::vector<std::shared_ptr<ov::Node>>& nodes,
                          const ov::ParameterVector& parameters,
                          const ov::op::util::VariableVector& variables,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  const auto sinks = cast_to_sink_vector(nodes);
                  return std::make_shared<ov::Model>(results, sinks, parameters, variables, name);
              }),
              py::arg("results"),
              py::arg("sinks"),
              py::arg("parameters"),
              py::arg("variables"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of results.
            :type results: List[op.Result]
            :param sinks: List of Nodes to be used as Sinks (e.g. Assign ops).
            :type sinks: List[openvino.runtime.Node]
            :param parameters: List of parameters.
            :type parameters: List[op.Parameter]
            :param variables: List of variables.
            :type variables: List[op.util.Variable]
            :param name: String to set as model's friendly name.
            :type name: str
            )");

    model.def(py::init([](const ov::OutputVector& results,
                          const std::vector<std::shared_ptr<ov::Node>>& nodes,
                          const ov::ParameterVector& parameters,
                          const ov::op::util::VariableVector& variables,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  const auto sinks = cast_to_sink_vector(nodes);
                  return std::make_shared<ov::Model>(results, sinks, parameters, variables, name);
              }),
              py::arg("results"),
              py::arg("sinks"),
              py::arg("parameters"),
              py::arg("variables"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of results.
            :type results: List[openvino.runtime.Output]
            :param sinks: List of Nodes to be used as Sinks (e.g. Assign ops).
            :type sinks: List[openvino.runtime.Node]
            :param variables: List of variables.
            :type variables: List[op.util.Variable]
            :param name: String to set as model's friendly name.
            :type name: str
        )");

    model.def(py::init([](const ov::ResultVector& results,
                          const ov::ParameterVector& parameters,
                          const ov::op::util::VariableVector& variables,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  return std::make_shared<ov::Model>(results, parameters, variables, name);
              }),
              py::arg("results"),
              py::arg("parameters"),
              py::arg("variables"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of results.
            :type results: List[op.Result]
            :param parameters: List of parameters.
            :type parameters: List[op.Parameter]
            :param variables: List of variables.
            :type variables: List[op.util.Variable]
            :param name: String to set as model's friendly name.
            :type name: str
        )");

    model.def(py::init([](const ov::OutputVector& results,
                          const ov::ParameterVector& parameters,
                          const ov::op::util::VariableVector& variables,
                          const std::string& name) {
                  set_tensor_names(parameters);
                  return std::make_shared<ov::Model>(results, parameters, variables, name);
              }),
              py::arg("results"),
              py::arg("parameters"),
              py::arg("variables"),
              py::arg("name") = "",
              R"(
            Create user-defined Model which is a representation of a model

            :param results: List of results.
            :type results: List[openvino.runtime.Output]
            :param parameters: List of parameters.
            :type parameters: List[op.Parameter]
            :param name: String to set as model's friendly name.
            :type name: str
        )");

    model.def("validate_nodes_and_infer_types", &ov::Model::validate_nodes_and_infer_types);

    model.def(
        "reshape",
        [](ov::Model& self, const ov::PartialShape& partial_shape) {
            self.reshape(partial_shape);
        },
        py::call_guard<py::gil_scoped_release>(),
        py::arg("partial_shape"),
        R"(
                Reshape model input.

                GIL is released while running this function.

                :param partial_shape: New shape.
                :type partial_shape: openvino.runtime.PartialShape
                :return : void
             )");

    model.def(
        "reshape",
        [](ov::Model& self, const py::list& partial_shape) {
            ov::PartialShape new_shape(Common::partial_shape_from_list(partial_shape));
            py::gil_scoped_release release;
            self.reshape(new_shape);
        },
        py::arg("partial_shape"),
        R"(
                Reshape model input.

                GIL is released while running this function.

                :param partial_shape: New shape.
                :type partial_shape: list
                :return : void
             )");

    model.def(
        "reshape",
        [](ov::Model& self, const py::tuple& partial_shape) {
            ov::PartialShape new_shape(Common::partial_shape_from_list(partial_shape.cast<py::list>()));
            py::gil_scoped_release release;
            self.reshape(new_shape);
        },
        py::arg("partial_shape"),
        R"(
                Reshape model input.

                GIL is released while running this function.

                :param partial_shape: New shape.
                :type partial_shape: tuple
                :return : void
             )");

    model.def(
        "reshape",
        [](ov::Model& self, const std::string& partial_shape) {
            self.reshape(ov::PartialShape(partial_shape));
        },
        py::call_guard<py::gil_scoped_release>(),
        py::arg("partial_shape"),
        R"(
                Reshape model input.

                GIL is released while running this function.

                :param partial_shape: New shape.
                :type partial_shape: str
                :return : void
             )");

    model.def(
        "reshape",
        [](ov::Model& self, const py::dict& partial_shapes) {
            std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
            for (const auto& item : partial_shapes) {
                std::pair<ov::Output<ov::Node>, ov::PartialShape> new_shape;
                // check keys
                if (py::isinstance<py::int_>(item.first)) {
                    new_shape.first = self.input(item.first.cast<size_t>());
                } else if (py::isinstance<py::str>(item.first)) {
                    new_shape.first = self.input(item.first.cast<std::string>());
                } else if (py::isinstance<ov::Output<ov::Node>>(item.first)) {
                    new_shape.first = item.first.cast<ov::Output<ov::Node>>();
                } else {
                    throw py::type_error("Incorrect key type " + std::string(item.first.get_type().str()) +
                                         " to reshape a model, expected keys as openvino.runtime.Output, int or str.");
                }
                // check values
                if (py::isinstance<ov::PartialShape>(item.second)) {
                    new_shape.second = item.second.cast<ov::PartialShape>();
                } else if (py::isinstance<py::list>(item.second) || py::isinstance<py::tuple>(item.second)) {
                    new_shape.second = Common::partial_shape_from_list(item.second.cast<py::list>());
                } else if (py::isinstance<py::str>(item.second)) {
                    new_shape.second = ov::PartialShape(item.second.cast<std::string>());
                } else {
                    throw py::type_error(
                        "Incorrect value type " + std::string(item.second.get_type().str()) +
                        " to reshape a model, expected values as openvino.runtime.PartialShape, str, list or tuple.");
                }
                new_shapes.insert(new_shape);
            }
            py::gil_scoped_release release;
            self.reshape(new_shapes);
        },
        py::arg("partial_shapes"),
        R"( Reshape model inputs.

            The allowed types of keys in the `partial_shapes` dictionary are:

            (1) `int`, input index
            (2) `str`, input tensor name
            (3) `openvino.runtime.Output`

            The allowed types of values in the `partial_shapes` are:

            (1) `openvino.runtime.PartialShape`
            (2) `list` consisting of dimensions
            (3) `tuple` consisting of dimensions
            (4) `str`, string representation of `openvino.runtime.PartialShape`

            When list or tuple are used to describe dimensions, each dimension can be written in form:

            (1) non-negative `int` which means static value for the dimension
            (2) `[min, max]`, dynamic dimension where `min` specifies lower bound and `max` specifies upper bound; the range includes both `min` and `max`; using `-1` for `min` or `max` means no known bound
            (3) `(min, max)`, the same as above
            (4) `-1` is a dynamic dimension without known bounds
            (4) `openvino.runtime.Dimension`
            (5) `str` using next syntax:
                '?' - to define fully dynamic dimension
                '1' - to define dimension which length is 1
                '1..10' - to define bounded dimension
                '..10' or '1..' to define dimension with only lower or only upper limit

            Reshape model input.

            GIL is released while running this function.

            :param partial_shapes: New shapes.
            :type partial_shapes: Dict[keys, values]
        )");

    model.def("get_output_size",
              &ov::Model::get_output_size,
              R"(
                    Return the number of outputs for the model.

                    :return: Number of outputs.
                    :rtype: int
                 )");
    model.def("get_ops",
              &ov::Model::get_ops,
              R"(
                    Return ops used in the model.

                    :return: List of Nodes representing ops used in model.
                    :rtype: List[openvino.runtime.Node]
                 )");
    model.def("get_ordered_ops",
              &ov::Model::get_ordered_ops,
              R"(
                    Return ops used in the model in topological order.

                    :return: List of sorted Nodes representing ops used in model.
                    :rtype: List[openvino.runtime.Node]
                 )");
    model.def("get_output_op",
              &ov::Model::get_output_op,
              py::arg("index"),
              R"(
                    Return the op that generates output i

                    :param index: output index
                    :type index: output index
                    :return: Node object that generates output i
                    :rtype: openvino.runtime.Node
                )");
    model.def("get_output_element_type",
              &ov::Model::get_output_element_type,
              py::arg("index"),
              R"(
                    Return the element type of output i

                    :param index: output index
                    :type index: int
                    :return: Type object of output i
                    :rtype: openvino.runtime.Type
                 )");
    model.def("get_output_shape",
              &ov::Model::get_output_shape,
              py::arg("index"),
              R"(
                    Return the shape of element i

                    :param index: element index
                    :type index: int
                    :return: Shape object of element i
                    :rtype: openvino.runtime.Shape
                 )");
    model.def("get_output_partial_shape",
              &ov::Model::get_output_partial_shape,
              py::arg("index"),
              R"(
                    Return the partial shape of element i

                    :param index: element index
                    :type index: int
                    :return: PartialShape object of element i
                    :rtype: openvino.runtime.PartialShape
                 )");
    model.def("get_parameters",
              &ov::Model::get_parameters,
              R"(
                    Return the model parameters.
                    
                    :return: ParameterVector containing model parameters.
                    :rtype: ParameterVector
                 )");
    model.def("get_results",
              &ov::Model::get_results,
              R"(
                    Return a list of model outputs.

                    :return: ResultVector containing model parameters.
                    :rtype: ResultVector
                 )");
    model.def("get_result",
              &ov::Model::get_result,
              R"(
                    Return single result.

                    :return: Node object representing result.
                    :rtype: openvino.runtime.Node
                 )");
    model.def("get_result_index",
              (int64_t(ov::Model::*)(const ov::Output<ov::Node>&) const) & ov::Model::get_result_index,
              py::arg("value"),
              R"(
                    Return index of result.

                    Return -1 if `value` not matched.

                    :param value: Output containing Node
                    :type value: openvino.runtime.Output
                    :return: Index for value referencing it.
                    :rtype: int
                 )");
    model.def("get_result_index",
              (int64_t(ov::Model::*)(const ov::Output<const ov::Node>&) const) & ov::Model::get_result_index,
              py::arg("value"),
              R"(
                    Return index of result.

                    Return -1 if `value` not matched.

                    :param value: Output containing Node
                    :type value: openvino.runtime.Output
                    :return: Index for value referencing it.
                    :rtype: int
                 )");

    model.def("get_name",
              &ov::Model::get_name,
              R"(
                    Get the unique name of the model.

                    :return: String with a name of the model.
                    :rtype: str
                 )");
    model.def("get_friendly_name",
              &ov::Model::get_friendly_name,
              R"(
                    Gets the friendly name for a model. If no
                    friendly name has been set via set_friendly_name
                    then the model's unique name is returned.

                    :return: String with a friendly name of the model.
                    :rtype: str
                 )");
    model.def("set_friendly_name",
              &ov::Model::set_friendly_name,
              py::arg("name"),
              R"(
                    Sets a friendly name for a model. This does
                    not overwrite the unique name of the model and
                    is retrieved via get_friendly_name(). Used mainly
                    for debugging.

                    :param name: String to set as the friendly name.
                    :type name: str
                 )");
    model.def("is_dynamic",
              &ov::Model::is_dynamic,
              R"(
                    Returns true if any of the op's defined in the model
                    contains partial shape.

                    :rtype: bool
                 )");
    model.def("input", (ov::Output<ov::Node>(ov::Model::*)()) & ov::Model::input);

    model.def("input", (ov::Output<ov::Node>(ov::Model::*)(size_t)) & ov::Model::input, py::arg("index"));

    model.def("input",
              (ov::Output<ov::Node>(ov::Model::*)(const std::string&)) & ov::Model::input,
              py::arg("tensor_name"));

    model.def("input", (ov::Output<const ov::Node>(ov::Model::*)() const) & ov::Model::input);

    model.def("input", (ov::Output<const ov::Node>(ov::Model::*)(size_t) const) & ov::Model::input, py::arg("index"));

    model.def("input",
              (ov::Output<const ov::Node>(ov::Model::*)(const std::string&) const) & ov::Model::input,
              py::arg("tensor_name"));

    model.def("output", (ov::Output<ov::Node>(ov::Model::*)()) & ov::Model::output);

    model.def("output", (ov::Output<ov::Node>(ov::Model::*)(size_t)) & ov::Model::output, py::arg("index"));

    model.def("output",
              (ov::Output<ov::Node>(ov::Model::*)(const std::string&)) & ov::Model::output,
              py::arg("tensor_name"));

    model.def("output", (ov::Output<const ov::Node>(ov::Model::*)() const) & ov::Model::output);

    model.def("output", (ov::Output<const ov::Node>(ov::Model::*)(size_t) const) & ov::Model::output, py::arg("index"));

    model.def("output",
              (ov::Output<const ov::Node>(ov::Model::*)(const std::string&) const) & ov::Model::output,
              py::arg("tensor_name"));

    model.def(
        "add_outputs",
        [](ov::Model& self, py::handle& outputs) {
            int i = 0;
            std::vector<ov::Output<ov::Node>> new_outputs;
            py::list _outputs;
            if (!py::isinstance<py::list>(outputs)) {
                if (py::isinstance<py::str>(outputs)) {
                    _outputs.append(outputs.cast<py::str>());
                } else if (py::isinstance<py::tuple>(outputs)) {
                    _outputs.append(outputs.cast<py::tuple>());
                } else if (py::isinstance<ov::Output<ov::Node>>(outputs)) {
                    _outputs.append(outputs.cast<ov::Output<ov::Node>>());
                } else {
                    throw py::type_error("Incorrect type of a value to add as output.");
                }
            } else {
                _outputs = outputs.cast<py::list>();
            }

            for (py::handle output : _outputs) {
                ov::Output<ov::Node> out;
                if (py::isinstance<py::str>(_outputs[i])) {
                    out = self.add_output(output.cast<std::string>());
                } else if (py::isinstance<py::tuple>(output)) {
                    py::tuple output_tuple = output.cast<py::tuple>();
                    out = self.add_output(output_tuple[0].cast<std::string>(), output_tuple[1].cast<int>());
                } else if (py::isinstance<ov::Output<ov::Node>>(_outputs[i])) {
                    out = self.add_output(output.cast<ov::Output<ov::Node>>());
                } else {
                    throw py::type_error("Incorrect type of a value to add as output at index " + std::to_string(i) +
                                         ".");
                }
                new_outputs.emplace_back(out);
                i++;
            }
            return new_outputs;
        },
        py::arg("outputs"));

    model.def("replace_parameter",
              &ov::Model::replace_parameter,
              py::arg("parameter_index"),
              py::arg("parameter"),
              R"(
                    Replace the `parameter_index` parameter of the model with `parameter`

                    All users of the `parameter_index` parameter are redirected to `parameter` , and the
                    `parameter_index` entry in the model parameter list is replaced with `parameter`

                    :param parameter_index: The index of the parameter to replace.
                    :type parameter_index: int
                    :param parameter: The parameter to substitute for the `parameter_index` parameter.
                    :type parameter: op.Parameter
        )");

    model.def(
        "get_parameter_index",
        (int64_t(ov::Model::*)(const std::shared_ptr<ov::op::v0::Parameter>&) const) & ov::Model::get_parameter_index,
        py::arg("parameter"),
        R"(
                    Return the index position of `parameter`

                    Return -1 if parameter not matched.

                    :param parameter: Parameter, which index is to be found.
                    :type parameter: op.Parameter
                    :return: Index for parameter
                    :rtype: int
                 )");

    model.def(
        "evaluate",
        [](ov::Model& self,
           ov::TensorVector& output_tensors,
           const ov::TensorVector& input_tensors,
           PyRTMap evaluation_context) -> bool {
            return self.evaluate(output_tensors, input_tensors, evaluation_context);
        },
        py::arg("output_tensors"),
        py::arg("input_tensors"),
        py::arg("evaluation_context") = PyRTMap(),
        R"(
            Evaluate the model on inputs, putting results in outputs

            :param output_tensors: Tensors for the outputs to compute. One for each result
            :type output_tensors: List[openvino.runtime.Tensor]
            :param input_tensors: Tensors for the inputs. One for each inputs.
            :type input_tensors: List[openvino.runtime.Tensor]
            :param evaluation_context: Storage of additional settings and attributes that can be used
                                       when evaluating the model. This additional information can be
                                       shared across nodes.
            :type evaluation_context: openvino.runtime.RTMap
            :rtype: bool
        )");

    model.def("clone",
              &ov::Model::clone,
              R"(
            Return a copy of self.
            :return: A copy of self.
            :rtype: openvino.runtime.Model
        )");

    model.def("__repr__", [](const ov::Model& self) {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();

        auto inputs_str = Common::docs::container_to_string(self.inputs(), ",\n");
        auto outputs_str = Common::docs::container_to_string(self.outputs(), ",\n");

        return "<" + class_name + ": '" + self.get_friendly_name() + "'\ninputs[\n" + inputs_str + "\n]\noutputs[\n" +
               outputs_str + "\n]>";
    });
    model.def("get_rt_info",
              (PyRTMap & (ov::Model::*)()) & ov::Model::get_rt_info,
              py::return_value_policy::reference_internal,
              R"(
                Returns PyRTMap which is a dictionary of user defined runtime info.

                :return: A dictionary of user defined data.
                :rtype: openvino.runtime.RTMap
             )");
    model.def(
        "get_rt_info",
        [](const ov::Model& self, const py::list& path) -> py::object {
            std::vector<std::string> cpp_args(path.size());
            for (size_t i = 0; i < path.size(); i++) {
                cpp_args[i] = path[i].cast<std::string>();
            }
            return Common::utils::from_ov_any(self.get_rt_info<ov::Any>(cpp_args));
        },
        py::arg("path"),
        R"(
                Returns runtime attribute.

                :param path: List of strings which defines a path to runtime info.
                :type path: List[str]

                :return: A runtime attribute.
                :rtype: Any
             )");
    model.def(
        "get_rt_info",
        [](const ov::Model& self, const py::str& path) -> py::object {
            return Common::utils::from_ov_any(self.get_rt_info<ov::Any>(path.cast<std::string>()));
        },
        py::arg("path"),
        R"(
                Returns runtime attribute.

                :param path: List of strings which defines a path to runtime info.
                :type path: str

                :return: A runtime attribute.
                :rtype: Any
             )");
    model.def(
        "has_rt_info",
        [](const ov::Model& self, const py::list& path) -> bool {
            // FIXME: understand why has_rt_info causes Python crash
            try {
                std::vector<std::string> cpp_args(path.size());
                for (size_t i = 0; i < path.size(); i++) {
                    cpp_args[i] = path[i].cast<std::string>();
                }
                self.get_rt_info<ov::Any>(cpp_args);
                return true;
            } catch (ov::Exception&) {
                return false;
            }
        },
        py::arg("path"),
        R"(
                Checks if given path exists in runtime info of the model.

                :param path: List of strings which defines a path to runtime info.
                :type path: List[str]

                :return: `True` if path exists, otherwise `False`.
                :rtype: bool
             )");
    model.def(
        "has_rt_info",
        [](const ov::Model& self, const py::str& path) -> bool {
            return self.has_rt_info(path.cast<std::string>());
        },
        py::arg("path"),
        R"(
                Checks if given path exists in runtime info of the model.

                :param path: List of strings which defines a path to runtime info.
                :type path: str

                :return: `True` if path exists, otherwise `False`.
                :rtype: bool
             )");
    model.def(
        "set_rt_info",
        [](ov::Model& self, const py::object& obj, const py::list& path) -> void {
            std::vector<std::string> cpp_args(path.size());
            for (size_t i = 0; i < path.size(); i++) {
                cpp_args[i] = path[i].cast<std::string>();
            }
            self.set_rt_info<ov::Any>(py_object_to_any(obj), cpp_args);
        },
        py::arg("obj"),
        py::arg("path"),
        R"(
                Add value inside runtime info

                :param obj: value for the runtime info
                :type obj: py:object
                :param path: List of strings which defines a path to runtime info.
                :type path: List[str]
             )");
    model.def(
        "set_rt_info",
        [](ov::Model& self, const py::object& obj, const py::str& path) -> void {
            self.set_rt_info<ov::Any>(py_object_to_any(obj), path.cast<std::string>());
        },
        py::arg("obj"),
        py::arg("path"),
        R"(
                Add value inside runtime info

                :param obj: value for the runtime info
                :type obj: Any
                :param path: String which defines a path to runtime info.
                :type path: str
             )");

    model.def_property_readonly("inputs", (std::vector<ov::Output<ov::Node>>(ov::Model::*)()) & ov::Model::inputs);
    model.def_property_readonly("outputs", (std::vector<ov::Output<ov::Node>>(ov::Model::*)()) & ov::Model::outputs);
    model.def_property_readonly("name", &ov::Model::get_name);
    model.def_property_readonly("rt_info",
                                (PyRTMap & (ov::Model::*)()) & ov::Model::get_rt_info,
                                py::return_value_policy::reference_internal);
    model.def_property("friendly_name", &ov::Model::get_friendly_name, &ov::Model::set_friendly_name);
}
