// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/profiler.hpp"

#include <pybind11/chrono.h>

#include "openvino/runtime/profiler.hpp"

namespace py = pybind11;

class _Profiler {
    std::shared_ptr<void> prof;
    std::string name;
public:
    _Profiler(std::string name) : name(name) {
        //ov::GetProfilerManager().set_enable(true);
    }
    void enter() {
        prof = ov::Profile(name);
    }
    void exit() {
        prof.reset();
    }
};

void regclass_Profiler(py::module m) {
    py::class_<_Profiler, std::shared_ptr<_Profiler>> cls(m, "Profiler");
    cls.doc() = "openvino.runtime.Profiler create a profiler.";

    cls.def(py::init([](std::string name) {
                return _Profiler(name);
            }),
            py::arg("name"),
            R"(
                Profiler constructor.
            )");

    cls.def(
        "__enter__",
        [](std::shared_ptr<_Profiler>& self) {
            self->enter();
            return self;
        });

    cls.def(
        "__exit__",
        [](std::shared_ptr<_Profiler>& self, py::object exc_type, py::object exc_value, py::object traceback) {
            self->exit();
        });
}
