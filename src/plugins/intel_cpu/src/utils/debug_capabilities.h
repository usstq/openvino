// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>

#include <onednn/dnnl.h>
#include <dnnl_debug.h>
#include "onednn/iml_type_mapper.h"
#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"
#include "edge.h"

namespace ov {
namespace intel_cpu {


// OV_CPU_DEBUG_LOG controls DEBUG_LOGs to output
//
// positive filter: enables patterns in filter
//   [+]foo;bar:line2;  enables  "foo:*" and "bar:line2"
//   -                  enables all debug log
//
// negative filter: disable patterns in filter
//   -f1;f2:l;          disables  "foo:*" and "bar:line2"
//
class DebugLogEnabled {
    bool enabled;
    std::string tag;

public:
    DebugLogEnabled(const char* file, const char* func, int line, const char* name = nullptr);

    const std::string & get_tag() const { return tag; }
    operator bool() const { return enabled; }
    void break_at(const std::string & log);
};

class NodeDesc;
class MemoryDesc;
class Node;
class Edge;

class PrintableModel {
public:
    PrintableModel(const ov::Model& model, std::string tag = "", std::string prefix = "") : model(model), tag(tag), prefix(prefix) {}
    const ov::Model& model;
    const std::string tag;
    const std::string prefix;
};

template<typename T>
class PrintableVector {
public:
    PrintableVector(const std::vector<T>& values, int maxsize = 80) : values(values), maxsize(maxsize) {}
    const std::vector<T>& values;
    int maxsize;
};

struct PrintableDelta {
    uint64_t us_last;
    uint64_t us_all;
};

class PrintableTimer {
public:
    PrintableTimer(): t0(std::chrono::high_resolution_clock::now()) {
        t1 = t0;
    }

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;

    PrintableDelta delta() {
        PrintableDelta ret;
        auto now = std::chrono::high_resolution_clock::now();
        ret.us_last = std::chrono::duration_cast<std::chrono::microseconds>(now - t1).count();
        ret.us_all = std::chrono::duration_cast<std::chrono::microseconds>(now - t0).count();
        t1 = now;
        return ret;
    }
};

std::ostream & operator<<(std::ostream & os, const NodeDesc& desc);
std::ostream & operator<<(std::ostream & os, const Node& node);
std::ostream & operator<<(std::ostream & os, const MemoryDesc& desc);
std::ostream & operator<<(std::ostream & os, const Edge& edge);
std::ostream & operator<<(std::ostream & os, const PrintableModel& model);
std::ostream & operator<<(std::ostream & os, const PrintableDelta& us);
std::ostream & operator<<(std::ostream & os, const Edge::ReorderStatus reorderStatus);

std::ostream & operator<<(std::ostream & os, const dnnl::primitive_desc& desc);
std::ostream & operator<<(std::ostream & os, const dnnl::memory::desc& desc);
std::ostream & operator<<(std::ostream & os, const impl_desc_type impl_type);
std::ostream & operator<<(std::ostream & os, const dnnl::memory::data_type dtype);
std::ostream & operator<<(std::ostream & os, const dnnl::memory::format_tag dtype);

class DumpModel : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("DumpModel");
    DumpModel(const std::string& file_name);
    void dump_cpp_style(std::ostream & os, const std::shared_ptr<ov::Model>& model);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
protected:
    std::string file_name;
};

template<typename T>
std::ostream & operator<<(std::ostream & os, const PrintableVector<T>& vec) {
    std::stringstream ss;
    auto N = vec.values.size();
    for (int i = 0; i < N; i++) {
        if (i > 0)
            ss << ",";
        if (ss.tellp() > vec.maxsize) {
            ss << "..." << N << "in total";
            break;
        }
        if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value)
            ss << static_cast<int>(vec.values[i]);
        else
            ss << vec.values[i];
    }
    os << ss.str();
    return os;
}

static inline std::ostream& write_all_to_stream(std::ostream& os) {
    return os;
}
template <typename T, typename... TS>
static inline std::ostream& write_all_to_stream(std::ostream& os, const T& arg, TS&&... args) {
    return ov::intel_cpu::write_all_to_stream(os << arg, std::forward<TS>(args)...);
}

}   // namespace intel_cpu
}   // namespace ov

#define DEBUG_ENABLE_NAME debug_enable_##__LINE__

#define DEBUG_LOG_EXT(name, ostream, prefix, ...)                        \
        do {                                                                                               \
            static DebugLogEnabled DEBUG_ENABLE_NAME(__FILE__, __func__, __LINE__, name);                  \
            if (DEBUG_ENABLE_NAME) {                                                                       \
                ::std::stringstream ss___;                                                                 \
                ov::intel_cpu::write_all_to_stream(ss___, prefix, DEBUG_ENABLE_NAME.get_tag(), " ", __VA_ARGS__); \
                ostream << ss___.str() << std::endl;                                                     \
                DEBUG_ENABLE_NAME.break_at(ss___.str());                                                   \
            }                                                                                              \
        } while (0)

#define CPU_DEBUG_CAP_ENABLE(...) __VA_ARGS__
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) true

#define DEBUG_LOG(...) DEBUG_LOG_EXT(nullptr, std::cout, "[ DEBUG ] ", __VA_ARGS__)
#define ERROR_LOG(...) DEBUG_LOG_EXT(nullptr, std::cerr, "[ ERROR ] ", __VA_ARGS__)

#define CREATE_DEBUG_TIMER(x) PrintableTimer x

#else // !CPU_DEBUG_CAPS

#include <string>
#include <iostream>
#include <sstream>
#include <chrono>

#include <onednn/dnnl.h>
#include <dnnl_debug.h>
#include "onednn/iml_type_mapper.h"
#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"

class DumpModel : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("DumpModel");
    DumpModel(const std::string& file_name);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

#define CPU_DEBUG_CAP_ENABLE(...)
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) x

#define DEBUG_LOG(...)
#define ERROR_LOG(...)
#define DEBUG_LOG_EXT(name, ...)

#define CREATE_DEBUG_TIMER(x)

#endif // CPU_DEBUG_CAPS

// To avoid "unused variable" warnings `when debug caps
// need more information than non-debug caps version
#define CPU_DEBUG_CAPS_MAYBE_UNUSED(x) (void)x
