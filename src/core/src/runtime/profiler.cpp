// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/runtime/profiler.hpp"

namespace ov {

ProfilerManager::ProfilerManager() : alive_cnt(0) {
    const char* str_enable = std::getenv("OV_PROFILE");
    if (!str_enable)
        str_enable = "0";
    closed = false;
    set_enable(atoi(str_enable));
}

ProfilerManager::~ProfilerManager() {
    std::lock_guard<std::mutex> lock_g(vector_mutex);
    flush();
    if (fw.is_open()) {
        fw << R"({
            "name": "Profiler End",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Trace OV Profiler",
            "ts":)"
            << get_usec() << "}",
            fw << "]\n";
        fw << "}\n";
        fw.close();
    }
    closed = true;
}

void ProfilerManager::flush() {
    if (fw.is_open()) {
        for (auto p : all_targets) {
            ProfileData prof_data = p->get_info();
            fw << "{\n";
            fw << "\"ph\": \"X\",\n"
                << "\"cat\": \"" << prof_data.cat << "\",\n"
                << "\"name\": \"" << prof_data.name << "\",\n"
                << "\"pid\": " << 0 << ",\n"
                << "\"tid\": " << p->tid << ",\n"
                << "\"ts\": " << p->start << ",\n"
                << "\"dur\": " << p->end - p->start << ",\n"
                << "\"args\": {\n";
            const char* sep = "";
            for (auto& a : prof_data.args) {
                if (a.first.substr(0,5) == "file:") {
                    // special args:
                    //     args["file:graph1.dump"] = content
                    // will dump the content to a file named `graph1.dump`
                    std::ofstream fdump(a.first.substr(5), std::ios::out);
                    if (fdump.is_open()) {
                        fdump << a.second;
                        fdump.close();
                    }
                } else {
                    fw << sep << "      \"" << a.first << "\":"
                       << "\"" << a.second << "\"";
                    sep = ",\n";
                }
            }
            fw << "\n          }\n";
            fw << "},\n";
        }
    }
    for (auto& p : all_targets) {
        delete p;
    }
    all_targets.clear();
}

void Profiler::move_into_manager(Profiler * p) {
    ProfilerManager::instance().add_profile(p);
}

void ProfilerManager::add_profile(Profiler * p) {
    if (closed) {
        std::cout <<" ?????????????? \n";
        return;
    }
    p->end = get_usec();
    std::lock_guard<std::mutex> lock_g(vector_mutex);
    all_targets.push_back(p);
    if (--alive_cnt == 0) {
        flush();
    }
}

void ProfilerManager::set_enable(bool on) {
    if (enabled != on) {
        enabled = on;
        if (enabled) {
            if (!fw.is_open()) {
                fw.open("ov_profile.json", std::ios::out);
                if (fw.is_open()) {
                    fw << "{\n";
                    fw << "\"schemaVersion\": 1,\n";
                    fw << "\"traceEvents\": [\n";
                }
            }
        }
        std::cout << "================ ProfilerManager is " << (enabled?"enabled":"disabled") << ".\n";
    }
}

ProfilerManager& ProfilerManager::instance() {
    static ProfilerManager inst;
    return inst;
}

};