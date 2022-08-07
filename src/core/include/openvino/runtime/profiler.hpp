// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides profile interface
 *
 * @file openvino/runtime/profile.hpp
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <chrono>
#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>

#include "openvino/core/core_visibility.hpp"

namespace ov {

struct OPENVINO_API ProfileData {
    std::string name;  // Title
    std::string cat;   // Category
    std::map<std::string, std::string> args;
};

class Profiler;

class OPENVINO_API ProfilerManager {
    bool enabled;
    bool closed;
    std::ofstream fw;
    std::atomic_int alive_cnt;
    std::mutex vector_mutex;
    std::vector<Profiler *> all_targets;

    ProfilerManager();

public:
    static ProfilerManager& instance();

    ~ProfilerManager();
    static int64_t get_usec() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }

    void add_profile(Profiler *p);
    void flush();
    void set_enable(bool on);
    bool is_enabled() {
        return enabled;
    }


    friend class Profiler;
};

class OPENVINO_API Profiler {
public:

    int64_t start;
    int64_t end;
    std::thread::id tid;

    Profiler() {
        tid = std::this_thread::get_id();
        ProfilerManager::instance().alive_cnt++;
        start = ProfilerManager::get_usec();
    }

    static void move_into_manager(Profiler * p);

    virtual ~Profiler(){}
    virtual ProfileData get_info() = 0;
};


class ProfilerStringLiteral : public Profiler {
public:
    ProfilerStringLiteral(const char * name) : name(name) {}
    const char * name;
    ProfileData get_info() override {
        ProfileData info;
        info.name = name;
        info.cat = "Simple";
        return info;
    }
};

class ProfilerString : public Profiler {
public:
    ProfilerString(const std::string & name) : name(name) {}
    std::string name;
    ProfileData get_info() override {
        ProfileData info;
        info.name = name;
        info.cat = "Simple";
        return info;
    }
};

OPENVINO_API inline std::shared_ptr<void> Profile(const char * name) {
    if (!ProfilerManager::instance().is_enabled()) return nullptr;
    return std::shared_ptr<void>(new ProfilerStringLiteral(name), Profiler::move_into_manager);
}

OPENVINO_API inline std::shared_ptr<void> Profile(const std::string & name) {
    if (!ProfilerManager::instance().is_enabled()) return nullptr;
    return std::shared_ptr<void>(new ProfilerString(name), Profiler::move_into_manager);
}

template <class T, class... Args>
OPENVINO_API inline std::shared_ptr<void> Profile(Args&&... args) {
    if (!ProfilerManager::instance().is_enabled()) return nullptr;
    return std::shared_ptr<void>(new T(std::forward<Args>(args)...), Profiler::move_into_manager);
}

};