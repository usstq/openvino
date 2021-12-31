// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <ratio>
#include <cstdlib>

namespace MKLDNNPlugin {

class PerfCount {
    uint64_t total_duration;
    uint32_t num;

    std::chrono::high_resolution_clock::time_point __start = {};
    std::chrono::high_resolution_clock::time_point __finish = {};

public:
    PerfCount(): total_duration(0), num(0) {}

    std::chrono::duration<double, std::milli> duration() const {
        return __finish - __start;
    }

    uint64_t avg() const { return (num == 0) ? 0 : total_duration / num; }

private:
    void start_itr() {
        __start = std::chrono::high_resolution_clock::now();
    }

    void finish_itr() {
        __finish = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::microseconds>(__finish - __start).count();
        num++;
    }

    friend class PerfHelper;
};
class PerfHelper {
    PerfCount &counter;
    bool b_active;

public:
    explicit PerfHelper(PerfCount& count, int mode) : counter(count) {
        static int perf_mode = std::getenv("PERFM") ? std::atoi(std::getenv("PERFM")) : 0;
        if (perf_mode == mode) {
            b_active = true;
            counter.start_itr();
        } else {
            b_active = false;
        }
    }

    ~PerfHelper() { if (b_active) counter.finish_itr(); }
};

}  // namespace MKLDNNPlugin

#define GET_PERF(_node, mode) std::unique_ptr<PerfHelper>(new PerfHelper(_node->PerfCounter(), mode))
#define PERF(_node, _need) auto pc = _need ? GET_PERF(_node, 0) : nullptr;
