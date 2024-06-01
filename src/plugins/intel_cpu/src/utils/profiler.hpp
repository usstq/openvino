// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define USE_PROFILER 1

#if defined(OPENVINO_ARCH_X86_64) && USE_PROFILER

#    include <atomic>
#    include <chrono>
#    include <cstddef>
#    include <deque>
#    include <fstream>
#    include <functional>
#    include <iostream>
#    include <map>
#    include <memory>
#    include <mutex>
#    include <set>
#    include <sstream>
#    include <thread>
#    include <vector>
extern "C" {
#    ifdef _WIN32
#        include <intrin.h>
#    else
#        include <x86intrin.h>
#    endif
}

namespace ov {
namespace intel_cpu {

namespace detail {
struct ProfileData {
    uint64_t start;
    uint64_t end;
    std::string cat;
    std::string name;
    uint64_t data[4] = {0};

    ProfileData(const std::string& cat, const std::string& name) : cat(cat), name(name) {
        start = __rdtsc();
    }
    static void record_end(ProfileData* p) {
        p->end = __rdtsc();
    }
};

struct chromeTrace {
    std::ostream& os;
    int fake_tid;
    uint64_t ts;
    chromeTrace(std::ostream& os, int fake_tid) : os(os), fake_tid(fake_tid) {}
    void addCompleteEvent(std::string name, std::string cat, double start, double dur) {
        // chrome tracing will show & group-by to name, so we use cat as name
        os << "{\"ph\": \"X\", \"name\": \"" << name << "\", \"cat\":\"" << cat << "\","
           << "\"pid\": " << fake_tid << ", \"tid\": 0,"
           << "\"ts\": " << std::setprecision (15) << start << ", \"dur\": " << dur << "},\n";
    }
};

struct TscCounter {
    uint64_t tsc_ticks_per_second;
    uint64_t tsc_ticks_base;
    double tsc_to_usec(uint64_t tsc_ticks) const {
        return (tsc_ticks - tsc_ticks_base) * 1000000.0 / tsc_ticks_per_second;
    }
    double tsc_to_usec(uint64_t tsc_ticks0, uint64_t tsc_ticks1) const {
        return (tsc_ticks1 - tsc_ticks0) * 1000000.0 / tsc_ticks_per_second;
    }
    TscCounter() {
        static std::once_flag flag;
        std::call_once(flag, [&]() {
            uint64_t start_ticks = __rdtsc();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            tsc_ticks_per_second = (__rdtsc() - start_ticks);
            std::cout << "[OV_CPU_PROFILE] tsc_ticks_per_second = " << tsc_ticks_per_second << std::endl;
            tsc_ticks_base = __rdtsc();
        });
    }
};

class ProfilerBase {
public:
    virtual void save(std::ofstream& fw, TscCounter& tsc) = 0;
};

struct ProfilerFinalizer {
    std::mutex g_mutex;
    std::set<ProfilerBase*> all_managers;
    const char* dump_file_name = "ov_profile.json";
    bool dump_file_over = false;
    bool not_finalized = true;
    std::ofstream fw;
    std::atomic_int totalProfilerManagers{0};
    TscCounter tsc;

    ~ProfilerFinalizer() {
        if (not_finalized)
            finalize();
    }

    void finalize() {
        if (!not_finalized)
            return;
        std::lock_guard<std::mutex> guard(g_mutex);
        if (dump_file_over || all_managers.empty())
            return;

        // start dump
        fw.open(dump_file_name, std::ios::out);
        fw << "{\n";
        fw << "\"schemaVersion\": 1,\n";
        fw << "\"traceEvents\": [\n";
        fw.flush();

        for (auto& pthis : all_managers) {
            pthis->save(fw, tsc);
        }
        all_managers.clear();

        fw << R"({
            "name": "Profiler End",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Trace OV Profiler",
            "ts":)"
           << tsc.tsc_to_usec(__rdtsc()) << "}",
            fw << "]\n";
        fw << "}\n";
        auto total_size = fw.tellp();
        fw.close();
        dump_file_over = true;
        not_finalized = false;
        std::cout << "[OV_CPU_PROFILE] Dumpped " << total_size / (1024 * 1024) << " (MB) to " << dump_file_name
                  << std::endl;
    }

    int register_manager(ProfilerBase* pthis) {
        std::lock_guard<std::mutex> guard(g_mutex);
        std::stringstream ss;
        auto serial_id = totalProfilerManagers.fetch_add(1);
        ss << "[OV_CPU_PROFILE] #" << serial_id << "(" << pthis << ") : is registed." << std::endl;
        std::cout << ss.str();
        all_managers.emplace(pthis);
        return serial_id;
    }

    static ProfilerFinalizer& get() {
        static ProfilerFinalizer inst;
        return inst;
    }
};

}  // namespace detail

class Profiler : public detail::ProfilerBase {
    bool enabled;
    std::deque<detail::ProfileData> all_data;
    int serial;

public:
    Profiler() {
        const char* str_enable = std::getenv("OV_CPU_PROFILE");
        if (!str_enable)
            str_enable = "0";
        enabled = atoi(str_enable) > 0;
        if (enabled)
            serial = detail::ProfilerFinalizer::get().register_manager(this);
    }
    ~Profiler() {
        detail::ProfilerFinalizer::get().finalize();
    }

    void save(std::ofstream& fw, detail::TscCounter& tsc) override {
        if (!enabled)
            return;
        auto data_size = all_data.size();
        if (!data_size)
            return;

        detail::chromeTrace ct(fw, serial);
        for (auto& d : all_data) {
            std::stringstream ss;
            ss << d.name;
            for (int i = 0; i < sizeof(d.data)/sizeof(d.data[0]); i++)
                ss << "," << d.data[i];

            ct.addCompleteEvent(d.cat,
                                ss.str(),
                                tsc.tsc_to_usec(d.start),
                                tsc.tsc_to_usec(d.start, d.end));
        }
        all_data.clear();
        std::cout << "[OV_CPU_PROFILE] #" << serial << "(" << this << ") finalize: dumpped " << data_size << std::endl;
    }

    using ProfileDataHandle = std::unique_ptr<detail::ProfileData, void (*)(detail::ProfileData*)>;

    static ProfileDataHandle startProfile(const std::string& cat, const std::string& name = {}) {
        thread_local Profiler inst;
        if (!inst.enabled) {
            return ProfileDataHandle(nullptr, [](detail::ProfileData*) {});
        }
        inst.all_data.emplace_back(cat, name);
        return ProfileDataHandle(&inst.all_data.back(), detail::ProfileData::record_end);
    }

    friend class ProfilerFinalizer;
};

// performance counter
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>

__attribute__((weak)) int perf_event_open(struct perf_event_attr* attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

struct read_format {
  uint64_t nr;
  struct {
    uint64_t value;
    uint64_t id;
  } values[];
};

struct linux_perf_event {
    uint32_t type;
    uint32_t config;
    int fd = -1;
    struct perf_event_mmap_page* buf = nullptr;
    const char* name;

    pid_t tid;

    linux_perf_event(const linux_perf_event&) = delete;
    linux_perf_event(linux_perf_event&&) = delete;

    std::vector<int> fds;
    std::vector<uint64_t> event_ids;
    std::vector<uint64_t> event_values;

    linux_perf_event(const std::vector<std::tuple<uint32_t, uint32_t, const char*>> type_config_names = {}) {
        struct perf_event_attr attr;

        tid = syscall(__NR_gettid);
        auto num_events = type_config_names.size();
        fds.resize(num_events, 0);
        event_ids.resize(num_events, 0);
        event_values.resize(num_events, 0);
        int group_fd = -1;
        for(int i = 0; i < num_events; i++) {
            auto& tc = type_config_names[i];
            auto type = std::get<0>(tc);
            auto config = std::get<1>(tc);
            auto* name = std::get<2>(tc);
            memset(&attr, 0, sizeof(struct perf_event_attr));
            attr.size = sizeof(struct perf_event_attr);
            if (type == PERF_TYPE_SOFTWARE) {
                attr.config = config;
                attr.disabled = 1;
                attr.exclude_kernel = 0;
                attr.sample_period = 1000;
                attr.exclude_hv = 1;
            } else {
                attr.type = type;
                attr.size = PERF_ATTR_SIZE_VER0;
                attr.config = config;
                attr.sample_type = PERF_SAMPLE_READ;
                attr.exclude_kernel = 1;
            }
            attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;

            // pid == 0 and cpu == -1
            // This measures the calling process/thread on any CPU.
            int fd = syscall(__NR_perf_event_open, &attr, 0, -1, group_fd, 0);
            if (fd < 0) {
                perror("perf_event_open failed!");
                abort();
            }
            if (group_fd == -1) {
                group_fd = fd;
                buf = (struct perf_event_mmap_page*)mmap(NULL, sysconf(_SC_PAGESIZE), PROT_READ, MAP_SHARED, fd, 0);
                if (buf == MAP_FAILED) {
                    perror("mmap perf_event_mmap_page failed!");
                    close(fd);
                    abort();
                }                
            }

            fds[i] = fd;
            if (ioctl(fd, PERF_EVENT_IOC_ID, &event_ids[i]) == -1) {
                perror("ioctl failed!");
                abort();
            }
        }
    }
/*
    linux_perf_event(uint32_t type, uint32_t config, const char* name) : type(type), config(config), fd(-1), buf(nullptr), name(name) {
        struct perf_event_attr attr = {};
        attr.size = sizeof(struct perf_event_attr);
        if (type == PERF_TYPE_SOFTWARE) {
            attr.config = config;
            attr.disabled = 1;
            attr.exclude_kernel = 0;
            attr.sample_period = 1000;
            attr.exclude_hv = 1;
        } else {
            attr.type = type;
            attr.size = PERF_ATTR_SIZE_VER0;
            attr.config = config;
            attr.sample_type = PERF_SAMPLE_READ;
            attr.exclude_kernel = 1;
        }
        tid = syscall(__NR_gettid);
        fd = perf_event_open(&attr, 0, -1, -1, 0);
        if (fd < 0) {
            perror("perf_event_open, consider:  echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid");
            abort();
            return;
        }
        buf = (struct perf_event_mmap_page*)mmap(NULL, sysconf(_SC_PAGESIZE), PROT_READ, MAP_SHARED, fd, 0);
        if (buf == MAP_FAILED) {
            perror("mmap");
            close(fd);
            fd = -1;
            abort();
            return;
        }
        if (0)
        {
            std::stringstream ss;
            ss << static_cast<void*>(this) << " rdpmc=" << rdpmc_read() << ". tid=" << tid << ",fd=" << fd << std::endl;
            std::cout << ss.str();
        }
        fds.push_back(fd);
        // std::ios::fmtflags f(std::cout.flags());
        // std::cout << std::hex << "Linux perf event " << name << " (type=" << type << ",config=" << config << ")" << " is opened!" << std::endl;
        // std::cout.flags(f);
    }
*/
    void start() {
        ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    uint64_t stop() {
        char buf[4096];
        ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        read(fds[0], buf, sizeof(buf));
        struct read_format* rf = (struct read_format*) buf;
        for (int i = 0; i < event_ids.size(); i++) {
            event_values[i] = 0;
            for (int n = 0; n < rf->nr; n++) {
                if (rf->values[n].id == event_ids[i]) {
                    event_values[i] = rf->values[n].value;
                }
            }
        }
        return event_values[0];
    }

    uint64_t operator[](int i) {
        return event_values[i];
    }

    ~linux_perf_event() {
        if (buf) munmap(buf, sysconf(_SC_PAGESIZE));
        for (auto fd : fds) {
            close(fd);
        }
    }

    uint64_t rdpmc_read() {
        auto cur_tid = syscall(__NR_gettid);
        if (tid != cur_tid) {
            std::stringstream ss;
            ss << static_cast<void*>(this) << std::dec << " cur_tid (" << cur_tid << ") != tid(" << tid << ")" << std::endl;
            std::cout << ss.str();
            abort();
        }
        uint64_t val, offset;
        uint32_t seq, index;
        do {
            seq = buf->lock;
            std::atomic_thread_fence(std::memory_order_acquire);
            index = buf->index;   //
            offset = buf->offset; // used to compensate the initial counter value
            if (index == 0) {     /* rdpmc not allowed */
                val = 0;
                std::stringstream ss;
                ss << static_cast<void*>(this) << " rdpmc failed.  cur_tid=" << cur_tid << ", tid=" << tid << ",fd=" << fd << ",offset=" << offset << std::endl;
                std::cout << ss.str();
                abort();
                break;
            }
            val = _rdpmc(index - 1);
            std::atomic_thread_fence(std::memory_order_acquire);
        } while (buf->lock != seq);
        uint64_t ret = (val + offset) & 0xffffffffffff;
        return ret;
    }
};


}  // namespace intel_cpu
}  // namespace ov

#    define PROFILE(var_name, ...)                                          \
        auto var_name = ov::intel_cpu::Profiler::startProfile(__VA_ARGS__); \
        (void)var_name;

#    define PROFILE2(var_name, ...) var_name = ov::intel_cpu::Profiler::startProfile(__VA_ARGS__);

#else

#    define PROFILE(var_name, ...)
#    define PROFILE2(var_name, ...)

#endif