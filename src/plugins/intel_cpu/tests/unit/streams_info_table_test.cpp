// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_system_conf.h>

#include <common_test_utils/test_common.hpp>

#include "cpu_map_scheduling.hpp"
#include "cpu_streams_calculation.hpp"

using namespace testing;
using namespace InferenceEngine;
using namespace ov;

namespace {

struct SchedulingCoreTypeTestCase {
    ov::hint::SchedulingCoreType input_type;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<std::vector<int>> result_table;
    ov::hint::SchedulingCoreType output_type;
};

class SchedulingCoreTypeTests : public CommonTestUtils::TestsCommon,
                                public testing::WithParamInterface<std::tuple<SchedulingCoreTypeTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());
        auto test_input_type = test_data.input_type;

        std::vector<std::vector<int>> test_result_table =
            ov::intel_cpu::apply_scheduling_core_type(test_input_type, test_data.proc_type_table);

        ASSERT_EQ(test_data.result_table, test_result_table);
        ASSERT_EQ(test_data.output_type, test_input_type);
    }
};

SchedulingCoreTypeTestCase _2sockets_ALL = {
    ov::hint::SchedulingCoreType::ANY_CORE,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    ov::hint::SchedulingCoreType::ANY_CORE,
};

SchedulingCoreTypeTestCase _2sockets_P_CORE_ONLY = {
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
};

SchedulingCoreTypeTestCase _2sockets_E_CORE_ONLY = {
    ov::hint::SchedulingCoreType::ECORE_ONLY,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    // ov::hint::scheduling_core_type returns ANY_CORE because the platform has no Ecores available to satisfy the
    // user's request.
};

SchedulingCoreTypeTestCase _1sockets_ALL = {
    ov::hint::SchedulingCoreType::ANY_CORE,
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    ov::hint::SchedulingCoreType::ANY_CORE,
};

SchedulingCoreTypeTestCase _1sockets_P_CORE_ONLY = {
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    {{20, 6, 8, 6}},
    {{12, 6, 0, 6}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
};

SchedulingCoreTypeTestCase _1sockets_P_CORE_ONLY_1 = {
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    {{8, 0, 8, 0}},
    {{8, 0, 8, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    // ov::hint::scheduling_core_type returns ANY_CORE because the platform has no Pcore available to satisfy the
    // user's request.
};

SchedulingCoreTypeTestCase _1sockets_E_CORE_ONLY = {
    ov::hint::SchedulingCoreType::ECORE_ONLY,
    {{20, 6, 8, 6}},
    {{8, 0, 8, 0}},
    ov::hint::SchedulingCoreType::ECORE_ONLY,
};

TEST_P(SchedulingCoreTypeTests, SchedulingCoreType) {}

INSTANTIATE_TEST_SUITE_P(SchedulingCoreTypeTable,
                         SchedulingCoreTypeTests,
                         testing::Values(_2sockets_ALL,
                                         _2sockets_P_CORE_ONLY,
                                         _2sockets_E_CORE_ONLY,
                                         _1sockets_ALL,
                                         _1sockets_P_CORE_ONLY,
                                         _1sockets_P_CORE_ONLY_1,
                                         _1sockets_E_CORE_ONLY));

struct UseHTTestCase {
    bool input_ht_value;
    bool input_ht_changed;
    std::string input_pm_hint;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<std::vector<int>> result_table;
    bool output_ht_value;
};

class UseHTTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<UseHTTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> test_result_table =
            ov::intel_cpu::apply_hyper_threading(test_data.input_ht_value,
                                                 test_data.input_ht_changed,
                                                 test_data.input_pm_hint,
                                                 test_data.proc_type_table);

        ASSERT_EQ(test_data.result_table, test_result_table);
        ASSERT_EQ(test_data.input_ht_value, test_data.output_ht_value);
    }
};

UseHTTestCase _2sockets_false_latency = {
    false,
    true,
    "LATENCY",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_false_throughput = {
    false,
    true,
    "THROUGHPUT",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_true_latency = {
    true,
    true,
    "LATENCY",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    true,
};

UseHTTestCase _2sockets_true_throughput = {
    true,
    true,
    "THROUGHPUT",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    true,
};

UseHTTestCase _2sockets_default_1_latency = {
    false,
    false,
    "LATENCY",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_default_1_throughput = {
    false,
    false,
    "THROUGHPUT",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_default_2_latency = {
    true,
    false,
    "LATENCY",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_default_2_throughput = {
    true,
    false,
    "THROUGHPUT",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _1sockets_1_false_latency = {
    false,
    true,
    "LATENCY",
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
    false,
};

UseHTTestCase _1sockets_1_false_throughput = {
    false,
    true,
    "THROUGHPUT",
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
    false,
};

UseHTTestCase _1sockets_1_true_latency = {
    true,
    true,
    "LATENCY",
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    true,
};

UseHTTestCase _1sockets_1_true_throughput = {
    true,
    true,
    "THROUGHPUT",
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    true,
};

UseHTTestCase _1sockets_1_default_1_latency = {
    false,
    false,
    "LATENCY",
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
    false,
};

UseHTTestCase _1sockets_1_default_1_throughput = {
    false,
    false,
    "THROUGHPUT",
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    true,
};

UseHTTestCase _1sockets_1_default_2_latency = {
    true,
    false,
    "LATENCY",
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
    false,
};

UseHTTestCase _1sockets_1_default_2_throughput = {
    true,
    false,
    "THROUGHPUT",
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    true,
};

UseHTTestCase _1sockets_2_false_latency = {
    false,
    true,
    "LATENCY",
    {{12, 6, 0, 6}},
    {{6, 6, 0, 0}},
    false,
};

UseHTTestCase _1sockets_2_false_throughput = {
    false,
    true,
    "THROUGHPUT",
    {{12, 6, 0, 6}},
    {{6, 6, 0, 0}},
    false,
};

UseHTTestCase _1sockets_2_true_latency = {
    true,
    true,
    "LATENCY",
    {{12, 6, 0, 6}},
    {{12, 6, 0, 6}},
    true,
};

UseHTTestCase _1sockets_2_true_throughput = {
    true,
    true,
    "THROUGHPUT",
    {{12, 6, 0, 6}},
    {{12, 6, 0, 6}},
    true,
};

UseHTTestCase _1sockets_2_default_1_latency = {
    false,
    false,
    "LATENCY",
    {{12, 6, 0, 6}},
    {{6, 6, 0, 0}},
    false,
};

UseHTTestCase _1sockets_2_default_1_throughput = {
    false,
    false,
    "THROUGHPUT",
    {{12, 6, 0, 6}},
    {{12, 6, 0, 6}},
    true,
};

UseHTTestCase _1sockets_2_default_2_latency = {
    true,
    false,
    "LATENCY",
    {{12, 6, 0, 6}},
    {{6, 6, 0, 0}},
    false,
};

UseHTTestCase _1sockets_2_default_2_throughput = {
    true,
    false,
    "THROUGHPUT",
    {{12, 6, 0, 6}},
    {{12, 6, 0, 6}},
    true,
};

TEST_P(UseHTTests, UseHT) {}

INSTANTIATE_TEST_SUITE_P(UseHTTable,
                         UseHTTests,
                         testing::Values(_2sockets_false_latency,
                                         _2sockets_true_latency,
                                         _2sockets_default_1_latency,
                                         _2sockets_default_2_latency,
                                         _1sockets_1_false_latency,
                                         _1sockets_1_true_latency,
                                         _1sockets_1_default_1_latency,
                                         _1sockets_1_default_2_latency,
                                         _1sockets_2_false_latency,
                                         _1sockets_2_true_latency,
                                         _1sockets_2_default_1_latency,
                                         _1sockets_2_default_2_latency,
                                         _2sockets_false_throughput,
                                         _2sockets_true_throughput,
                                         _2sockets_default_1_throughput,
                                         _2sockets_default_2_throughput,
                                         _1sockets_1_false_throughput,
                                         _1sockets_1_true_throughput,
                                         _1sockets_1_default_1_throughput,
                                         _1sockets_1_default_2_throughput,
                                         _1sockets_2_false_throughput,
                                         _1sockets_2_true_throughput,
                                         _1sockets_2_default_1_throughput,
                                         _1sockets_2_default_2_throughput));

struct StreamsCalculationTestCase {
    int input_streams;
    bool input_streams_chaged;
    int input_threads;
    int input_infer_requests;
    int model_prefer_threads;
    std::string input_perf_hint;
    ov::intel_cpu::Config::LatencyThreadingMode scopeOflatencyCandidate;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<std::vector<int>> stream_info_table;
};

class StreamsCalculationTests : public CommonTestUtils::TestsCommon,
                                public testing::WithParamInterface<std::tuple<StreamsCalculationTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> test_stream_info_table =
            ov::intel_cpu::get_streams_info_table(test_data.input_streams,
                                                  test_data.input_streams_chaged,
                                                  test_data.input_threads,
                                                  test_data.input_infer_requests,
                                                  test_data.model_prefer_threads,
                                                  test_data.input_perf_hint,
                                                  test_data.scopeOflatencyCandidate,
                                                  test_data.proc_type_table);

        ASSERT_EQ(test_data.stream_info_table, test_stream_info_table);
    }
};

StreamsCalculationTestCase _2sockets_104cores_latency_platform_1 = {
    1,      // param[in]: the number of streams in this simulation.
    false,  // param[in]: Whether the user explicitly sets the number of streams is higher priority in streams
            // calculation logic. If this param is true, the following performance hint and LatencyThreadingMode will be
            // ignored.
    0,      // param[in]: the number of threads in this simulation
    0,      // param[in]: the number of infer requests in this simulation
    0,      // param[in]: the model preferred number of threads in this simulation
    "LATENCY",                                                  // param[in]: the performance hint in this simulation
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,  // param[in]: the LatencyThreadingMode in this
                                                                // simulation
    {{208, 104, 0, 104, -1, -1},
     {104, 52, 0, 52, 0, 0},
     {104, 52, 0, 52, 1, 1}},    // param[in]: the proc_type_table in this simulation
    {{1, MAIN_CORE_PROC, 208}},  // param[expected out]: the expected result of streams_info_table in this simulation
};

StreamsCalculationTestCase _2sockets_104cores_latency_platform_2 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_platform_3 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1},
     {52, 26, 0, 26, 0, 0},
     {52, 26, 0, 26, 1, 0},
     {52, 26, 0, 26, 2, 1},
     {52, 26, 0, 26, 3, 1}},
    {{1, MAIN_CORE_PROC, 208}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_platform_4 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {26, 13, 0, 0, 0, 0}, {26, 13, 0, 0, 1, 0}, {26, 13, 0, 0, 2, 1}, {26, 13, 0, 0, 3, 1}},
    {{1, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_socket_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_socket_2 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_socket_3 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{208, 104, 0, 104, -1, -1},
     {52, 26, 0, 26, 0, 0},
     {52, 26, 0, 26, 1, 0},
     {52, 26, 0, 26, 2, 1},
     {52, 26, 0, 26, 3, 1}},
    {{2, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_socket_4 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{104, 104, 0, 0, -1, -1}, {26, 13, 0, 0, 0, 0}, {26, 13, 0, 0, 1, 0}, {26, 13, 0, 0, 2, 1}, {26, 13, 0, 0, 3, 1}},
    {{2, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_socket_5 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{60, 60, 0, 0, -1, -1}, {10, 10, 0, 0, 0, 0}, {10, 10, 0, 0, 1, 0}, {20, 20, 0, 0, 2, 1}, {20, 20, 0, 0, 3, 1}},
    {{1, MAIN_CORE_PROC, 40}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_socket_6 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{60, 60, 0, 0, -1, -1}, {10, 10, 0, 0, 0, 0}, {20, 20, 0, 0, 1, 1}, {10, 10, 0, 0, 2, 0}, {20, 20, 0, 0, 3, 1}},
    {{1, MAIN_CORE_PROC, 40}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_socket_7 = {
    1,
    true,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{104, 104, 0, 0, -1, -1}, {26, 13, 0, 0, 0, 0}, {26, 13, 0, 0, 1, 0}, {26, 13, 0, 0, 2, 1}, {26, 13, 0, 0, 3, 1}},
    {{1, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_node_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_node_2 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_node_3 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{208, 104, 0, 104, -1, -1},
     {52, 26, 0, 26, 0, 0},
     {52, 26, 0, 26, 1, 0},
     {52, 26, 0, 26, 2, 1},
     {52, 26, 0, 26, 3, 1}},
    {{4, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_node_4 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{104, 104, 0, 0, -1, -1}, {26, 13, 0, 0, 0, 0}, {26, 13, 0, 0, 1, 0}, {26, 13, 0, 0, 2, 1}, {26, 13, 0, 0, 3, 1}},
    {{4, MAIN_CORE_PROC, 26}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_node_5 = {
    1,
    true,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{104, 104, 0, 0, -1, -1}, {26, 13, 0, 0, 0, 0}, {26, 13, 0, 0, 1, 0}, {26, 13, 0, 0, 2, 1}, {26, 13, 0, 0, 3, 1}},
    {{1, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_1 = {
    1,
    false,
    20,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 20}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_2 = {
    1,
    false,
    20,
    5,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 20}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_3 = {
    1,
    false,
    208,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 208}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_4 = {
    1,
    true,
    20,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 20}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_5 = {
    1,
    true,
    20,
    5,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 20}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_6 = {
    1,
    true,
    208,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 208}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{52, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_3 = {
    1,
    false,
    20,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{5, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_4 = {
    2,
    true,
    20,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 10}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_5 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{208, MAIN_CORE_PROC, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_6 = {
    1,
    false,
    0,
    0,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{104, MAIN_CORE_PROC, 2}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_7 = {
    1,
    false,
    0,
    0,
    8,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{26, MAIN_CORE_PROC, 8}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_8 = {
    1,
    false,
    40,
    0,
    8,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{5, MAIN_CORE_PROC, 8}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_9 = {
    5,
    true,
    20,
    2,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 10}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_10 = {
    1,
    false,
    0,
    2,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_11 = {
    2,
    true,
    0,
    5,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_12 = {
    1,
    false,
    0,
    2,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_13 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{26, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_14 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_15 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{104, MAIN_CORE_PROC, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_16 = {
    1,
    false,
    0,
    0,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{52, MAIN_CORE_PROC, 2}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_17 = {
    1,
    false,
    0,
    0,
    8,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{13, MAIN_CORE_PROC, 8}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_18 = {
    1,
    false,
    0,
    2,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_19 = {
    2,
    true,
    0,
    5,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_20 = {
    1,
    false,
    0,
    2,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_48cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 48}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{12, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_2 = {
    100,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{48, MAIN_CORE_PROC, 1}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_3 = {
    1,
    false,
    100,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{12, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_4 = {
    2,
    true,
    20,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 10}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_2 = {
    1,
    false,
    10,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 10}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_3 = {
    1,
    false,
    0,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 12}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_4 = {
    1,
    false,
    0,
    0,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}, {0, HYPER_THREADING_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_5 = {
    1,
    false,
    0,
    2,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}, {0, HYPER_THREADING_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_6 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}, {0, HYPER_THREADING_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_7 = {
    1,
    false,
    0,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_8 = {
    1,
    false,
    0,
    0,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_9 = {
    1,
    false,
    0,
    2,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_10 = {
    1,
    true,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_11 = {
    1,
    true,
    10,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 10}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_12 = {
    1,
    true,
    0,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 12}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_13 = {
    1,
    true,
    0,
    0,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}, {0, HYPER_THREADING_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_14 = {
    1,
    true,
    0,
    2,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}, {0, HYPER_THREADING_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_15 = {
    1,
    true,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}, {0, HYPER_THREADING_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_16 = {
    1,
    true,
    0,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_17 = {
    1,
    true,
    0,
    0,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_18 = {
    1,
    true,
    0,
    2,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}, {2, HYPER_THREADING_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}, {1, EFFICIENT_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_3 = {
    4,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_4 = {
    1,
    false,
    12,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_5 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1}, {4, EFFICIENT_CORE_PROC, 2}, {6, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_6 = {
    1,
    false,
    0,
    0,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{3, MAIN_CORE_PROC, 2}, {4, EFFICIENT_CORE_PROC, 2}, {3, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_7 = {
    100,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1}, {8, EFFICIENT_CORE_PROC, 1}, {6, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_8 = {
    1,
    false,
    100,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}, {2, HYPER_THREADING_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_9 = {
    4,
    true,
    0,
    8,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_10 = {
    6,
    true,
    0,
    4,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_11 = {
    1,
    false,
    0,
    2,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}, {1, EFFICIENT_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_12 = {
    1,
    false,
    0,
    2,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}, {1, EFFICIENT_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_13 = {
    1,
    false,
    1,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_14 = {
    1,
    false,
    9,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1}, {1, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_15 = {
    1,
    false,
    12,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1}, {3, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_16 = {
    1,
    false,
    15,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1}, {4, EFFICIENT_CORE_PROC, 2}, {1, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, ALL_PROC, 12}, {0, MAIN_CORE_PROC, 2}, {0, EFFICIENT_CORE_PROC, 8}, {0, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_2 = {
    1,
    false,
    8,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, ALL_PROC, 8}, {0, MAIN_CORE_PROC, 2}, {0, EFFICIENT_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_3 = {
    1,
    false,
    0,
    0,
    2,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_4 = {
    1,
    false,
    0,
    0,
    10,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, ALL_PROC, 12}, {0, MAIN_CORE_PROC, 2}, {0, EFFICIENT_CORE_PROC, 8}, {0, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_5 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{10, 2, 8, 0, 0, 0}},
    {{1, ALL_PROC, 10}, {0, MAIN_CORE_PROC, 2}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_6 = {
    1,
    false,
    0,
    0,
    2,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{10, 2, 8, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_7 = {
    1,
    false,
    0,
    0,
    10,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{10, 2, 8, 0, 0, 0}},
    {{1, ALL_PROC, 10}, {0, MAIN_CORE_PROC, 2}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2}, {4, EFFICIENT_CORE_PROC, 2}, {1, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2}, {1, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_3 = {
    4,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2}, {3, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_4 = {
    1,
    false,
    6,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_5 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{2, MAIN_CORE_PROC, 1}, {4, EFFICIENT_CORE_PROC, 2}, {2, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_6 = {
    1,
    false,
    0,
    0,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2}, {4, EFFICIENT_CORE_PROC, 2}, {1, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, ALL_PROC, 12}, {0, MAIN_CORE_PROC, 4}, {0, EFFICIENT_CORE_PROC, 4}, {0, HYPER_THREADING_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_2 = {
    1,
    false,
    100,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, ALL_PROC, 12}, {0, MAIN_CORE_PROC, 4}, {0, EFFICIENT_CORE_PROC, 4}, {0, HYPER_THREADING_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_3 = {
    1,
    false,
    0,
    0,
    4,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, MAIN_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_4 = {
    1,
    false,
    0,
    0,
    8,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, ALL_PROC, 12}, {0, MAIN_CORE_PROC, 4}, {0, EFFICIENT_CORE_PROC, 4}, {0, HYPER_THREADING_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_5 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{8, 4, 4, 0, 0, 0}},
    {{1, ALL_PROC, 8}, {0, MAIN_CORE_PROC, 4}, {0, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_6 = {
    1,
    false,
    0,
    0,
    4,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{8, 4, 4, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_7 = {
    1,
    false,
    0,
    0,
    8,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{8, 4, 4, 0, 0, 0}},
    {{1, ALL_PROC, 8}, {0, MAIN_CORE_PROC, 4}, {0, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, MAIN_CORE_PROC, 4}, {1, EFFICIENT_CORE_PROC, 4}, {1, HYPER_THREADING_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, MAIN_CORE_PROC, 4}, {1, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_3 = {
    4,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{2, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_4 = {
    6,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{2, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}, {2, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_5 = {
    1,
    false,
    6,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{2, MAIN_CORE_PROC, 2}, {1, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_6 = {
    1,
    false,
    8,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{2, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_7 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{4, MAIN_CORE_PROC, 1}, {2, EFFICIENT_CORE_PROC, 2}, {4, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_8 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{8, 4, 4, 0, 0, 0}},
    {{2, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_6cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 12}},
};

StreamsCalculationTestCase _1sockets_6cores_latency_2 = {
    1,
    false,
    100,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 12}},
};

StreamsCalculationTestCase _1sockets_6cores_latency_3 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{6, 6, 0, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {2, HYPER_THREADING_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}, {1, HYPER_THREADING_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_3 = {
    1,
    false,
    8,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_4 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1}, {6, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_ecores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 16}},
};

StreamsCalculationTestCase _1sockets_ecores_latency_2 = {
    1,
    false,
    4,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_ecores_latency_3 = {
    1,
    false,
    0,
    4,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 16}},
};

StreamsCalculationTestCase _1sockets_ecores_latency_4 = {
    1,
    false,
    0,
    0,
    4,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_1 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{16, EFFICIENT_CORE_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_2 = {
    1,
    false,
    0,
    0,
    4,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{4, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_3 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{2, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_4 = {
    8,
    true,
    0,
    4,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{4, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_5 = {
    2,
    true,
    0,
    0,
    4,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{2, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_mock_tput_1 = {
    1,
    false,
    15,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 7, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1}, {3, EFFICIENT_CORE_PROC, 2}, {3, HYPER_THREADING_PROC, 1}},
};

TEST_P(StreamsCalculationTests, StreamsCalculation) {}

INSTANTIATE_TEST_SUITE_P(StreamsInfoTable,
                         StreamsCalculationTests,
                         testing::Values(_2sockets_104cores_latency_platform_1,
                                         _2sockets_104cores_latency_platform_2,
                                         _2sockets_104cores_latency_platform_3,
                                         _2sockets_104cores_latency_platform_4,
                                         _2sockets_104cores_latency_socket_1,
                                         _2sockets_104cores_latency_socket_2,
                                         _2sockets_104cores_latency_socket_3,
                                         _2sockets_104cores_latency_socket_4,
                                         _2sockets_104cores_latency_socket_5,
                                         _2sockets_104cores_latency_socket_6,
                                         _2sockets_104cores_latency_socket_7,
                                         _2sockets_104cores_latency_node_1,
                                         _2sockets_104cores_latency_node_2,
                                         _2sockets_104cores_latency_node_3,
                                         _2sockets_104cores_latency_node_4,
                                         _2sockets_104cores_latency_node_5,
                                         _2sockets_104cores_latency_1,
                                         _2sockets_104cores_latency_2,
                                         _2sockets_104cores_latency_3,
                                         _2sockets_104cores_latency_4,
                                         _2sockets_104cores_latency_5,
                                         _2sockets_104cores_latency_6,
                                         _2sockets_104cores_tput_1,
                                         _2sockets_104cores_tput_2,
                                         _2sockets_104cores_tput_3,
                                         _2sockets_104cores_tput_4,
                                         _2sockets_104cores_tput_5,
                                         _2sockets_104cores_tput_6,
                                         _2sockets_104cores_tput_7,
                                         _2sockets_104cores_tput_8,
                                         _2sockets_104cores_tput_9,
                                         _2sockets_104cores_tput_10,
                                         _2sockets_104cores_tput_11,
                                         _2sockets_104cores_tput_12,
                                         _2sockets_104cores_tput_13,
                                         _2sockets_104cores_tput_14,
                                         _2sockets_104cores_tput_15,
                                         _2sockets_104cores_tput_16,
                                         _2sockets_104cores_tput_17,
                                         _2sockets_104cores_tput_18,
                                         _2sockets_104cores_tput_19,
                                         _2sockets_104cores_tput_20,
                                         _2sockets_48cores_latency_1,
                                         _2sockets_48cores_tput_1,
                                         _2sockets_48cores_tput_2,
                                         _2sockets_48cores_tput_3,
                                         _2sockets_48cores_tput_4,
                                         _1sockets_14cores_latency_1,
                                         _1sockets_14cores_latency_2,
                                         _1sockets_14cores_latency_3,
                                         _1sockets_14cores_latency_4,
                                         _1sockets_14cores_latency_5,
                                         _1sockets_14cores_latency_6,
                                         _1sockets_14cores_latency_7,
                                         _1sockets_14cores_latency_8,
                                         _1sockets_14cores_latency_9,
                                         _1sockets_14cores_latency_10,
                                         _1sockets_14cores_latency_11,
                                         _1sockets_14cores_latency_12,
                                         _1sockets_14cores_latency_13,
                                         _1sockets_14cores_latency_14,
                                         _1sockets_14cores_latency_15,
                                         _1sockets_14cores_latency_16,
                                         _1sockets_14cores_latency_17,
                                         _1sockets_14cores_latency_18,
                                         _1sockets_14cores_tput_1,
                                         _1sockets_14cores_tput_2,
                                         _1sockets_14cores_tput_3,
                                         _1sockets_14cores_tput_4,
                                         _1sockets_14cores_tput_5,
                                         _1sockets_14cores_tput_6,
                                         _1sockets_14cores_tput_7,
                                         _1sockets_14cores_tput_8,
                                         _1sockets_14cores_tput_9,
                                         _1sockets_14cores_tput_10,
                                         _1sockets_14cores_tput_11,
                                         _1sockets_14cores_tput_12,
                                         _1sockets_14cores_tput_13,
                                         _1sockets_14cores_tput_14,
                                         _1sockets_14cores_tput_15,
                                         _1sockets_14cores_tput_16,
                                         _1sockets_10cores_latency_1,
                                         _1sockets_10cores_latency_2,
                                         _1sockets_10cores_latency_3,
                                         _1sockets_10cores_latency_4,
                                         _1sockets_10cores_latency_5,
                                         _1sockets_10cores_latency_6,
                                         _1sockets_10cores_latency_7,
                                         _1sockets_10cores_tput_1,
                                         _1sockets_10cores_tput_2,
                                         _1sockets_10cores_tput_3,
                                         _1sockets_10cores_tput_4,
                                         _1sockets_10cores_tput_5,
                                         _1sockets_10cores_tput_6,
                                         _1sockets_8cores_latency_1,
                                         _1sockets_8cores_latency_2,
                                         _1sockets_8cores_latency_3,
                                         _1sockets_8cores_latency_4,
                                         _1sockets_8cores_latency_5,
                                         _1sockets_8cores_latency_6,
                                         _1sockets_8cores_latency_7,
                                         _1sockets_8cores_tput_1,
                                         _1sockets_8cores_tput_2,
                                         _1sockets_8cores_tput_3,
                                         _1sockets_8cores_tput_4,
                                         _1sockets_8cores_tput_5,
                                         _1sockets_8cores_tput_6,
                                         _1sockets_8cores_tput_7,
                                         _1sockets_8cores_tput_8,
                                         _1sockets_6cores_latency_1,
                                         _1sockets_6cores_latency_2,
                                         _1sockets_6cores_latency_3,
                                         _1sockets_6cores_tput_1,
                                         _1sockets_6cores_tput_2,
                                         _1sockets_6cores_tput_3,
                                         _1sockets_6cores_tput_4,
                                         _1sockets_ecores_latency_1,
                                         _1sockets_ecores_latency_2,
                                         _1sockets_ecores_latency_3,
                                         _1sockets_ecores_latency_4,
                                         _1sockets_ecores_tput_1,
                                         _1sockets_ecores_tput_2,
                                         _1sockets_ecores_tput_3,
                                         _1sockets_ecores_tput_4,
                                         _1sockets_ecores_tput_5,
                                         _1sockets_mock_tput_1));

struct StreamGenerateionTestCase {
    int input_stream;
    bool input_stream_changed;
    int input_thread;
    int input_request;
    int input_model_prefer;
    ov::hint::SchedulingCoreType input_type;
    bool input_ht_value;
    bool input_ht_changed;
    bool input_cpu_value;
    bool input_cpu_changed;
    ov::hint::PerformanceMode input_pm_hint;
    ov::intel_cpu::Config::LatencyThreadingMode input_latency_scope;
    ov::threading::IStreamsExecutor::ThreadBindingType input_binding_type;
    std::vector<std::vector<int>> input_proc_type_table;
    ov::hint::SchedulingCoreType output_type;
    bool output_ht_value;
    bool output_cpu_value;
    ov::hint::PerformanceMode output_pm_hint;
    std::vector<std::vector<int>> output_proc_type_table;
    std::vector<std::vector<int>> output_stream_info_table;
};

void make_config(StreamGenerateionTestCase& test_data, ov::intel_cpu::Config& config) {
    config.schedulingCoreType = test_data.input_type;
    config.enableCpuPinning = test_data.input_cpu_value;
    config.changedCpuPinning = test_data.input_cpu_changed;
    config.enableHyperThreading = test_data.input_ht_value;
    config.changedHyperThreading = test_data.input_ht_changed;
    config.perfHintsConfig.ovPerfHint = ov::util::to_string(test_data.input_pm_hint);
    config.scopeOflatencyCandidate = test_data.input_latency_scope;
    config.perfHintsConfig.ovPerfHintNumRequests = test_data.input_request;
    config.streamExecutorConfig._streams = test_data.input_stream;
    config.streamExecutorConfig._streams_changed = test_data.input_stream_changed;
    config.streamExecutorConfig._threads = test_data.input_thread;
    config.streamExecutorConfig._threadBindingType = test_data.input_binding_type;
    config.streamExecutorConfig._orig_proc_type_table = test_data.input_proc_type_table;
}

class StreamGenerationTests : public CommonTestUtils::TestsCommon,
                              public testing::WithParamInterface<std::tuple<StreamGenerateionTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());
        ov::intel_cpu::Config config;
        make_config(test_data, config);

        ov::intel_cpu::generate_stream_info(test_data.input_stream, nullptr, config, test_data.input_model_prefer);

        ASSERT_EQ(test_data.output_stream_info_table, config.streamExecutorConfig._streams_info_table);
        ASSERT_EQ(test_data.output_proc_type_table, config.streamExecutorConfig._proc_type_table);
        ASSERT_EQ(test_data.output_cpu_value, config.streamExecutorConfig._cpu_pinning);
        ASSERT_EQ(test_data.output_ht_value, config.enableHyperThreading);
        ASSERT_EQ(test_data.output_type, config.schedulingCoreType);
        ASSERT_EQ(test_data.output_pm_hint,
                  ov::util::from_string(config.perfHintsConfig.ovPerfHint, ov::hint::performance_mode));
    }
};

TEST_P(StreamGenerationTests, StreamsGeneration) {}

StreamGenerateionTestCase generation_latency_1sockets_14cores_1 = {
    1,                                       // param[in]: simulated settting for streams number
    false,                                   // param[in]: simulated settting for streams number changed
    0,                                       // param[in]: simulated setting for threads number
    0,                                       // param[in]: simulated setting for inference request number
    0,                                       // param[in]: simulated setting for model prefer threads number
    ov::hint::SchedulingCoreType::ANY_CORE,  // param[in]: simulated setting for scheduling core type
                                             // (PCORE_ONLY/ECORE_ONLY/ANY_CORE)
    true,                                    // param[in]: simulated setting for enableHyperThreading
    true,                                    // param[in]: simulated settting for changedHyperThreading
    true,                                    // param[in]: simulated setting for enableCpuPinning
    true,                                    // param[in]: simulated setting for changedCpuPinning
    ov::hint::PerformanceMode::LATENCY,      // param[in]: simulated setting for performance mode (throughput/latency)
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,  // param[in]: simulated setting for scope of candidate processors
                                                       // on latency mode
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,  // param[in]: simulated setting for
                                                                       // threadBindingType
    {{20, 6, 8, 6, 0, 0}},  // param[in]: simulated proc_type_table for platform which has one socket, 6 Pcores, 8
                            // Ecores and hyper threading enabled
    ov::hint::SchedulingCoreType::ANY_CORE,  // param[expected out]: scheduling core type needs to be the same as input
    true,                                    // param[expected out]: enableHyperThreading needs to be the same as input
    true,                                    // param[expected out]: enableCpuPinning needs to be the same as input
    ov::hint::PerformanceMode::LATENCY,      // param[expected out]: performance mode needs to be the same as input
    {{20, 6, 8, 6, 0, 0}},  // param[expected out]: since hyper threading is enabled and all core type is used,
                            // proc_type_table needs to be the same as input
    {{1, ALL_PROC, 20},
     {0, MAIN_CORE_PROC, 6},
     {0, EFFICIENT_CORE_PROC, 8},
     {0,
      HYPER_THREADING_PROC,
      6}},  // param[expected out]: since performance mode is latency and all cores is used, the final streams is 1
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_2 = {
    1,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    true,
    true,
    ov::hint::PerformanceMode::LATENCY,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{14, 6, 8, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_3 = {
    1,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{14, 6, 8, 0, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{6, 6, 0, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}},
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_4 = {
    1,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{12, 6, 0, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 12}},
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_5 = {
    1,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{6, 6, 0, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_6 = {
    1,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 24}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_7 = {
    1,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 24}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_8 = {
    1,
    true,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 48}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_9 = {
    1,
    true,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 48}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_1 = {
    0,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    true,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE,
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}, {2, HYPER_THREADING_PROC, 3}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_2 = {
    0,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{6, 6, 0, 0, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_3 = {
    10,
    true,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{12, 6, 0, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1}, {4, HYPER_THREADING_PROC, 1}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_4 = {
    0,
    false,
    10,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{12, 6, 0, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3}, {1, HYPER_THREADING_PROC, 3}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_5 = {
    0,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    {{24, MAIN_CORE_PROC, 4}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_6 = {
    0,
    false,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{12, MAIN_CORE_PROC, 4}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_7 = {
    100,
    true,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{48, MAIN_CORE_PROC, 1}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_8 = {
    2,
    true,
    20,
    0,
    1,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 10}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_9 = {
    0,
    false,
    0,
    0,
    1,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    false,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{48, MAIN_CORE_PROC, 1}},
};

INSTANTIATE_TEST_SUITE_P(smoke_StreamsGeneration,
                         StreamGenerationTests,
                         ::testing::Values(generation_latency_1sockets_14cores_1,
                                           generation_latency_1sockets_14cores_2,
                                           generation_latency_1sockets_14cores_3,
                                           generation_latency_1sockets_14cores_4,
                                           generation_latency_1sockets_14cores_5,
                                           generation_latency_2sockets_48cores_6,
                                           generation_latency_2sockets_48cores_7,
                                           generation_latency_2sockets_48cores_8,
                                           generation_latency_2sockets_48cores_9,
                                           generation_tput_1sockets_14cores_1,
                                           generation_tput_1sockets_14cores_2,
                                           generation_tput_1sockets_14cores_3,
                                           generation_tput_1sockets_14cores_4,
                                           generation_tput_2sockets_48cores_5,
                                           generation_tput_2sockets_48cores_6,
                                           generation_tput_2sockets_48cores_7,
                                           generation_tput_2sockets_48cores_8,
                                           generation_tput_2sockets_48cores_9));

}  // namespace