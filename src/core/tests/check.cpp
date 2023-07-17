// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/check.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"

using namespace std;

TEST(check, check_true_string_info) {
    OPENVINO_ASSERT(true, "this should not throw");
}

TEST(check, check_true_non_string_info) {
    OPENVINO_ASSERT(true, "this should not throw", 123);
}

TEST(check, check_true_no_info) {
    OPENVINO_ASSERT(true);
}

TEST(check, check_false_string_info) {
    EXPECT_THROW({ OPENVINO_ASSERT(false, "this should throw"); }, ov::AssertFailure);
}

TEST(check, check_false_non_string_info) {
    EXPECT_THROW({ OPENVINO_ASSERT(false, "this should throw", 123); }, ov::AssertFailure);
}

TEST(check, check_false_no_info) {
    EXPECT_THROW({ OPENVINO_ASSERT(false); }, ov::AssertFailure);
}

TEST(check, check_with_explanation) {
    bool check_failure_thrown = false;

    try {
        OPENVINO_ASSERT(false, "xyzzyxyzzy", 123);
    } catch (const ov::AssertFailure& e) {
        check_failure_thrown = true;
        EXPECT_PRED_FORMAT2(testing::IsSubstring, "Check 'false' failed at", e.what());
        EXPECT_PRED_FORMAT2(testing::IsSubstring, "xyzzyxyzzy123", e.what());
    }

    EXPECT_TRUE(check_failure_thrown);
}

TEST(check, ngraph_check_true_string_info) {
    NGRAPH_CHECK(true, "this should not throw");
}

TEST(check, ngraph_check_true_non_string_info) {
    NGRAPH_CHECK(true, "this should not throw", 123);
}

TEST(check, ngraph_check_true_no_info) {
    NGRAPH_CHECK(true);
}

TEST(check, ngraph_check_false_string_info) {
    EXPECT_THROW({ NGRAPH_CHECK(false, "this should throw"); }, ngraph::CheckFailure);
}

TEST(check, ngraph_check_false_non_string_info) {
    EXPECT_THROW({ NGRAPH_CHECK(false, "this should throw", 123); }, ngraph::CheckFailure);
}

TEST(check, ngraph_check_false_no_info) {
    EXPECT_THROW({ NGRAPH_CHECK(false); }, ngraph::CheckFailure);
}

TEST(check, ngraph_check_with_explanation) {
    bool check_failure_thrown = false;

    try {
        NGRAPH_CHECK(false, "xyzzyxyzzy", 123);
    } catch (const ngraph::CheckFailure& e) {
        check_failure_thrown = true;
        EXPECT_PRED_FORMAT2(testing::IsSubstring, "Check 'false' failed at", e.what());
        EXPECT_PRED_FORMAT2(testing::IsSubstring, "xyzzyxyzzy123", e.what());
    }

    EXPECT_TRUE(check_failure_thrown);
}

TEST(check, ov_throw_exception_check_relative_path_to_source) {
    const auto path = ov::util::path_join({"src", "core", "tests", "check.cpp"});
    const auto exp_msg = "Exception from " + path + ":" + std::to_string(__LINE__ + 1) + ":\nTest message";
    OV_EXPECT_THROW(OPENVINO_THROW("Test message"), ov::Exception, testing::HasSubstr(exp_msg));
}
