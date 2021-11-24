// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <openvino/runtime/variable_state.hpp>

using namespace ::testing;
using namespace std;

using VariableStateOVTests = ::testing::Test;

TEST_F(VariableStateOVTests, throwsOnUninitializedReset) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.reset(), ov::Exception);
}

TEST_F(VariableStateOVTests, throwsOnUninitializedGetname) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.get_name(), ov::Exception);
}

TEST_F(VariableStateOVTests, throwsOnUninitializedGetState) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.get_state(), ov::Exception);
}

TEST_F(VariableStateOVTests, throwsOnUninitializedSetState) {
    ov::runtime::VariableState state;
    ov::runtime::Tensor tensor;
    ASSERT_THROW(state.set_state(tensor), ov::Exception);
}
