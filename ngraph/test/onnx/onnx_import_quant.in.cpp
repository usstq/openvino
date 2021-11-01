// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "onnx_import/onnx.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear_const_scale_const_zero_p) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear_const.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<float>{32.25f, 48.34f, 50.f, 83.f});

    test_case.add_expected_output(std::vector<std::uint8_t>{64, 97, 100, 166});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<float>{32.25f, 48.34f, 50.f, 83.f});
    test_case.add_input(std::vector<float>{0.5f});

    test_case.add_expected_output(std::vector<std::uint8_t>{64, 97, 100, 166});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear_zero_point) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear_zero_point.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<float>{0.f, 2.f, 3.f, 1000.f, -254.f, -1000.f});  // x
    test_case.add_input(std::vector<float>{2.0f});                                    // y_scale
    test_case.add_input(std::vector<std::uint8_t>{128});                              // y_zero_point

    test_case.add_expected_output<std::uint8_t>({6}, std::vector<std::uint8_t>{128, 129, 130, 255, 1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear_axis_zero) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear_axis_zero.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<float>{0.f, 2.f, 3.f, 1000.f, 0.f, 2.f, 3.f, 1000.f, 0.f, 2.f, 3.f, 1000.f});  // x
    test_case.add_input(std::vector<float>{1.f, 2.f, 4.f});   // y_scale
    test_case.add_input(std::vector<std::uint8_t>{0, 0, 0});  // y_zero_point

    //  std::vector<std::uint8_t>{0, 2, 3, 255, 0, 1, 2, 255, 0, 1, 1, 250}}; <- bad expected output
    //                                                                           given HALF_TO_EVEN
    //                                                                           round mode
    test_case.add_expected_output<std::uint8_t>({3, 4},
                                                std::vector<std::uint8_t>{0, 2, 3, 255, 0, 1, 2, 255, 0, 0, 1, 250});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quantize_linear_axis_negative) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/quantize_linear_axis_negative.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<float>{0.f, 2.f, 3.f, 1000.f, 0.f, 2.f, 3.f, 1000.f, 0.f, 2.f, 3.f, 1000.f});  // x
    test_case.add_input(std::vector<float>{1.f, 2.f, 4.f});   // y_scale
    test_case.add_input(std::vector<std::uint8_t>{0, 0, 0});  // y_zero_point

    //  std::vector<std::uint8_t>{0, 2, 3, 255, 0, 1, 2, 255, 0, 1, 1, 250}}; <- bad expected output
    //                                                                           given HALF_TO_EVEN
    //                                                                           round mode
    test_case.add_expected_output<std::uint8_t>({3, 4},
                                                std::vector<std::uint8_t>{0, 2, 3, 255, 0, 1, 2, 255, 0, 0, 1, 250});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dequant_lin.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<std::uint8_t>{19, 210, 21, 10});

    test_case.add_expected_output(std::vector<float>{76.f, 840.f, 84.f, 40.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_scalar_scale_and_zero_point) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_scalar_scale_and_zero_point.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<std::uint8_t>{19, 210, 21, 10});  // x
    test_case.add_input(std::vector<float>{2.0f});                    // scale
    test_case.add_input(std::vector<uint8_t>{128});                   // zero_point

    test_case.add_expected_output<float>(std::vector<float>{-218, 164, -214, -236});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_scalar_scale) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_scalar_scale.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<std::uint8_t>{19, 210, 21, 10});  // x
    test_case.add_input(std::vector<float>{2.0f});                    // scale
    test_case.add_input(std::vector<uint8_t>{128, 7});                // zero_point

    test_case.add_expected_output<float>(std::vector<float>{-218, 406, -214, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_scalar_inputs) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_scalar_inputs.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<std::uint8_t>{19});  // x
    test_case.add_input(std::vector<float>{2.0f});       // scale
    test_case.add_input(std::vector<uint8_t>{128});      // zero_point

    test_case.add_expected_output<float>(std::vector<float>{-218});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_scalar_zero_point) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_scalar_zero_point.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<std::uint8_t>{19, 210, 21, 10});  // x
    test_case.add_input(std::vector<float>{2.0f, 1.0f});              // scale
    test_case.add_input(std::vector<uint8_t>{128});                   // zero_point

    test_case.add_expected_output<float>(std::vector<float>{-218, 82, -214, -118});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_scalar_zero_scale_uint8) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_0.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input(std::vector<uint8_t>{0, 3, 128, 255});  // x
    test_case.add_input(std::vector<float>{2.0f});              // scale
    test_case.add_input(std::vector<uint8_t>{128});             // zero_point

    test_case.add_expected_output<float>({4}, std::vector<float>{-256.0f, -250.0f, 0.0f, 254.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_scalar_zero_scale_int8) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_1.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input(std::vector<int8_t>{-30, -3, 100, 127});  // x
    test_case.add_input(std::vector<float>{2.0f});                // scale
    test_case.add_input(std::vector<int8_t>{-10});                // zero_point

    test_case.add_expected_output<float>({4}, std::vector<float>{-40.0f, 14.0f, 220.0f, 274.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_1d_zero_scale_uint8) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_2.onnx"));

    auto test_case = ngraph::test::TestCase<TestEngine>(function);

    test_case.add_input(std::vector<uint8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 10, 20, 30});  // x
    test_case.add_input(std::vector<float>{1.0f, 2.0f, 4.0f});                         // scale
    test_case.add_input(std::vector<uint8_t>{0, 0, 0});                                // zero_point

    test_case.add_expected_output<float>(
        {3, 4},
        std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 4.0f, 6.0f, 0.0f, 40.0f, 80.0f, 120.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_1d_zero_scale_int8) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_3.onnx"));

    auto test_case = ngraph::test::TestCase<TestEngine>(function);

    test_case.add_input(std::vector<int8_t>{0, 1, 2, 3, 0, 2, 4, 6, 0, 10, 20, 30});  // x
    test_case.add_input(std::vector<float>{1.0f, 2.0f, 4.0f, 8.0f});                  // scale
    test_case.add_input(std::vector<int8_t>{0, -10, -20, -30});                       // zero_point

    test_case.add_expected_output<float>(
        {3, 4},
        std::vector<float>{0.0f, 22.0f, 88.0f, 264.0f, 0.0f, 24.0f, 96.0f, 288.0f, 0.0f, 40.0f, 160.0f, 480.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_1d_zero_scale_int8_4d) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_4.onnx"));

    auto test_case = ngraph::test::TestCase<TestEngine>(function);

    test_case.add_input(std::vector<uint8_t>{7, 9, 10, 10, 5,  8, 9, 1, 8, 6, 7, 9, 10, 0, 7, 10,
                                             8, 2, 6,  0,  5,  9, 8, 1, 2, 7, 5, 3, 2,  4, 1, 3,
                                             8, 7, 4,  8,  10, 1, 5, 5, 7, 7, 0, 2, 4,  4, 0, 5});  // x
    test_case.add_input(std::vector<float>{1.0f, 10.0f, 7.0f});                                     // scale
    test_case.add_input(std::vector<uint8_t>{10, 2, 1});                                            // zero_point

    test_case.add_expected_output<float>(
        {2, 3, 2, 4},
        std::vector<float>{-3.0f, -1.0f,  0.0f,  0.0f,  -5.0f, -2.0f, -1.0f, -9.0f, 60.0f, 40.0f, 50.0f, 70.0f,
                           80.0f, -20.0f, 50.0f, 80.0f, 49.0f, 7.0f,  35.0f, -7.0f, 28.0f, 56.0f, 49.0f, 0.0f,
                           -8.0f, -3.0f,  -5.0f, -7.0f, -8.0f, -6.0f, -9.0f, -7.0f, 60.0f, 50.0f, 20.0f, 60.0f,
                           80.0f, -10.0f, 30.0f, 30.0f, 42.0f, 42.0f, -7.0f, 7.0f,  21.0f, 21.0f, -7.0f, 28.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_dequantize_linear_1d_zero_scale_uint8_negative_axis) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/dequantize_linear_5.onnx"));

    auto test_case = ngraph::test::TestCase<TestEngine>(function);

    test_case.add_input(std::vector<uint8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 10, 20, 30});  // x
    test_case.add_input(std::vector<float>{1.0f, 2.0f, 4.0f});                         // scale
    test_case.add_input(std::vector<uint8_t>{0, 0, 0});                                // zero_point

    test_case.add_expected_output<float>(
        {3, 4},
        std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 2.0f, 4.0f, 6.0f, 0.0f, 40.0f, 80.0f, 120.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quant_conv_linear) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/quant_conv_lin.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    std::vector<std::vector<std::uint8_t>> inputs;
    test_case.add_input(std::vector<uint8_t>{ 1,  2,  3,  4,  5,  6,  7,  8,  9,
                                             10, 11, 12, 13, 14, 15, 16, 17, 18,
                                             19, 20, 21, 22, 23, 24, 25, 26, 27,
                                             28, 29, 30, 31, 32, 33, 34, 35, 36,
                                             37, 38, 39, 40, 41, 42, 43, 44, 45,
                                             46, 47, 48, 49, 50, 51, 52, 53, 54,
                                             55, 56, 57, 58, 59, 60, 61, 62, 63,
                                             64, 65, 66, 67, 68, 69, 70, 71, 72,
                                             73, 74, 75, 76, 77, 78, 79, 80, 81});

    test_case.add_expected_output<uint8_t>({1, 1, 9, 9}, std::vector<uint8_t>{ 2,  3,  3,  3,  4,  4,  4,  5,  2,
                                                                               4,  6,  7,  8,  8,  9,  9, 10,  3,
                                                                               8, 11, 12, 13, 13, 14, 14, 15,  5,
                                                                              11, 16, 17, 18, 18, 19, 19, 20,  7,
                                                                              14, 22, 22, 23, 23, 24, 24, 25,  8,
                                                                              18, 27, 27, 28, 28, 29, 29, 30, 10,
                                                                              21, 32, 32, 33, 33, 34, 34, 35, 12,
                                                                              24, 37, 37, 38, 38, 39, 40, 40, 13,
                                                                              17, 26, 27, 27, 27, 28, 28, 28, 9});
    //clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quant_conv_linear_2d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/qlinear_conv_2d.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input_from_file<uint8_t>(TEST_FILES, "onnx/qlinearconv2d/x.bin");
    test_case.add_input(std::vector<float>{0.00369204697199166f});  // x_scale
    test_case.add_input(std::vector<uint8_t>{132});                 // x_zero_point
    test_case.add_input(std::vector<uint8_t>{0});                   // w
    test_case.add_input(std::vector<float>{0.00172794575337321f});  // w_scale
    test_case.add_input(std::vector<uint8_t>{255});                 // w_zero_point
    test_case.add_input(std::vector<float>{0.00162681262008846f});  // y_scale
    test_case.add_input(std::vector<uint8_t>{123});                 // y_zero_point

    test_case.add_expected_output_from_file<uint8_t>({1, 1, 7, 7}, TEST_FILES, "onnx/qlinearconv2d/y.bin");
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quant_conv_linear_3d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/qlinear_conv_3d.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{130,  14, 244,  53,
                                             244, 119, 236,  79,
                                               9, 138,  93,  62,
                                              66, 158,  81, 176,

                                             225, 118, 160, 117,
                                             246,  69, 172,  50,
                                              23,  42, 139,  0,
                                             146, 157, 248, 251,

                                              30, 112,  99, 138,
                                             190,  22, 143, 186,
                                             199, 148, 190, 148,
                                              89,  16, 134, 220,

                                             191,  69,  34,   5,
                                             156, 255, 196, 134,
                                              49, 233, 220, 129,
                                             107, 220, 172, 124});  // x
    test_case.add_input(std::vector<float>{0.00389225385151803f});  // x_scale
    test_case.add_input(std::vector<uint8_t>{127});                 // x_zero_point
    test_case.add_input(std::vector<uint8_t>{255});                 // w
    test_case.add_input(std::vector<float>{0.00128723995294422f});  // w_scale
    test_case.add_input(std::vector<uint8_t>{0});                   // w_zero_point
    test_case.add_input(std::vector<float>{0.0011764180380851f});   // y_scale
    test_case.add_input(std::vector<uint8_t>{128});                 // y_zero_point

    test_case.add_expected_output<uint8_t>({1, 1, 4, 4, 4},
                                           {128, 128, 128, 128,
                                            128, 128, 128, 128,
                                            128, 128, 128, 128,
                                            128, 128, 128, 128,

                                            128, 128, 128, 128,
                                            128, 131, 255, 128,
                                            128,   0,  91, 128,
                                            128, 128, 128, 128,

                                            128, 128, 128, 128,
                                            128,  23,  98, 128,
                                            128, 206, 196, 128,
                                            128, 128, 128, 128,

                                            128, 128, 128, 128,
                                            128, 128, 128, 128,
                                            128, 128, 128, 128,
                                            128, 128, 128, 128});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_quant_conv_linear_onnx_example) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantization/quant_conv_linear_onnx_example.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{255, 174, 162,  25, 203, 168,  58,
                                              15,  59, 237,  95, 129,   0,  64,
                                              56, 242, 153, 221, 168,  12, 166,
                                             232, 178, 186, 195, 237, 162, 237,
                                             188,  39, 124,  77,  80, 102,  43,
                                             127, 230,  21,  83,  41,  40, 134,
                                             255, 154,  92, 141,  42, 148, 247});  // x
    test_case.add_input(std::vector<float>{0.00369204697f});                       // x_scale
    test_case.add_input(std::vector<uint8_t>{132});                                // x_zero_point
    test_case.add_input(std::vector<uint8_t>{0});                                  // w
    test_case.add_input(std::vector<float>{0.00172794575f});                       // w_scale
    test_case.add_input(std::vector<uint8_t>{255});                                // w_zero_point
    test_case.add_input(std::vector<float>{0.00162681262f});                       // y_scale
    test_case.add_input(std::vector<uint8_t>{123});                                // y_zero_point

    test_case.add_expected_output<uint8_t>({1, 1, 7, 7}, std::vector<uint8_t>{  0,  81,  93, 230,  52,  87, 197,
                                                                              240, 196,  18, 160, 126, 255, 191,
                                                                              199,  13, 102,  34,  87, 243,  89,
                                                                               23,  77,  69,  60,  18,  93,  18,
                                                                               67, 216, 131, 178, 175, 153, 212,
                                                                              128,  25, 234, 172, 214, 215, 121,
                                                                                0, 101, 163, 114, 213, 107,   8});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_qlinear_matmul_2d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/qlinear_matmul.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input(std::vector<uint8_t>{208, 236, 0, 238, 3, 214, 255, 29});                      // T1
    test_case.add_input(std::vector<float>{0.0066f});                                                  // a_scale
    test_case.add_input(std::vector<uint8_t>{113});                                                    // a_zero_point
    test_case.add_input(std::vector<uint8_t>{152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247});  // T2
    test_case.add_input(std::vector<float>{0.00705f});                                                 // b_scale
    test_case.add_input(std::vector<uint8_t>{114});                                                    // b_zero_point
    test_case.add_input(std::vector<float>{0.0107f});                                                  // y_scale
    test_case.add_input(std::vector<uint8_t>{118});                                                    // y_zero_point

    test_case.add_expected_output({2, 3}, std::vector<uint8_t>{168, 115, 255, 1, 66, 151});  // T3
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_2d_simple_zero_point) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{11, 7, 3,
                                             10, 6, 2,
                                              9, 5, 1,
                                              8, 4, 0});                        // A
    test_case.add_input(std::vector<uint8_t>{1, 4,
                                             2, 5,
                                             3, 6});                            // B
    test_case.add_input(std::vector<uint8_t>{12});                              // a_zero_point
    test_case.add_input(std::vector<uint8_t>{0});                               // b_zero_point

    test_case.add_expected_output({4, 2}, std::vector<int32_t>{-38,  -83,
                                                               -44,  -98,
                                                               -50, -113,
                                                               -56, -128});     // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_int8) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_int8.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<int8_t>{-3,  7, 5, -6,
                                             4, -5, 8,  7});                            // A
    test_case.add_input(std::vector<int8_t>{ 5, -3,  7,  8,
                                            -6, -8, -3,  6,
                                             7,  9,  9, -5,
                                             8,  7, -6,  7});                           // B
    test_case.add_input(std::vector<int8_t>{5});                                        // a_zero_point
    test_case.add_input(std::vector<int8_t>{5});                                        // b_zero_point

    test_case.add_expected_output({2, 4}, std::vector<int32_t>{-55,  16, 89, -44,
                                                               122, 154, 68, -39});     // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_vectorized_zero_point) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_vectorized_zero_point.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{11, 22, 33, 44, 55,
                                             22, 33, 44, 55, 66,
                                             33, 44, 55, 66, 77,
                                             44, 55, 66, 77, 88});                  // A
    test_case.add_input(std::vector<uint8_t>{ 13,  1,   3,
                                              21, 49,  31,
                                               9,  0,   2,
                                             107,  7,  94,
                                               1, 63, 127});                        // B
    test_case.add_input(std::vector<uint8_t>{33, 44, 55, 66});                      // a_zero_point
    test_case.add_input(std::vector<uint8_t>{10, 20, 30});                          // b_zero_point

    test_case.add_expected_output({4, 3}, std::vector<int32_t>{682, 902, 3421,
                                                               682, 902, 3421,
                                                               682, 902, 3421,
                                                               682, 902, 3421});     // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_no_zero_point) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_no_zero_point.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{11, 22, 33, 44, 55,
                                             22, 33, 44, 55, 66,
                                             33, 44, 55, 66, 77,
                                             44, 55, 66, 77, 88});                      // A
    test_case.add_input(std::vector<uint8_t>{ 13,  1,   3,
                                              21, 49,  31,
                                               9,  0,   2,
                                             107,  7,  94,
                                               1, 63, 127});                            // B

    test_case.add_expected_output({4, 3}, std::vector<int32_t>{ 5665, 4862, 11902,
                                                                7326, 6182, 14729,
                                                                8987, 7502, 17556,
                                                               10648, 8822, 20383});    // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_2d_x_3d) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_2d_x_3d.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<int8_t>{7, -3,  1, 2,
                                            0,  2, -4, 6});                         // A
    test_case.add_input(std::vector<int8_t>{1, -13, 10,
                                            2, -16, 14,
                                            3, -19, 18,
                                            4, -22,  22,

                                            -1, 13, -10,
                                            -2, 16, -14,
                                            -3, 19, -18,
                                            -4, 22, -22});                          // B
    test_case.add_input(std::vector<int8_t>{-4});                                   // a_zero_point

    test_case.add_expected_output({2, 2, 3}, std::vector<int32_t>{52, -386, 346,
                                                                  56, -368, 344,

                                                                  -52, 386, -346,
                                                                  -56, 368, -344}); // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_3d_x_2d) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_3d_x_2d.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<int8_t>{-13, 11, -1, -2,
                                              4, -2,  3, 10,

                                              8, -2,  4,  5,
                                             -4, -3,  1,  2});                         // A
    test_case.add_input(std::vector<int8_t>{  1, -3,   5,
                                              7, -2, -10,
                                            -13,  9,   7,
                                             11,  3,  -3});                             // B
    test_case.add_input(std::vector<int8_t>{4});                                        // a_zero_point
    test_case.add_input(std::vector<int8_t>{-3});                                       // a_zero_point

    test_case.add_expected_output({2, 2, 3}, std::vector<int32_t>{-32, -89, -235,
                                                                   34,  18,   32,

                                                                  -30,   0,   74,
                                                                 -100, -55,  -45});     // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_3d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_3d.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{125, 135, 145, 155,
                                             130, 140, 150, 160,

                                             125, 135, 145, 155,
                                             130, 140, 150, 160});                          // A
    test_case.add_input(std::vector<int8_t>{-10, -5,  0,   5,
                                             -5,  0,  5,  10,
                                             -5, -4, -3,  -2,
                                             -1,  0,  1,   2,

                                             10,  5,  0,  -5,
                                              5,  0, -5, -10,
                                              5,  4,  3,   2,
                                              1,  0, -1,  -2});                             // B
    test_case.add_input(std::vector<uint8_t>{150});                                         // a_zero_point
    test_case.add_input(std::vector<int8_t>{5,  10,  15,  20,
                                           -5, -10, -15, -20});                             // b_zero_point

    test_case.add_expected_output({2, 2, 4}, std::vector<int32_t>{545,  545,  545,  545,
                                                                  340,  300,  260,  220,

                                                                 -545, -545, -545, -545,
                                                                 -340, -300, -260, -220});  // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_4d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_4d.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{0, 1,  2,  3,
                                             4, 5,  6,  7,
                                             8, 9, 10, 11,

                                             12, 13, 14,
                                             15, 16, 17,
                                             18, 19, 20,
                                             21, 22, 23});                                      // A
    test_case.add_input(std::vector<uint8_t>{0,  1,  2,
                                             3,  4,  5,
                                             6,  7,  8,
                                             9,  10, 11,

                                             12, 13, 14,
                                             15, 16, 17,
                                             18, 19, 20,
                                             21, 22, 23});                                      // B
    test_case.add_input(std::vector<uint8_t>{0});                                               // a_zero_point
    test_case.add_input(std::vector<uint8_t>{0});                                               // b_zero_point

    test_case.add_expected_output<int32_t>({1, 2, 3, 3}, std::vector<int32_t> {42,  48,  54,
                                                                              114, 136, 158,
                                                                              186, 224, 262,

                                                                               906,  960, 1014,
                                                                              1170, 1240, 1310,
                                                                              1434, 1520, 1606});  // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_4d_zero_point) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_4d.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{0,  1,  2,  3,
                                             4,  5,  6,  7,
                                             8,  9, 10, 11,

                                            12, 13, 14, 15,
                                            16, 17, 18, 19,
                                            20, 21, 22, 23});                                   // A
    test_case.add_input(std::vector<uint8_t>{0,  1,  2,
                                             3,  4,  5,
                                             6,  7,  8,
                                             9, 10, 11,

                                            12, 13, 14,
                                            15, 16, 17,
                                            18, 19, 20,
                                            21, 22, 23});                                       // B
    test_case.add_input(std::vector<uint8_t>{1});                                               // a_zero_point
    test_case.add_input(std::vector<uint8_t>{1});                                               // b_zero_point

    test_case.add_expected_output<int32_t>({1, 2, 3, 3}, std::vector<int32_t>{22,   24,   26,
                                                                              78,   96,  114,
                                                                             134,  168,  202,

                                                                             790,  840,  890,
                                                                            1038, 1104, 1170,
                                                                            1286, 1368, 1450}); // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_matmul_integer_matrix_zero_point) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_integer_matrix_zero_point.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{0, 1, 2, 3,
                                             4, 5, 6, 7,

                                             8,  9, 10, 11,
                                            12, 13, 14, 15});                                   // A
    test_case.add_input(std::vector<uint8_t>{0,  1,  2,
                                             3,  4,  5,
                                             6,  7,  8,
                                             9, 10, 11,

                                            12, 13, 14,
                                            15, 16, 17,
                                            18, 19, 20,
                                            21, 22, 23});                                       // B
    test_case.add_input(std::vector<uint8_t>{1,
                                             2,

                                             3,
                                             4});                                               // a_zero_point
    test_case.add_input(std::vector<uint8_t>{1, 2, 3,

                                             4, 5, 6});                                         // b_zero_point

    test_case.add_expected_output<int32_t>({1, 2, 2, 3}, std::vector<int32_t>{22,  22,  22,
                                                                              64,  64,  64,

                                                                             340, 340, 340,
                                                                             490, 490, 490});   // Y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_qlinear_matmul_3d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/qlinear_matmul_3d.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input(
        std::vector<uint8_t>{208, 236, 0, 238, 3, 214, 255, 29, 208, 236, 0, 238, 3, 214, 255, 29});  // T1
    test_case.add_input(std::vector<float>{0.0066f});                                                 // a_scale
    test_case.add_input(std::vector<uint8_t>{113});                                                   // a_zero_point
    test_case.add_input(std::vector<uint8_t>{152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247,
                                             152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247});  // T2
    test_case.add_input(std::vector<float>{0.00705f});                                                 // b_scale
    test_case.add_input(std::vector<uint8_t>{114});                                                    // b_zero_point
    test_case.add_input(std::vector<float>{0.0107f});                                                  // y_scale
    test_case.add_input(std::vector<uint8_t>{118});                                                    // y_zero_point

    test_case.add_expected_output({2, 2, 3},
                                  std::vector<uint8_t>{168, 115, 255, 1, 66, 151, 168, 115, 255, 1, 66, 151});  // T3
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer_simple_zero_point) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{11, 22, 33,
                                             44, 55, 66,
                                             77, 88, 99});                          // x
    test_case.add_input(std::vector<uint8_t>{1, 2,
                                             3, 4});                                // w
    test_case.add_input(std::vector<uint8_t>{111});                                 // x_zero_point
    test_case.add_input(std::vector<uint8_t>{1});                                   // w_zero_point

    test_case.add_expected_output({1, 1, 2, 2}, std::vector<int32_t>{-391, -325,
                                                                     -193, -127});  // y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer_int8) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer_int8.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<int8_t>{-11,  22, -33,
                                              44, -55,  66,
                                             -77,  88, -99});                       // x
    test_case.add_input(std::vector<int8_t>{ 1, -2,
                                             -3,  4});                              // w
    test_case.add_input(std::vector<int8_t>{-5});                                   // x_zero_point
    test_case.add_input(std::vector<int8_t>{-5});                                   // w_zero_point

    test_case.add_expected_output({1, 1, 2, 2}, std::vector<int32_t>{-307,  617,
                                                                      837, -747});  // y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer_no_zero_point) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer_no_zero_point.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<int8_t>{-100, -89, -78,
                                             -67, -56, -45,
                                             -34, -23, -12});                       // x
    test_case.add_input(std::vector<int8_t>{0, 1,
                                            2, 3});                                 // w

    test_case.add_expected_output({1, 1, 2, 2}, std::vector<int32_t>{-391, -325,
                                                                     -193, -127});  // y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer_vector_w_zero_point) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer_vector_w_zero_point.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{11, 22, 33, 44,
                                            55, 66, 77, 88,
                                            99, 88, 77, 66,
                                            55, 44, 33, 22,

                                             1,  2,  3,  4,
                                             5,  6,  7,  8,
                                             9, 10, 11, 12,
                                            13, 14, 15, 16});                       // x
    test_case.add_input(std::vector<uint8_t>{2, 2, 3,
                                            4, 5, 6,
                                            7, 8, 9,

                                            2, 2, 3,
                                            4, 5, 6,
                                            7, 8, 9});                              // w

    test_case.add_input(std::vector<uint8_t>{1});                                   // x_zero_point
    test_case.add_input(std::vector<uint8_t>{1, 2});                                // w_zero_point

    test_case.add_expected_output({2, 2, 2, 2}, std::vector<int32_t>{2702, 2647,
                                                                     2174, 1855,

                                                                     2183, 2095,
                                                                     1589, 1303,


                                                                      258,  295,
                                                                      406,  443,

                                                                      213,  241,
                                                                      325,  353});  // y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_conv_integer_overload) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/conv_integer_overload.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);

    // don't change style for better readibility
    // clang-format off
    test_case.add_input(std::vector<uint8_t>{255, 255, 255,
                                               0,   0,   0,
                                             255, 255, 255});                           // x
    test_case.add_input(std::vector<int8_t>{127, -128,
                                            -128, 127});                                // w
    test_case.add_input(std::vector<uint8_t>{255});                                     // x_zero_point
    test_case.add_input(std::vector<int8_t>{-128});                                     // w_zero_point

    test_case.add_expected_output({1, 1, 2, 2}, std::vector<int32_t>{-65025, -65025,
                                                                     -65025, -65025});  // y
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_fake_quantize_import_only) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantization/fake_quantize_const_inputs.onnx"));

    const Shape expected_output_shape{1, 2, 3, 4};
    EXPECT_EQ(function->get_output_size(), 1);
    EXPECT_EQ(function->get_output_shape(0), expected_output_shape);
    EXPECT_EQ(count_ops_of_type<op::v0::FakeQuantize>(function), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(function), 4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_fake_quantize_const_inputs_infer) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantization/fake_quantize_const_inputs.onnx"));

    const Shape data_shape{1, 2, 3, 4};
    const auto n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    std::iota(std::begin(input_data), std::end(input_data), 0);

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        data_shape,
        std::vector<float>{2.f,   2.f,   2.f,   2.f,   2.f,  5.5f, 5.5f, 5.5f, 5.5f, 9.f,  9.f,  9.f,
                           12.5f, 12.5f, 12.5f, 12.5f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_fake_quantize_nonconst_inputs_infer) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/quantization/fake_quantize_nonconst_inputs.onnx"));

    const Shape data_shape{1, 2, 3, 4};
    const size_t n_elements = shape_size(data_shape);
    std::vector<float> input_data(n_elements);
    std::iota(std::begin(input_data), std::end(input_data), 0);

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({3.f});
    // input_high
    test_case.add_input<float>({17.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    test_case.add_expected_output<float>(
        data_shape,
        std::vector<float>{2.f,   2.f,   2.f,   2.f,   2.f,  5.5f, 5.5f, 5.5f, 5.5f, 9.f,  9.f,  9.f,
                           12.5f, 12.5f, 12.5f, 12.5f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f});
    test_case.run();
}
