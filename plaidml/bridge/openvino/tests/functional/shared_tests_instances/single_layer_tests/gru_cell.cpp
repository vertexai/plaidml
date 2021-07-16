// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/gru_cell.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<bool> should_decompose{
    // true will decompose GRUCell to component ops and skip plaidml GRUCell implementation
    true,
    false,
};
std::vector<size_t> batch{5};
std::vector<size_t> hidden_size{
    1,
    10,
};
std::vector<size_t> input_size{
    1,
    30,
};
// When set `relu` as Gate gate activation and forbid clip, it
// potentially will get a big error.
// This precision error seems potentially related to
// differences between fp16 PlaidML and fp32 reference implementations.
std::vector<std::vector<std::string>> activations = {
    {"relu", "tanh"},
    {"tanh", "sigmoid"},
    {"sigmoid", "tanh"},
    {"tanh", "relu"},
};
std::vector<float> clips = {
    // 0.0f,
    0.7f,
};
std::vector<bool> linear_before_reset = {
    true,
    false,
};
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(GRUCellCommon, GRUCellTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(should_decompose),                //
                            ::testing::ValuesIn(batch),                           //
                            ::testing::ValuesIn(hidden_size),                     //
                            ::testing::ValuesIn(input_size),                      //
                            ::testing::ValuesIn(activations),                     //
                            ::testing::ValuesIn(clips),                           //
                            ::testing::ValuesIn(linear_before_reset),             //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        GRUCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, GRUCellTest,
                        ::testing::Combine(                                                 //
                            ::testing::Values(false),                                       //
                            ::testing::Values(3),                                           //
                            ::testing::Values(32),                                          //
                            ::testing::Values(16),                                          //
                            ::testing::Values(std::vector<std::string>({"tanh", "relu"})),  //
                            ::testing::Values(1.0f),                                        //
                            ::testing::Values(true),                                        //
                            ::testing::Values(InferenceEngine::Precision::FP32),            //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),            //
                        GRUCellTest::getTestCaseName);
}  // namespace
