// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/lstm_cell.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<bool> should_decompose{
    // true will decompose LSTMCell to component ops and skip plaidml LSTMCell implementation
    true,
    false,
};
std::vector<size_t> batch{5};
std::vector<size_t> hidden_size{1, 10};
std::vector<size_t> input_size{1, 30};
std::vector<std::vector<std::string>> activations = {
    {"relu", "sigmoid", "tanh"},       {"sigmoid", "tanh", "tanh"}, {"tanh", "relu", "sigmoid"},
    {"sigmoid", "sigmoid", "sigmoid"}, {"tanh", "tanh", "tanh"},    {"relu", "relu", "relu"},
};
std::vector<float> clips{0.f, 0.7f};
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

// Note: For f16, skip relu/relu/relu as it creates tests that overflow 65504, the max value of FP16
std::vector<std::vector<std::string>> f16Activations = {
    {"relu", "sigmoid", "tanh"},       {"sigmoid", "tanh", "tanh"}, {"tanh", "relu", "sigmoid"},
    {"sigmoid", "sigmoid", "sigmoid"}, {"tanh", "tanh", "tanh"},
};

INSTANTIATE_TEST_CASE_P(LSTMCellCommon, LSTMCellTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(should_decompose),                //
                            ::testing::ValuesIn(batch),                           //
                            ::testing::ValuesIn(hidden_size),                     //
                            ::testing::ValuesIn(input_size),                      //
                            ::testing::ValuesIn(activations),                     //
                            ::testing::ValuesIn(clips),                           //
                            ::testing::Values(InferenceEngine::Precision::FP32),  //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        LSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(LSTMCellF16, LSTMCellTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(should_decompose),                //
                            ::testing::ValuesIn(batch),                           //
                            ::testing::ValuesIn(hidden_size),                     //
                            ::testing::ValuesIn(input_size),                      //
                            ::testing::ValuesIn(f16Activations),                  //
                            ::testing::ValuesIn(clips),                           //
                            ::testing::Values(InferenceEngine::Precision::FP16),  //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        LSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, LSTMCellTest,
                        ::testing::Combine(                                                            //
                            ::testing::Values(false),                                                  //
                            ::testing::Values(3),                                                      //
                            ::testing::Values(64),                                                     //
                            ::testing::Values(32),                                                     //
                            ::testing::Values(std::vector<std::string>({"relu", "sigmoid", "tanh"})),  //
                            ::testing::Values(std::numeric_limits<float>::infinity()),                 //
                            ::testing::ValuesIn(netPrecisions),                                        //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),                       //
                        LSTMCellTest::getTestCaseName);
}  // namespace
