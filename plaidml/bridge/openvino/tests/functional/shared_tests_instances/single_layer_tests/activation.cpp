// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/activation.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
    {Sigmoid, {}},
    {Tanh, {}},
    {Relu, {}},
    {Exp, {}},
    {Log, {}},
    {Sign, {}},
    {Abs, {}},
    {Clamp, {{-2.0f, 2.0f}}},
    {Negative, {}},
    {Cos, {}},
    {Sin, {}},
    {Sqrt, {}},
    {Elu, {{0.1f}}},
    {HardSigmoid, {{0.2f, 0.5f}}},
    {Selu, {{1.6732f, 1.0507f}}},
    {Ceiling, {}},
    {Swish, {{1.0f}}},
    {Mish, {}},
    {HSwish, {}},
    {SoftPlus, {}},
    {HSigmoid, {}},
};

const std::vector<InferenceEngine::Precision> f32Precision = {
    InferenceEngine::Precision::FP32,
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationF32Types = {
    {Acos, {}},
    {Acosh, {}},
    {Asin, {}},
    {Asinh, {}},
    {Atan, {}},
    {Atanh, {}},
    {Cosh, {}},
    {Floor, {}},
    {Sinh, {}},
    {Tan, {}},
    {Erf, {}},
    {RoundHalfToEven, {}},
    {RoundHalfAwayFromZero, {}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
    {PReLu, {{-0.01f}}},
    {LeakyRelu, {{0.01f}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
    {{1, 50}, {{}}},
    {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
    {{1, 50}, {{1}, {50}}},
    {{1, 128}, {{1}, {128}}},
};

const auto basicCases = ::testing::Combine(                                //
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),  //
    ::testing::ValuesIn(netPrecisions),                                    //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),            //
    ::testing::Values(InferenceEngine::Layout::ANY),                       //
    ::testing::Values(InferenceEngine::Layout::ANY),                       //
    ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),            //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                     //
);

const auto f32OnlyCases = ::testing::Combine(                                 //
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationF32Types)),  //
    ::testing::ValuesIn(f32Precision),                                        //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),               //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),               //
    ::testing::Values(InferenceEngine::Layout::ANY),                          //
    ::testing::Values(InferenceEngine::Layout::ANY),                          //
    ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),               //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                        //
);

const auto basicPreluCases = ::testing::Combine(                                //
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)),  //
    ::testing::ValuesIn(netPrecisions),                                         //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),                 //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),                 //
    ::testing::Values(InferenceEngine::Layout::ANY),                            //
    ::testing::Values(InferenceEngine::Layout::ANY),                            //
    ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),            //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                          //
);

INSTANTIATE_TEST_CASE_P(                  //
    smoke_Activation_Basic,               //
    ActivationLayerTest,                  //
    basicCases,                           //
    ActivationLayerTest::getTestCaseName  //
);

INSTANTIATE_TEST_CASE_P(                  //
    smoke_Activation_F32_Only,            //
    ActivationLayerTest,                  //
    f32OnlyCases,                         //
    ActivationLayerTest::getTestCaseName  //
);

INSTANTIATE_TEST_CASE_P(                  //
    smoke_Activation_Basic_Prelu,         //
    ActivationLayerTest,                  //
    basicPreluCases,                      //
    ActivationLayerTest::getTestCaseName  //
);

INSTANTIATE_TEST_CASE_P(                  //
    smoke_Activation_Param_Basic_Prelu,   //
    ActivationParamLayerTest,             //
    basicPreluCases,                      //
    ActivationLayerTest::getTestCaseName  //
);

}  // namespace
