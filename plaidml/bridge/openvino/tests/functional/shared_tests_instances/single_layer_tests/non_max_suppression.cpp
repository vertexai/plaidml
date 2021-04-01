// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/non_max_suppression.hpp"

using LayerTestsDefinitions::InputShapeParams;
using LayerTestsDefinitions::NmsLayerTest;

namespace {
// Each InputShapeParams consists of Number of batches, Number of boxes, Number of classes
const std::vector<InputShapeParams> inShapeParams = {InputShapeParams{2, 9, 3}};

const std::vector<int32_t> maxOutBoxPerClass = {5, 10};
const std::vector<float> threshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
const std::vector<ngraph::op::v5::NonMaxSuppression::BoxEncodingType> encodType = {
    ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
    ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<ngraph::element::Type> outType = {ngraph::element::i32, ngraph::element::i64};

const auto nmsParams = ::testing::Combine(
    ::testing::ValuesIn(inShapeParams),
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),
                       ::testing::Values(InferenceEngine::Precision::I32),
                       ::testing::Values(InferenceEngine::Precision::FP32)),
    ::testing::ValuesIn(maxOutBoxPerClass), ::testing::ValuesIn(threshold), ::testing::ValuesIn(threshold),
    ::testing::ValuesIn(sigmaThreshold), ::testing::ValuesIn(encodType), ::testing::ValuesIn(sortResDesc),
    ::testing::ValuesIn(outType), ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(NMS, NmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);

const auto smokeParams =
    ::testing::Combine(::testing::Values(InputShapeParams{1, 5, 1}),
                       ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),
                                          ::testing::Values(InferenceEngine::Precision::I32),
                                          ::testing::Values(InferenceEngine::Precision::FP32)),
                       ::testing::Values(5), ::testing::Values(0.2f), ::testing::Values(0.3f), ::testing::Values(0.4f),
                       ::testing::ValuesIn(encodType), ::testing::Values(true), ::testing::Values(ngraph::element::i32),
                       ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(smoke, NmsLayerTest, smokeParams, NmsLayerTest::getTestCaseName);

}  // namespace
