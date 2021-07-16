// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NB: IE_SET_METRIC requires this header to be included first
#include "ie_metric_helpers.hpp"

#include "plaidml_executable_network.hpp"

#include <fstream>  // NOLINT[build/include_order]
#include <memory>   // NOLINT[build/include_order]
#include <vector>   // NOLINT[build/include_order]

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/convert_broadcast3.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>

#include "plaidml_builder.hpp"
#include "plaidml_infer_request.hpp"

using namespace InferenceEngine;

namespace PlaidMLPlugin {

#define IE_SET_METRIC(name, ...) [&] { IE_SET_METRIC_RETURN(name, __VA_ARGS__); }()

static plaidml::Program buildProgram(const CNNNetwork& network) {
  InputsDataMap inputsInfo = network.getInputsInfo();

  OutputsDataMap outputsInfo = network.getOutputsInfo();

  std::shared_ptr<ngraph::Function> func = ngraph::clone_function(*network.getFunction());
  ngraph::pass::Manager manager;
  manager.register_pass<ngraph::pass::BatchNormDecomposition>();
  manager.register_pass<ngraph::pass::ConvertBroadcast3>();
  manager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4>();
  manager.run_passes(func);
  ngraph::pass::ConstantFolding().run_on_function(func);
  return buildProgram(func, network.getName(), inputsInfo, outputsInfo);
}

IInferRequestInternal::Ptr PlaidMLExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                            OutputsDataMap networkOutputs) {
  return std::make_shared<PlaidMLInferRequest>(networkInputs, networkOutputs, program_);
}

PlaidMLExecutableNetwork::PlaidMLExecutableNetwork(const CNNNetwork& network, const std::string& device)
    : program_(buildProgram(network)) {
  program_.compile();
}

InferenceEngine::Parameter PlaidMLExecutableNetwork::GetMetric(const std::string& name) const {
  if (name == METRIC_KEY(SUPPORTED_METRICS)) {
    std::vector<std::string> metrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
    };
    return IE_SET_METRIC(SUPPORTED_METRICS, metrics);
  } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
    return IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, 1);
  } else {
    THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
  }
}

void PlaidMLExecutableNetwork::Export(std::ostream& model) {  //
  model << program_.str() << std::endl;
}

void PlaidMLExecutableNetwork::Export(const std::string& modelFileName) {
  std::ofstream modelFile(modelFileName, std::ios::out);
  if (modelFile.is_open()) {
    Export(modelFile);
  } else {
    THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export";
  }
}

}  // namespace PlaidMLPlugin
