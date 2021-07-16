// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

#include "plaidml/core/core.h"

namespace PlaidMLPlugin {

class PlaidMLExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
 public:
  using Ptr = std::shared_ptr<PlaidMLExecutableNetwork>;

  PlaidMLExecutableNetwork(const InferenceEngine::CNNNetwork& network, const std::string& device);
  virtual ~PlaidMLExecutableNetwork() = default;

  InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
      InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) final;

  InferenceEngine::Parameter GetMetric(const std::string& name) const override;

  void Export(const std::string& modelFileName) final;
  void Export(std::ostream& networkModel) final;

 private:
  plaidml::Program program_;
};

}  // namespace PlaidMLPlugin
