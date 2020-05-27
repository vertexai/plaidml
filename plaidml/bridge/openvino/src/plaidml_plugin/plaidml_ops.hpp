// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "ie_layers.h"  // NOLINT[build/include_subdir]

// TODO: Clean & order nGraph includes
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "details/caseless.hpp"

#include "plaidml/edsl/edsl.h"

namespace PlaidMLPlugin {

// IE_SUPPRESS_DEPRECATED_START

struct Context {
  ngraph::Node* layer;
  std::vector<plaidml::edsl::Tensor> operands;
};

// IE_SUPPRESS_DEPRECATED_END

using Op = std::function<plaidml::edsl::Value(const Context& ctx)>;

class OpsRegistry {
 public:
  static OpsRegistry* instance() {
    static OpsRegistry registry;
    return &registry;
  }

  void registerOp(const std::string& name, Op op) {  //
    registry[name] = op;
  }

  const Op resolve(const std::string& name) {
    auto it = registry.find(name);
    if (it == registry.end()) {
      return nullptr;
    }
    return it->second;
  }

 private:
  InferenceEngine::details::caseless_unordered_map<std::string, Op> registry;
};

struct OpRegistration {
  OpRegistration(const std::string& name, Op op) { OpsRegistry::instance()->registerOp(name, op); }
};

}  // namespace PlaidMLPlugin
