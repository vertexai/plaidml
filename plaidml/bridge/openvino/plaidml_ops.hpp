// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ie_icnn_network.hpp"  // TODO

// #include "plugin_api/caseless.hpp"  // Could be replaced with some work if necessary!!

#include "ngraph/node.hpp"

#include "plaidml/edsl/edsl.h"

namespace PlaidMLPlugin {

struct Context {
  ngraph::Node* layer;
  std::vector<plaidml::edsl::Tensor> operands;
};

using Op = std::function<plaidml::edsl::Value(const Context& ctx)>;

class OpsRegistry {
 public:
  static OpsRegistry* instance() {
    static OpsRegistry registry;
    return &registry;
  }

  void registerOp(const std::string& name, Op op) {  //
    registry[normalizedName(name)] = op;
  }

  const Op resolve(const std::string& name) {
    auto it = registry.find(normalizedName(name));
    if (it == registry.end()) {
      return nullptr;
    }
    return it->second;
  }

 private:
  std::unordered_map<std::string, Op> registry;

  // TODO: Would perhaps be nicer to revert to caseless_unordered_map from plugin_api/caseless.hpp if we can get it
  // building
  std::string normalizedName(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) { return std::tolower(ch); });
    return name;
  }
};

struct OpRegistration {
  OpRegistration(const std::string& name, Op op) { OpsRegistry::instance()->registerOp(name, op); }
};

}  // namespace PlaidMLPlugin
