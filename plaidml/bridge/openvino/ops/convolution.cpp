// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"
#include <iostream>

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerConvolution() {
  registerOp("Convolution", [](const Context& ctx) {
    std::cout << "Making OV convolution" << std::endl;
    auto* layer = ngraph::as_type<ngraph::opset1::Convolution>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto F = ctx.operands.at(1);
    std::vector<int> strides;
    for (auto stride : layer->get_strides()) {
      strides.push_back(stride);
    }
    std::vector<int> dilations;
    for (auto dilation : layer->get_dilations()) {
      dilations.push_back(dilation);
    }
    auto autopad_mode = to_plaidml(layer->get_auto_pad());
    auto result = op::convolution(I, F)
                      .strides(strides)
                      .dilations(dilations)
                      .autopad_mode(autopad_mode)
                      .input_layout(plaidml::op::TensorLayout::NCX)
                      .filter_layout(plaidml::op::TensorLayout::KCX)
                      .name(ctx.layer->get_friendly_name());
    if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
      std::vector<int> padding;
      for (auto pad : layer->get_pads_begin()) {
        padding.push_back(pad);
      }
      for (auto pad : layer->get_pads_end()) {
        padding.push_back(pad);
      }
      result.manual_padding(padding);
    }
    return edsl::make_tuple(result);
  });
}

}  // namespace PlaidMLPlugin
