// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerRnnCell() {
  registerOp("RnnCell", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 5);
    auto xt = ctx.operands.at(0);    // input tensor
    auto ht_1 = ctx.operands.at(1);  // hidden state tensor
    auto W = ctx.operands.at(2);     // weight tensor [hidden_size, input_size]
    auto R = ctx.operands.at(3);     // recurrence weight tensor [hidden_size, input_size]
    auto B = ctx.operands.at(4);     // bias tensor [hidden_size]

    auto* layer = ngraph::as_type<ngraph::opset4::RNNCell>(ctx.layer);

    auto activations = layer->get_activations();
    auto activation = activations.at(0);

    // TODO: activation_alpha and activation_beta are not used
    auto activations_alpha = layer->get_activations_alpha();
    auto activations_beta = layer->get_activations_beta();

    auto clip = layer->get_clip();
    auto should_clip = (clip > 0.f) && (clip != std::numeric_limits<float>::infinity());

    auto ht = op::dot(xt, op::transpose(W)) + op::dot(ht_1, op::transpose(R)) + B;
    ht = clip_activation(activation, should_clip, clip, ht);

    return edsl::make_tuple(ht);
  });
}

}  // namespace PlaidMLPlugin
