// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerLstmCell() {
  registerOp("LstmCell", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 6);
    auto xt = ctx.operands.at(0);    // input tensor
    auto ht_1 = ctx.operands.at(1);  // hidden state tensor
    auto ct_1 = ctx.operands.at(2);  // cell state tensor
    auto W = ctx.operands.at(3);     // weight tensor [4 * hidden_size, input_size]
    auto R = ctx.operands.at(4);     // recurrence weight tensor [4 * hidden_size, input_size]
    auto B = ctx.operands.at(5);     // bias tensor [4 * hidden_size]

    auto* layer = ngraph::as_type<ngraph::opset4::LSTMCell>(ctx.layer);
    auto hidden_size = layer->get_hidden_size();
    auto batch_size = xt.compute_shape().sizes()[0];

    auto activations = layer->get_activations();
    auto activation_f = activations.at(0);
    auto activation_g = activations.at(1);
    auto activation_h = activations.at(2);

    // TODO: activation_alpha and activation_beta are not used
    auto activation_alpha = layer->get_activations_alpha();
    auto activation_beta = layer->get_activations_beta();

    auto clip = layer->get_clip();
    auto should_clip = (clip > 0.f) && (clip != std::numeric_limits<float>::infinity());

    auto gates_output = op::dot(xt, op::transpose(W)) + op::dot(ht_1, op::transpose(R)) + op::unsqueeze(B, {0});

    edsl::Tensor ft = op::slice(gates_output).add_dim(0, batch_size).add_dim(0, hidden_size);
    edsl::Tensor it = op::slice(gates_output).add_dim(0, batch_size).add_dim(hidden_size, 2 * hidden_size);
    edsl::Tensor ct = op::slice(gates_output).add_dim(0, batch_size).add_dim(2 * hidden_size, 3 * hidden_size);
    edsl::Tensor ot = op::slice(gates_output).add_dim(0, batch_size).add_dim(3 * hidden_size, 4 * hidden_size);

    ft = clip_activation(activation_f, should_clip, clip, ft);
    it = clip_activation(activation_f, should_clip, clip, it);
    ct = clip_activation(activation_g, should_clip, clip, ct);
    ot = clip_activation(activation_f, should_clip, clip, ot);

    auto Ct = ft * ct_1 + it * ct;
    auto Ht = ot * clip_activation(activation_h, false, clip, Ct);

    return edsl::make_tuple(Ht, Ct);
  });
}

}  // namespace PlaidMLPlugin
