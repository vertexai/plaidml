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

void registerGruCell() {
  registerOp("GruCell", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 5);
    auto xt = ctx.operands.at(0);    // input tensor
    auto ht_1 = ctx.operands.at(1);  // hidden state tensor
    auto W = ctx.operands.at(2);     // weight tensor [3*hidden_size, input_size]
    auto R = ctx.operands.at(3);     // recurrence weight tensor [3*hidden_size, input_size]
    auto B = ctx.operands.at(4);     // bias tensor linear_before_reset ? [4*hidden_size] : [3*hidden_size]

    auto* layer = ngraph::as_type<ngraph::opset4::GRUCell>(ctx.layer);
    auto hidden_size = layer->get_hidden_size();
    auto batch_size = xt.compute_shape().sizes()[0];

    auto activations = layer->get_activations();
    auto activation_f = activations.at(0);
    auto activation_g = activations.at(1);

    // TODO: activation_alpha and activation_beta are not used
    auto activations_alpha = layer->get_activations_alpha();
    auto activations_beta = layer->get_activations_beta();

    auto clip = layer->get_clip();
    auto should_clip = (clip > 0.f) && (clip != std::numeric_limits<float>::infinity());
    auto linear_before_reset = layer->get_linear_before_reset();

    auto xt_w = op::dot(xt, op::transpose(W));
    auto ht_r = op::dot(ht_1, op::transpose(R));

    edsl::Tensor xt_wz = op::slice(xt_w).add_dim(0, batch_size).add_dim(0, hidden_size);
    edsl::Tensor ht_rz = op::slice(ht_r).add_dim(0, batch_size).add_dim(0, hidden_size);
    edsl::Tensor bz = op::slice(B).add_dim(0, hidden_size);
    auto zt = xt_wz + ht_rz + bz;
    zt = clip_activation(activation_f, should_clip, clip, zt);

    edsl::Tensor xt_wr = op::slice(xt_w).add_dim(0, batch_size).add_dim(hidden_size, 2 * hidden_size);
    edsl::Tensor ht_rr = op::slice(ht_r).add_dim(0, batch_size).add_dim(hidden_size, 2 * hidden_size);
    edsl::Tensor br = op::slice(B).add_dim(hidden_size, 2 * hidden_size);
    auto rt = xt_wr + ht_rr + br;
    rt = clip_activation(activation_f, should_clip, clip, rt);

    edsl::Tensor xt_wh = op::slice(xt_w).add_dim(0, batch_size).add_dim(2 * hidden_size, 3 * hidden_size);
    edsl::Tensor bhw = op::slice(B).add_dim(2 * hidden_size, 3 * hidden_size);
    edsl::Tensor ht;
    if (linear_before_reset) {
      edsl::Tensor ht_rh = op::slice(ht_r).add_dim(0, batch_size).add_dim(2 * hidden_size, 3 * hidden_size);
      edsl::Tensor bhr = op::slice(B).add_dim(3 * hidden_size, 4 * hidden_size);
      ht = xt_wh + rt * (ht_rh + bhr) + bhw;
    } else {
      edsl::Tensor rh = op::slice(R).add_dim(2 * hidden_size, 3 * hidden_size).add_dim(0, hidden_size);
      ht = xt_wh + op::dot(rt * ht_1, op::transpose(rh)) + bhw;
    }
    ht = clip_activation(activation_g, should_clip, clip, ht);
    ht = (1 - zt) * ht + zt * ht_1;
    return edsl::make_tuple(ht);
  });
}

}  // namespace PlaidMLPlugin
