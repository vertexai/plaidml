// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("lrn", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::LRN*>(ctx.layer);
  // IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  return edsl::make_tuple(op::lrn(I, {static_cast<int64_t>(layer->get_nsize())}));  // TODO
});

}  // namespace PlaidMLPlugin
