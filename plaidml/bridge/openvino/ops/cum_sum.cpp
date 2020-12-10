// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace {
template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION
        << "Dynamic slicing not currently supported by PlaidML plugin; all of indices, offsets and default index"
           "must be Constants.";
  }
}

edsl::Tensor reverse_tensor(edsl::Tensor reverse_crop, int64_t seq_axis) {
  std::vector<edsl::TensorDim> dims(reverse_crop.rank());
  reverse_crop.bind_dims(dims);
  std::vector<edsl::TensorIndex> I_idxs(reverse_crop.rank());
  std::vector<edsl::TensorIndex> O_idxs(I_idxs);
  O_idxs[seq_axis] = dims[seq_axis] - 1 - I_idxs[seq_axis];
  return edsl::Contraction().outShape(dims).outAccess(O_idxs).assign(reverse_crop(I_idxs));
}

edsl::Tensor exclued_first(edsl::Tensor I, int64_t axis) {
  std::vector<edsl::TensorDim> dims(I.rank());
  I.bind_dims(dims);

  std::vector<int> lo_pad(I.rank(), 0);
  std::vector<int> hi_pad(lo_pad);
  lo_pad[axis] = 1;

  auto pad_I = op::explicit_padding(I, lo_pad, hi_pad).padval(edsl::Constant(0));
  return edsl::gather(pad_I, edsl::index({dims[axis]}, 0)).axis(axis);
}

}  // namespace

namespace PlaidMLPlugin {

void registerCumSum() {
  registerOp("cumsum", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto* layer = ngraph::as_type<ngraph::opset3::CumSum>(ctx.layer);
    auto axis = cast_constant_operand<int64_t>(1, layer)[0];
    if (axis < 0) {
      axis = axis + I.rank();
    }
    auto is_exclusive = layer->is_exclusive();
    auto is_reverse = layer->is_reverse();

    auto I_reverse = is_reverse ? reverse_tensor(I, axis) : I;
    auto sum_tensor = op::cumsum(I_reverse, axis);
    auto O = is_exclusive ? exclued_first(sum_tensor, axis) : sum_tensor;
    auto O_reverse = is_reverse ? reverse_tensor(O, axis) : O;
    return edsl::make_tuple(O_reverse);
  });
}

}  // namespace PlaidMLPlugin