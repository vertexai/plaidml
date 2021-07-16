// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace plaidml::edsl;    // NOLINT[build/namespaces]

namespace {

// This function can be replaced by the same function in plaidml/op/lib/ops.cc in the future.
std::pair<TensorDim, TensorDim> compute_padding_and_output_size(  //
    const TensorDim& input_size,                                  //
    const TensorDim& filter_size,                                 //
    int64_t stride,                                               //
    plaidml::op::AutoPadMode autopad_mode,                        //
    int64_t pad_lo,                                               //
    int64_t pad_hi,                                               //
    int64_t dilation,                                             //
    int64_t data_dilation,                                        //
    bool use_ceil_for_output_shape) {
  // Effective input and filter sizes are the sizes after dilations are
  // accounted for. So a 4x3 filter dilated by (3, 2) has an effective filter
  // size of 10 and 5 for its 2 spatial dims

  auto I_eff = (data_dilation * (input_size - 1)) + 1;  // Effective Input Size
  auto F_eff = (dilation * (filter_size - 1)) + 1;      // Effective Filter Size
  int64_t ceil_term =
      use_ceil_for_output_shape ? stride - 1 : 0;  // TODO: Will need to confirm that this is the intended behavior
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    TensorDim pad_before(pad_lo);
    TensorDim output_size((I_eff + pad_lo + pad_hi - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == plaidml::op::AutoPadMode::VALID) {
    TensorDim pad_before(0);
    TensorDim output_size((I_eff - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == plaidml::op::AutoPadMode::SAME_LOWER || autopad_mode == plaidml::op::AutoPadMode::SAME_UPPER) {
    TensorDim output_size((I_eff + stride - 1 + ceil_term) / stride);
    int64_t lower_term = (autopad_mode == plaidml::op::AutoPadMode::SAME_LOWER) ? 1 : 0;
    TensorDim pad_before((max(0, (output_size - 1) * stride + F_eff - I_eff) + lower_term) / 2);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  THROW_IE_EXCEPTION << "Unexpected autopadding mode.";
}

edsl::Tensor compute_deformable_convolution(edsl::Tensor I,                      //
                                            edsl::Tensor OFF,                    //
                                            edsl::Tensor F,                      //
                                            std::vector<int64_t> I_shape,        //
                                            std::vector<int64_t> OFF_shape,      //
                                            std::vector<int64_t> F_shape,        //
                                            int64_t G,                           //
                                            int64_t DG,                          //
                                            int64_t rank,                        //
                                            std::vector<size_t> strides,         //
                                            std::vector<size_t> dilations,       //
                                            std::vector<TensorDim> pad_befores,  //
                                            plaidml::DType plaidml_type) {
  auto N = I_shape[0];
  auto CI = I_shape[1];
  auto CO = F_shape[0];
  auto OFF_C = OFF_shape[1];
  if (CI % G != 0 || CO % G != 0 || CI % DG != 0 || OFF_C % DG != 0) {
    THROW_IE_EXCEPTION << "Incorrect shape for DeformableConvolution.";
  }
  std::vector<int64_t> deformed_dims;
  for (auto i = 2; i < rank; ++i) {
    deformed_dims.push_back(F_shape[i] * OFF_shape[i]);
  }

  std::vector<int64_t> OFF_reshape_dims = {N, DG, rank - 2};
  OFF_reshape_dims.insert(OFF_reshape_dims.end() - 1, F_shape.begin() + 2, F_shape.end());
  OFF_reshape_dims.insert(OFF_reshape_dims.end(), OFF_shape.begin() + 2, OFF_shape.end());
  // For example, in 2D, after edsl::reshape, the shape of offset will be {N, DG, F_H, F_W, 2, OFF_H, OFF_W}.
  edsl::Tensor offset = edsl::reshape(OFF, OFF_reshape_dims);

  std::vector<int64_t> OFF_transpose_dims;
  OFF_transpose_dims.push_back(0);
  OFF_transpose_dims.push_back(1);
  for (auto i = 0; i < rank - 2; ++i) {
    OFF_transpose_dims.push_back(rank + 1 + i);
    OFF_transpose_dims.push_back(2 + i);
  }
  OFF_transpose_dims.push_back(rank);
  // For example, in 2D, after op::transpose, the shape of offset will be {N, DG, OFF_H, F_H, OFF_W, F_W, 2}.
  offset = op::transpose(offset, make_tuple<int64_t>(OFF_transpose_dims));

  std::vector<int64_t> OFF_multiply_dims = {N, DG, 1, rank - 2};
  OFF_multiply_dims.insert(OFF_multiply_dims.end() - 1, deformed_dims.begin(), deformed_dims.end());
  // For example, in 2D, after edsl::reshape, the shape of offset will be {N, DG, 1, OFF_H*F_H, OFF_W*F_W, 2}.
  offset = edsl::reshape(offset, OFF_multiply_dims);

  std::vector<int64_t> OFF_broadcast_dims = {N, DG, CI / DG, rank - 2};
  OFF_broadcast_dims.insert(OFF_broadcast_dims.end() - 1, deformed_dims.begin(), deformed_dims.end());
  std::vector<int64_t> OFF_broadcast_axes;
  for (auto i = 0; i < rank + 2; ++i) {
    OFF_broadcast_axes.push_back(i);
  }
  // For example, in 2D, after op::broadcast, the shape of offset will be {N, DG, CI/DG, OFF_H*F_H, OFF_W*F_W, 2}.
  offset = op::broadcast(offset, OFF_broadcast_dims, OFF_broadcast_axes);

  std::vector<int64_t> OFF_reshape_channel = {N, CI, rank - 2};
  OFF_reshape_channel.insert(OFF_reshape_channel.end() - 1, deformed_dims.begin(), deformed_dims.end());
  // For example, in 2D, after edsl::reshape, the shape of offset will be {N, CI, OFF_H*F_H, OFF_W*F_W, 2}.
  offset = edsl::reshape(offset, OFF_reshape_channel);

  std::vector<edsl::Tensor> index_vec(rank - 2);
  for (auto i = 0; i < rank - 2; ++i) {
    edsl::Tensor index_vec_0 =
        cast(edsl::index({TensorDim(OFF_shape[i + 2]), TensorDim(F_shape[i + 2])}, 0), plaidml_type);
    edsl::Tensor index_vec_1 =
        cast(edsl::index({TensorDim(OFF_shape[i + 2]), TensorDim(F_shape[i + 2])}, 1), plaidml_type);
    index_vec[i] = index_vec_0 * strides[i] + index_vec_1 * dilations[i] - pad_befores[i];

    std::vector<int64_t> index_reshape_dims(rank + 1, 1);
    index_reshape_dims[i + 2] = deformed_dims[i];
    index_vec[i] = edsl::reshape(index_vec[i], index_reshape_dims);

    std::vector<int64_t> index_broadcast_dims(rank + 1, 1);
    for (auto j = 2; j < rank; ++j) {
      index_broadcast_dims[j] = deformed_dims[j - 2];
    }
    std::vector<int64_t> index_broadcast_axes;
    for (auto j = 0; j < rank + 1; ++j) {
      index_broadcast_axes.push_back(j);
    }
    // For example, in 2D, after op::broadcast, the shape of index_vec[i] will be {1, 1, OFF_H*F_H, OFF_W*F_W, 1}.
    index_vec[i] = op::broadcast(index_vec[i], index_broadcast_dims, index_broadcast_axes);
  }
  edsl::Tensor index = op::concatenate(index_vec, -1);  // The shape of index is {1, 1, NEW_DIM, rank-2}
  edsl::Tensor deformed_index = offset + index;

  deformed_index = edsl::select(deformed_index < 0, cast(Tensor{-1.0}, plaidml_type), deformed_index);
  std::vector<edsl::Tensor> deformed_index_vec(rank - 2);
  for (auto i = 0; i < rank - 2; ++i) {
    deformed_index_vec[i] = edsl::gather(deformed_index, i).axis(-1);
    deformed_index_vec[i] = edsl::select(deformed_index_vec[i] > I_shape[i + 2],
                                         cast(Tensor{I_shape[i + 2]}, plaidml_type), deformed_index_vec[i]);
    deformed_index_vec[i] =
        edsl::select((deformed_index_vec[i] > (I_shape[i + 2] - 1)) && (deformed_index_vec[i] < I_shape[i + 2]),
                     cast(Tensor{I_shape[i + 2] - 1}, plaidml_type), deformed_index_vec[i]);
  }
  deformed_index = op::concatenate(deformed_index_vec, -1);
  int64_t deformed_dims_size = 1;
  for (auto i = 0; i < rank - 2; ++i) {
    deformed_dims_size *= deformed_dims[i];
  }
  deformed_index = edsl::reshape(deformed_index, {N, CI, deformed_dims_size, rank - 2});
  edsl::Tensor deformed_input =
      op::gatherND(I, deformed_index).batchDims(2).interpolationMode(InterpolationMode::LINEAR);
  std::vector<int64_t> deformed_input_reshape_dims;
  deformed_input_reshape_dims.push_back(N);
  deformed_input_reshape_dims.push_back(CI);
  deformed_input_reshape_dims.insert(deformed_input_reshape_dims.end(), deformed_dims.begin(), deformed_dims.end());
  deformed_input = edsl::reshape(deformed_input, deformed_input_reshape_dims);
  // Compute DeformableConvolution.
  edsl::Tensor result = op::convolution(deformed_input, F)
                            .strides(std::vector<int64_t>{F_shape.begin() + 2, F_shape.end()})
                            .autopad_mode(plaidml::op::AutoPadMode::VALID)
                            .input_layout(plaidml::op::TensorLayout::NCX)
                            .filter_layout(plaidml::op::TensorLayout::KCX)
                            .groups(G)
                            .autogroup_mode(plaidml::op::AutoGroupMode::EXPLICIT)
                            .group_layout(plaidml::op::GroupLayout::IN_K);
  return result;
}

}  // namespace

namespace PlaidMLPlugin {

void registerDeformableConvolution() {
  registerOp("DeformableConvolution", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::DeformableConvolution>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3);

    auto I = ctx.operands.at(0);
    auto OFF = ctx.operands.at(1);  // OFF(for offset) is the deformable values tensor in OpenVINO doc.
    auto F = ctx.operands.at(2);
    auto I_shape = I.compute_shape().sizes();
    auto OFF_shape = OFF.compute_shape().sizes();
    auto F_shape = F.compute_shape().sizes();

    auto G = layer->get_group();
    auto DG = layer->get_deformable_group();
    auto strides = layer->get_strides();
    auto dilations = layer->get_dilations();
    auto autopad_mode = to_plaidml(layer->get_auto_pad());
    auto type = layer->get_input_element_type(0);
    auto plaidml_type = to_plaidml(type);
    auto I_rank = I.rank();
    // Compute the spatial size of filter;
    auto F_spatial_size = 1;
    for (auto i = 0; i < F_shape.size() - 2; ++i) {
      F_spatial_size *= F_shape[i + 2];
    }
    // Validate the shape of offset.
    if (OFF_shape[1] != (I_rank - 2) * DG * F_spatial_size) {
      THROW_IE_EXCEPTION << "Incorrect shape for DeformableConvolution.";
    }
    // Compute manual_padding.
    std::vector<int64_t> manual_padding;
    if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
      auto pads_begin = layer->get_pads_begin();
      auto pads_end = layer->get_pads_end();
      manual_padding.insert(manual_padding.end(), pads_begin.begin(), pads_begin.end());
      manual_padding.insert(manual_padding.end(), pads_end.begin(), pads_end.end());
    }
    while (manual_padding.size() < 2 * (I_rank - 2)) {
      manual_padding.push_back(0);
    }
    // Compute pad_before and the shape of output.
    std::vector<TensorDim> pad_befores;
    for (auto i = 0; i < I_rank - 2; ++i) {
      auto pad_before_and_output = compute_padding_and_output_size(TensorDim(I_shape[i + 2]),       //
                                                                   TensorDim(F_shape[i + 2]),       //
                                                                   strides[i],                      //
                                                                   autopad_mode,                    //
                                                                   manual_padding[i],               //
                                                                   manual_padding[i + I_rank - 2],  //
                                                                   dilations[i],                    //
                                                                   1,                               //
                                                                   false);
      pad_befores.push_back(pad_before_and_output.first);
    }
    // Compute DeformableConvolution.
    edsl::Tensor O = compute_deformable_convolution(I, OFF, F, I_shape, OFF_shape, F_shape, G, DG, I_rank, strides,
                                                    dilations, pad_befores, plaidml_type);
    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
