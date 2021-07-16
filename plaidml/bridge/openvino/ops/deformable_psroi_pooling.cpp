// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerDeformablePSROIPooling() {
  registerOp("DeformablePSROIPooling", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 3 || ctx.operands.size() == 2);
    auto has_offset = ctx.operands.size() == 3;
    auto* layer = ngraph::as_type<ngraph::opset1::DeformablePSROIPooling>(ctx.layer);
    auto data = ctx.operands.at(0);
    auto rois = ctx.operands.at(1);
    auto output_dim = static_cast<int>(layer->get_output_dim());
    auto group_size = static_cast<int>(layer->get_group_size());
    auto spatial_scale = layer->get_spatial_scale();
    auto mode = layer->get_mode();
    auto spatial_bins_x = static_cast<int>(layer->get_spatial_bins_x());
    auto spatial_bins_y = static_cast<int>(layer->get_spatial_bins_y());
    auto trans_std = layer->get_trans_std();
    auto part_size = layer->get_part_size();

    auto data_shape = data.compute_shape().sizes();
    auto channels_in = data_shape[1];
    auto height_in = data_shape[2];
    auto width_in = data_shape[3];
    IE_ASSERT(channels_in == output_dim * group_size * group_size);

    auto rois_shape = rois.compute_shape().sizes();
    auto num_rois = static_cast<int>(rois_shape[0]);
    auto dim_one = edsl::TensorDim(1);

    rois = edsl::round(rois);
    // Left top ROI corner - (x1, y1)
    edsl::Tensor roi_lt = edsl::cast(op::slice(rois).add_dim(0, num_rois).add_dim(1, 3), DType::FLOAT32);
    roi_lt = roi_lt * spatial_scale - 0.5f;
    // Right bottom ROI corner - (x2, y2)
    edsl::Tensor roi_rb = edsl::cast(op::slice(rois).add_dim(0, num_rois).add_dim(3, 5), DType::FLOAT32);
    roi_rb = (roi_rb + 1.0f) * spatial_scale - 0.5f;

    // Calculate coordinates of each bin of roi boxes
    auto total_bins = num_rois * output_dim * group_size * group_size;
    auto roi_sizes = op::maximum(roi_rb - roi_lt, edsl::cast(edsl::Tensor(0.1f), DType::FLOAT32));
    auto bin_sizes = roi_sizes / static_cast<float>(group_size);
    bin_sizes = op::tile(bin_sizes, {1, output_dim * group_size * group_size});
    bin_sizes = edsl::reshape(bin_sizes, {total_bins, 2});

    auto bin_group_offsets = edsl::index({edsl::TensorDim(group_size), dim_one}, 0);
    auto bin_w_offsets = op::tile(bin_group_offsets, {group_size});
    auto bin_h_offsets = edsl::reshape(op::tile(bin_group_offsets, {1, group_size}), {group_size * group_size, 1});
    auto bin_offsets = op::concatenate({bin_w_offsets, bin_h_offsets}, 1);
    bin_offsets = op::tile(bin_offsets, {num_rois * output_dim});

    roi_lt = op::tile(roi_lt, {1, output_dim * group_size * group_size});
    roi_lt = edsl::reshape(roi_lt, {total_bins, 2});
    auto bin_lt_indices = roi_lt + bin_offsets * bin_sizes;

    if (has_offset) {
      // Calculate indices for offsets tensor
      auto offsets = ctx.operands.at(2);
      auto offset_shape = offsets.compute_shape().sizes();
      // offset tensor shape should be (#rois, 2*#classes, part_size, part_size)
      IE_ASSERT(offset_shape[2] == part_size);

      auto offset_bin_idx = bin_offsets * part_size / group_size;
      auto coords_sub_channels = offset_shape[1] / 2;
      auto class_sub_channels = output_dim / coords_sub_channels;
      auto output_channels = edsl::index({edsl::TensorDim(output_dim), dim_one}, 0);
      auto offset_channel_idx = output_channels / class_sub_channels;
      offset_channel_idx = op::tile(offset_channel_idx, {num_rois, group_size * group_size});
      offset_channel_idx = edsl::reshape(offset_channel_idx, {total_bins, 1});

      auto offset_roi_idx = edsl::index({edsl::TensorDim(num_rois), dim_one}, 0);
      offset_roi_idx = op::tile(offset_roi_idx, {1, output_dim * group_size * group_size});
      offset_roi_idx = edsl::reshape(offset_roi_idx, {total_bins, 1});
      auto bin_offset_indices = op::concatenate({offset_roi_idx, offset_channel_idx, offset_bin_idx}, 1);

      // Transform offset tensor, make offset x, y pair to be at last dimension
      auto offsets_reshaped = edsl::reshape(offsets, {num_rois, coords_sub_channels, 2, part_size, part_size});
      offsets_reshaped = op::transpose(offsets_reshaped, edsl::make_tuple<int64_t>({0, 1, 4, 3, 2}));

      // Gather offset values and apply them onto left top bin coordinate indices
      edsl::Tensor bin_offset_values = op::gatherND(offsets_reshaped, bin_offset_indices);
      roi_sizes = op::tile(roi_sizes, {1, output_dim * group_size * group_size});
      roi_sizes = edsl::reshape(roi_sizes, {total_bins, 2});
      bin_offset_values = bin_offset_values * trans_std * roi_sizes;
      bin_lt_indices = bin_lt_indices + bin_offset_values;
    }

    // Process spatial sub bins for each roi bin
    auto total_sub_bins = total_bins * spatial_bins_x * spatial_bins_y;
    auto zero_idx = edsl::index({dim_one}, 0);
    edsl::Tensor bin_width = edsl::gather(bin_sizes, 0).axis(1);
    edsl::Tensor bin_height = edsl::gather(bin_sizes, 1).axis(1);
    auto sub_bin_width = bin_width / static_cast<float>(spatial_bins_x);
    auto sub_bin_height = bin_height / static_cast<float>(spatial_bins_y);
    sub_bin_width = op::tile(sub_bin_width, {1, spatial_bins_x * spatial_bins_y});
    sub_bin_height = op::tile(sub_bin_height, {1, spatial_bins_x * spatial_bins_y});
    sub_bin_width = edsl::reshape(sub_bin_width, {total_sub_bins, 1});
    sub_bin_height = edsl::reshape(sub_bin_height, {total_sub_bins, 1});

    auto sub_bin_x_idx = edsl::index({edsl::TensorDim(spatial_bins_x), dim_one}, 0);
    sub_bin_x_idx = op::tile(sub_bin_x_idx, {total_bins * spatial_bins_y});
    sub_bin_x_idx = edsl::reshape(sub_bin_x_idx, {total_sub_bins, 1});
    auto sub_bin_y_idx = edsl::index({edsl::TensorDim(spatial_bins_y), dim_one}, 0);
    sub_bin_y_idx = op::tile(sub_bin_y_idx, {total_bins, spatial_bins_x});
    sub_bin_y_idx = edsl::reshape(sub_bin_y_idx, {total_sub_bins, 1});

    bin_lt_indices = op::tile(bin_lt_indices, {1, spatial_bins_x * spatial_bins_y});
    bin_lt_indices = edsl::reshape(bin_lt_indices, {total_sub_bins, 2});
    auto bin_lt_x_idx = edsl::gather(bin_lt_indices, 0).axis(1);
    auto bin_lt_y_idx = edsl::gather(bin_lt_indices, 1).axis(1);
    sub_bin_x_idx = bin_lt_x_idx + sub_bin_x_idx * sub_bin_width;
    sub_bin_y_idx = bin_lt_y_idx + sub_bin_y_idx * sub_bin_height;

    auto invalid_sub_x_idx = (sub_bin_x_idx < -0.5f) * (sub_bin_x_idx > (width_in - 0.5f));
    auto invalid_sub_y_idx = (sub_bin_y_idx < -0.5f) * (sub_bin_y_idx > (height_in - 0.5f));
    auto invalid_sub_idx = invalid_sub_x_idx * invalid_sub_y_idx;
    invalid_sub_idx = op::squeeze(invalid_sub_idx, {-1});

    auto fp_zero = edsl::cast(edsl::Tensor(0), DType::FLOAT32);
    auto width_bound = edsl::cast(edsl::Tensor(width_in - 1), DType::FLOAT32);
    auto height_bound = edsl::cast(edsl::Tensor(height_in - 1), DType::FLOAT32);
    sub_bin_x_idx = op::clip(sub_bin_x_idx, fp_zero, width_bound);
    sub_bin_y_idx = op::clip(sub_bin_y_idx, fp_zero, height_bound);

    auto sub_bin_indices = op::concatenate({sub_bin_y_idx, sub_bin_x_idx}, 1);

    edsl::Tensor batch_idx = edsl::gather(rois, 0).axis(1);
    batch_idx = op::tile(batch_idx, {1, output_dim * group_size * group_size * spatial_bins_x * spatial_bins_y});
    batch_idx = edsl::reshape(batch_idx, {total_sub_bins, 1});

    auto channel_indices = edsl::index({edsl::TensorDim(channels_in), dim_one}, 0);
    channel_indices = op::tile(channel_indices, {num_rois, spatial_bins_x * spatial_bins_y});
    channel_indices = edsl::reshape(channel_indices, {total_sub_bins, 1});

    // Calculate each sub-bin by bilinear interpolation
    sub_bin_indices = op::concatenate({batch_idx, channel_indices, sub_bin_indices}, 1);
    edsl::Tensor sub_bins = op::gatherND(data, sub_bin_indices).interpolationMode(edsl::InterpolationMode::LINEAR);

    sub_bins = edsl::select(invalid_sub_idx, edsl::cast(edsl::Tensor(0.0f), sub_bins.dtype()), sub_bins);
    sub_bins = edsl::reshape(sub_bins, {total_bins, spatial_bins_x * spatial_bins_y});
    // Do pooling on each group of sub-bins
    auto bins = op::sum(sub_bins, edsl::make_tuple(1)) / (spatial_bins_x * spatial_bins_y);

    auto roi_score_maps = edsl::reshape(bins, {num_rois, output_dim, group_size, group_size});
    return edsl::make_tuple(roi_score_maps);
  });
}
}  // namespace PlaidMLPlugin
