// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace plaidml::edsl;

namespace PlaidMLPlugin {

void registerDetectionOutput() {
  registerOp("DetectionOutput", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::DetectionOutput>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3 || ctx.operands.size() == 5);
    auto Location = ctx.operands.at(0);
    auto Confidence = ctx.operands.at(1);
    auto Priors = ctx.operands.at(2);

    edsl::Tensor ArmConfidence;
    edsl::Tensor ArmLocation;
    bool with_add_pred = false;
    if (ctx.operands.size() == 5) {
      ArmConfidence = ctx.operands.at(3);
      ArmLocation = ctx.operands.at(4);
      with_add_pred = true;
    }

    auto num_classes = layer->get_attrs().num_classes;
    auto background_label_id = layer->get_attrs().background_label_id;
    auto top_k = layer->get_attrs().top_k;
    auto variance_encoded_in_target = layer->get_attrs().variance_encoded_in_target;
    auto keep_top_k = layer->get_attrs().keep_top_k;
    auto code_type = layer->get_attrs().code_type;
    auto share_location = layer->get_attrs().share_location;
    auto nms_threshold = layer->get_attrs().nms_threshold;
    auto confidence_threshold = layer->get_attrs().confidence_threshold;
    auto clip_after_nms = layer->get_attrs().clip_after_nms;
    auto clip_before_nms = layer->get_attrs().clip_before_nms;
    auto decrease_label_id = layer->get_attrs().decrease_label_id;
    auto normalized = layer->get_attrs().normalized;
    auto input_height = layer->get_attrs().input_height;
    auto input_width = layer->get_attrs().input_width;
    auto objectness_score = layer->get_attrs().objectness_score;

    int prior_size = normalized ? 4 : 5;
    int prior_offset = normalized ? 0 : 1;
    int num_loc_classes = share_location ? 1 : num_classes;
    int i_h = normalized ? 1 : input_height;
    int i_w = normalized ? 1 : input_width;
    auto batch = Location.compute_shape().sizes()[0];
    auto num_priors = Priors.compute_shape().sizes()[2] / prior_size;
    auto priors_shape_variance = Priors.compute_shape().sizes()[1];

    std::vector<int64_t> location_shape = {batch, num_priors, num_loc_classes, 4};
    std::vector<int64_t> confidence_shape = {batch, num_priors, num_classes};
    std::vector<int64_t> priors_shape = {batch, priors_shape_variance, num_priors, prior_size};

    edsl::Tensor location = edsl::reshape(Location, location_shape);
    edsl::Tensor confidence = edsl::reshape(Confidence, confidence_shape);

    // priors tensor consists of prior_boxes and prior_variances.
    edsl::Tensor priors = edsl::reshape(Priors, priors_shape);
    edsl::Tensor prior_boxes =
        op::slice(priors).add_dim(0, batch).add_dim(0, 1).add_dim(0, num_priors).add_dim(prior_offset, prior_size);
    edsl::Tensor prior_variances;
    if (variance_encoded_in_target) {
      // Fill prior_variances with 1s.
      auto one = cast(edsl::Tensor{1}, prior_boxes.dtype());
      auto var_shape = prior_boxes.compute_shape().sizes();
      prior_variances = op::broadcast(one, var_shape, {});
    } else {
      prior_variances = op::slice(priors).add_dim(0, batch).add_dim(1, 2).add_dim(0, num_priors).add_dim(0, prior_size);
      if (!normalized) {
        // Reshape the prior_variances tensor from {batch, 1, num_priors, 5} to {batch, 1, num_priors, 4};
        prior_variances = edsl::reshape(prior_variances, {batch, 1, num_priors * prior_size});
        // Truncate the prior_variances tensor and drop the trailing invalid data.
        prior_variances = op::slice(prior_variances).add_dim(0, batch).add_dim(0, 1).add_dim(0, num_priors * 4);
        prior_variances = edsl::reshape(prior_variances, {batch, 1, num_priors, 4});
      }
    }

    prior_variances = op::squeeze(prior_variances, {1});
    prior_boxes = op::squeeze(prior_boxes, {1});

    edsl::Tensor decoded_bboxes;
    edsl::Tensor arm_loc;
    if (with_add_pred) {
      // Update confidence if there are 5 inputs.
      Tensor IX = edsl::index({edsl::TensorDim(num_priors)}, 0);
      IX = IX * 2 + 1;
      edsl::Tensor arm_conf = edsl::gather(ArmConfidence, IX).axis(1);
      arm_conf = op::repeat(op::unsqueeze(arm_conf, {-1})).count(num_classes).axis(2);
      confidence = edsl::select(arm_conf < objectness_score, cast(edsl::Tensor{0.0f}, DType::FLOAT32), confidence);

      arm_loc = edsl::reshape(ArmLocation, location_shape);
      arm_loc = op::transpose(arm_loc, edsl::make_tuple<int64_t>({0, 2, 1, 3}));
    }

    // Transpose the confidence to match the input shape of `scores` in NMS.
    // confidence -> {batch, num_boxes, num_classes}
    // scores -> {batch, num_classes, num_boxes}
    edsl::Tensor nms_conf = op::transpose(confidence, edsl::make_tuple<int64_t>({0, 2, 1}));
    // Transpose location to match the indices order of 'selected_indices' in NMS.
    // location -> {batch, num_boxes, num_classes, 4}
    // selected_indices {batch, num_classes, num_boxes}
    location = op::transpose(location, edsl::make_tuple<int64_t>({0, 2, 1, 3}));
    // Set the scores of background class to zeros.
    if (background_label_id > -1) {
      auto zero = cast(edsl::Tensor{0.0f}, nms_conf.dtype());
      std::vector<int64_t> slice_shape = {batch, 1, num_priors};
      auto bg_slice = op::broadcast(zero, slice_shape, {});
      edsl::Tensor idxs = edsl::index({edsl::TensorDim(num_priors)}, 0);
      edsl::Tensor scatter_idx = op::slice(idxs).add_dim(background_label_id, background_label_id + 1);
      nms_conf = edsl::scatter(nms_conf, scatter_idx, bg_slice).axis(1).mode(edsl::ScatterMode::UPDATE_SLICE);
    }

    bool center_decode_mode = false;
    if (code_type == "caffe.PriorBoxParameter.CENTER_SIZE") {
      center_decode_mode = true;
    }

    edsl::Tensor iou_threshold = cast(edsl::Tensor{nms_threshold}, DType::FLOAT32);
    edsl::Tensor score_threshold = cast(edsl::Tensor{confidence_threshold}, DType::FLOAT32);
    std::vector<edsl::Tensor> result = op::nms(prior_boxes, nms_conf, iou_threshold, score_threshold, top_k)
                                           .soft_nms_sigma(0.0f)
                                           .center_point_box(center_decode_mode)
                                           .sort_result_descending(false)
                                           .box_output_type(DType::INT32)
                                           .boxes_decode_mode(op::BoxesDecodeMode::SSD)
                                           .clip_before_nms(clip_before_nms)
                                           .clip_after_nms(clip_after_nms)
                                           .ssd_input_height(i_h)
                                           .ssd_input_width(i_w)
                                           .ssd_variances(prior_variances)
                                           .ssd_location(location)
                                           .ssd_with_arm_loc(with_add_pred)
                                           .ssd_arm_location(arm_loc)
                                           .nms_style(decrease_label_id ? op::NmsStyle::MXNET : op::NmsStyle::CAFFE)
                                           .share_location(share_location)
                                           .hard_suppression(false)
                                           .build();
    edsl::Tensor selected_indices = result[0];
    auto selected_indices_shape = selected_indices.compute_shape().sizes();
    edsl::Tensor selected_scores = result[1];
    edsl::Tensor valid_outputs = result[2];

    edsl::Tensor topk_results;
    if (keep_top_k[0] > -1 && selected_indices_shape[0] > keep_top_k[0]) {
      edsl::Tensor scores_slice = op::slice(selected_scores).add_dim(0, selected_indices_shape[0]).add_dim(2, 3);
      auto sorted_idxs = op::squeeze(edsl::argsort(scores_slice, 0, edsl::SortDirection::DESC), {-1});
      edsl::Tensor idxs_topk = op::slice(sorted_idxs).add_dim(0, keep_top_k[0]);
      auto idxs_topk_sorted = op::sort(idxs_topk, 0, edsl::SortDirection::ASC);
      topk_results = edsl::gather(selected_scores, idxs_topk_sorted).axis(0);
    } else {
      // Pad -1 at the end the valid data.
      auto neg_one = cast(edsl::Tensor{-1}, selected_indices.dtype());
      // The output is a 7-element tuple.
      int output_tuple_size = 7;
      std::vector<int64_t> pad_shape = {1, output_tuple_size};
      auto pad_slice = op::broadcast(neg_one, pad_shape, {});
      topk_results = op::concatenate({selected_scores, pad_slice}, 0);
    }

    return edsl::make_tuple(topk_results);
  });
}

}  // namespace PlaidMLPlugin
