// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset5::NonMaxSuppression;

namespace PlaidMLPlugin {

void registerNonMaxSuppression() {
  registerOp("NonMaxSuppression", [](const Context& ctx) {
    auto* layer = ngraph::as_type<NonMaxSuppression>(ctx.layer);
    // OPSET5 provides multiple templates, follow setup() choice now.
    IE_ASSERT(ctx.operands.size() == 6);
    if (ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(2)) == nullptr) {
      THROW_IE_EXCEPTION << "Dynamic max_output_boxes_per_class of NMS not currentlly supported by PlaidML plugin";
    }
    if (ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(5)) == nullptr) {
      THROW_IE_EXCEPTION << "Dynamic soft_nms_sigma of NMS not currentlly supported by PlaidML plugin";
    }
    auto Boxes = ctx.operands.at(0);
    auto Scores = ctx.operands.at(1);
    int max_output_boxes_per_class = layer->max_boxes_output_from_input();
    auto IOU_threshold = ctx.operands.at(3);
    auto Score_threshold = ctx.operands.at(4);
    float soft_nms_sigma = layer->soft_nms_sigma_from_input();
    bool center_point_box =
        layer->get_box_encoding() == ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER ? true : false;
    bool sort_result_descending = layer->get_sort_result_descending();
    DType box_output_type = to_plaidml(layer->get_output_type());

    std::vector<edsl::Tensor> Outputs =
        op::nms(Boxes, Scores, IOU_threshold, Score_threshold, max_output_boxes_per_class)
            .soft_nms_sigma(soft_nms_sigma)
            .center_point_box(center_point_box)
            .sort_result_descending(sort_result_descending)
            .box_output_type(box_output_type)
            .build();

    return edsl::make_tuple(Outputs);
  });
}

}  // namespace PlaidMLPlugin
