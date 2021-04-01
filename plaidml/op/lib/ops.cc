// Copyright 2019 Intel Corporation.

#include "plaidml/op/lib/ops.h"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml/op/op.h"
#include "pmlc/util/logging.h"

using namespace plaidml::edsl;  // NOLINT
using namespace plaidml::op;    // NOLINT

namespace plaidml::op::lib {

// Forward declare the operations here so they can call each other
Value abs(const Value&);
Value all(const Value&);
Value any(const Value&);
Value argmax(const Value&);
Value binary_crossentropy(const Value&);
Value broadcast(const Value&);
Value clip(const Value&);
Value concatenate(const Value&);
Value convolution(const Value&);
Value cumprod(const Value&);
Value cumsum(const Value&);
Value dot(const Value&);
Value elu(const Value&);
Value explicit_padding(const Value&);
Value flip(const Value&);
Value hard_sigmoid(const Value&);
Value image_resize(const Value&);
Value max(const Value&);
Value maximum(const Value&);
Value mean(const Value&);
Value min(const Value&);
Value minimum(const Value&);
Value mvn(const Value&);
Value nms(const Value&);
Value l2norm(const Value&);
Value pool(const Value&);
Value prod(const Value&);
Value relu(const Value&);
Value reorg_yolo(const Value&);
Value repeat(const Value&);
Value reshape(const Value&);
Value sigmoid(const Value&);
Value slice(const Value&);
Value softmax(const Value&);
Value sort(const Value&);
Value spatial_padding(const Value&);
Value square(const Value&);
Value squeeze(const Value&);
Value sum(const Value&);
Value tile(const Value&);
Value transpose(const Value&);
Value unsqueeze(const Value&);
Value variance(const Value&);

namespace {

struct AggregationAxes {
  std::vector<TensorIndex> src_idxs;
  std::vector<TensorIndex> dst_idxs;
  std::vector<TensorIndex> reduce_idxs;
  std::vector<TensorDim> src_dims;
  std::vector<TensorDim> dst_dims;
  std::vector<TensorDim> reduce_dims;
  std::set<size_t> axes;

  AggregationAxes(size_t ndims, const Value& in_axes, bool keepdims) : src_idxs(ndims), src_dims(ndims) {
    IVLOG(5, "Received agg axes request with\n\tndims = " << ndims << "\n\tin_axes = " << in_axes
                                                          << "\n\tkeepdims = " << keepdims);
    if (in_axes.is_none()) {
      for (size_t i = 0; i < ndims; i++) {
        axes.insert(i);
      }
    } else if (in_axes.is_tuple()) {
      for (const auto& axis : in_axes.as_tuple()) {
        auto int_axis = axis.as_int();
        if (int_axis < 0) {
          int_axis = ndims + int_axis;
        }
        if (int_axis < 0 || ndims < static_cast<size_t>(int_axis)) {
          throw std::out_of_range(llvm::formatv("axis out of range: {0}", int_axis));
        }
        axes.insert(int_axis);
      }
    } else if (in_axes.is_int()) {
      auto axis = in_axes.as_int();
      if (axis < 0) {
        axis = ndims + axis;
      }
      axes = {static_cast<size_t>(axis)};
    } else {
      throw std::runtime_error("Invalid Value type for AggregationAxes: in_axes");
    }
    for (auto axis : axes) {
      reduce_idxs.push_back(src_idxs[axis]);
      reduce_dims.push_back(src_dims[axis]);
    }
    if (keepdims) {
      dst_idxs = src_idxs;
      dst_dims = src_dims;
      for (auto axis : axes) {
        dst_idxs[axis] = TensorIndex();
        dst_dims[axis] = TensorDim{1};
      }
    } else {
      for (size_t i = 0; i < ndims; i++) {
        if (!axes.count(i)) {
          dst_idxs.push_back(src_idxs[i]);
          dst_dims.push_back(src_dims[i]);
        }
      }
    }
  }
};

struct LRNAxes {
  std::vector<TensorIndex> src_idxs;
  std::vector<TensorIndex> dst_idxs;
  std::vector<Constraint> constraints;
  std::set<int64_t> axes;
  std::vector<int64_t> widths;

  LRNAxes(size_t ndims, const std::vector<int64_t>& in_axes, const std::vector<int64_t>& widths) : src_idxs(ndims) {
    IVLOG(5, "Received agg axes request with\n\tndims = " << ndims << "\n\tin_axes = " << in_axes
                                                          << "\n\twidths = " << widths);
    dst_idxs = src_idxs;
    for (int64_t axis : in_axes) {
      if (axis < 0) {
        axis += ndims;
      }
      if (axis < 0 || ndims < static_cast<size_t>(axis)) {
        throw std::out_of_range(llvm::formatv("axis out of range: {0}", axis));
      }
      axes.insert(axis);
    }
    if (axes.size() != widths.size()) {
      throw std::runtime_error(llvm::formatv("Inconsistent axis count and window width count in LRN ({0} vs {1})",
                                             axes.size(), widths.size()));
    }
    std::vector<TensorIndex> window_idxs(widths.size());
    size_t i = 0;  // to iterate through window_idxs and widths in tandem with axes
    for (auto ax_it = axes.begin(); ax_it != axes.end(); ax_it++, i++) {
      constraints.push_back(window_idxs[i] < widths[i]);
      src_idxs[*ax_it] = src_idxs[*ax_it] + window_idxs[i] - widths[i] / 2;
    }
  }
};

template <typename T>
T validate(int raw) {
  if (raw < 0 || raw >= static_cast<int>(T::_LAST)) {
    throw std::runtime_error("Invalid enumeration value");
  }
  return static_cast<T>(raw);
}

std::string to_string(AutoGroupMode mode) {
  switch (mode) {
    case AutoGroupMode::UNGROUPED:
      return "AutoGroupMode::UNGROUPED";
    case AutoGroupMode::EXPLICIT:
      return "AutoGroupMode::EXPLICIT";
    case AutoGroupMode::AUTO:
      return "AutoGroupMode::AUTO";
    case AutoGroupMode::DEPTHWISE:
      return "AutoGroupMode::DEPTHWISE";
    default:
      return "<<UNRECOGNIZED AutoGroupMode>>";
  }
}

std::ostream& operator<<(std::ostream& os, const AutoGroupMode& mode) {
  os << to_string(mode);
  return os;
}

std::string to_string(AutoPadMode mode) {
  switch (mode) {
    case AutoPadMode::EXPLICIT:
      return "AutoPadMode::EXPLICIT";
    case AutoPadMode::SAME_LOWER:
      return "AutoPadMode::SAME_LOWER";
    case AutoPadMode::SAME_UPPER:
      return "AutoPadMode::SAME_UPPER";
    case AutoPadMode::VALID:
      return "AutoPadMode::VALID";
    default:
      return "<<UNRECOGNIZED AutoPadMode>>";
  }
}

std::string to_string(PadMode mode) {
  switch (mode) {
    case PadMode::CONSTANT:
      return "PadMode::CONSTANT";
    case PadMode::EDGE:
      return "PadMode::EDGE";
    case PadMode::REFLECT:
      return "PadMode::REFLECT";
    case PadMode::SYMMETRIC:
      return "PadMode::SYMMETRIC";
    default:
      return "<<UNRECOGNIZED AutoPadMode>>";
  }
}

std::ostream& operator<<(std::ostream& os, const AutoPadMode& mode) {
  os << to_string(mode);
  return os;
}

std::string to_string(ConvDerivMode mode) {
  switch (mode) {
    case ConvDerivMode::NONE:
      return "ConvDerivMode::NONE";
    case ConvDerivMode::DATA:
      return "ConvDerivMode::DATA";
    case ConvDerivMode::FILTER:
      return "ConvDerivMode::FILTER";
    default:
      return "<<UNRECOGNIZED ConvDerivMode>>";
  }
}

std::ostream& operator<<(std::ostream& os, const ConvDerivMode& mode) {
  os << to_string(mode);
  return os;
}

std::string to_string(GroupLayout mode) {
  switch (mode) {
    case GroupLayout::NONE:
      return "GroupLayout::NONE";
    case GroupLayout::SEPARATE:
      return "GroupLayout::SEPARATE";
    case GroupLayout::IN_C:
      return "GroupLayout::IN_C";
    case GroupLayout::IN_K:
      return "GroupLayout::IN_K";
    default:
      return "<<UNRECOGNIZED GroupLayout>>";
  }
}

std::ostream& operator<<(std::ostream& os, const GroupLayout& mode) {
  os << to_string(mode);
  return os;
}

// TODO: Enable when needed
// std::string to_string(PoolMode m) {
//   switch (m) {
//    case PoolMode::AVG:
//     return "avg";
//    case PoolMode::MAX:
//     return "max";
//    case PoolMode::MIN:
//     return "min";
//    case PoolMode::SUM:
//     return "sum";
//   }
//   throw std::runtime_error("Unable to convert pooling mode to string due to unrecognized mode");
// }

size_t nonspatial_dims(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::NXC:
    case TensorLayout::NCX:
    case TensorLayout::KCX:
    case TensorLayout::XCK:
      return 2;
    case TensorLayout::GKCX:
    case TensorLayout::XGCK:
      return 3;
    default:
      throw std::runtime_error("Unrecognized layout when attempting to count non-spatial dimensions");
  }
}

std::string to_string(TensorLayout mode) {
  switch (mode) {
    case TensorLayout::NXC:
      return "TensorLayout::NXC";
    case TensorLayout::NCX:
      return "TensorLayout::NCX";
    case TensorLayout::KCX:
      return "TensorLayout::KCX";
    case TensorLayout::XCK:
      return "TensorLayout::XCK";
    case TensorLayout::GKCX:
      return "TensorLayout::GKCX";
    case TensorLayout::XGCK:
      return "TensorLayout::XGCK";
    default:
      return "<<UNRECOGNIZED TensorLayout>>";
  }
}

std::ostream& operator<<(std::ostream& os, const TensorLayout& mode) {
  os << to_string(mode);
  return os;
}

bool is_input_layout(TensorLayout layout) {  //
  return (layout == TensorLayout::NCX || layout == TensorLayout::NXC);
}

bool is_filter_layout(TensorLayout layout) {
  return (layout == TensorLayout::KCX || layout == TensorLayout::XCK || layout == TensorLayout::GKCX ||
          layout == TensorLayout::XGCK);
}

bool is_filter_layout_with_separate_groups(TensorLayout layout) {
  return (layout == TensorLayout::GKCX || layout == TensorLayout::XGCK);
}

size_t normalize_axis(int64_t axis, size_t ndims, std::string op_name = "") {
  bool negative_axis_given = false;
  if (axis < 0) {
    axis += ndims;
    negative_axis_given = true;
  }
  if (axis < 0 || ndims <= static_cast<size_t>(axis)) {
    if (negative_axis_given) {
      axis -= ndims;
    }
    auto op_name_str = op_name.empty() ? "" : llvm::formatv("for {0} op ", op_name).str();
    throw std::runtime_error(llvm::formatv("Axis out of range {0}(axis {1} requested for tensors with {2} dimensions)",
                                           op_name_str, axis, ndims));
  }
  return axis;
}

std::pair<TensorDim, TensorDim> compute_padding_and_output_size(  //
    const TensorDim& input_size,                                  //
    const TensorDim& filter_size,                                 //
    int64_t stride,                                               //
    AutoPadMode autopad_mode,                                     //
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
  if (autopad_mode == AutoPadMode::EXPLICIT) {
    TensorDim pad_before(pad_lo);
    TensorDim output_size((I_eff + pad_lo + pad_hi - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == AutoPadMode::VALID) {
    TensorDim pad_before(0);
    TensorDim output_size((I_eff - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == AutoPadMode::SAME_LOWER || autopad_mode == AutoPadMode::SAME_UPPER) {
    TensorDim output_size((I_eff + stride - 1 + ceil_term) / stride);
    int64_t lower_term = (autopad_mode == AutoPadMode::SAME_LOWER) ? 1 : 0;
    TensorDim pad_before((max(0, (output_size - 1) * stride + F_eff - I_eff) + lower_term) / 2);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  throw std::runtime_error(llvm::formatv("Unexpected autopadding mode: {0}", to_string(autopad_mode)));
}

std::pair<TensorDim, TensorDim> compute_padding_and_input_size(  //
    const TensorDim& output_size,                                //
    const TensorDim& filter_size,                                //
    int64_t stride,                                              //
    AutoPadMode autopad_mode,                                    //
    int64_t pad_lo,                                              //
    int64_t pad_hi,                                              //
    int64_t dilation,                                            //
    int64_t data_dilation) {
  // Note that this computes the smallest input_size that would produce the given output_size

  if (data_dilation != 1) {
    throw std::runtime_error("Cannot infer input data size for transposed convolution with data dilation");
  }

  auto O_eff = output_size;
  auto F_eff = (dilation * (filter_size - 1)) + 1;  // Effective Filter Size
  if (autopad_mode == AutoPadMode::EXPLICIT) {
    TensorDim pad_before(pad_lo);
    TensorDim input_size(O_eff * stride - pad_lo - pad_hi + F_eff - stride);
    return std::pair<TensorDim, TensorDim>(pad_before, input_size);
  }
  if (autopad_mode == AutoPadMode::VALID) {
    TensorDim pad_before(0);
    TensorDim input_size(O_eff * stride + F_eff - stride);
    return std::pair<TensorDim, TensorDim>(pad_before, input_size);
  }
  if (autopad_mode == AutoPadMode::SAME_LOWER || autopad_mode == AutoPadMode::SAME_UPPER) {
    TensorDim input_size(O_eff * stride - stride + 1);
    int64_t lower_term = (autopad_mode == AutoPadMode::SAME_LOWER) ? 1 : 0;
    TensorDim pad_before((max(0, (O_eff - 1) * stride + F_eff - input_size) + lower_term) / 2);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  throw std::runtime_error(llvm::formatv("Unexpected autopadding mode: {0}", to_string(autopad_mode)));
}

std::vector<int64_t>* extend_manual_padding(std::vector<int64_t>* pads, size_t rank) {
  // TODO: Perhaps we should throw for sizes != 0, rank, 2*rank?
  if (pads->size() > 2 * rank) {
    throw std::runtime_error(llvm::formatv(
        "Inconsistent spatial rank: operation with {0} spatial dimensions had {1} manual padding values given", rank,
        pads->size()));
  }
  while (pads->size() < rank) {
    pads->push_back(0);
  }
  while (pads->size() < 2 * rank) {
    // Where pad_hi isn't set, copy the pad_lo value
    pads->push_back(pads->at(pads->size() - rank));
  }
  return pads;
}

}  // namespace

Value abs(const Value& value) {
  IVLOG(1, "abs");
  auto args = value.as_tuple();
  if (args.size() != 1) {
    throw std::runtime_error("abs expects 1 argument");
  }
  auto I = args[0].as_tensor();
  auto O = select(I < 0., -I, I);
  return Value{O};
}

Value all(const Value& value) {
  IVLOG(1, "all");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("all expects 3 arguments");
  }
  auto I = args[0].as_tensor();
  auto axes = args[1];
  auto keepdims = args[2].as_bool();

  auto one = cast(Tensor{1}, I.dtype());
  auto zero = cast(Tensor{0}, I.dtype());
  auto I_as_bool = select(I == 0, zero, one);
  if (I.rank() == 0) {
    return Value{cast(I_as_bool, DType::UINT8)};
  }
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    return Value{cast(I_as_bool, DType::UINT8)};
  }

  AggregationAxes agg(I.rank(), axes, keepdims);

  I.bind_dims(agg.src_dims);
  Tensor O = Contraction(agg.dst_dims, agg.dst_idxs).product(I_as_bool(agg.src_idxs));
  return Value{cast(O, DType::UINT8)};
}

Value any(const Value& value) {
  IVLOG(1, "any");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("any expects 3 arguments");
  }
  auto I = args[0].as_tensor();
  auto axes = args[1];
  auto keepdims = args[2].as_bool();

  auto one = cast(Tensor{1}, I.dtype());
  auto zero = cast(Tensor{0}, I.dtype());
  auto I_as_bool = select(I == 0, zero, one);
  if (I.rank() == 0) {
    return Value{cast(I_as_bool, DType::UINT8)};
  }
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    return Value{cast(I_as_bool, DType::UINT8)};
  }

  AggregationAxes agg(I.rank(), axes, keepdims);

  I.bind_dims(agg.src_dims);
  Tensor S = Contraction(agg.dst_dims, agg.dst_idxs).sum(I_as_bool(agg.src_idxs));
  auto O = select(S == 0, zero, one);
  return Value{cast(O, DType::UINT8)};
}

Value argmax(const Value& value) {
  IVLOG(1, "argmax");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("argmax expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  auto axes = args[1];
  AggregationAxes agg(I.rank(), axes, false);
  I.bind_dims(agg.src_dims);
  Tensor M = Contraction(agg.dst_dims, agg.dst_idxs).max(I(agg.src_idxs));
  auto IX = index(agg.reduce_dims, 0);
  Tensor AM = Contraction(agg.dst_dims, agg.dst_idxs).max(cond(I(agg.src_idxs), M(agg.dst_idxs), IX(agg.reduce_idxs)));
  auto O = cast(AM, DType::UINT32);
  return Value{O};
}

Value binary_crossentropy(const Value& value) {
  IVLOG(1, "binary_crossentropy")
  auto args = value.as_tuple();

  // Read arguments
  if (args.size() != 3) {
    throw std::runtime_error("binary_crossentropy expects 3 arguments");
  }
  auto T = args[0].as_tensor();  // Targets Tensor
  auto raw_P = args[1];          // Predictions Tensor, before clipping
  auto epsilon = args[2].as_float();

  // Check args & set useful values
  if (epsilon < 0. || epsilon >= 0.5) {
    throw std::runtime_error(
        llvm::formatv("The epsilon used in binary_crossentropy must be between 0 and 0.5, received {0}", epsilon));
  }
  auto clip_inputs = make_tuple(raw_P, Value{epsilon}, Value{1. - epsilon});
  auto P = clip(clip_inputs).as_tensor();
  auto O = -T * log(P) - (1 - T) * log(1 - P);
  return Value{O};
}

Value broadcast(const Value& value) {
  IVLOG(1, "broadcast");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error(llvm::formatv("PlaidML broadcast op expects 3 arguments (received {0})", args.size()));
  }

  auto I = args[0].as_tensor();
  auto broadcast_axes = args[2].as_int_tuple();
  if (I.rank() != broadcast_axes.size()) {
    throw std::runtime_error(
        llvm::formatv("Mismatched broadcast axes (received {0} broadcast axes for input tensor of rank {1})",
                      broadcast_axes.size(), I.rank()));
  }

  auto input_shape = I.compute_shape().sizes();
  auto target_shape = args[1].as_int_tuple();

  std::vector<TensorDim> O_dims(target_shape.begin(), target_shape.end());
  std::vector<TensorDim> I_dims(I.rank());
  I.bind_dims(I_dims);

  std::vector<TensorIndex> I_idxs;
  std::vector<TensorIndex> O_idxs(target_shape.size());

  // Define input dims and indexes
  for (size_t i = 0; i < broadcast_axes.size(); i++) {
    if (input_shape[i] == 1) {
      I_idxs.emplace_back(0);
    } else {
      I_idxs.emplace_back(O_idxs.at(broadcast_axes[i]));
    }
  }

  Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return Value{O};
}

Value clip(const Value& value) {
  IVLOG(1, "clip");
  auto args = value.as_tuple();

  // Read arguments
  if (args.size() != 3) {
    throw std::runtime_error("clip expects 3 arguments");
  }
  auto I = args[0].as_tensor();
  auto raw_min = args[1];
  auto raw_max = args[2];

  auto O = I;
  if (!raw_min.is_none()) {
    auto min = raw_min.as_tensor();
    O = select(O > min, O, cast(min, O.dtype()));
  }
  if (!raw_max.is_none()) {
    auto max = raw_max.as_tensor();
    O = select(O < max, O, cast(max, O.dtype()));
  }
  return Value{O};
}

Value concatenate(const Value& value) {
  // TODO: Make errors nicer (e.g. when bind_dims fails)
  IVLOG(1, "concatenate")

  // Read Arguments
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("concatenate expects 2 arguments");
  }
  auto tensor_vals = args[0].as_tuple();
  auto raw_axis = args[1].as_int();

  // Initialize useful values
  std::vector<Tensor> tensors;
  for (auto tensor_val : tensor_vals) {
    tensors.push_back(tensor_val.as_tensor());
  }
  if (tensors.empty()) {
    throw std::runtime_error("The concatenate op requires at least one input tensor");
  }
  auto ndims = tensors[0].rank();
  auto axis = normalize_axis(raw_axis, ndims, "concatenate");
  std::vector<TensorDim> dims(ndims);
  std::vector<TensorIndex> I_idxs;  // These will be named
  std::vector<TensorIndex> O_idxs;
  std::vector<TensorDim> axis_dims(tensors.size());
  std::vector<TensorDim> axis_dim_subtotals;
  std::vector<Tensor> results;
  TensorIndex axis_idx("a");
  axis_dim_subtotals.emplace_back(0);
  for (size_t i = 0; i < ndims; ++i) {
    if (i == axis) {
      I_idxs.push_back(axis_idx);
    } else {
      I_idxs.emplace_back(llvm::formatv("n{0}", i));
    }
  }

  // Bind the various input dimensions
  for (size_t i = 0; i < tensors.size(); ++i) {
    dims[axis] = axis_dims[i];
    tensors[i].bind_dims(dims);
    axis_dim_subtotals.push_back(axis_dims[i] + axis_dim_subtotals.back());
  }
  // Set dims to the final output dims
  dims[axis] = axis_dim_subtotals[tensors.size()];  // There are tensors.size() + 1 TensorDims in the RHS vector
  O_idxs = I_idxs;                                  // The axis index will be overwritten during the loop

  // Compute each intermediate output
  for (size_t i = 0; i < tensors.size(); ++i) {
    O_idxs[axis] = axis_idx + axis_dim_subtotals[i];
    Tensor R = Contraction(dims, O_idxs).assign(tensors[i](I_idxs));
    results.emplace_back(R);
  }
  auto final_result = results[0];
  for (size_t i = 1; i < tensors.size(); ++i) {
    final_result = final_result + results[i];
  }

  return Value{final_result};
}

namespace {
// Convolution helper functions

size_t compute_conv_rank_validating_strides(std::vector<int64_t>* strides, const Tensor& I_or_O,
                                            TensorLayout input_layout) {
  // As a side effect of computing the spatial_rank, this fills in the default 1s to strides if it's empty
  size_t spatial_rank = strides->size();
  if (spatial_rank == 0) {
    // We are probably* being asked to infer a default strides value using the spatial rank from the tensors. So examine
    // the `I_or_O` tensor rank and reconstruct `strides`.
    // (*: It's also possible that we're dealing with a rank 0 conv, but the inferred strides will be unchanged in that
    // case anyway, so this is safe.)
    // If we ever extend possible layouts so that I and O may have different layouts, we will need to do this check in
    // different ways depending on whether deriv_mode is DATA or not

    // Infer spatial rank from input/output tensor rank
    spatial_rank = I_or_O.rank() - nonspatial_dims(input_layout);
    for (size_t i = 0; i < spatial_rank; i++) {
      strides->push_back(1);
    }
  }
  return spatial_rank;
}

void validate_conv_padding(const std::vector<int64_t>& manual_padding, AutoPadMode autopad_mode,
                           std::stringstream& args_log) {
  if (!manual_padding.empty() && autopad_mode != AutoPadMode::EXPLICIT) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error("Autopadding and manual padding both requested for single conv operation");
  }
}

void validate_conv_dilations_rank(size_t spatial_rank, std::vector<int64_t>* dilations, std::stringstream& args_log) {
  // Verifies dilations rank and loads default value (all 1s) if dilations empty
  if (dilations->empty()) {
    // We're being asked to infer a default value for dilations
    for (size_t i = 0; i < spatial_rank; i++) {
      dilations->push_back(1);
    }
  } else if (dilations->size() != spatial_rank) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error(
        llvm::formatv("Inconsistent spatial rank in conv op (expecting rank {0}, received {1}D dilations)",
                      spatial_rank, dilations->size()));
  }
}

void validate_conv_data_dilations_rank(size_t spatial_rank, std::vector<int64_t>* data_dilations,
                                       std::stringstream& args_log) {
  if (data_dilations->empty()) {
    // We're being asked to infer a default value for data_dilations
    for (size_t i = 0; i < spatial_rank; i++) {
      data_dilations->push_back(1);
    }
  } else if (data_dilations->size() != spatial_rank) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error(
        llvm::formatv("Inconsistent spatial rank in conv op (expecting rank {0}, received {1}D data_dilations)",
                      spatial_rank, data_dilations->size()));
  }
}

void validate_conv_input_layout(TensorLayout input_layout, std::stringstream& args_log) {
  if (!is_input_layout(input_layout)) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error("Input tensor layout requested in conv op does not apply to convolution input tensors");
  }
}

void validate_conv_filter_layout(TensorLayout filter_layout, std::stringstream& args_log) {
  if (!is_filter_layout(filter_layout)) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error("Filter tensor layout requested in conv op does not apply to convolution filter tensors");
  }
}

void validate_conv_input_rank(size_t const spatial_rank, const Tensor& I, TensorLayout input_layout,
                              ConvDerivMode deriv_mode, std::stringstream& args_log) {
  if (deriv_mode != ConvDerivMode::DATA && I.rank() - spatial_rank != nonspatial_dims(input_layout)) {
    // If we ever extend possible layouts so that I and O may have different layouts, we will
    // need to do this check in different ways depending on whether deriv_mode is DATA or not
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error(llvm::formatv(
        "Inconsistent spatial rank in conv op (expected spatial rank {0} but input tensor has {1} dimensions, and thus "
        "{2} spatial dims). (This error can also occur if the layout of I is incorrectly specified or interpreted.)",
        spatial_rank, I.rank(), (I.rank() - nonspatial_dims(input_layout))));
  }
}

void validate_conv_filter_rank(size_t spatial_rank, const Tensor& F, TensorLayout filter_layout,
                               ConvDerivMode deriv_mode, std::stringstream& args_log) {
  if (deriv_mode != ConvDerivMode::FILTER && F.rank() - spatial_rank != nonspatial_dims(filter_layout)) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error(
        llvm::formatv("Inconsistent spatial rank in conv op (expected spatial rank {0} but filter tensor has {1} "
                      "dimensions, and thus {2} spatial dims). (This error can also occur if the layout of F is "
                      "incorrectly specified or interpreted.)",
                      spatial_rank, F.rank(), (F.rank() - nonspatial_dims(filter_layout))));
  }
}

void validate_conv_filter_shape_rank(size_t spatial_rank, std::vector<int64_t> filter_shape,
                                     std::stringstream& args_log) {
  if (filter_shape.size() && (filter_shape.size() != spatial_rank)) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error(
        llvm::formatv("Filter shape manually specified with inconsistent rank (expected spatial rank {0} but "
                      "filter_shape has {1} dimensions)",
                      spatial_rank, filter_shape.size()));
  }
}

void validate_conv_group_layout(TensorLayout filter_layout, GroupLayout group_layout, std::stringstream& args_log) {
  if (is_filter_layout_with_separate_groups(filter_layout) && group_layout != GroupLayout::SEPARATE) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error("Filter_layout specifies separate groups but group_layout isn't SEPARATE");
  }
  if (!is_filter_layout_with_separate_groups(filter_layout) && group_layout == GroupLayout::SEPARATE) {
    IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
    throw std::runtime_error("Filter_layout lacks separate groups but group_layout is SEPARATE");
  }
}

void validate_conv_result_shape(size_t spatial_rank, const std::vector<int64_t>& result_shape, ConvDerivMode deriv_mode,
                                bool infer_result_shape, std::stringstream& args_log) {
  if (result_shape.empty()) {
    if (deriv_mode != ConvDerivMode::NONE && !infer_result_shape) {
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error(
          "Transposed/gradient convolutions require specifying the result_shape. This can be bypassed by setting "
          "infer_result_shape = true, but be warned that infered result shapes do not necessarily match the input "
          "shape used in the forward convolution, as multiple input shapes produce the same output shape.");
    }
  } else {
    if (result_shape.size() != spatial_rank) {
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error(
          llvm::formatv("Inconsistent spatial rank in conv op (received {0} spatial dimensions based on strides "
                        "but result shape has {1} spatial dims).",
                        spatial_rank, result_shape.size()));
    }
  }
}

void normalize_grouping_strategy(int64_t* groups, AutoGroupMode* autogroup_mode, GroupLayout* group_layout) {
  // This normalization enforces:
  //  * If group_layout is NONE:
  //      - autogroup_mode is UNGROUPED
  //        (AUTO is converted to UNGROUPED, as is EXPLICIT if groups == 1)
  //        (DEPTHWISE throws, as does EXPLICIT if groups != 1)
  //      - groups is 1
  //        (If non-1, it is converted after autogroup_mode conversion succeeds)
  //  * If autogroup_mode is UNGROUPED:
  //      - groups is 1
  //        (If non-1, it throws)
  //      - group_layout allowed to vary
  //  * If autogroup_mode is EXPLICIT:
  //      - groups is > 1
  //        (If < 1, throw; if == 1, autogroup_mode is converted to UNGROUPED)
  //      - group_layout is allowed to vary, but may not be NONE
  //        (If group_layout is NONE and groups != 1, this throws (see above))
  //        (If group_layout is NONE and groupd == 1, autogroup_mode is converted to UNGROUPED (see above))
  //  * If autogroup_mode is AUTO:
  //      - groups is to be ignored
  //      - group_layout is SEPARATE or IN_K
  //        (throws if group_layout is IN_C)
  //        (if group_layout is NONE, autogroup_mode is converted to UNGROUPED (see above))
  //  * If autogroup_mode is DEPTHWISE:
  //      - groups is to be ignored
  //      - group_layout is not NONE
  switch (*autogroup_mode) {
    case AutoGroupMode::UNGROUPED:
      if (*groups != 1) {
        throw std::runtime_error("Convolution AutoGroupMode::UNGROUPED requires groups == 1");
      }
      break;
    case AutoGroupMode::AUTO:
      if (*group_layout == GroupLayout::NONE) {
        *groups = 1;
        *autogroup_mode = AutoGroupMode::UNGROUPED;
      }
      if (*group_layout == GroupLayout::IN_C) {
        // TODO: This and related cases may depend on the deriv_mode; take that into account
        throw std::runtime_error("Cannot automatically detect group size of convolution with IN_C GroupLayout");
      }
      break;
    case AutoGroupMode::EXPLICIT:
      if (*groups < 1) {
        throw std::runtime_error("Requested grouped convolution with fewer than 1 groups");
      }
      if (*groups == 1) {
        *autogroup_mode = AutoGroupMode::UNGROUPED;
      }
      if (*group_layout == GroupLayout::NONE && *groups != 1) {
        throw std::runtime_error("GroupLayout not specified for grouped convolution");
      }
      break;
    case AutoGroupMode::DEPTHWISE:
      if (*group_layout == GroupLayout::NONE) {
        throw std::runtime_error("Convolution GroupLayout must be specified to use DEPTHWISE AutoGroupMode");
      }
      break;
    default:
      throw std::runtime_error("Unrecognized AutoGroupMode");
  }
}

}  // namespace

Value convolution(const Value& value) {
  IVLOG(1, "convolution");
  // Parameters:
  //  0. Input Tensor
  //  1. Filter Tensor
  //  2. Strides
  //  3. Dilations
  //  4. Data Dilations
  //  5. Kernel Shape
  //  6. Groups
  //  7. Autopad Mode
  //  8. Manual Padding
  //  9. Input Tensor Layout
  // 10. Filter Tensor Layout
  // 11. Grouping Layout
  // 12. Winograd allowed
  // 13. Name
  // 14. Autogrouping (? Unclear if we really need this)
  // 15. Deriv Mode (DATA is equivalent to transposed conv)
  // 16. Result Shape (a.k.a. output shape, used for transposed/derivative convs)
  // 17. Infer Result Shape (is it legal to omit result shape for transposed convs)

  // Read Arguments
  auto args = value.as_tuple();
  if (args.size() != 18) {
    throw std::runtime_error("Convolution op expects 18 arguments");
  }
  IVLOG(5, "Requesting convolution with raw args: " << args);
  auto I_or_O = args[0].as_tensor();  // O if deriv_mode is DATA, else I
  auto F_or_O = args[1].as_tensor();  // O if deriv_mode is FILTER, else F
  auto strides = args[2].as_int_tuple_or_empty();
  auto dilations = args[3].as_int_tuple_or_empty();
  auto data_dilations = args[4].as_int_tuple_or_empty();
  // TODO: Perhaps could upgrade use of filter_shape?
  // This is the shape of the _spatial_ filter dims _only_
  auto filter_shape = args[5].as_int_tuple_or_empty();
  auto groups = args[6].as_int();  // will be 1 for non-grouped convolutions
  auto autopad_mode = validate<AutoPadMode>(args[7].as_int());
  auto manual_padding = args[8].as_int_tuple_or_empty();
  auto input_layout = validate<TensorLayout>(args[9].as_int());
  auto filter_layout = validate<TensorLayout>(args[10].as_int());
  auto group_layout = validate<GroupLayout>(args[11].as_int());
  auto winograd_allowed = args[12].as_bool();  // TODO: Implement Winograd
  auto name = args[13].as_str();
  auto autogroup_mode = validate<AutoGroupMode>(args[14].as_int());
  auto deriv_mode = validate<ConvDerivMode>(args[15].as_int());
  auto result_shape = args[16].as_int_tuple_or_empty();
  auto infer_result_shape = args[17].as_bool();

  // Construct a string to log the arguments if something throws
  std::stringstream args_log;
  if (VLOG_IS_ON(1)) {
    args_log << "  Input Tensor: " << I_or_O << "\n";
    args_log << "  Filter Tensor: " << F_or_O << "\n";
    args_log << "  Strides: " << std::to_string(strides) << "\n";
    args_log << "  Dilations: " << std::to_string(dilations) << "\n";
    args_log << "  Data Dilations: " << std::to_string(data_dilations) << "\n";
    args_log << "  Filter Shape (optional): " << std::to_string(filter_shape) << "\n";
    args_log << "  Number of Groups (1 if not grouped): " << groups << "\n";
    args_log << "  Autopadding Mode: " << autopad_mode << "\n";
    args_log << "  Manual Padding (if used): " << std::to_string(manual_padding) << "\n";
    args_log << "  Input Layout: " << input_layout << "\n";
    args_log << "  Filter Layout: " << filter_layout << "\n";
    args_log << "  Group Layout: " << group_layout << "\n";
    args_log << "  Winograd Permitted?: " << winograd_allowed << "\n";
    args_log << "  Name: " << name << "\n";
    args_log << "  Autogroup Mode: " << autogroup_mode << "\n";
    args_log << "  Derivative Mode: " << deriv_mode << "\n";
    args_log << "  Result Shape (for transposed/differentiated convs): " << std::to_string(result_shape);
  }
  IVLOG(3, "Requesting convolution with args:\n" << args_log.str());

  Tensor I;  // Inputs (i.e. Data) tensor
  Tensor F;  // Filters (i.e. Weights i.e. Kernel) tensor
  Tensor O;  // Output (i.e. of a forward pass) tensor
  Contraction OC;

  // Connect the inputs to the right names
  switch (deriv_mode) {
    case ConvDerivMode::NONE:
      I = I_or_O;
      F = F_or_O;
      break;
    case ConvDerivMode::DATA:
      O = I_or_O;
      F = F_or_O;
      break;
    case ConvDerivMode::FILTER:
      I = I_or_O;
      O = F_or_O;
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Invalid ConvDerivMode");
  }

  // Determine the number of spatial dimensions
  auto spatial_rank = compute_conv_rank_validating_strides(&strides, I_or_O, input_layout);

  // Verify inputs are consistent & generate default arguments if needed
  validate_conv_padding(manual_padding, autopad_mode, args_log);
  validate_conv_dilations_rank(spatial_rank, &dilations, args_log);
  validate_conv_data_dilations_rank(spatial_rank, &data_dilations, args_log);
  validate_conv_input_layout(input_layout, args_log);
  validate_conv_filter_layout(filter_layout, args_log);
  validate_conv_input_rank(spatial_rank, I, input_layout, deriv_mode, args_log);
  validate_conv_filter_rank(spatial_rank, F, filter_layout, deriv_mode, args_log);
  validate_conv_filter_shape_rank(spatial_rank, filter_shape, args_log);
  validate_conv_group_layout(filter_layout, group_layout, args_log);
  validate_conv_result_shape(spatial_rank, result_shape, deriv_mode, infer_result_shape, args_log);
  extend_manual_padding(&manual_padding, spatial_rank);
  if (name.empty()) {
    name = "conv";
  }
  normalize_grouping_strategy(&groups, &autogroup_mode, &group_layout);

  // Prepare dimension and index variables
  TensorDim N, CI, CO, G;
  // The channel dimensions as used by the filters, adjusted for group layout
  TensorDim F_CI, F_CO;
  TensorIndex n("n");
  TensorIndex ci("ci");
  TensorIndex co("co");
  TensorIndex g("g");
  // The spatial dimensions of I
  std::vector<TensorDim> I_spatial_dims(spatial_rank);
  // The spatial indexes of I
  std::vector<TensorIndex> x;
  for (size_t i = 0; i < spatial_rank; ++i) {
    x.emplace_back(TensorIndex(llvm::formatv("x{0}", i)));
  }
  // The spatial dimensions of O; nearly unused
  std::vector<TensorDim> O_spatial_dims(spatial_rank);
  // The spatial dimensions of F
  std::vector<TensorDim> F_spatial_dims(spatial_rank);
  // The spatial indexs of F
  std::vector<TensorIndex> k;
  for (size_t i = 0; i < spatial_rank; ++i) {
    k.emplace_back(TensorIndex(llvm::formatv("k{0}", i)));
  }
  std::vector<TensorDim> I_dims;
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorDim> F_dims;
  // this ensures that the inferred filter shape matches filter_shape if the latter is passed in
  std::vector<TensorDim> F_explicit_dims;
  std::vector<TensorIndex> F_idxs;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  // G may be explicit or automatically set, based on autogroup_mode
  TensorDim G_explicit(groups);
  switch (autogroup_mode) {
    case AutoGroupMode::EXPLICIT:
    case AutoGroupMode::UNGROUPED:
      G = G_explicit;
      break;
    case AutoGroupMode::DEPTHWISE:
      G = CI;
      if (group_layout == GroupLayout::IN_K || group_layout == GroupLayout::SEPARATE) {
        F_CI = TensorDim(1);
      } else if (group_layout == GroupLayout::IN_C) {
        // Everything can be inferred, do nothing  // nolint(whitespace/empty_if_body)
      } else {
        IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
        throw std::runtime_error(llvm::formatv("Unsupported group layout '{0}' used with autogroup mode DEPTHWISE",
                                               to_string(group_layout)));
      }
      break;
    case AutoGroupMode::AUTO:
      if (group_layout == GroupLayout::SEPARATE || group_layout == GroupLayout::IN_K) {
        // just let G be inferred; i.e. do nothing  // nolint(whitespace/empty_if_body)
      } else {
        IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
        throw std::runtime_error(
            llvm::formatv("Unsupported group layout '{0}' used with autogroup mode AUTO", to_string(group_layout)));
      }
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Unrecognized AutoGroupMode");
  }

  // Set up dimensions of the inputs first so they can be bound Group layout
  // affects the size of filter dimensions; we pass through the dims that don't
  // need to be adjusted here, and we will calculate those dimensions that will
  // be adjusted later (after some more dims are bound).
  // TODO: This needs more thorough test converage
  switch (group_layout) {
    case GroupLayout::NONE:
      F_CO = CO;
      F_CI = CI;
      break;
    case GroupLayout::IN_C:
      // Later: F_CO = CO / G;
      F_CI = CI;
      break;
    case GroupLayout::IN_K:
      F_CO = CO;
      // Later: F_CI = CI / G;
      break;
    case GroupLayout::SEPARATE:
      // Later: F_CO = CO / G;
      // Later: F_CI = CI / G;
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Invalid group_layout");
  }

  // The input data dims
  if (deriv_mode != ConvDerivMode::DATA) {
    switch (input_layout) {
      case TensorLayout::NCX:
        I_dims.push_back(N);
        I_dims.push_back(CI);
        for (size_t i = 0; i < spatial_rank; ++i) {
          I_dims.push_back(I_spatial_dims[i]);
        }
        break;
      case TensorLayout::NXC:
        I_dims.push_back(N);
        for (size_t i = 0; i < spatial_rank; ++i) {
          I_dims.push_back(I_spatial_dims[i]);
        }
        I_dims.push_back(CI);
        break;
      default:
        IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
        throw std::runtime_error("Invalid input_layout");
    }
    I.bind_dims(I_dims);
  }

  // The filter dims
  if (deriv_mode != ConvDerivMode::FILTER) {
    switch (filter_layout) {
      case TensorLayout::GKCX:
        F_dims.push_back(G);
        F_explicit_dims.push_back(G);
        // Fall through deliberately
      case TensorLayout::KCX:
        F_dims.push_back(F_CO);
        F_explicit_dims.push_back(F_CO);
        F_dims.push_back(F_CI);
        F_explicit_dims.push_back(F_CI);
        for (size_t i = 0; i < spatial_rank; ++i) {
          F_dims.push_back(F_spatial_dims[i]);
          if (filter_shape.size()) {
            F_explicit_dims.push_back(TensorDim(filter_shape[i]));
          }
        }
        break;
      case TensorLayout::XCK:
      case TensorLayout::XGCK:
        for (size_t i = 0; i < spatial_rank; ++i) {
          F_dims.push_back(F_spatial_dims[i]);
          if (filter_shape.size()) {
            F_explicit_dims.push_back(TensorDim(filter_shape[i]));
          }
        }
        if (filter_layout == TensorLayout::XGCK) {
          F_dims.push_back(G);
          F_explicit_dims.push_back(G);
        }
        F_dims.push_back(F_CI);
        F_explicit_dims.push_back(F_CI);
        F_dims.push_back(F_CO);
        F_explicit_dims.push_back(F_CO);
        break;
      default:
        IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
        throw std::runtime_error("Invalid filter_layout");
    }
    F.bind_dims(F_dims);
    if (filter_shape.size()) {
      F.bind_dims(F_explicit_dims);
    }
  }

  // The output data dims
  if (deriv_mode != ConvDerivMode::NONE) {
    // This assumes we infer the output layout from the input layout. So if we
    // change that, the output data dims section will need to be adapted.
    switch (input_layout) {
      case TensorLayout::NCX:
        O_dims.push_back(N);
        O_dims.push_back(CO);
        for (size_t i = 0; i < spatial_rank; ++i) {
          O_dims.push_back(O_spatial_dims[i]);
        }
        break;
      case TensorLayout::NXC:
        O_dims.push_back(N);
        for (size_t i = 0; i < spatial_rank; ++i) {
          O_dims.push_back(O_spatial_dims[i]);
        }
        O_dims.push_back(CO);
        break;
      default:
        IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
        throw std::runtime_error("Invalid input_layout");
    }
    O.bind_dims(O_dims);
  }

  // Compute the adjustments to the filter channel dimensions that come from group size
  switch (group_layout) {
    case GroupLayout::NONE:
      break;
    case GroupLayout::IN_C:
      CO = F_CO * G;
      break;
    case GroupLayout::IN_K:
      CI = F_CI * G;
      break;
    case GroupLayout::SEPARATE:
      CO = F_CO * G;
      CI = F_CI * G;
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Invalid group_layout");
  }

  // Determine the padding and the shape of the result tensor
  // Note that this is a different shape computed in a different way depending on the DerivMode. In most cases with DATA
  // and FILTER, `result_shape` will be set, in which case we use the NONE mode logic for the fully determined padding
  // amounts it enables. If we need to we can infer result_shape for these cases (currently only implemented for DATA),
  // but we have to make the assumption that the size is the minimal possible, so we avoid this path where possible.
  std::vector<TensorDim> pad_before;
  // Replace the unset defaults
  switch (deriv_mode) {
    case ConvDerivMode::NONE:
      O_spatial_dims.clear();
      break;
    case ConvDerivMode::DATA:
      I_spatial_dims.clear();
      break;
    default:
      break;
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    TensorDim local_pad_before;
    TensorDim local_output_size;
    TensorDim local_input_size;
    TensorDim local_filter_size;
    if (deriv_mode == ConvDerivMode::DATA && result_shape.empty()) {
      TensorDim local_output_size = O_spatial_dims[i];
      TensorDim local_filter_size = F_spatial_dims[i];
      std::tie(local_pad_before, local_input_size) = compute_padding_and_input_size(
          local_output_size, local_filter_size, strides[i], autopad_mode, manual_padding[i],
          manual_padding[i + spatial_rank], dilations[i], data_dilations[i]);
      pad_before.push_back(local_pad_before);
      I_spatial_dims.push_back(local_input_size);
    } else {
      if (deriv_mode == ConvDerivMode::FILTER && result_shape.empty()) {
        IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
        throw std::runtime_error(
            "Result shape inference not yet supported for filter transposed/derivative convolutions");
      }
      local_input_size = (deriv_mode == ConvDerivMode::DATA) ? TensorDim(result_shape[i]) : I_spatial_dims[i];
      local_filter_size = (deriv_mode == ConvDerivMode::FILTER) ? TensorDim(result_shape[i]) : F_spatial_dims[i];
      std::tie(local_pad_before, local_output_size) = compute_padding_and_output_size(
          local_input_size, local_filter_size, strides[i], autopad_mode, manual_padding[i],
          manual_padding[i + spatial_rank], dilations[i], data_dilations[i], false);
      pad_before.push_back(local_pad_before);
      switch (deriv_mode) {
        case ConvDerivMode::NONE:
          O_spatial_dims.push_back(local_output_size);
          break;
        case ConvDerivMode::DATA:
          I_spatial_dims.push_back(TensorDim(result_shape[i]));
          break;
        default:
          break;
      }
    }
  }

  // Now set up the dimensions of the result to be returned
  switch (deriv_mode) {
    case ConvDerivMode::NONE:
      // This assumes we infer the output layout from the input layout. So if we
      // change that, the below switch will need to be adapted.
      switch (input_layout) {
        case TensorLayout::NCX:
          O_dims.push_back(N);
          O_dims.push_back(CO);
          for (size_t i = 0; i < spatial_rank; ++i) {
            O_dims.push_back(O_spatial_dims[i]);
          }
          break;
        case TensorLayout::NXC:
          O_dims.push_back(N);
          for (size_t i = 0; i < spatial_rank; ++i) {
            O_dims.push_back(O_spatial_dims[i]);
          }
          O_dims.push_back(CO);
          break;
        default:
          IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
          throw std::runtime_error("Invalid input_layout");
      }
      OC = Contraction(name).outShape(O_dims);
      break;
    case ConvDerivMode::DATA:
      switch (input_layout) {
        case TensorLayout::NCX:
          I_dims.push_back(N);
          I_dims.push_back(CI);
          for (size_t i = 0; i < spatial_rank; ++i) {
            I_dims.push_back(I_spatial_dims[i]);
          }
          break;
        case TensorLayout::NXC:
          I_dims.push_back(N);
          for (size_t i = 0; i < spatial_rank; ++i) {
            I_dims.push_back(I_spatial_dims[i]);
          }
          I_dims.push_back(CI);
          break;
        default:
          IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
          throw std::runtime_error("Invalid input_layout");
      }
      OC = Contraction(name).outShape(I_dims);
      break;
    case ConvDerivMode::FILTER:
      switch (filter_layout) {
        // TODO: This won't always work for grouped convolutions, will have to update
        case TensorLayout::GKCX:
          F_dims.push_back(G);
          // Fall through deliberately
        case TensorLayout::KCX:
          F_dims.push_back(F_CO);
          F_dims.push_back(F_CI);
          for (size_t i = 0; i < spatial_rank; ++i) {
            F_dims.push_back(TensorDim(result_shape[i]));
          }
          break;
        case TensorLayout::XCK:
        case TensorLayout::XGCK:
          for (size_t i = 0; i < spatial_rank; ++i) {
            F_dims.push_back(TensorDim(result_shape[i]));
          }
          if (filter_layout == TensorLayout::XGCK) {
            F_dims.push_back(G);
          }
          F_dims.push_back(F_CI);
          F_dims.push_back(F_CO);
          break;
        default:
          IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
          throw std::runtime_error("Invalid filter_layout");
      }
      OC = Contraction(name).outShape(F_dims);
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Invalid deriv_mode");
  }

  // Set up index formulas
  // Input data indexes
  switch (input_layout) {
    case TensorLayout::NCX:
      I_idxs.push_back(n);
      if (group_layout == GroupLayout::NONE) {
        I_idxs.push_back(ci);
      } else {
        I_idxs.push_back((CI / G) * g + ci);
      }
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_idxs.emplace_back((strides[i] * x[i] + dilations[i] * k[i] - pad_before[i]) / data_dilations[i]);
      }
      break;
    case TensorLayout::NXC:
      I_idxs.push_back(n);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_idxs.emplace_back((strides[i] * x[i] + dilations[i] * k[i] - pad_before[i]) / data_dilations[i]);
      }
      if (group_layout == GroupLayout::NONE) {
        I_idxs.push_back(ci);
      } else {
        I_idxs.push_back((CI / G) * g + ci);
      }
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Invalid input_layout");
  }

  std::vector<Constraint> constraints;

  // Filter indexes
  TensorIndex f_co, f_ci;  // Filter index formulas for out/in channels; depend on group layout
  switch (group_layout) {
    case GroupLayout::NONE:
    case GroupLayout::SEPARATE:
      f_co = co;
      f_ci = ci;
      break;
    case GroupLayout::IN_C:
      f_co = co;
      f_ci = (CI / G) * g + ci;
      constraints.push_back(ci < CI / G);
      break;
    case GroupLayout::IN_K:
      f_co = (CO / G) * g + co;
      f_ci = ci;
      constraints.push_back(co < CO / G);
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Unrecognized group layout");
  }
  switch (filter_layout) {
    case TensorLayout::GKCX:
      F_idxs.push_back(g);
      // Fall through deliberately
    case TensorLayout::KCX:
      F_idxs.push_back(f_co);
      F_idxs.push_back(f_ci);
      for (size_t i = 0; i < spatial_rank; ++i) {
        F_idxs.push_back(k[i]);
      }
      break;
    case TensorLayout::XCK:
    case TensorLayout::XGCK:
      for (size_t i = 0; i < spatial_rank; ++i) {
        F_idxs.push_back(k[i]);
      }
      if (filter_layout == TensorLayout::XGCK) {
        F_idxs.push_back(g);
      }
      F_idxs.push_back(f_ci);
      F_idxs.push_back(f_co);
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Invalid filter_layout");
  }

  // Output data indexes
  // This assumes we infer the output layout from the input layout. So if we
  // change that, the below switch will need to be adapted.
  switch (input_layout) {
    case TensorLayout::NCX:
      O_idxs.push_back(n);
      if (group_layout == GroupLayout::NONE) {
        O_idxs.push_back(co);
      } else {
        O_idxs.push_back((CO / G) * g + co);
      }
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_idxs.push_back(x[i]);
      }
      break;
    case TensorLayout::NXC:
      O_idxs.push_back(n);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_idxs.push_back(x[i]);
      }
      if (group_layout == GroupLayout::NONE) {
        O_idxs.push_back(co);
      } else {
        O_idxs.push_back((CO / G) * g + co);
      }
      break;
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Invalid input_layout");
  }

  // Return the contraction
  switch (deriv_mode) {
    case ConvDerivMode::NONE:
      OC.outAccess(O_idxs).sum(I(I_idxs) * F(F_idxs)).add_constraints(constraints);
      return Value{OC};
    case ConvDerivMode::DATA:
      OC.outAccess(I_idxs).sum(O(O_idxs) * F(F_idxs)).add_constraints(constraints);
      return Value{OC};
    case ConvDerivMode::FILTER:
      OC.outAccess(F_idxs).sum(I(I_idxs) * O(O_idxs)).add_constraints(constraints);
      return Value{OC};
    default:
      IVLOG(2, "Bad convolution, arguments:\n" << args_log.str());
      throw std::runtime_error("Unrecognized deriv_mode");
  }
}

Value cumprod(const Value& value) {
  IVLOG(1, "cumprod");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("cumprod expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  auto raw_axis = args[1].as_int();

  auto ndims = I.rank();
  auto axis = normalize_axis(raw_axis, ndims, "cumprod");
  std::vector<TensorDim> dims(ndims);
  I.bind_dims(dims);
  std::vector<TensorIndex> I_idxs(ndims);
  std::vector<TensorIndex> O_idxs(I_idxs);
  TensorIndex cumulator_idx;
  I_idxs[axis] = I_idxs[axis] - cumulator_idx;
  Tensor O = Contraction(dims, O_idxs).product(I(I_idxs)).add_constraint(cumulator_idx < dims[axis]);
  return Value{O};
}

Value cumsum(const Value& value) {
  IVLOG(1, "cumsum");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("cumsum expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  auto raw_axis = args[1].as_int();
  auto is_exclusive = args[2].as_bool();

  auto ndims = I.rank();
  auto axis = normalize_axis(raw_axis, ndims, "cumsum");
  std::vector<TensorDim> dims(ndims);
  I.bind_dims(dims);
  std::vector<TensorIndex> I_idxs(ndims);
  std::vector<TensorIndex> O_idxs(I_idxs);
  TensorIndex cumulator_idx;
  I_idxs[axis] = is_exclusive ? I_idxs[axis] - cumulator_idx - 1 : I_idxs[axis] - cumulator_idx;
  Tensor O = Contraction(dims, O_idxs).sum(I(I_idxs)).add_constraint(cumulator_idx < dims[axis]);
  return Value{O};
}

Value dot(const Value& value) {
  IVLOG(1, "dot");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("dot expects 2 arguments");
  }
  auto X = args[0].as_tensor();
  auto Y = args[1].as_tensor();
  if (X.dtype() != Y.dtype()) {
    throw std::runtime_error(llvm::formatv("Invalid dtype in dot: X.dtype = '{0}', Y.dtype = '{1}'",
                                           to_string(X.dtype()), to_string(Y.dtype())));
  }
  if (X.rank() == 1 && Y.rank() == 1) {
    TensorDim I;
    TensorIndex i;
    X.bind_dims(I);
    Y.bind_dims(I);
    Tensor O = Contraction({I}, {i}).sum(X(i) * Y(i));
    return Value{O};
  }
  if (1 <= X.rank() && 2 <= Y.rank()) {
    std::vector<TensorDim> X_dims(X.rank());
    std::vector<TensorDim> Y_dims(Y.rank());
    TensorIndex z;
    std::vector<TensorIndex> X_idxs(X.rank());
    std::vector<TensorIndex> Y_idxs(Y.rank());
    X_idxs[X.rank() - 1] = z;
    Y_idxs[Y.rank() - 2] = z;
    X.bind_dims(X_dims);
    Y.bind_dims(Y_dims);
    std::vector<TensorDim> O_dims;
    std::vector<TensorIndex> O_idxs;
    for (size_t i = 0; i < X.rank() - 1; i++) {
      O_dims.push_back(X_dims[i]);
      O_idxs.push_back(X_idxs[i]);
    }
    for (size_t i = 0; i < Y.rank() - 2; i++) {
      O_dims.push_back(Y_dims[i]);
      O_idxs.push_back(Y_idxs[i]);
    }
    O_dims.push_back(Y_dims[Y.rank() - 1]);
    O_idxs.push_back(Y_idxs[Y.rank() - 1]);
    Tensor O = Contraction(O_dims, O_idxs).sum(X(X_idxs) * Y(Y_idxs));
    return Value{O};
  }
  throw std::runtime_error(
      llvm::formatv("Unsupported dims for dot operation: X.dims = {0}, Y.dims = {1}", X.rank(), Y.rank()));
}

Value elu(const Value& value) {
  IVLOG(1, "elu");

  // Read arguments
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error(llvm::formatv("PlaidML elu op expects 2 arguments (received {0})", args.size()));
  }
  auto I = args[0].as_tensor();

  // Same algorithm, but alpha may be either int or float
  if (args[1].is_float()) {
    auto alpha = args[1].as_float();
    auto O = select(I < 0, alpha * exp(I) - alpha, I);
    return Value{O};
  } else if (args[1].is_int()) {
    auto alpha = args[1].as_int();
    auto O = select(I < 0, alpha * exp(I) - alpha, I);
    return Value{O};
  }
  throw std::runtime_error("Unexpected type for alpha in elu");
}

Value explicit_padding(const Value& value) {
  IVLOG(1, "explicit_padding");
  auto args = value.as_tuple();
  if (args.size() < 5) {
    throw std::runtime_error("explicit_padding expects 5 arguments");
  }

  auto I = args[0].as_tensor();
  auto lo_pads = args[1].as_int_tuple();
  auto hi_pads = args[2].as_int_tuple();
  auto mode = validate<PadMode>(args[3].as_int());

  // validate inputs

  if (lo_pads.size() != I.rank()) {
    IVLOG(2, lo_pads.size())
    IVLOG(2, I.rank())
    IVLOG(2, lo_pads[0])
    throw std::runtime_error(
        llvm::formatv("Inconsistent shapes in explicit_padding op (received an input tensor with {0} dims, "
                      "but received lower padding for {1} dims.)",
                      I.rank(), lo_pads.size()));
  }
  if (hi_pads.size() != I.rank()) {
    throw std::runtime_error(
        llvm::formatv("Inconsistent shapes in explicit_padding op (received an input tensor with {0} dims, "
                      "but received higher padding for {1} dims.)",
                      I.rank(), hi_pads.size()));
  }

  std::vector<TensorDim> I_dims;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorIndex> O_idxs;

  // Assign dimensions & indices
  std::vector<TensorDim> X(I.rank());
  std::vector<TensorIndex> x;
  for (size_t i = 0; i < I.rank(); ++i) {
    x.emplace_back(llvm::formatv("x{0}", i));
  }

  for (size_t i = 0; i < I.rank(); ++i) {
    I_dims.push_back(X[i]);
    I_idxs.push_back(x[i]);
  }
  I.bind_dims(I_dims);

  for (size_t i = 0; i < I.rank(); ++i) {
    O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
    O_idxs.push_back(x[i] + lo_pads[i]);
  }

  Tensor O;

  switch (mode) {
    case PadMode::CONSTANT: {
      IVLOG(2, "Constant padding requested");

      auto padval = args[4].as_tensor();
      O = Contraction(O_dims, O_idxs).assign(I(I_idxs)).init(padval);
    } break;
    case PadMode::EDGE:
    case PadMode::SYMMETRIC:
    case PadMode::REFLECT: {
      throw std::runtime_error(llvm::formatv("Unimplemented padding mode: {0}", to_string(mode)));
    } break;
    default:
      throw std::runtime_error(llvm::formatv("Unrecognized padding mode: {0}", to_string(mode)));
  }

  return Value{O};
}

Value flip(const Value& value) {
  IVLOG(1, "flip");
  // This is numpy-style `flip`; Keras calls it `repeat`

  // Read arguments
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error(llvm::formatv("PlaidML flip op expects 2 arguments (received {0})", args.size()));
  }
  auto I = args[0].as_tensor();
  std::vector<int64_t> raw_axes;
  // Hold off on reading the axis arg

  // Set up useful variables
  auto ndims = I.rank();
  if (args[1].is_int()) {
    raw_axes.push_back(args[1].as_int());
  } else if (args[1].is_none()) {
    for (uint64_t i = 0; i < ndims; ++i) {
      raw_axes.push_back(i);
    }
  } else {
    raw_axes = args[1].as_int_tuple();
  }
  std::vector<size_t> axes;
  for (auto& raw_axis : raw_axes) {
    axes.push_back(normalize_axis(raw_axis, ndims, "flip"));
  }

  // Perform the flip
  std::vector<TensorDim> dims(ndims);
  std::vector<TensorIndex> I_idxs(ndims);
  std::vector<TensorIndex> O_idxs(I_idxs);
  I.bind_dims(dims);
  for (const auto& axis : axes) {
    O_idxs[axis] = dims[axis] - 1 - I_idxs[axis];
  }
  Tensor O = Contraction(dims, O_idxs).assign(I(I_idxs));
  return Value{O};
}

Value hard_sigmoid(const Value& value) {
  IVLOG(1, "hard_sigmoid");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("hard_sigmoid expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  auto slope = args[1].as_float();
  if (slope <= 0) {
    throw std::runtime_error(llvm::formatv("hard_sigmoid expects positive slope, received {0}", slope));
  }
  auto hi_cusp = 1. / (2. * slope);
  auto lo_cusp = -hi_cusp;
  auto lo = cast(Tensor(0.), I.dtype());
  auto hi = cast(Tensor(1.), I.dtype());
  auto O = select(I < lo_cusp, lo, select(I > hi_cusp, hi, slope * I + 0.5));
  return Value{O};
}

Value image_resize(const Value& value) {
  // Resize a 2D image's spatial dimensions, each by a positive integer factor
  IVLOG(1, "image_resize");
  auto args = value.as_tuple();
  if (args.size() != 4) {
    throw std::runtime_error("image_resize expects 4 arguments");
  }
  auto raw_I = args[0];
  auto factors = args[1].as_int_tuple();
  auto interp = validate<InterpolationMode>(args[2].as_int());
  auto layout = validate<TensorLayout>(args[3].as_int());

  for (const auto& scale_factor : factors) {
    if (scale_factor <= 0) {
      throw std::runtime_error(
          llvm::formatv("All scaling factors in image_resize must be positive (received {0})", scale_factor));
    }
  }

  // The total number of spatial dimensions and how many non-spatial dimensions are before & after the spatial dims
  size_t rank;  // an error if this isn't 2
  int64_t pre_axes;
  auto I = raw_I.as_tensor();
  auto ndims = I.rank();
  switch (layout) {
    case TensorLayout::NCX:
      rank = ndims - 2;
      pre_axes = 2;
      break;
    case TensorLayout::NXC:
      rank = ndims - 2;
      pre_axes = 1;
      break;
    default:
      throw std::runtime_error(
          llvm::formatv("Unable to apply image_resize to a tensor with layout '%1'", to_string(layout)));
  }
  if (rank != 2) {
    throw std::runtime_error(llvm::formatv("Expected 2 spatial dims for resize_images, received {0}", rank));
  }
  if (factors.size() != 2) {
    throw std::runtime_error(
        llvm::formatv("Cannot resize a 2D image using {1} spatial scaling factors", rank, factors.size()));
  }

  Tensor O;
  switch (interp) {
    case InterpolationMode::NEAREST: {
      auto R = repeat(make_tuple(raw_I, Value{factors[0]}, Value{pre_axes}));
      O = repeat(make_tuple(R, Value{factors[1]}, Value{pre_axes + 1})).as_tensor();
    } break;
    case InterpolationMode::BILINEAR: {
      // This aligns the corners to 0 and <factor> * (<dim> - 1), and assumes zero-padding, which is a bit weird. But
      // it's easy to code and for ML the weirdness probably doesn't particularly matter.
      // Could likely eke out a bit more perf by precomputing K instead of doing it at runtime on device.

      // Setup K
      Tensor HCoeff = cast(Tensor{1.0 / factors[0]}, DType::FLOAT32);
      Tensor WCoeff = cast(Tensor{1.0 / factors[1]}, DType::FLOAT32);
      TensorDim HFactor{factors[0]};
      TensorDim WFactor{factors[1]};
      TensorIndex j{"j"}, i{"i"}, y{"y"}, x{"x"};
      Tensor HCoeffVec = Contraction({HFactor}, {y}).assign(HCoeff());
      Tensor WCoeffVec = Contraction({WFactor}, {x}).assign(WCoeff());
      TensorDim HK_dim = 2 * HFactor - 1;
      TensorDim WK_dim = 2 * WFactor - 1;
      Tensor HK = Contraction({HK_dim}, {y}).sum(HCoeffVec(j + y - HFactor + 1)).add_constraint(j < HFactor);
      Tensor WK = Contraction({WK_dim}, {x}).sum(WCoeffVec(i + x - WFactor + 1)).add_constraint(i < WFactor);
      Tensor K = Contraction({HK_dim, WK_dim}, {y, x}).assign(HK(y) * WK(x));

      // Resize
      std::vector<TensorDim> I_dims(ndims);
      std::vector<TensorIndex> I_idxs(ndims);
      I.bind_dims(I_dims);
      std::vector<TensorDim> O_dims;
      std::vector<TensorIndex> O_idxs;
      for (int64_t ax = 0; ax < pre_axes; ++ax) {
        O_dims.push_back(I_dims[ax]);
        O_idxs.push_back(I_idxs[ax]);
      }
      O_dims.push_back(HFactor * I_dims[pre_axes]);
      O_dims.push_back(WFactor * I_dims[pre_axes + 1]);
      O_idxs.push_back(HFactor * I_idxs[pre_axes] + j - HFactor + 1);
      O_idxs.push_back(WFactor * I_idxs[pre_axes + 1] + i - WFactor + 1);
      for (size_t ax = pre_axes + 2; ax < ndims; ++ax) {
        O_dims.push_back(I_dims[ax]);
        O_idxs.push_back(I_idxs[ax]);
      }
      O = Contraction(O_dims, O_idxs).sum(I(I_idxs) * K(j, i));
    } break;
    default:
      throw std::runtime_error("Unrecognized InterpolationMode in image_resize");
  }
  return Value{O};
}

Value lrn(const Value& value) {
  IVLOG(1, "lrn");
  auto args = value.as_tuple();
  if (args.size() != 6) {
    throw std::runtime_error("lrn expects 6 arguments");
  }
  auto I = args[0].as_tensor();
  auto window_size = args[1].as_int_tuple();
  auto axes = args[2].as_int_tuple();
  auto alpha = args[3].as_float();
  auto beta = args[4].as_float();
  auto epsilon = args[5].as_float();

  LRNAxes agg(I.rank(), axes, window_size);
  std::vector<TensorDim> dims(I.rank());
  I.bind_dims(dims);

  auto I_sqr = I * I;
  Tensor local_sum_sqr = Contraction(dims, agg.dst_idxs).sum(I_sqr(agg.src_idxs)).add_constraints(agg.constraints);
  return Value{I / edsl::pow(alpha * local_sum_sqr + epsilon, Tensor(beta))};
}

Value max(const Value& value) {
  IVLOG(1, "max");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("max expects 3 arguments");
  }
  auto I = args[0].as_tensor();
  auto axes = args[1];
  auto keepdims = args[2].as_bool();
  AggregationAxes agg(I.rank(), axes, keepdims);
  I.bind_dims(agg.src_dims);
  Tensor O = Contraction(agg.dst_dims, agg.dst_idxs).max(I(agg.src_idxs));
  return Value{O};
}

Value maximum(const Value& value) {
  IVLOG(1, "maximum");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("maximum expects 2 arguments");
  }
  auto X = args[0].as_tensor();
  auto Y = args[1].as_tensor();
  auto O = select(X < Y, Y, X);
  return Value{O};
}

Value mean(const Value& value) {
  IVLOG(1, "mean");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("mean expects 3 arguments");
  }

  auto I = args[0].as_tensor();
  if (I.rank() == 0) {
    return Value{I};
  }

  // TODO: Move this commented block to Keras?
  // if (I_shape.dtype() == DType::BOOLEAN) {
  //   I = cast(I, floatx());
  // }

  auto axes = args[1];
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    return Value{I};
  }

  auto keepdims = args[2].as_bool();

  AggregationAxes agg(I.rank(), axes, keepdims);

  I.bind_dims(agg.src_dims);
  Tensor SO = Contraction(agg.dst_dims, agg.dst_idxs).sum(I(agg.src_idxs));
  auto denom = Tensor{1};
  for (const auto& axis : agg.axes) {
    denom = denom * agg.src_dims.at(axis);
  }
  return Value{SO / denom};
}

Value min(const Value& value) {
  IVLOG(1, "min");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("min expects 3 arguments");
  }
  auto I = args[0].as_tensor();
  auto axes = args[1];
  auto keepdims = args[2].as_bool();
  AggregationAxes agg(I.rank(), axes, keepdims);
  I.bind_dims(agg.src_dims);
  Tensor O = Contraction(agg.dst_dims, agg.dst_idxs).min(I(agg.src_idxs));
  return Value{O};
}

Value minimum(const Value& value) {
  IVLOG(1, "minimum");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("minimum expects 2 arguments");
  }
  auto X = args[0].as_tensor();
  auto Y = args[1].as_tensor();
  auto O = select(X < Y, X, Y);
  return Value{O};
}

namespace {
// NMS helper function

Tensor compute_iou(Tensor Boxes, bool center_point_box) {
  // Get intersection over union of boxes
  // The shape of the result tensor is [num_batches, num_boxes, num_boxes]
  // IOU[batch, box_1, box_2] is the IOU of box_1 and box_2 for the image batch.
  Tensor Boxes_y1;
  Tensor Boxes_x1;
  Tensor Boxes_y2;
  Tensor Boxes_x2;
  TensorDim one(1);
  // The memref is used for gather
  Tensor Zero = edsl::cast(edsl::index({one}, 0), Boxes.dtype());
  Tensor Zero_int = edsl::cast(Zero, DType::INT32);
  if (center_point_box) {
    // The box data is [x, y, width, height] in center mode.
    Tensor Boxes_xcenter = edsl::gather(Boxes, Zero_int).axis(2);
    Tensor Boxes_ycenter = edsl::gather(Boxes, Zero_int + 1).axis(2);
    Tensor Boxes_width_half = edsl::gather(Boxes, Zero_int + 2).axis(2);
    Boxes_width_half = Boxes_width_half / 2.0f;
    Tensor Boxes_height_half = edsl::gather(Boxes, Zero_int + 3).axis(2);
    Boxes_height_half = Boxes_height_half / 2.0f;
    Boxes_x1 = Boxes_xcenter - Boxes_width_half;
    Boxes_x2 = Boxes_xcenter + Boxes_width_half;
    Boxes_y1 = Boxes_ycenter - Boxes_height_half;
    Boxes_y2 = Boxes_ycenter + Boxes_height_half;
  } else {
    // The box data is [y1, x1, y2, x2] in corner mode
    Boxes_y1 = edsl::gather(Boxes, Zero_int).axis(2);
    Boxes_x1 = edsl::gather(Boxes, Zero_int + 1).axis(2);
    Boxes_y2 = edsl::gather(Boxes, Zero_int + 2).axis(2);
    Boxes_x2 = edsl::gather(Boxes, Zero_int + 3).axis(2);
  }

  // Boxes dims is [num_batches, num_boxes, 4]
  std::vector<TensorDim> dims(Boxes.rank());
  Boxes.bind_dims(dims);
  TensorDim num_batches = dims[0];
  TensorDim num_boxes = dims[1];

  // Create IOU of each box pair.
  Boxes_y1 = edsl::reshape(Boxes_y1, {num_batches, num_boxes});
  Boxes_x1 = edsl::reshape(Boxes_x1, {num_batches, num_boxes});
  Boxes_y2 = edsl::reshape(Boxes_y2, {num_batches, num_boxes});
  Boxes_x2 = edsl::reshape(Boxes_x2, {num_batches, num_boxes});

  // Calc box local area
  Tensor IOU_areai = (Boxes_y2 - Boxes_y1) * (Boxes_x2 - Boxes_x1);
  // Calc intersecion area of box pair
  Tensor IOU_intersection_ymin = op::maximum(edsl::reshape(Boxes_y1, {num_batches, num_boxes, one}),
                                             edsl::reshape(Boxes_y1, {num_batches, one, num_boxes}));
  Tensor IOU_intersection_xmin = op::maximum(edsl::reshape(Boxes_x1, {num_batches, num_boxes, one}),
                                             edsl::reshape(Boxes_x1, {num_batches, one, num_boxes}));
  Tensor IOU_intersection_ymax = op::minimum(edsl::reshape(Boxes_y2, {num_batches, num_boxes, one}),
                                             edsl::reshape(Boxes_y2, {num_batches, one, num_boxes}));
  Tensor IOU_intersection_xmax = op::minimum(edsl::reshape(Boxes_x2, {num_batches, num_boxes, one}),
                                             edsl::reshape(Boxes_x2, {num_batches, one, num_boxes}));
  Tensor IOU_intersection_area_ygap = IOU_intersection_ymax - IOU_intersection_ymin;
  Tensor IOU_intersection_area_xgap = IOU_intersection_xmax - IOU_intersection_xmin;
  Tensor IOU_intersection_area = select(IOU_intersection_area_ygap > 0.0f, IOU_intersection_area_ygap, Zero) *
                                 select(IOU_intersection_area_xgap > 0.0f, IOU_intersection_area_xgap, Zero);
  // Compute IOU of box pair
  Tensor IOU_denominator = op::unsqueeze(IOU_areai, {-1}) + op::unsqueeze(IOU_areai, {-2}) - IOU_intersection_area;
  edsl::Tensor IOU_denominator_reciprocal = edsl::select(IOU_denominator <= 0.0f, Zero, 1.0f / IOU_denominator);
  edsl::Tensor IOU = IOU_intersection_area * IOU_denominator_reciprocal;
  return IOU;
}

}  // namespace

// Non-maximum suppression
Value nms(const Value& value) {
  IVLOG(1, "nms");
  auto args = value.as_tuple();
  if (args.size() != 9) {
    throw std::runtime_error("nms expects 9 arguments");
  }
  auto Boxes = args[0].as_tensor();
  auto Scores = args[1].as_tensor();
  auto IOU_threshold = args[2].as_tensor();
  auto Score_threshold = args[3].as_tensor();
  int64_t max_output_boxes_per_class = args[4].as_int();
  float soft_nms_sigma = args[5].as_float();
  auto center_point_box = args[6].as_bool();
  auto sort_result_descending = args[7].as_bool();
  auto box_output_type = validate<DType>(args[8].as_int());
  auto box_input_type = Boxes.dtype();
  auto thres_type = IOU_threshold.dtype();

  std::vector<int64_t> scores_shape = Scores.compute_shape().sizes();
  int64_t boxes_count = scores_shape[2];

  std::vector<TensorDim> Scores_dims(Scores.rank());
  Scores.bind_dims(Scores_dims);
  TensorDim num_batches = Scores_dims[0];
  TensorDim num_classes = Scores_dims[1];
  TensorDim num_boxes = Scores_dims[2];
  TensorDim one(1);

  // Used for gather and select
  Tensor Zero = edsl::cast(edsl::index({one}, 0), box_input_type);
  Tensor Zero_int = edsl::cast(Zero, DType::INT32);
  Tensor One = Zero + 1;
  Tensor One_int = Zero_int + 1;
  Tensor NEG1 = -One;
  Tensor Neg1_o = edsl::cast(NEG1, box_output_type);

  std::vector<Tensor> boxes;
  std::vector<Tensor> scores;
  Tensor Valid_outputs = Zero;
  Tensor IOU = compute_iou(Boxes, center_point_box);

  float weight = 0.0f;
  if (soft_nms_sigma != 0) {
    weight = -0.5 / soft_nms_sigma;
  }

  TensorShape Node_shape(DType::FLOAT32, {1, 1, 2});
  std::vector<float> invalid_node_index = {-1, -1};
  Buffer buffer_invalid_node(Node_shape);
  buffer_invalid_node.copy_from(invalid_node_index.data());
  auto Invalid_node = edsl::cast(edsl::Constant(buffer_invalid_node, "Invalid_node"), box_output_type);

  int64_t num_boxes_per_class = boxes_count;
  if (max_output_boxes_per_class != 0) {
    num_boxes_per_class = std::min(boxes_count, max_output_boxes_per_class);
  }

  auto Index_b = edsl::index({num_batches, num_classes, one}, 0);
  auto Index_c = edsl::index({num_batches, num_classes, one}, 1);
  auto Index_bc = edsl::cast(op::concatenate({Index_b, Index_c}, 2), box_output_type);

  Tensor Score_threshold_i = edsl::cast(Score_threshold, box_input_type);
  Tensor IOU_threshold_i = edsl::cast(IOU_threshold, box_input_type);
  Tensor New_scores = edsl::select(Scores > Score_threshold_i, Scores, Zero);

  std::vector<TensorDim> Scatter_dims = {num_batches, num_classes, one};
  std::vector<TensorIndex> Scatter_idxs(3);
  Tensor Zero_scatter = edsl::Contraction(Scatter_dims, Scatter_idxs).assign(Zero(0));
  // Select box
  for (int64_t k = 0; k < num_boxes_per_class; k++) {
    // Select the box with largest score
    Tensor Candidate_index = edsl::gather(edsl::argsort(New_scores, 2, edsl::SortDirection::DESC), Zero_int).axis(2);
    Tensor SCORE = edsl::reshape(op::max(New_scores, edsl::Value(2)), {num_batches, num_classes, one});
    Tensor Current_node = edsl::select(SCORE > 0.0f, Index_bc, Invalid_node);

    // Update count of selected box
    Tensor Valid = op::sum(edsl::select(SCORE != 0.0f, One, Zero));
    Valid_outputs = Valid_outputs + Valid;

    // Add selected box to scores
    scores.push_back(edsl::cast(Current_node, thres_type));
    SCORE = edsl::select(SCORE > 0.0f, SCORE, NEG1);
    scores.push_back(edsl::cast(SCORE, thres_type));

    // Set scores of current box and boxes which have IOU larger than threshold to zero
    // The scores can be same, use scatter to update current node
    Tensor New_scores_update =
        edsl::scatter(New_scores, Candidate_index, Zero_scatter).axis(2).mode(edsl::ScatterMode::UPDATE_ELT);
    New_scores = edsl::reshape(New_scores_update, {num_batches, num_classes, num_boxes});

    Tensor IOU_candidate = edsl::gather(IOU, Candidate_index).mode(edsl::GatherMode::ND).batchDims(1);
    // use >= to include suppose_hard_suppresion case
    New_scores = edsl::select(IOU_candidate >= IOU_threshold_i, Zero, New_scores);

    // Add selected box to boxes
    Tensor Box_index = edsl::select(SCORE > 0.0f, edsl::cast(Candidate_index, box_output_type), Neg1_o);
    boxes.push_back(Current_node);
    boxes.push_back(edsl::reshape(Box_index, {num_batches, num_classes, one}));

    // update scores for current class
    Tensor Scale = edsl::exp(IOU_candidate * IOU_candidate * weight);
    New_scores = New_scores * Scale;
    New_scores = edsl::select(New_scores > Score_threshold_i, New_scores, Zero);
  }

  Valid_outputs = edsl::cast(Valid_outputs, box_output_type);

  TensorDim num_results = num_batches * num_classes * num_boxes_per_class;
  // concatenate scores
  Tensor Scores_result = edsl::reshape(op::concatenate(scores, 2), {num_results, edsl::TensorDim(3)});
  // concatenate boxes
  Tensor Boxes_result = edsl::reshape(op::concatenate(boxes, 2), {num_results, edsl::TensorDim(3)});

  if (sort_result_descending) {
    // Sort across batch
    Tensor Scores_slice = edsl::gather(Scores_result, One_int + 1).axis(1);
    Scores_slice = edsl::reshape(Scores_slice, {num_results});
    Tensor Indexes = edsl::argsort(Scores_slice, 0, edsl::SortDirection::DESC);

    Scores_result = edsl::gather(Scores_result, Indexes).axis(0);
    Boxes_result = edsl::gather(Boxes_result, Indexes).axis(0);
  } else {
    // Put all -1 to the end, we now just have -1 at end of each class
    TensorDim dst = num_results * 3;
    auto Index = edsl::index({dst}, 0);
    auto Max = edsl::cast(Tensor{dst}, DType::INT32);
    Scores_result = edsl::reshape(Scores_result, {dst});
    Boxes_result = edsl::reshape(Boxes_result, {dst});
    Tensor Neg1_thres = edsl::cast(Tensor(-1), thres_type);
    auto Index2 = edsl::select(Scores_result != Neg1_thres, Index, Max);
    auto Index3 = edsl::argsort(edsl::cast(Index2, DType::FLOAT32), 0);
    Scores_result = edsl::gather(Scores_result, Index3).axis(0);
    Boxes_result = edsl::gather(Boxes_result, Index3).axis(0);
    Scores_result = edsl::reshape(Scores_result, {num_results, TensorDim(3)});
    Boxes_result = edsl::reshape(Boxes_result, {num_results, TensorDim(3)});
  }

  return edsl::make_tuple(Boxes_result, Scores_result, Valid_outputs);
}

Value mvn(const Value& value) {
  IVLOG(1, "mvn");
  auto args = value.as_tuple();
  if (args.size() != 6) {
    throw std::runtime_error("mvn expects 6 arguments");
  }
  auto I_raw = args[0];
  auto I = args[0].as_tensor();
  auto axes = args[1];
  auto normalize_variance = args[2].as_bool();
  auto epsilon = args[3].as_float();
  auto across_channels = args[4].as_bool();
  auto layout = args[5].as_str();
  if (axes.is_none()) {
    if (layout.empty()) {
      throw std::runtime_error("Either axes or layout must be specified for MVN");
    }
    std::vector<int64_t> raw_axes;
    for (size_t i = 0; i < layout.size(); i++) {
      auto dim = layout[i];
      if (dim == 'N') continue;
      if (dim == 'C' && !across_channels) continue;
      raw_axes.push_back(i);
    }
    axes = edsl::make_tuple(raw_axes);
  }
  auto R = I - mean(edsl::make_tuple(I_raw, axes, /*keepdims=*/true)).as_tensor();

  if (normalize_variance) {
    auto stdev = edsl::sqrt(variance(edsl::make_tuple(I_raw, axes, /*keepdims=*/true)).as_tensor());
    R = R / maximum(edsl::make_tuple(stdev, edsl::cast(Tensor(epsilon), I.dtype()))).as_tensor();
  }

  return Value{R};
}

Value l2norm(const Value& value) {
  IVLOG(1, "l2norm");
  auto args = value.as_tuple();
  if (args.size() != 4) {
    throw std::runtime_error("norm expects 4 arguments");
  }

  auto I = args[0].as_tensor();
  auto axes = args[1].as_int_tuple();
  auto epsilon = args[2].as_float();
  auto eps_mode = validate<EpsMode>(args[3].as_int());

  auto X = op::sum((I * I), edsl::make_tuple(axes), 1);
  switch (eps_mode) {
    case EpsMode::ADD:
      X = X + epsilon;
      break;
    case EpsMode::MAX:
      X = op::maximum(X, edsl::Tensor{epsilon});
      break;
    default:
      throw std::runtime_error("Unrecognized eps_mode in l2norm op");
  }
  auto N = edsl::sqrt(X);
  return Value(N);
}

Value prod(const Value& value) {
  IVLOG(1, "prod");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("prod expects 3 arguments");
  }

  auto I = args[0].as_tensor();
  auto raw_axes = args[1];
  auto keepdims = args[2].as_bool();

  if (I.rank() == 0) {
    return Value{I};
  }
  if (raw_axes.is_tuple() && raw_axes.as_tuple().empty()) {
    return Value{I};
  }

  // TODO: Move this commented block to Keras?
  // if (I_shape.dtype() == DType::BOOLEAN) {
  //   I = cast(I, floatx());  // TODO: cast if * is not && for bools, don't if it is &&
  // }

  AggregationAxes agg(I.rank(), raw_axes, keepdims);

  I.bind_dims(agg.src_dims);
  Tensor O = Contraction(agg.dst_dims, agg.dst_idxs).product(I(agg.src_idxs));
  return Value{O};
}

Value pool(const Value& value) {
  // The parameters of pool:
  //    0. Input Tensor
  //    1. Pool Mode (avg/max)
  //    2. Pool Size
  //    3. Strides
  //    4. Autopad Mode (explicit, same_lower, same_upper, valid, [maybe full?])
  //    5. Manual Padding
  //    6. Layout (i.e. Channel Order) (minimally NXC v NCX)
  //    7. Include Padding in Avg Computation (bool)
  //    8. Ceil Mode (i.e. as in ONNX)
  //
  // N.B. We determine the number of spatial dimensions from the Pool Size and
  // confirm it is consistent with other parameters that imply a spatial
  // dimension size, specifically strides. We do also check this against the
  // input tensor shape and the manual padding, but these are less strict:
  // manual padding may omit some padding values (which are then assumed to be
  // 0), and the input tensor shape may have multiple channel dimensions (i.e.
  // for cases like tensors going into or coming out of grouped convolutions).

  // Read arguments
  auto args = value.as_tuple();
  if (args.size() != 9) {
    throw std::runtime_error(llvm::formatv("PlaidML pool op expects 9 arguments (received {0})", args.size()));
  }
  auto I = args[0].as_tensor();
  auto pool_mode = validate<PoolMode>(args[1].as_int());
  auto pool_size = args[2].as_int_tuple();
  auto strides = args[3].as_int_tuple();
  auto autopad_mode = validate<AutoPadMode>(args[4].as_int());
  auto manual_padding = args[5].as_int_tuple();
  auto input_layout = validate<TensorLayout>(args[6].as_int());
  auto include_padding_in_avg = args[7].as_bool();
  auto use_ceil_for_output_shape = args[8].as_bool();

  // Initialize useful values
  auto spatial_rank = pool_size.size();
  auto I_channel_dims = I.rank() - spatial_rank - 1;

  // Verify inputs are consistent
  if (manual_padding.size() && autopad_mode != AutoPadMode::EXPLICIT) {
    throw std::runtime_error("Autopadding and manual padding both requested for single pool operation");
  }
  if (strides.size() != spatial_rank) {
    throw std::runtime_error(
        llvm::formatv("Inconsistent spatial rank in pool op (received {0}D pool_size and {1}D strides)", spatial_rank,
                      strides.size()));
  }
  if (I_channel_dims != 1) {
    throw std::runtime_error(
        llvm::formatv("Inconsistent spatial rank in pool op (pool_size has {0} spatial dimensions but input tensor "
                      "has {1} dimensions, and thus {2} spatial dims)",
                      spatial_rank, I.rank(), (I.rank() - 2)));
  }
  if (!is_input_layout(input_layout)) {
    throw std::runtime_error("Tensor layout requested in pool op does not apply to pooling");
  }

  extend_manual_padding(&manual_padding, spatial_rank);

  TensorDim N, C;
  TensorIndex n, c;
  std::vector<TensorDim> X(spatial_rank);
  std::vector<TensorIndex> x(spatial_rank);
  std::vector<TensorIndex> k(spatial_rank);  // within-pool spatial indexes

  std::vector<TensorDim> pad_before;
  std::vector<TensorDim> I_dims = {N};
  std::vector<TensorIndex> I_idxs = {n};
  std::vector<TensorDim> O_dims = {N};
  std::vector<TensorIndex> O_idxs = {n};
  std::vector<Constraint> constraints;
  if (input_layout == TensorLayout::NCX) {
    I_dims.push_back(C);
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    I_dims.push_back(X[i]);
  }
  if (input_layout == TensorLayout::NXC) {
    I_dims.push_back(C);
  }
  I.bind_dims(I_dims);
  if (input_layout == TensorLayout::NCX) {
    I_idxs.push_back(c);
    O_dims.push_back(C);
    O_idxs.push_back(c);
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    O_idxs.push_back(x[i]);
    TensorDim local_pad_before;
    TensorDim local_output_size;
    TensorIndex local_index;
    std::tie(local_pad_before, local_output_size) =
        compute_padding_and_output_size(X[i], TensorDim(pool_size[i]), strides[i], autopad_mode, manual_padding[i],
                                        manual_padding[spatial_rank + i], 1, 1, use_ceil_for_output_shape);
    pad_before.emplace_back(local_pad_before);
    local_index = strides[i] * x[i] + k[i] - pad_before[i];
    O_dims.emplace_back(local_output_size);
    I_idxs.emplace_back(local_index);
    constraints.push_back(k[i] < pool_size[i]);
  }
  if (input_layout == TensorLayout::NXC) {
    I_idxs.push_back(c);
    O_dims.push_back(C);
    O_idxs.push_back(c);
  }
  Contraction O = Contraction(O_dims, O_idxs).add_constraints(constraints);
  if (pool_mode == PoolMode::MAX) {
    O.max(I(I_idxs));
    return Value{O};
  } else if (pool_mode == PoolMode::MIN) {
    O.min(I(I_idxs));
    return Value{O};
  } else if (pool_mode == PoolMode::SUM) {
    O.sum(I(I_idxs));
    return Value{O};
  } else if (pool_mode == PoolMode::AVG) {
    O.sum(I(I_idxs));
    if (include_padding_in_avg) {
      int64_t total_pool_size = 1;
      for (const auto& sz : pool_size) {
        total_pool_size *= sz;
      }
      return Value{O / total_pool_size};
    } else {
      auto One = cast(Tensor{1}, I.dtype());
      // Note: O_idxs is used in both cases b/c both need indexes of the form
      // x0, x1, ... However, they do not represent the same index values (and
      // notably do not iterate over the same size of dimensions as I_dims !=
      // O_dims)
      Tensor Ones = Contraction(I_dims, O_idxs).assign(One());
      Tensor Count = Contraction(O_dims, O_idxs).sum(Ones(I_idxs)).add_constraints(constraints);
      // Ones(O_idxs) = One(std::vector<TensorIndex>());
      return Value{O / Count};
    }
  } else {
    throw std::runtime_error("Unrecognized pool_mode in pool op");
  }
}

Value relu(const Value& value) {
  IVLOG(1, "relu");
  auto args = value.as_tuple();
  if (args.size() != 4) {
    throw std::runtime_error("relu expects 4 arguments");
  }
  auto I = args[0].as_tensor();
  auto alpha = args[1];
  auto max_value = args[2];
  auto threshold = args[3].as_float();
  Tensor A;
  if (alpha.is_none()) {
    A = Tensor(0.0);
  } else {
    A = alpha.as_tensor();
  }
  auto O = select(I < threshold, edsl::cast(A * (I - threshold), I.dtype()), I);
  if (!max_value.is_none()) {
    auto M = cast(max_value.as_tensor(), I.dtype());
    O = select(O < M, O, M);
  }
  return Value{O};
}

Value reorg_yolo(const Value& value) {
  IVLOG(1, "reorg_yolo");

  auto args = value.as_tuple();
  if (args.size() != 4) {
    throw std::runtime_error(llvm::formatv("PlaidML reorg_yolo op expects 4 arguments (received {0})", args.size()));
  }
  auto I = args[0].as_tensor();
  auto stride = args[1].as_int();
  auto decrease = args[2].as_bool();
  auto layout = args[3].as_str();

  auto ndims = I.rank();
  if (ndims != 4) {
    throw std::runtime_error(
        llvm::formatv("PlaidML reorg_yolo op expects I to have 4 dimensions (received {0})", ndims));
  }

  TensorLens lens(layout, "NCHW");
  I = I.use(lens);

  TensorDim N, C, H, W;
  I.bind_dims(N, C, H, W);

  TensorIndex b, i, j, k, x, y;
  auto h = j * stride + y;
  auto w = i * stride + x;

  Tensor O;
  if (decrease) {
    auto C_out = C / (stride * stride);
    auto c = k + ((x + y * stride) * C_out);
    O = Contraction(lens)
            .outShape(N, C_out, H * stride, W * stride)
            .outAccess(b, k, h, w)
            .assign(I(b, c, j, i))
            .add_constraint(y < stride)
            .add_constraint(x < stride);
  } else {
    auto C_out = C * (stride * stride);
    auto c = k + ((x + y * stride) * C);
    O = Contraction(lens)
            .outShape(N, C_out, H / stride, W / stride)
            .outAccess(b, c, j, i)
            .assign(I(b, k, h, w))
            .add_constraint(y < stride)
            .add_constraint(x < stride);
  }

  return Value{O};
}

Value repeat(const Value& value) {
  IVLOG(1, "repeat");
  // This is numpy-style `repeat`; Keras calls it `repeat_elements`
  // This is more limited than in numpy (both repeats & axis required, both must
  // be ints)

  // Read arguments
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error(llvm::formatv("PlaidML repeat op expects 3 arguments (received {0})", args.size()));
  }
  auto I = args[0].as_tensor();
  auto raw_axis = args[2].as_int();

  TensorDim repeats;
  if (args[1].is_int()) {
    repeats = TensorDim(args[1].as_int());
  } else if (args[1].is_dim()) {
    repeats = args[1].as_dim();
  } else {
    throw std::runtime_error("Repeat only accepts int or TensorDim for repeats parameter");
  }

  // Set up useful variables
  auto ndims = I.rank();
  auto axis = normalize_axis(raw_axis, ndims, "repeat");

  std::vector<TensorDim> I_dims(ndims);
  I.bind_dims(I_dims);
  std::vector<TensorDim> O_dims(I_dims);
  O_dims[axis] = repeats * I_dims[axis];
  std::vector<TensorIndex> I_idxs(ndims);
  std::vector<TensorIndex> O_idxs(I_idxs);
  TensorIndex inner;
  O_idxs[axis] = repeats * I_idxs[axis] + inner;
  Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs)).add_constraint(inner < repeats);
  return Value{O};
}

Value reshape(const Value& value) {
  IVLOG(1, "reshape");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error(llvm::formatv("PlaidML reshape op expects 2 arguments (received {0})", args.size()));
  }

  auto I = args[0].as_tensor();
  std::vector<TensorDim> O_dims;
  std::vector<TensorDim> I_dims(I.rank());
  I.bind_dims(I_dims);

  size_t fill_dim = 0;
  bool fill_dim_set = false;

  auto target_shape = args[1].as_tuple();
  for (size_t i = 0; i < target_shape.size(); i++) {
    if (target_shape[i].is_int()) {
      auto dim = target_shape[i].as_int();
      switch (dim) {
        case (AUTO_DIM_MATCH):
          if (i < I_dims.size()) {
            O_dims.emplace_back(I_dims[i]);
          } else {
            throw std::runtime_error(llvm::formatv("matching dimension requested at {0} from {1}-dimensional tensor",
                                                   (i + 1), I_dims.size()));
          }
          break;

        case (AUTO_DIM_FILL):
          if (fill_dim_set) {
            throw std::runtime_error("at most one dimension's size may be inferred");
          }
          fill_dim_set = true;
          O_dims.emplace_back(1);
          fill_dim = i;
          break;
        default:
          O_dims.emplace_back(dim);
          break;
      }
    } else if (target_shape[i].is_dim()) {
      O_dims.emplace_back(target_shape[i].as_dim());
    } else if (target_shape[i].is_none()) {
      if (i < I_dims.size()) {
        O_dims.emplace_back(I_dims[i]);
      } else {
        throw std::runtime_error(
            llvm::formatv("matching dimension requested at {0} from {1}-dimensional tensor", (i + 1), I_dims.size()));
      }
    }
  }

  if (fill_dim_set) {
    TensorDim num(1);
    for (size_t i = 0; i < I_dims.size(); i++) {
      num = I_dims[i] * num;
    }
    TensorDim den(1);
    for (size_t i = 0; i < O_dims.size(); i++) {
      den = O_dims[i] * den;
    }
    O_dims[fill_dim] = TensorDim(num / den);
  }
  return Value{edsl::reshape(I, O_dims)};
}

Value scale_gradient(const Value& value) {
  IVLOG(1, "scale_gradient");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("scale_gradient expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  Tensor scale;
  if (args[1].is_float()) {
    // Cast scale to Tensor if it's given as a float
    scale = Tensor{args[1].as_float()};
  } else {
    scale = args[1].as_tensor();
  }
  auto O = I;  // Forward pass is NoOp
  return Value{O};
}

Value sigmoid(const Value& value) {
  IVLOG(1, "sigmoid");
  auto args = value.as_tuple();
  if (args.size() != 1) {
    throw std::runtime_error("sigmoid expects 1 argument");
  }
  auto I = args[0].as_tensor();
  auto O = 1.0 / (1.0 + exp(-I));
  return Value{O};
}

Value slice(const Value& value) {
  // This code avoids using max/min ops to keep start/stop values in the [-dim - 1, dim] range
  // This means requesting a slice with a start/stop index outside the valid range will give bizarre behavior
  IVLOG(1, "slice");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("slice expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  auto slices = args[1].as_tuple();

  // Useful values and correctness checks
  auto ndims = I.rank();
  // First, handle the case of a scalar
  if (ndims == 0) {
    return Value{I};
  }
  if (slices.size() != ndims) {
    throw std::runtime_error(
        llvm::formatv("{0} slice axes provided to slice {1}-dimensional tensor", slices.size(), ndims));
  }

  // Initialize dims & indexes
  std::vector<TensorDim> I_dims{ndims};
  I.bind_dims(I_dims);
  std::vector<TensorIndex> O_idxs{ndims};
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> I_idxs;

  // For each axis, compute output size and formula for extracting requested values
  size_t skipped_O_idxs = 0;  // How many indexes are unneeded because the slice is a single integer
  for (size_t i = 0; i < ndims; ++i) {
    // If this "slice" is just an integer, extract that entry in this axis and continue to the next axis
    if (slices[i].is_int()) {
      auto idx_val = slices[i].as_int();
      TensorIndex idx{idx_val};
      if (idx_val < 0) {
        idx = idx + I_dims[i];
      }
      // Do not push to O_dims -- this dim is omitted; instead, skip using the index
      O_idxs.erase(O_idxs.begin() + i - skipped_O_idxs);
      skipped_O_idxs++;
      I_idxs.push_back(TensorIndex{idx});
      continue;
    }

    // Parse the slice as a tuple of 3 ints or Nones
    if (!slices[i].is_tuple()) {
      throw std::runtime_error("Attempted to define slice with invalid type");
    }
    auto slice = slices[i].as_tuple();
    if (slice.size() != 3) {
      throw std::runtime_error(llvm::formatv("Attempted to slice with {0} entries, 3 required", slice.size()));
    }

    // Read slice, filling in appropriate default values where Nones appear
    TensorDim start;
    TensorDim stop;
    int64_t step;
    // Setup step first (needed to determine start/end defaults)
    if (slice[2].is_none()) {
      step = 1;
    } else {
      try {
        step = slice[2].as_int();
      } catch (std::runtime_error& e) {
        throw std::runtime_error(
            llvm::formatv("Unable to parse step of slice as int, original message: {0}", e.what()));
      }
    }
    if (step == 0) {
      throw std::runtime_error("Cannot slice with step size 0");
    }
    // Setup start
    if (slice[0].is_none()) {
      if (step > 0) {
        start = TensorDim{0};
      } else {
        start = I_dims[i] - 1;
      }
    } else {
      int64_t int_start;
      try {
        int_start = slice[0].as_int();
      } catch (std::runtime_error& e) {
        throw std::runtime_error(
            llvm::formatv("Unable to parse start of slice as int, original message: {0}", e.what()));
      }
      if (int_start < 0) {
        start = I_dims[i] + int_start;
      } else {
        start = TensorDim{int_start};
      }
    }
    // Setup stop
    if (slice[1].is_none()) {
      if (step > 0) {
        stop = I_dims[i];
      } else {
        stop = TensorDim{-1};
      }
    } else {
      int64_t int_stop;
      try {
        int_stop = slice[1].as_int();
      } catch (std::runtime_error& e) {
        throw std::runtime_error(
            llvm::formatv("Unable to parse stop of slice as int, original message: {0}", e.what()));
      }
      if (int_stop < 0) {
        stop = I_dims[i] + int_stop;
      } else {
        stop = TensorDim{int_stop};
      }
    }
    // Set output size for this axis
    auto offset_to_ceil = step;  // Make this a ceil division w/ this offset
    if (step > 0) {
      offset_to_ceil -= 1;
    } else {
      offset_to_ceil += 1;
    }
    O_dims.push_back((stop - start + offset_to_ceil) / step);
    // Set index to read for this axis
    I_idxs.push_back(start + step * O_idxs[i - skipped_O_idxs]);
  }

  // Perform the slice
  Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return Value{O};
}

Value softmax(const Value& value) {
  IVLOG(1, "softmax");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("softmax expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  auto raw_axis = args[1].as_int();

  int64_t ndims = I.rank();
  int64_t axis = normalize_axis(raw_axis, ndims, "softmax");

  // Ensure the axis is the last dimension, to make the derivative code happy
  bool transposed = false;
  std::vector<Value> pattern(ndims);  // Will be reused at the end to return to original order
  if (axis != ndims - 1) {
    transposed = true;
    for (int64_t i = 0; i < ndims; ++i) {
      if (i == axis) {
        pattern[i] = Value{ndims - 1};
      } else if (i == ndims - 1) {
        pattern[i] = Value{axis};
      } else {
        pattern[i] = Value{i};
      }
    }
    I = transpose(make_tuple(args[0], Value{pattern})).as_tensor();
    axis = ndims - 1;  // we've moved the softmax axis to be the last axis
  }

  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs(ndims);
  I.bind_dims(I_dims);
  // R_dims & R_idxs are the dims/idxs reduced along the specified axis; used in the inner contractions
  std::vector<TensorDim> R_dims = I_dims;
  std::vector<TensorIndex> R_idxs = I_idxs;
  R_dims[axis] = TensorDim{1};
  R_idxs[axis] = TensorIndex{0};
  Tensor M = Contraction(R_dims, R_idxs).max(I(I_idxs));
  auto E = exp(I - M);
  Tensor N = Contraction(R_dims, R_idxs).sum(E(I_idxs));
  auto O = E / N;
  // If we reordered, return to original order
  if (transposed) {
    return transpose(make_tuple(Value{O}, Value{pattern}));
  }
  return Value{O};
}

Value sort(const Value& value) {
  IVLOG(1, "sort");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("sort expects 3 arguments");
  }
  auto I = args[0].as_tensor();
  auto raw_axis = args[1].as_int();
  auto direction = validate<SortDirection>(args[2].as_int());

  int64_t ndims = I.rank();
  int64_t axis = normalize_axis(raw_axis, ndims, "sort");

  std::vector<TensorDim> I_dims(ndims);
  I.bind_dims(I_dims);

  std::vector<Tensor> IX_dims;
  for (int64_t i = 0; i < ndims; ++i) {
    if (i == axis) {
      IX_dims.push_back(op::unsqueeze(argsort(I, axis, direction), {ndims}));
    } else {
      IX_dims.push_back(op::unsqueeze(edsl::index(I_dims, i), {ndims}));
    }
  }
  auto IX = op::concatenate(IX_dims, -1);
  Tensor O = edsl::gather(I, IX).mode(GatherMode::ND);
  return Value{O};
}

Value spatial_padding(const Value& value) {
  IVLOG(1, "spatial_padding");
  auto args = value.as_tuple();
  if (args.size() != 4) {
    throw std::runtime_error("spatial_padding expects 4 arguments");
  }
  auto I = args[0].as_tensor();
  auto lo_pads = args[1].as_int_tuple();
  auto hi_pads = args[2].as_int_tuple();
  auto data_layout = validate<TensorLayout>(args[3].as_int());

  // validate inputs
  auto nonspatial_ndims = nonspatial_dims(data_layout);
  auto spatial_rank = I.rank() - nonspatial_ndims;

  if (spatial_rank < 1) {
    throw std::runtime_error(llvm::formatv(
        "Insufficient spatial rank in spatial_padding op (At least 1 spatial dim required; received {0} spatial "
        "dims based on an input tensor with {1} dims with a specified layout with {2} nonspatial dims.)",
        spatial_rank, I.rank(), nonspatial_ndims));
  }
  if (lo_pads.size() != spatial_rank) {
    throw std::runtime_error(
        llvm::formatv("Inconsistent spatial rank in spatial_padding op (received {0} spatial dim(s) based on an "
                      "input tensor with {1} dims with a specified layout with {2} nonspatial dims, but received "
                      "lower padding for {3} spatial dims.)",
                      spatial_rank, I.rank(), nonspatial_ndims, lo_pads.size()));
  }
  if (hi_pads.size() != spatial_rank) {
    throw std::runtime_error(
        llvm::formatv("Inconsistent spatial rank in spatial_padding op (received {0} spatial dim(s) based on an "
                      "input tensor with {1} dims with a specified layout with {2} nonspatial dims, but received "
                      "upper padding for {3} spatial dims.)",
                      spatial_rank, I.rank(), nonspatial_ndims, hi_pads.size()));
  }

  std::vector<TensorDim> I_dims;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorIndex> O_idxs;

  switch (data_layout) {
    case TensorLayout::GKCX: {
      IVLOG(2, "Spatial padding requested for tensor with kernel-style layout.");
      // Initialize dims & indexes
      TensorDim G, K, C;
      TensorIndex g("g");
      TensorIndex k("k");
      TensorIndex c("c");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(llvm::formatv("x{0}", i));
      }

      // Assign input dims & indexes
      I_dims.push_back(G);
      I_idxs.push_back(g);
      I_dims.push_back(K);
      I_idxs.push_back(k);
      I_dims.push_back(C);
      I_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      O_dims.push_back(G);
      O_idxs.push_back(g);
      O_dims.push_back(K);
      O_idxs.push_back(k);
      O_dims.push_back(C);
      O_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
    } break;
    case TensorLayout::KCX: {
      IVLOG(2, "Spatial padding requested for tensor with kernel-style layout.");
      // Initialize dims & indexes
      TensorDim K, C;
      TensorIndex k("k");
      TensorIndex c("c");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(llvm::formatv("x{0}", i));
      }

      // Assign input dims & indexes
      I_dims.push_back(K);
      I_idxs.push_back(k);
      I_dims.push_back(C);
      I_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      O_dims.push_back(K);
      O_idxs.push_back(k);
      O_dims.push_back(C);
      O_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
    } break;
    case TensorLayout::NCX: {
      TensorDim N, C;
      TensorIndex n("n");
      TensorIndex c("c");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(llvm::formatv("x{0}", i));
      }

      // Assign input dims & indexes
      I_dims.push_back(N);
      I_idxs.push_back(n);
      I_dims.push_back(C);
      I_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      O_dims.push_back(N);
      O_idxs.push_back(n);
      O_dims.push_back(C);
      O_idxs.push_back(c);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
    } break;
    case TensorLayout::NXC: {
      TensorDim N, C;
      TensorIndex n("n");
      TensorIndex c("c");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(llvm::formatv("x{0}", i));
      }

      // Assign input dims & indexes
      I_dims.push_back(N);
      I_idxs.push_back(n);
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I_dims.push_back(C);
      I_idxs.push_back(c);
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      O_dims.push_back(N);
      O_idxs.push_back(n);
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
      O_dims.push_back(C);
      O_idxs.push_back(c);
    } break;
    case TensorLayout::XCK: {
      IVLOG(2, "Spatial padding requested for tensor with kernel-style layout.");
      TensorDim C, K;
      TensorIndex c("c");
      TensorIndex k("k");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(llvm::formatv("x{0}", i));
      }

      // Assign input dims & indexes
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I_dims.push_back(C);
      I_idxs.push_back(c);
      I_dims.push_back(K);
      I_idxs.push_back(k);
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
      O_dims.push_back(C);
      O_idxs.push_back(c);
      O_dims.push_back(K);
      O_idxs.push_back(k);
    } break;
    case TensorLayout::XGCK: {
      IVLOG(2, "Spatial padding requested for tensor with kernel-style layout.");
      TensorDim G, C, K;
      TensorIndex g("g");
      TensorIndex c("c");
      TensorIndex k("k");
      std::vector<TensorDim> X(spatial_rank);
      std::vector<TensorIndex> x;
      for (size_t i = 0; i < spatial_rank; ++i) {
        x.emplace_back(llvm::formatv("x{0}", i));
      }

      // Assign input dims & indexes
      for (size_t i = 0; i < spatial_rank; ++i) {
        I_dims.push_back(X[i]);
        I_idxs.push_back(x[i]);
      }
      I_dims.push_back(G);
      I_idxs.push_back(g);
      I_dims.push_back(C);
      I_idxs.push_back(c);
      I_dims.push_back(K);
      I_idxs.push_back(k);
      I.bind_dims(I_dims);

      // Assign output dims & indexes
      for (size_t i = 0; i < spatial_rank; ++i) {
        O_dims.push_back(X[i] + lo_pads[i] + hi_pads[i]);
        O_idxs.push_back(x[i] + lo_pads[i]);
      }
      O_dims.push_back(G);
      O_idxs.push_back(g);
      O_dims.push_back(C);
      O_idxs.push_back(c);
      O_dims.push_back(K);
      O_idxs.push_back(k);
    } break;
    default:
      throw std::runtime_error("Unrecognized TensorLayout in spatial_padding");
  }
  Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return Value{O};
}

Value square(const Value& value) {
  IVLOG(1, "square");
  auto x = value.as_tensor();
  return Value(x * x);
}

Value squeeze(const Value& value) {
  IVLOG(1, "squeeze");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("Squeeze expects 2 arguments");
  }

  // argument 0: tensor to be squeezed
  auto I = args[0].as_tensor();
  auto ndims = I.rank();
  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorDim> O_dims;
  I.bind_dims(I_dims);

  // argument 1: axes to squeeze upon
  std::vector<int64_t> raw_axes;
  if (args[1].is_int()) {
    raw_axes.push_back(args[1].as_int());
  } else {
    raw_axes = args[1].as_int_tuple();
  }
  std::set<size_t> axes;
  for (auto& raw_axis : raw_axes) {
    axes.insert(normalize_axis(raw_axis, ndims, "squeeze"));
  }

  for (size_t i = 0; i < ndims; ++i) {
    if (!axes.count(i)) {
      O_dims.push_back(I_dims[i]);
    }
  }
  std::vector<Value> O_dims_values;
  for (const TensorDim& dim : O_dims) {
    O_dims_values.push_back(Value{dim});
  }
  return reshape(make_tuple(Value{I}, Value{O_dims_values}));
}

Value sum(const Value& value) {
  IVLOG(1, "sum");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("sum expects 3 arguments");
  }

  auto I = args[0].as_tensor();
  if (I.rank() == 0) {
    return Value{I};
  }

  // TODO: Move this commented block to Keras?
  // if (I_shape.dtype() == DType::BOOLEAN) {
  //   I = cast(I, floatx());
  // }

  auto axes = args[1];
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    return Value{I};
  }

  auto keepdims = args[2].as_bool();

  AggregationAxes agg(I.rank(), axes, keepdims);
  I.bind_dims(agg.src_dims);
  Tensor O = Contraction(agg.dst_dims, agg.dst_idxs).sum(I(agg.src_idxs));
  return Value{O};
}

Value tile(const Value& value) {
  IVLOG(1, "tile");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("Tile expects 2 arguments");
  }

  // Read arguments
  auto I = args[0].as_tensor();
  auto reps = args[1].as_int_tuple();

  // Validate args
  auto ndims = I.rank();
  if (reps.empty()) {
    throw std::runtime_error("No tiling factors provided to tile operation");
  }
  while (reps.size() < ndims) {
    // Numpy style: extend to full rank with 1s
    reps.push_back(1);
  }
  if (reps.size() > ndims) {
    throw std::runtime_error("More tiling factors provided to tile operation than tensor dimensions");
  }

  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs(ndims);
  I.bind_dims(I_dims);
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  for (size_t i = 0; i < ndims; ++i) {
    O_dims.push_back(I_dims[i] * reps[i]);
    O_idxs.push_back(TensorIndex() * I_dims[i] + I_idxs[i]);
  }
  Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return Value{O};
}

Value transpose(const Value& value) {
  // Reorders dimensions so dim i of the output is dim pattern[i] of the input
  IVLOG(1, "transpose");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("Transpose expects 2 arguments");
  }

  // Read arguments
  auto I = args[0].as_tensor();
  auto pattern_val = args[1];

  // Normalize pattern value
  auto ndims = I.rank();
  std::vector<int64_t> pattern;
  if (pattern_val.is_none()) {
    // Default is to reverse the dimensions
    for (size_t i = 0; i < ndims; ++i) {
      pattern.push_back(ndims - 1 - i);
    }
  } else if (pattern_val.is_tuple()) {
    pattern = pattern_val.as_int_tuple();
  } else {
    throw std::runtime_error("Transpose 2nd argument must be none or integer tuple");
  }

  // Ensure pattern is sane
  for (const auto& i : pattern) {
    if (i < 0 || ndims <= static_cast<size_t>(i)) {
      throw std::runtime_error(
          llvm::formatv("Transpose of nonexistent axis {0} requested (input has {1} dimensions)", i, ndims));
    }
  }

  // Setup inputs and outputs
  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs(ndims);
  I.bind_dims(I_dims);
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  for (size_t i = 0; i < ndims; ++i) {
    O_dims.push_back(I_dims[pattern[i]]);
    O_idxs.push_back(I_idxs[pattern[i]]);
  }
  Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return Value{O};
}

Value unsqueeze(const Value& value) {
  // Read arguments
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error(llvm::formatv("PlaidML unsqueeze op expects 2 arguments (received {0})", args.size()));
  }

  auto I = args[0].as_tensor();
  auto ndims = I.rank();

  std::vector<int64_t> raw_axes;
  if (args[1].is_int()) {
    raw_axes.push_back(args[1].as_int());
  } else {
    raw_axes = args[1].as_int_tuple();
  }

  std::set<size_t> axes;
  for (auto& raw_axis : raw_axes) {
    axes.insert(normalize_axis(raw_axis, ndims + raw_axes.size(), "unsqueeze"));
  }

  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs;
  std::vector<TensorDim> O_dims;
  std::vector<TensorIndex> O_idxs;
  I.bind_dims(I_dims);
  size_t src_loc = 0;
  size_t new_rank = ndims + axes.size();
  for (size_t i = 0; i < new_rank; i++) {
    if (axes.count(i)) {
      O_dims.push_back(edsl::TensorDim(1));
      O_idxs.emplace_back(llvm::formatv("a{0}", i));
    } else {
      I_idxs.emplace_back(llvm::formatv("n{0}", i));
      O_dims.push_back(I_dims[src_loc]);
      O_idxs.push_back(I_idxs[src_loc]);
      src_loc++;
    }
  }
  if (src_loc != I.rank()) {
    throw std::runtime_error(llvm::formatv("Unsqueeze did not replicate entirety of input into output"));
  }
  Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return Value{O};
}

Value variance(const Value& value) {
  // This computes the *uncorrected* sample variance (i.e. denominator = n
  // rather than = n-1) to match tensorflow
  IVLOG(1, "variance");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("Variance expects 3 arguments");
  }

  // Read arguments
  auto I = args[0].as_tensor();
  auto axes = args[1];
  auto keepdims = args[2].as_bool();

  IVLOG(2, "I: " << I.str());
  IVLOG(2, "axes: " << axes);
  IVLOG(2, "keep_dims: " << keepdims);

  // Handle trivial cases
  if (I.rank() == 0) {
    // TODO: Adjust for dtype?
    return Value{0};
  }
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    throw std::runtime_error("Variance expects nonempty axis list");
  }

  // TODO: Move this commented block to Keras?
  // if (I.shape().dtype() == DType::BOOLEAN) {
  //   I = cast(I, floatx());
  // }

  auto Mean = mean(edsl::make_tuple(I, axes, true)).as_tensor();
  AggregationAxes agg(I.rank(), axes, keepdims);

  I.bind_dims(agg.src_dims);

  auto SquaredDifference = (I - Mean) * (I - Mean);
  Tensor SumSqDiff = Contraction(agg.dst_dims, agg.dst_idxs).sum(SquaredDifference(agg.src_idxs));
  auto denom = Tensor{1};
  for (const auto& axis : agg.axes) {
    denom = denom * agg.src_dims.at(axis);
  }
  return Value{SumSqDiff / denom};
}

void RegisterOps() {
  auto registry = OperationRegistry::Instance();
  registry->Register("abs", abs);
  registry->Register("all", all);
  registry->Register("any", any);
  registry->Register("argmax", argmax);
  registry->Register("binary_crossentropy", binary_crossentropy);
  registry->Register("broadcast", broadcast);
  registry->Register("clip", clip);
  registry->Register("concatenate", concatenate);
  registry->Register("convolution", convolution);
  registry->Register("cumprod", cumprod);
  registry->Register("cumsum", cumsum);
  registry->Register("dot", dot);
  registry->Register("elu", elu);
  registry->Register("explicit_padding", explicit_padding);
  registry->Register("flip", flip);
  registry->Register("hard_sigmoid", hard_sigmoid);
  registry->Register("image_resize", image_resize);
  registry->Register("lrn", lrn);
  registry->Register("max", max);
  registry->Register("maximum", maximum);
  registry->Register("mean", mean);
  registry->Register("min", min);
  registry->Register("minimum", minimum);
  registry->Register("mvn", mvn);
  registry->Register("nms", nms);
  registry->Register("l2norm", l2norm);
  registry->Register("pool", pool);
  registry->Register("prod", prod);
  registry->Register("relu", relu);
  registry->Register("reorg_yolo", reorg_yolo);
  registry->Register("repeat", repeat);
  registry->Register("reshape", reshape);
  registry->Register("scale_gradient", scale_gradient);
  registry->Register("sigmoid", sigmoid);
  registry->Register("slice", slice);
  registry->Register("softmax", softmax);
  registry->Register("sort", sort);
  registry->Register("spatial_padding", spatial_padding);
  registry->Register("square", square);
  registry->Register("squeeze", squeeze);
  registry->Register("sum", sum);
  registry->Register("tile", tile);
  registry->Register("transpose", transpose);
  registry->Register("unsqueeze", unsqueeze);
  registry->Register("variance", variance);
}

}  // namespace plaidml::op::lib
