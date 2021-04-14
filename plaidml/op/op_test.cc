// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>

#include "llvm/ADT/StringRef.h"

#include "plaidml/op/op.h"
#include "plaidml/testenv.h"
#include "pmlc/util/logging.h"

using ::testing::Eq;

using namespace plaidml;        // NOLINT
using namespace plaidml::edsl;  // NOLINT

class OpTest : public TestFixture {};

// TEST_F(OpTest, Abs) {
//   auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
//   auto program = makeProgram("abs", {I}, {op::abs(I)});
//   runProgram(program);
// }

// TEST_F(OpTest, All) {
//   auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
//   auto program = makeProgram("all", {I}, {op::all(I)});
//   runProgram(program);
// }

// TEST_F(OpTest, Any) {
//   auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
//   auto program = makeProgram("any", {I}, {op::any(I)});
//   runProgram(program);
// }

// TEST_F(OpTest, Argmax) {
//   auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
//   auto program = makeProgram("argmax", {I}, {op::argmax(I)});
//   runProgram(program);
// }

// TEST_F(OpTest, BinaryCrossentropy) {
//   auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
//   auto O = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "O");
//   auto program = makeProgram("binary_crossentropy", {I, O}, {op::binary_crossentropy(I, O, 0.0)});
//   runProgram(program);
// }

// TEST_F(OpTest, BroadcastNoOp) {
//   auto A = Placeholder(DType::FLOAT32, {3});
//   std::vector<int64_t> rshape = {3};
//   std::vector<int64_t> bdims = {0};
//   auto C = op::broadcast(A, rshape, bdims);
//   auto program = makeProgram("broadcast_nop", {A}, {C});

//   std::vector<float> A_input = {0, 1, 2};
//   checkExact(program, {A_input}, {A_input});
// }

// TEST_F(OpTest, BroadcastScalar) {
//   auto A = Placeholder(DType::FLOAT32, {});
//   std::vector<int64_t> rshape = {3, 4};
//   std::vector<int64_t> bdims = {};
//   auto C = op::broadcast(A, rshape, bdims);
//   auto program = makeProgram("broadcast_scalar", {A}, {C});

//   std::vector<float> A_input = {3};
//   std::vector<float> C_output = {3, 3, 3, 3,  //
//                                  3, 3, 3, 3,  //
//                                  3, 3, 3, 3};
//   checkExact(program, {A_input}, {C_output});
// }

// TEST_F(OpTest, BroadcastNoOpLarge) {
//   auto A = Placeholder(DType::FLOAT32, {3, 4});
//   std::vector<int64_t> rshape = {3, 4};
//   std::vector<int64_t> bdims = {0, 1};
//   auto C = op::broadcast(A, rshape, bdims);
//   auto program = makeProgram("broadcast_nop_large", {A}, {C});

//   std::vector<float> A_input = {0, 1, 2, 3,  //
//                                 0, 1, 2, 3,  //
//                                 0, 1, 2, 3};
//   checkExact(program, {A_input}, {A_input});
// }

// TEST_F(OpTest, BroadcastNumpy) {
//   auto A = Placeholder(DType::FLOAT32, {1, 3, 3});
//   std::vector<int64_t> rshape = {1, 4, 3, 3};
//   std::vector<int64_t> bdims = {0, 2, 3};
//   auto C = op::broadcast(A, rshape, bdims);
//   auto program = makeProgram("broadcast_numpy", {A}, {C});

//   std::vector<float> A_input = {0, 1, 2,  //
//                                 0, 1, 2,  //
//                                 0, 1, 2};
//   std::vector<float> C_output = {0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2};
//   checkExact(program, {A_input}, {C_output});
// }

// TEST_F(OpTest, BroadcastNumpy2) {
//   auto A = Placeholder(DType::FLOAT32, {3, 1});
//   std::vector<int64_t> rshape = {3, 4};
//   std::vector<int64_t> bdims = {0, 1};
//   auto C = op::broadcast(A, rshape, bdims);
//   auto program = makeProgram("broadcast_numpy_2", {A}, {C});

//   std::vector<float> A_input = {0, 1, 2};
//   std::vector<float> C_output = {0, 0, 0, 0,  //
//                                  1, 1, 1, 1,  //
//                                  2, 2, 2, 2};
//   checkExact(program, {A_input}, {C_output});
// }

// TEST_F(OpTest, BroadcastNumpy3) {
//   auto A = Placeholder(DType::FLOAT32, {3});
//   std::vector<int64_t> rshape = {4, 3};
//   std::vector<int64_t> bdims = {1};
//   auto C = op::broadcast(A, rshape, bdims);
//   auto program = makeProgram("broadcast_numpy_3", {A}, {C});

//   std::vector<float> A_input = {0, 1, 2};
//   std::vector<float> C_output = {0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2,  //
//                                  0, 1, 2};
//   checkExact(program, {A_input}, {C_output});
// }

// TEST_F(OpTest, BroadcastNonNumpy) {
//   auto A = Placeholder(DType::FLOAT32, {3});
//   std::vector<int64_t> rshape = {3, 4};
//   std::vector<int64_t> bdims = {0};
//   auto C = op::broadcast(A, rshape, bdims);
//   auto program = makeProgram("broadcast_non_numpy", {A}, {C});

//   std::vector<float> A_input = {0, 1, 2};
//   std::vector<float> C_output = {0, 0, 0, 0,  //
//                                  1, 1, 1, 1,  //
//                                  2, 2, 2, 2};
//   checkExact(program, {A_input}, {C_output});
// }

// TEST_F(OpTest, Broadcast) {
//   auto I = Placeholder(DType::FLOAT32, {1, 224, 224}, "I");
//   std::vector<int64_t> result_shape = {1, 224, 224, 3};
//   std::vector<int64_t> bcast_axes = {0, 1, 2};
//   auto program = makeProgram("broadcast", {I}, {op::broadcast(I, result_shape, bcast_axes)});
//   runProgram(program);
// }

// TEST_F(OpTest, Clip) {
//   auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
//   auto raw_min = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "raw_min");
//   auto raw_max = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "raw_max");
//   auto program = makeProgram("clip", {I, raw_min, raw_max}, {op::clip(I, raw_min, raw_max)});
//   runProgram(program);
// }

// TEST_F(OpTest, Concatenate) {
//   auto A = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "A");
//   auto B = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "B");
//   auto program = makeProgram("concatenate", {A, B}, {op::concatenate({A, B}, 2)});
//   runProgram(program);
// }

struct ConvolutionParams {
  DType dType;
  std::vector<int64_t> I_dims;
  std::vector<int64_t> K_dims;
  std::function<void(op::convolution&)> builder;
};

struct ConvolutionTest : public OpTest, public ::testing::WithParamInterface<ConvolutionParams> {};

TEST_P(ConvolutionTest, Basic) {
  const auto& params = GetParam();
  auto I = Placeholder(params.dType, params.I_dims);
  auto K = Placeholder(params.dType, params.K_dims);
  auto conv = op::convolution(I, K);
  params.builder(conv);
  auto program = makeProgram("convolution", {I, K}, {conv});
  runProgram(program);
}

ConvolutionParams convParams[] = {
    // {
    //     DType::FLOAT32,
    //     {1, 224, 224, 3},
    //     {7, 7, 3, 64},
    //     [](op::convolution& conv) {
    //       conv.strides({2, 2}).autopad_mode(op::AutoPadMode::EXPLICIT).manual_padding({2, 2});
    //     },
    // },
    // {
    //     // resnet/conv1
    //     DType::FLOAT32,
    //     {1, 224, 224, 3},
    //     {7, 7, 3, 64},
    //     [](op::convolution& conv) {
    //       conv.name("conv1").strides({2, 2}).autopad_mode(op::AutoPadMode::EXPLICIT).manual_padding({3, 3});
    //     },
    // },
    // {
    //     // resnet/res2a_branch1
    //     DType::FLOAT32,
    //     {1, 56, 56, 64},
    //     {1, 1, 64, 256},
    //     [](op::convolution& conv) {
    //       conv.name("res2a_branch1").strides({1, 1}).autopad_mode(op::AutoPadMode::VALID);
    //     },
    // },
    {
        // MVLM conv
        DType::FLOAT16,
        {1, 3, 256, 256},
        {64, 3, 3, 3},
        [](op::convolution& conv) {
          conv.name("MVLM")
              .autopad_mode(op::AutoPadMode::EXPLICIT)
              .strides({1, 1})
              .manual_padding({1, 1, 1, 1})
              .input_layout(plaidml::op::TensorLayout::NCX)
              .filter_layout(plaidml::op::TensorLayout::KCX)
              .dilations({1, 1});
        },
    },
    // {
    //     // 3D convolution
    //     DType::FLOAT32,
    //     {1, 32, 224, 224, 3},
    //     {7, 7, 7, 3, 64},
    //     [](op::convolution& conv) {
    //       conv.strides({2, 2, 2}).autopad_mode(op::AutoPadMode::EXPLICIT).manual_padding({2, 3, 2, 3, 2, 3});
    //     },
    // },
    // {
    //     // 3D convolution
    //     DType::FLOAT32,
    //     {1, 3, 1, 1, 1024},
    //     {1, 1, 1, 1024, 400},
    //     [](op::convolution& conv) {
    //       conv.strides({1, 1, 1}).autopad_mode(op::AutoPadMode::EXPLICIT).manual_padding({0, 0, 0, 0, 0, 0});
    //     },
    // },
    // {
    //     // 3D convolution
    //     DType::FLOAT32,
    //     {1, 16, 56, 56, 64},
    //     {1, 1, 1, 64, 64},
    //     [](op::convolution& conv) {
    //       conv.strides({1, 1, 1}).autopad_mode(op::AutoPadMode::EXPLICIT).manual_padding({0, 0, 0, 0, 0, 0});
    //     },
    // },
    // {
    //     // 3D convolution
    //     DType::FLOAT32,
    //     {1, 16, 56, 56, 64},
    //     {3, 3, 3, 64, 192},
    //     [](op::convolution& conv) {
    //       conv.strides({1, 1, 1}).autopad_mode(op::AutoPadMode::EXPLICIT).manual_padding({1, 1, 1, 1, 1, 1});
    //     },
    // },
};

INSTANTIATE_TEST_CASE_P(Suite, ConvolutionTest, ::testing::ValuesIn(convParams));

// TEST_F(OpTest, CumProd) {
//   auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
//   auto program = makeProgram("cumprod", {I}, {op::cumprod(I, 2)});
//   runProgram(program);
// }

// TEST_F(OpTest, CumSum) {
//   auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
//   auto program = makeProgram("cumsum", {I}, {op::cumsum(I, 2)});
//   runProgram(program);
// }

// TEST_F(OpTest, CumSumExclusive) {
//   auto I = Placeholder(DType::FLOAT32, {4}, "I");
//   auto program = makeProgram("cumsum", {I}, {op::cumsum(I, 0, true)});
//   std::vector<float> input = {2, 4, 6, 8};
//   std::vector<float> output = {0, 2, 6, 12};
//   checkExact(program, {input}, {output});
// }

// TEST_F(OpTest, Dot) {
//   auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
//   auto K = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "K");
//   auto program = makeProgram("dot", {I, K}, {op::dot(I, K)});
//   runProgram(program);
// }

// TEST_F(OpTest, Elu) {
//   auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
//   auto program = makeProgram("elu", {I}, {op::elu(I, 0.1)});
//   runProgram(program);
// }

// TEST_F(OpTest, ExplicitPadding) {
//   auto I = Placeholder(DType::FLOAT32, {2, 3}, "A");
//   auto O = op::explicit_padding(I, {2, 1}, {2, 1}).padval(Constant(-1.0));
//   auto program = makeProgram("explicit_padding", {I}, {O});

//   std::vector<float> I_input = {1, 2, 3,  //
//                                 4, 5, 6};
//   std::vector<float> O_output = {-1, -1, -1, -1, -1,  //
//                                  -1, -1, -1, -1, -1,  //
//                                  -1, 1,  2,  3,  -1,  //
//                                  -1, 4,  5,  6,  -1,  //
//                                  -1, -1, -1, -1, -1,  //
//                                  -1, -1, -1, -1, -1};

//   checkExact(program, {I_input}, {O_output});
// }

// TEST_F(OpTest, ExplicitPaddingNegInf) {
//   auto I = Placeholder(DType::FLOAT32, {2, 3}, "A");
//   float neg_inf = -std::numeric_limits<float>::infinity();
//   auto O = op::explicit_padding(I, {2, 1}, {2, 1}).padval(Constant(neg_inf));
//   auto program = makeProgram("explicit_padding", {I}, {O});

//   std::vector<float> I_input = {1, 2, 3,  //
//                                 4, 5, 6};
//   std::vector<float> O_output = {neg_inf, neg_inf, neg_inf, neg_inf, neg_inf,  //
//                                  neg_inf, neg_inf, neg_inf, neg_inf, neg_inf,  //
//                                  neg_inf, 1,       2,       3,       neg_inf,  //
//                                  neg_inf, 4,       5,       6,       neg_inf,  //
//                                  neg_inf, neg_inf, neg_inf, neg_inf, neg_inf,  //
//                                  neg_inf, neg_inf, neg_inf, neg_inf, neg_inf};

//   checkExact(program, {I_input}, {O_output});
// }

// TEST_F(OpTest, ExplicitPaddingInf) {
//   auto I = Placeholder(DType::FLOAT32, {2, 3}, "A");
//   float inf = std::numeric_limits<float>::infinity();
//   auto O = op::explicit_padding(I, {2, 1}, {2, 1}).padval(Constant(inf));
//   auto program = makeProgram("explicit_padding", {I}, {O});

//   std::vector<float> I_input = {-1, -2, -3,  //
//                                 -4, -5, -6};
//   std::vector<float> O_output = {inf, inf, inf, inf, inf,  //
//                                  inf, inf, inf, inf, inf,  //
//                                  inf, -1,  -2,  -3,  inf,  //
//                                  inf, -4,  -5,  -6,  inf,  //
//                                  inf, inf, inf, inf, inf,  //
//                                  inf, inf, inf, inf, inf};

//   checkExact(program, {I_input}, {O_output});
// }

// // TODO: Consider writing a folder for this test case.
// TEST_F(OpTest, ExplicitPaddingNoOp) {
//   auto I = Placeholder(DType::FLOAT32, {2, 3}, "A");
//   auto O = op::explicit_padding(I, {0, 0}, {0, 0}).padval(Constant(0));
//   auto program = makeProgram("explicit_padding", {I}, {O});

//   std::vector<float> I_input = {1, 2, 3,  //
//                                 4, 5, 6};
//   std::vector<float> O_output = {1, 2, 3,  //
//                                  4, 5, 6};

//   checkExact(program, {I_input}, {O_output});
// }

// TEST_F(OpTest, Flip) {
//   auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
//   auto program = makeProgram("flip", {I}, {op::flip(I, 2)});
//   runProgram(program);
// }

// TEST_F(OpTest, HardSigmoid) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("hard_sigmoid", {A}, {op::hard_sigmoid(A, 0.05)});
//   runProgram(program);
// }

// TEST_F(OpTest, ImageResize) {
//   auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
//   auto O = op::image_resize(I, {5, 4}, op::InterpolationMode::BILINEAR, op::TensorLayout::NXC);
//   auto program = makeProgram("image_resize", {I}, {O});
//   runProgram(program);
// }

// TEST_F(OpTest, Max) {
//   auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
//   auto program = makeProgram("max", {I}, {op::max(I)});  // NOLINT(build/include_what_you_use)
//   runProgram(program);
// }

// TEST_F(OpTest, Maximum) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto B = Placeholder(DType::FLOAT32, {10, 20}, "B");
//   auto program = makeProgram("maximum", {A, B}, {op::maximum(A, B)});
//   runProgram(program);
// }

// TEST_F(OpTest, Mean) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("mean", {A}, {op::mean(A)});
//   runProgram(program);
// }

// TEST_F(OpTest, Min) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("min", {A}, {op::min(A)});  // NOLINT(build/include_what_you_use)
//   runProgram(program);
// }

// TEST_F(OpTest, Minimum) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto B = Placeholder(DType::FLOAT32, {10, 20}, "B");
//   auto program = makeProgram("minimum", {A, B}, {op::minimum(A, B)});
//   runProgram(program);
// }

// TEST_F(OpTest, L2Norm) {
//   auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
//   auto program = makeProgram("l2norm", {I}, {op::l2norm(I, {1}).epsilon(0.01).eps_mode(op::EpsMode::ADD)});
//   runProgram(program);
// }

// TEST_F(OpTest, Pool) {
//   auto I = Placeholder(DType::FLOAT32, {10, 20, 30, 40, 50}, "I");
//   auto program = makeProgram("pool", {I},
//                              {op::pool(I, op::PoolMode::SUM, {1, 2, 3}, {1, 2, 3}, op::AutoPadMode::EXPLICIT, {1, 2},
//                                        op::TensorLayout::NXC, true, true)});
//   runProgram(program);
// }

// TEST_F(OpTest, Prod) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("prod", {A}, {op::prod(A)});
//   runProgram(program);
// }

// TEST_F(OpTest, Relu) {
//   auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto M = Placeholder(DType::FLOAT32, {10, 20}, "M");
//   auto program = makeProgram("relu", {I, A, M}, {op::relu(I).alpha(A).max_value(M).threshold(0.05)});
//   runProgram(program);
// }

// TEST_F(OpTest, ReluNoAlpha) {
//   auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
//   auto M = Placeholder(DType::FLOAT32, {10, 20}, "M");
//   auto program = makeProgram("relu", {I, M}, {op::relu(I).max_value(M).threshold(0.05)});
//   runProgram(program);
// }

// TEST_F(OpTest, ReluNoMaxValue) {
//   auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("relu", {I, A}, {op::relu(I).alpha(A).threshold(0.05)});
//   runProgram(program);
// }

// TEST_F(OpTest, ReluOnlyThreshold) {
//   auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
//   auto program = makeProgram("relu", {I}, {op::relu(I).threshold(0.05)});
//   runProgram(program);
// }

// TEST_F(OpTest, ReluNoParams) {
//   auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
//   auto program = makeProgram("relu", {I}, {op::relu(I)});
//   runProgram(program);
// }

// // See: https://leimao.github.io/blog/Reorg-Layer-Explained/
// static std::vector<int64_t> reorgYoloRefImpl(const std::vector<int64_t>& I, unsigned N, unsigned C, unsigned H,
//                                              unsigned W, unsigned stride, bool decrease) {
//   std::vector<int64_t> O(I.size());
//   auto C_out = C / (stride * stride);
//   for (unsigned b = 0; b < N; b++) {
//     for (unsigned k = 0; k < C; k++) {
//       for (unsigned j = 0; j < H; j++) {
//         for (unsigned i = 0; i < W; i++) {
//           auto in_index = i + W * (j + H * (k + C * b));
//           auto c2 = k % C_out;
//           auto offset = k / C_out;
//           auto w2 = i * stride + offset % stride;
//           auto h2 = j * stride + offset / stride;
//           auto out_index = w2 + W * stride * (h2 + H * stride * (c2 + C_out * b));
//           if (decrease) {
//             O[out_index] = I[in_index];
//           } else {
//             O[in_index] = I[out_index];
//           }
//         }
//       }
//     }
//   }
//   return O;
// }

// TEST_F(OpTest, ReorgYoloDecrease) {
//   const unsigned N = 1, C = 4, H = 6, W = 6, S = 2;
//   const bool decrease = true;

//   auto I = Placeholder(DType::INT64, {N, C, H, W});
//   auto O = op::reorg_yolo(I, S, decrease);
//   auto program = makeProgram("reorg_yolo", {I}, {O});

//   std::vector<int64_t> I_input(N * C * H * W);
//   for (unsigned i = 0; i < I_input.size(); i++) {
//     I_input[i] = i;
//   }
//   auto O_expected = reorgYoloRefImpl(I_input, N, C, H, W, S, decrease);
//   checkExact(program, {I_input}, {O_expected});
// }

// TEST_F(OpTest, ReorgYoloIncrease) {
//   const unsigned N = 1, C = 4, H = 6, W = 6, S = 2;
//   const unsigned C_out = C * (S * S), H_out = H / S, W_out = W / S;
//   const bool decrease = false;

//   auto I = Placeholder(DType::INT64, {N, C, H, W});
//   auto O = op::reorg_yolo(I, S, decrease);
//   auto program = makeProgram("reorg_yolo", {I}, {O});

//   std::vector<int64_t> I_input(N * C * H * W);
//   for (unsigned i = 0; i < I_input.size(); i++) {
//     I_input[i] = i;
//   }
//   auto O_expected = reorgYoloRefImpl(I_input, N, C_out, H_out, W_out, S, decrease);
//   checkExact(program, {I_input}, {O_expected});
// }

// TEST_F(OpTest, ReorgYoloNHWC) {
//   const unsigned N = 1, C = 4, H = 6, W = 6, S = 2;
//   const bool decrease = true;

//   auto I = Placeholder(DType::INT64, {N, H, W, C});
//   auto O = op::reorg_yolo(I, S, decrease, /*layout=*/"NHWC");
//   auto program = makeProgram("reorg_yolo", {I}, {O});
//   runProgram(program);
// }

// TEST_F(OpTest, Repeat) {
//   auto A = Placeholder(DType::FLOAT32, {32, 1, 4, 1}, "A");
//   auto X = op::repeat(A).count(3).axis(2);
//   auto program = makeProgram("repeat", {A}, {X});
//   runProgram(program);
// }

// TEST_F(OpTest, Reshape) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   TensorDim I, J;
//   A.bind_dims(I, J);
//   auto program = makeProgram("reshape", {A}, {op::reshape(A, make_tuple(J, I))});
//   runProgram(program);
// }

// TEST_F(OpTest, Sigmoid) {
//   auto A = Placeholder(DType::FLOAT32, {10}, "A");
//   auto program = makeProgram("sigmoid", {A}, {op::sigmoid(A)});
//   runProgram(program);
// }

// TEST_F(OpTest, Slice) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto X = op::slice(A).add_dims({2, 10});
//   auto program = makeProgram("slice", {A}, {X});
//   runProgram(program);
// }

// TEST_F(OpTest, Slice2) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto X = op::slice(A)  //
//                .add_dim(/*start=*/0, /*stop=*/2)
//                .add_dim(/*start=*/2, /*stop=*/8, /*step=*/2);
//   auto program = makeProgram("slice", {A}, {X});
//   runProgram(program);
// }

// TEST_F(OpTest, Softmax) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("softmax", {A}, {op::softmax(A, 1)});
//   runProgram(program);
// }

// TEST_F(OpTest, Sort) {
//   auto A = Placeholder(DType::FLOAT32, {5, 4}, "A");
//   auto program = makeProgram("sort", {A}, {op::sort(A, 1, SortDirection::ASC)});

//   std::vector<float> input = {
//       81.69, 95.74, 27.74, 43.69,  //
//       55.79, 56.79, 57.52, 5.9,    //
//       39.48, 7.11,  14.81, 66.23,  //
//       20.25, 66.05, 64.5,  71.07,  //
//       67.6,  54.42, 87.59, 80.02,  //
//   };

//   std::vector<float> sorted = {
//       27.74, 43.69, 81.69, 95.74,  //
//       5.9,   55.79, 56.79, 57.52,  //
//       7.11,  14.81, 39.48, 66.23,  //
//       20.25, 64.5,  66.05, 71.07,  //
//       54.42, 67.6,  80.02, 87.59,  //
//   };

//   checkClose(program, {input}, {sorted});
// }

// TEST_F(OpTest, SpatialPadding) {
//   auto A = Placeholder(DType::FLOAT32, {64, 4, 32, 32}, "A");
//   auto X = op::spatial_padding(A, {1, 3}, {3, 3}, op::TensorLayout::NXC);
//   auto program = makeProgram("spatial_padding", {A}, {X});
//   runProgram(program);
// }

// TEST_F(OpTest, Square) {
//   auto A = Placeholder(DType::FLOAT32, {10}, "A");
//   auto program = makeProgram("square", {A}, {op::square(A)});
//   runProgram(program);
// }

// TEST_F(OpTest, Sum) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("sum", {A}, {op::sum(A)});
//   runProgram(program);
// }

// TEST_F(OpTest, Squeeze) {
//   auto A = Placeholder(DType::UINT64, {3, 1, 4, 1}, "A");
//   auto C = op::squeeze(A, {1, 3});
//   auto program = makeProgram("squeeze", {A}, {C});

//   std::vector<uint64_t> A_input = {0, 1, 2,  3,  //
//                                    4, 5, 6,  7,  //
//                                    8, 9, 10, 11};
//   checkExact(program, {A_input}, {A_input});
// }

// TEST_F(OpTest, Tile) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto X = op::tile(A, {5, 4});
//   auto program = makeProgram("tile", {A}, {X});
//   runProgram(program);
// }

// TEST_F(OpTest, Transpose) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("transpose", {A}, {op::transpose(A)});
//   runProgram(program);
// }

// TEST_F(OpTest, Unsqueeze) {
//   auto A = Placeholder(DType::UINT64, {3, 4}, "A");
//   auto C = op::unsqueeze(A, {0});
//   auto program = makeProgram("unsqueeze", {A}, {C});
//   std::vector<uint64_t> A_input = {0, 1, 2,  3,  //
//                                    4, 5, 6,  7,  //
//                                    8, 9, 10, 11};
//   checkExact(program, {A_input}, {A_input});
// }

// TEST_F(OpTest, Variance) {
//   auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
//   auto program = makeProgram("variance", {A}, {op::variance(A)});
//   runProgram(program);
// }
