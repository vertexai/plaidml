// RUN: pmlc-opt %s -test-conv2d-finder -mlir-disable-threading -split-input-file 2>&1 | FileCheck %s

#conv2d_1x1_0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#conv2d_1x1_1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#conv2d_1x1_2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>

// CHECK-LABEL: Testing : conv2d_1x1
func @conv2d_1x1(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<1x1x3x32xf32>) -> tensor<1x224x224x32xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-NEXT: You say you want a convolution.
  // CHECK-NEXT: paddings = 0 0
  // CHECK-NEXT: strides = 1 1
  // CHECK-NEXT: dilations = 1 1
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #conv2d_1x1_0, srcs = [#conv2d_1x1_1, #conv2d_1x1_2]} : tensor<f32>, tensor<1x224x224x3xf32>, tensor<1x1x3x32xf32> -> tensor<1x224x224x32xf32>
  return %0 : tensor<1x224x224x32xf32>
}

#conv2d_3x3_0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#conv2d_3x3_1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)>
#conv2d_3x3_2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>

// CHECK-LABEL: Testing : conv2d_3x3
func @conv2d_3x3(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<3x3x3x32xf32>) -> tensor<1x224x224x32xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-NEXT: You say you want a convolution.
  // CHECK-NEXT: paddings = -1 -1
  // CHECK-NEXT: strides = 1 1
  // CHECK-NEXT: dilations = 1 1
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #conv2d_3x3_0, srcs = [#conv2d_3x3_1, #conv2d_3x3_2]} : tensor<f32>, tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32> -> tensor<1x224x224x32xf32>
  return %0 : tensor<1x224x224x32xf32>
}

#conv2d_complex_0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#conv2d_complex_1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4 * 3 - 2, d2 * 2 + d5 * 3 - 2, d6)>
#conv2d_complex_2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>

// CHECK-LABEL: Testing : conv2d_complex
func @conv2d_complex(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-NEXT: You say you want a convolution.
  // CHECK-NEXT: paddings = -2 -2
  // CHECK-NEXT: strides = 2 2
  // CHECK-NEXT: dilations = 3 3
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #conv2d_complex_0, srcs = [#conv2d_complex_1, #conv2d_complex_2]} : tensor<f32>, tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32> -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

#conv2d_nchw_0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#conv2d_nchw_1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#conv2d_nchw_2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d5, d6)>

// CHECK-LABEL: Testing : conv2d_nchw
func @conv2d_nchw(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<3x32x1x1xf32>) -> tensor<1x32x224x224xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-NEXT: You say you want a convolution.
  // CHECK-NEXT: paddings = 0 0
  // CHECK-NEXT: strides = 1 1
  // CHECK-NEXT: dilations = 1 1
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #conv2d_nchw_0, srcs = [#conv2d_nchw_1, #conv2d_nchw_2]} : tensor<f32>, tensor<1x3x224x224xf32>, tensor<3x32x1x1xf32> -> tensor<1x32x224x224xf32>
  return %0 : tensor<1x32x224x224xf32>
}

// CHECK-LABEL: Testing : conv2d_negative
func @conv2d_negative(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<1x1x3x32xf32>) -> tensor<1x224x224x32xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-NEXT: Well, you know, we all want to change the world.
  // CHECK-NEXT: Invalid AggregationKind
  %0 = tile.contract mul, mul, %cst, %arg0, %arg1 {sink = #conv2d_1x1_0, srcs = [#conv2d_1x1_1, #conv2d_1x1_2]} : tensor<f32>, tensor<1x224x224x3xf32>, tensor<1x1x3x32xf32> -> tensor<1x224x224x32xf32>

  // CHECK-NEXT: Well, you know, we all want to change the world.
  // CHECK-NEXT: Invalid CombinationKind
  %1 = tile.contract add, add, %cst, %arg0, %arg1 {sink = #conv2d_1x1_0, srcs = [#conv2d_1x1_1, #conv2d_1x1_2]} : tensor<f32>, tensor<1x224x224x3xf32>, tensor<1x1x3x32xf32> -> tensor<1x224x224x32xf32>

  // CHECK-NEXT: Well, you know, we all want to change the world.
  // CHECK-NEXT: Unable to find batch dimension
  %2 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #conv2d_1x1_0, srcs = [#conv2d_1x1_2, #conv2d_1x1_1]} : tensor<f32>, tensor<1x1x3x32xf32>, tensor<1x224x224x3xf32> -> tensor<1x224x224x32xf32>

  return %1 : tensor<1x224x224x32xf32>
}

#dot2d_0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#dot2d_1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#dot2d_2 = affine_map<(d0, d1, d2) -> (d2, d1)>

// CHECK-LABEL: Testing : dot2d_negative
func @dot2d_negative(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-NEXT: Well, you know, we all want to change the world.
  // CHECK-NEXT: Invalid tensor rank
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #dot2d_0, srcs = [#dot2d_1, #dot2d_2]} : tensor<f32>, tensor<8x16xf32>, tensor<16x32xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

#dot4d_0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#dot4d_1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#dot4d_2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d2, d3)>

// CHECK-LABEL: Testing : dot4d_negative
func @dot4d_negative(%arg0: tensor<4x8x16x32xf32>, %arg1: tensor<32x16x8x4xf32>) -> tensor<4x8x8x4xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK-NEXT: Well, you know, we all want to change the world.
  // CHECK-NEXT: Invalid spatial dimensions
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #dot4d_0, srcs = [#dot4d_1, #dot4d_2]} : tensor<f32>, tensor<4x8x16x32xf32>, tensor<32x16x8x4xf32> -> tensor<4x8x8x4xf32>
  return %0 : tensor<4x8x8x4xf32>
}
