// RUN: pmlc-opt -tile-reorder-inputs %s | FileCheck %s

// CHECK-LABEL: @main
// CHECK:   %[[reorder0:.*]] = tile.contract assign, none, %{{.*}}, %{{.*}} {lowerBounds = #map0, sink = #map1, srcs = [#map1], tags = {layout = "kcx"}
// CHECK:   layer.box "ng.Convolution" (%{{.*}}, %{{.*}}) = (%{{.*}}, %[[reorder0]])
// CHECK:   tile.contract add, mul, %{{.*}}, %{{.*}}, %{{.*}} {sink = #map4, srcs = [#map5, #map6], tags = {layout = "ncx"}}
func @main(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32> {tile.const = 0 : index}, %arg2: tensor<1x64x1x1xf32> {tile.const = 1 : index}) -> tensor<1x64x112x112xf32> {
  %0 = layer.box "ng.Convolution" (%arg111, %arg112) = (%arg0, %arg1) : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> {
    %cst_109 = tile.constant(0.000000e+00 : f64) : tensor<f32>
    %conv = tile.contract add, mul, %cst_109, %arg111, %arg112 {sink = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>, srcs = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 * 2 + d5 - 3, d3 * 2 + d6 - 3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>] } : tensor<f32>, tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32> -> tensor<1x64x112x112xf32>
    layer.return %conv : tensor<1x64x112x112xf32>
  } {attrs = {dilations = [1, 1], pads_begin = [3, 3], pads_end = [3, 3], strides = [2, 2]}}
  %1 = layer.box "ng.Add" (%arg23, %arg24) = (%0, %arg2) : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32> {
    %35 = tile.add %arg23, %arg24 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    layer.return %35 : tensor<1x64x112x112xf32>
  }
  return %1 : tensor<1x64x112x112xf32>
}
