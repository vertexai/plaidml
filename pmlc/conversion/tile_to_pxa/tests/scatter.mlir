// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @scatter1d(%arg0: tensor<8xf32>, %arg1: tensor<4xsi32>, %arg2: tensor<4xf32>) -> tensor<8xf32> {
  %0 = tile.scatter %arg0 %arg1 %arg2 {axis = 0 : index, mode = 0 : i64} : (tensor<8xf32>, tensor<4xsi32>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @scatter1d
// CHECK: affine.parallel (%[[I:.*]]) = (0) to (4)
// CHECK: %[[SRC:.*]] = pxa.load %{{.*}}[%[[I]]] : memref<4xf32>
// CHECK: %[[IDX_RAW:.*]] = pxa.load %{{.*}}[%[[I]]] : memref<4xi32>
// CHECK: %[[IDX:.*]] = index_cast %[[IDX_RAW]] : i32 to index
// CHECK: pxa.store addf %[[SRC]], %{{.*}}[%[[IDX]]] : (f32, memref<8xf32>) -> memref<8xf32>
// CHECK: affine.yield %{{.*}} : memref<8xf32>

// -----

func @scatter1d_si32(%arg0: tensor<8xsi32>, %arg1: tensor<4xsi32>, %arg2: tensor<4xsi32>) -> tensor<8xsi32> {
  %0 = tile.scatter %arg0 %arg1 %arg2 {axis = 0 : index, mode = 0 : i64} : (tensor<8xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<8xsi32>
  return %0 : tensor<8xsi32>
}

// CHECK-LABEL: func @scatter1d_si32
// CHECK: affine.parallel (%[[I:.*]]) = (0) to (4)
// CHECK: %[[SRC:.*]] = pxa.load %{{.*}}[%[[I]]] : memref<4xi32>
// CHECK: %[[IDX_RAW:.*]] = pxa.load %{{.*}}[%[[I]]] : memref<4xi32>
// CHECK: %[[IDX:.*]] = index_cast %[[IDX_RAW]] : i32 to index
// CHECK: pxa.store addi %[[SRC]], %{{.*}}[%[[IDX]]] : (i32, memref<8xi32>) -> memref<8xi32>
// CHECK: affine.yield %{{.*}} : memref<8xi32>

// -----

func @scatter3d(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2xsi32>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
  %0 = tile.scatter %arg0 %arg1 %arg2 {axis = 0 : index, mode = 0 : i64} : (tensor<4x4x4xf32>, tensor<2xsi32>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>
}

// CHECK-LABEL: func @scatter3d
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]], %[[K:.*]]) = (0, 0, 0) to (2, 4, 4)
// CHECK: %[[SRC:.*]] = pxa.load %{{.*}}[%[[I]], %[[J]], %[[K]]] : memref<2x4x4xf32>
// CHECK: %[[IDX_RAW:.*]] = pxa.load %{{.*}}[%[[I]]]  : memref<2xi32>
// CHECK: %[[IDX:.*]] = index_cast %[[IDX_RAW]] : i32 to index
// CHECK: pxa.store addf %[[SRC]], %{{.*}}[%[[IDX]], %[[J]], %[[K]]] : (f32, memref<4x4x4xf32>) -> memref<4x4x4xf32>
// CHECK: affine.yield %{{.*}} : memref<4x4x4xf32>
