// RUN: pmlc-opt %s -convert-pxa-to-affine | FileCheck %s

func @scatter1d(%arg0: memref<8xf32>, %arg1: memref<4xi32>, %arg2: memref<4xf32>, %arg3: memref<8xf32>) -> memref<8xf32> {
  %0 = affine.parallel (%arg4) = (0) to (8) reduce ("assign") -> (memref<8xf32>) {
    %2 = pxa.load %arg0[%arg4] : memref<8xf32>
    %3 = pxa.reduce assign %2, %arg3[%arg4] : memref<8xf32>
    affine.yield %3 : memref<8xf32>
  }
  %1 = affine.parallel (%arg4) = (0) to (4) reduce ("assign") -> (memref<8xf32>) {
    %2 = pxa.load %arg2[%arg4] : memref<4xf32>
    %3 = pxa.load %arg1[%arg4] : memref<4xi32>
    %4 = index_cast %3 : i32 to index
    %5 = pxa.store addf %2, %0[%4] : (f32, memref<8xf32>) -> memref<8xf32>
    affine.yield %5 : memref<8xf32>
  }
  return %1 : memref<8xf32>
}

// CHECK-LABEL: @scatter1d
// CHECK-SAME: (%[[arg0:.*]]: memref<8xf32>, %[[arg1:.*]]: memref<4xi32>, %[[arg2:.*]]: memref<4xf32>, %[[arg3:.*]]: memref<8xf32>)
// CHECK: affine.for %[[arg4:.*]] = 0 to 8
// CHECK:   %[[r0:.*]] = affine.load %[[arg0]][%[[arg4]]] : memref<8xf32>
// CHECK:   %[[r1:.*]] = affine.load %[[arg3]][%[[arg4]]] : memref<8xf32>
// CHECK:   affine.store %[[r0]], %[[arg3]][%[[arg4]]] : memref<8xf32>
// CHECK: affine.for %[[arg4]] = 0 to 4
// CHECK:   %[[r0:.*]] = affine.load %[[arg2]][%[[arg4]]] : memref<4xf32>
// CHECK:   %[[r1:.*]] = affine.load %[[arg1]][%[[arg4]]] : memref<4xi32>
// CHECK:   %[[r2:.*]] = index_cast %[[r1]] : i32 to index
// CHECK:   %{{.*}} = atomic_rmw addf %[[r0]], %[[arg3]][%[[r2]]] : (f32, memref<8xf32>) -> f32
// CHECK: return

