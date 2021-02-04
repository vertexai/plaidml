// RUN: pmlc-opt -tile-scf-replaceArgument %s | FileCheck %s

func @main(%arg1: memref<4xf32>, %arg2: memref<4xf32>) -> memref<4xf32> {
  %cst = constant 1.000000e+00 : f32
  %zero = constant 0 : index
  %one = constant 1 : index
  %four = constant 4 : index
  %1 = scf.for %arg3 = %zero to %four step %one iter_args(%arg4 = %arg1) -> (memref<4xf32>) {
    %2 = alloc() : memref<4xf32>
    %3 = affine.parallel (%arg5) = (0) to (4) reduce ("assign") -> (memref<4xf32>) {
      %4 = pxa.load %arg4[%arg5] : memref<4xf32>
      %5 = addf %4, %cst : f32
      %6 = pxa.reduce assign %5, %2[%arg5] : memref<4xf32>
      affine.yield %6 : memref<4xf32>
    }
    scf.yield %3 : memref<4xf32>
  }
  return %1 : memref<4xf32>
}

// CHECK-LABEL: @main
// CHECK-SAME: (%[[arg1:.*]]: memref<4xf32>, %[[arg2:.*]]: memref<4xf32>)
// CHECK: %[[cst:.*]] = constant 1.000000e+00 : f32
// CHECK: %[[c0:.*]] = constant 0
// CHECK: %[[c1:.*]] = constant 1
// CHECK: %[[c4:.*]] = constant 4
// CHECK: scf.for {{.*}} = %[[c0]] to %[[c4]] step %[[c1]] iter_args({{.*}} = %[[arg1]])
// CHECK:   alloc() : memref<4xf32>
// CHECK:   %[[res:.*]] = affine.parallel (%[[I:.*]]) = (0) to (4)
// CHECK:     %[[t1:.*]] = pxa.load
// CHECK:     %[[t2:.*]] = addf
// CHECK:     pxa.reduce assign %[[t2]], %[[arg2]][%[[I]]] : memref<4xf32>
// CHECK:     affine.yield
// CHECK:   scf.yield %[[res]]
// CHECK: return
