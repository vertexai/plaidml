// RUN: pmlc-opt -convert-tile-to-pxa %s | FileCheck %s

func @loop(%arg0: !stdx.argpack, %arg1: tensor<4xsi32>) -> tensor<4xsi32> {
  %c1 = tile.constant(1 : i64) : tensor<si32>
  %0 = stdx.unpack %arg0 : tensor<1xsi32>
  %1 = tile.loop %0 iter_args(%arg2 = %arg1) -> (tensor<4xsi32>) {
    %2 = tile.add %arg2, %c1 : (tensor<4xsi32>, tensor<si32>) -> tensor<4xsi32>
    tile.yield %2 : tensor<4xsi32>
  }
  return %1 : tensor<4xsi32>
}

// CHECK-LABEL: func @loop
// CHECK: scf.for
// CHECK: alloc
// CHECK: affine.parallel
// CHECK: affine.yield
// CHECK: scf.yield
// CHECK: return
