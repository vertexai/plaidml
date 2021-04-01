// RUN: pmlc-opt -x86-convert-pxa-to-affine -loop-invariant-code-motion \
// RUN:          -canonicalize -split-input-file %s | FileCheck %s

#id_map = affine_map<(i, j) -> (i, j)>

// CHECK: func @dot
func @dot(%A: memref<4x8xf32>, %B: memref<8x6xf32>, %C: memref<4x6xf32>) -> memref<4x6xf32> {
  // CHECK: xsmm.gemm.dispatch.f32 [2, 2, 2], [8, 6, 6]
  // CHECK: affine.for
  // CHECK: affine.for
  // CHECK: affine.for
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) reduce ("assign") -> (memref<4x6xf32>) {
    %0 = pxa.gemm %C[%i, %j]:#id_map = %A[%i, %k]:#id_map, %B[%k, %j]:#id_map, [2, 2, 2], [1]
      : (memref<4x8xf32>, memref<8x6xf32>) -> memref<4x6xf32>
    // CHECK: xsmm.gemm.invoke.f32
    affine.yield %0 : memref<4x6xf32>
  }
  return %ret : memref<4x6xf32>
}

// -----

#O_tile = affine_map<(m, n) -> (0, m, 0, n)>
#I_tile = affine_map<(m, k) -> (0, m, 0, k)>
#K_tile = affine_map<(k, n) -> (0, 0, k, n)>

// CHECK: func @res2a_branch2a
func @res2a_branch2a(%I: memref<1x56x56x64xf32>, %K: memref<1x1x64x64xf32>, %O: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  // CHECK: xsmm.gemm.dispatch.f32 [14, 64, 64], [3584, 64, 3584]
  // CHECK: affine.for
  // CHECK: affine.for
  %ret = affine.parallel (%x, %y) = (0, 0) to (56, 56) step (14, 1) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %0 = pxa.gemm %O[0, %x, %y, 0]:#O_tile = %I[0, %x, %y, 0]:#I_tile, %K[0, %x, %y, 0]:#K_tile, [14, 64, 64], [1]
      : (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> memref<1x56x56x64xf32>
    // CHECK: xsmm.gemm.invoke.f32
    affine.yield %0 : memref<1x56x56x64xf32>
  }
  return %ret : memref<1x56x56x64xf32>
}
