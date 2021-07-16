// RUN: pmlc-opt -convert-stdx-to-llvm -canonicalize %s | FileCheck %s

module {
  func @acos_f16(%arg0: memref<3x10xf16>) -> memref<3x10xf16> {
    %0 = stdx.acos(%arg0) : (memref<3x10xf16>) -> memref<3x10xf16>
    return %0 : memref<3x10xf16>
  }
}

// CHECK-LABEL: func @acos_f16
// CHECK: %[[result:.*]] = llvm.call @acosf(%{{.*}})
// CHECK: return %[[result]] : !llvm.struct<(ptr<f16>, ptr<f16>, i64, array<2 x i64>, array<2 x i64>)>

// -----

module {
  func @pow_i32(%arg0: memref<1xi32>, %arg1: memref<5xi32>) -> memref<5xi32> {
    %0 = stdx.pow(%arg0, %arg1) : (memref<1xi32>, memref<5xi32>) -> memref<5xi32>
    return %0 : memref<5xi32>
  }
}

// CHECK-LABEL: func @pow_i32
// CHECK: %[[result:.*]] = llvm.call @powf(%{{.*}}, %{{.*}})
// CHECK: return %[[result]] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
