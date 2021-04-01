// RUN: pmlc-opt -pxa-dataflow-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple
func @simple(%out : memref<2xf32>) -> memref<2xf32> {
  // CHECK: constant 0.0
  %zero = constant 0.0 : f32
  %buf = alloc() : memref<2xf32>
  // CHECK-NEXT: affine.parallel
  %0 = affine.parallel (%i) = (0) to (2) reduce ("assign") -> (memref<2xf32>) {
    %1 = pxa.reduce assign %zero, %buf[%i] : memref<2xf32>
    %2 = pxa.load %1[%i] : memref<2xf32>
    // CHECK-NEXT: pxa.reduce addf
    %3 = pxa.reduce addf %2, %out[%i] : memref<2xf32>
    affine.yield %3 : memref<2xf32>
  }
  return %0 : memref<2xf32>
}

// CHECK-LABEL: func @grn
func @grn(%arg0: memref<1x4x4x3xf16>) -> memref<1x4x4x3xf16> {
  // CHECK: constant 1.001360e-05
  %cst = constant 1.001360e-05 : f16
  // CHECK-NEXT: constant 0.0
  %cst_0 = constant 0.000000e+00 : f16
  // CHECK-NEXT: constant 1.0
  %cst_1 = constant 1.000000e+00 : f16
  // CHECK-NEXT: alloc() : memref<1x4x4x1xf16>
  // CHECK-NEXT: alloc() : memref<1x4x4x3xf16>
  %0 = alloc() : memref<1x4x4x3xf16>
  %1 = alloc() : memref<1x4x4x1xf16>
  %2 = alloc() : memref<1x4x4x1xf16>
  %3 = alloc() : memref<1x4x4x1xf16>
  %4 = alloc() : memref<1x4x4x1xi1>
  %5 = alloc() : memref<1x4x4x1xf16>
  %6 = alloc() : memref<1x4x4x3xf16>
  // CHECK-NEXT: affine.parallel
  %7 = affine.parallel (%arg1, %arg2) = (0, 0) to (4, 4) reduce ("assign") -> (memref<1x4x4x3xf16>) {
    // CHECK-NEXT: pxa.reduce assign
    %8 = pxa.reduce assign %cst_0, %1[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
    // CHECK-NEXT: affine.parallel
    %9 = affine.parallel (%arg3) = (0) to (3) reduce ("assign") -> (memref<1x4x4x1xf16>) {
      // CHECK-NEXT: pxa.load
      %24 = pxa.load %arg0[0, %arg1, %arg2, %arg3] : memref<1x4x4x3xf16>
      // CHECK-NEXT: pxa.load
      %25 = pxa.load %arg0[0, %arg1, %arg2, %arg3] : memref<1x4x4x3xf16>
      // CHECK-NEXT: mulf
      %26 = mulf %24, %25 : f16
      %27 = pxa.reduce assign %26, %0[0, %arg1, %arg2, %arg3] : memref<1x4x4x3xf16>
      %28 = pxa.load %27[0, %arg1, %arg2, %arg3] : memref<1x4x4x3xf16>
      // CHECK-NEXT: pxa.reduce addf
      %29 = pxa.reduce addf %28, %8[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
      affine.yield %29 : memref<1x4x4x1xf16>
    }
    // CHECK: pxa.load
    %10 = pxa.load %9[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
    // CHECK-NEXT: addf
    %11 = addf %10, %cst_1 : f16
    %12 = pxa.reduce assign %11, %2[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
    %13 = pxa.load %12[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
    // CHECK-NEXT: math.sqrt
    %14 = math.sqrt %13 : f16
    %15 = pxa.reduce assign %14, %3[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
    %16 = pxa.load %15[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
    // CHECK-NEXT: cmpf olt
    %17 = cmpf olt, %16, %cst : f16
    %18 = pxa.reduce assign %17, %4[0, %arg1, %arg2, 0] : memref<1x4x4x1xi1>
    %19 = pxa.load %18[0, %arg1, %arg2, 0] : memref<1x4x4x1xi1>
    %20 = pxa.load %15[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
    // CHECK-NEXT: select
    %21 = select %19, %cst, %20 : f16
    %22 = pxa.reduce assign %21, %5[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
    // CHECK-NEXT: affine.parallel
    %23 = affine.parallel (%arg3) = (0) to (3) reduce ("assign") -> (memref<1x4x4x3xf16>) {
      // CHECK-NEXT: pxa.load
      %24 = pxa.load %arg0[0, %arg1, %arg2, %arg3] : memref<1x4x4x3xf16>
      %25 = pxa.load %22[0, %arg1, %arg2, 0] : memref<1x4x4x1xf16>
      // CHECK-NEXT: divf
      %26 = divf %24, %25 : f16
      // CHECK-NEXT: pxa.reduce assign
      %27 = pxa.reduce assign %26, %6[0, %arg1, %arg2, %arg3] : memref<1x4x4x3xf16>
      affine.yield %27 : memref<1x4x4x3xf16>
    }
    affine.yield %23 : memref<1x4x4x3xf16>
  }
  return %7 : memref<1x4x4x3xf16>
}
