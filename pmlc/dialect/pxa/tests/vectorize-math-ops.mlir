// RUN: pmlc-opt -pxa-vectorize="strategy=recursive vectorize-math-op=false" %s | FileCheck %s

// CHECK-LABEL: func @sigmoid
func @sigmoid(%arg0: memref<1x1152x1x1xf16>, %arg1: memref<1x1152x1x1xf16>) -> memref<1x1152x1x1xf16> {
  %cst = constant 1.000000e+00 : f16
  %0 = affine.parallel (%arg2) = (0) to (1152) reduce ("assign") -> (memref<1x1152x1x1xf16>) {
    %1 = pxa.load %arg0[0, %arg2, 0, 0] : memref<1x1152x1x1xf16>
    // CHECK: pxa.vector_load %{{.*}} : memref<1x1152x1x1xf16>, vector<8xf16>
    %2 = negf %1 : f16
    // CHECK: negf %{{.*}} : vector<8xf16>
    %3 = math.exp %2 : f16
    // CHECK: memref.alloc() : memref<8xf16>
    // CHECK: constant {{.*}}
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] : vector<8xf16>, memref<8xf16>
    // CHECK: affine.parallel (%{{.*}}) = (0) to (8) reduce ("assign") -> (memref<8xf16>)
    // CHECK:   affine.load %{{.*}}[%{{.*}}] : memref<8xf16>
    // CHECK:   math.exp %{{.*}} : f16
    // CHECK:   pxa.reduce assign %{{.*}}, %{{.*}}[%{{.*}}] : memref<8xf16>
    // CHECK:   affine.yield %{{.*}} : memref<8xf16>
    // CHECK: vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<8xf16>, vector<8xf16>
    %4 = addf %3, %cst : f16
    // CHECK: vector.broadcast %{{.*}} : f16 to vector<8xf16>
    // CHECK: addf %{{.*}}, %{{.*}} : vector<8xf16>
    %5 = divf %cst, %4 : f16
    // CHECK: vector.broadcast %{{.*}} : f16 to vector<8xf16>
    // CHECK: divf %{{.*}}, %{{.*}} : vector<8xf16>
    %6 = pxa.reduce assign %5, %arg1[0, %arg2, 0, 0] : memref<1x1152x1x1xf16>
    // CHECK: vector_reduce assign %{{.*}}, %{{.*}}[0, %{{.*}}, 0, 0] : memref<1x1152x1x1xf16>, vector<8xf16>
    affine.yield %6 : memref<1x1152x1x1xf16>
  }
  return %0 : memref<1x1152x1x1xf16>
}
