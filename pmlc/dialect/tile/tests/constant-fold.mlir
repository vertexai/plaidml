// RUN: pmlc-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @basic
func @basic(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = tile.constant(1.0 : f32) : tensor<f32>
  %0 = tile.add %arg0, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = tile.mul %0, %cst : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
  // CHECK-NEXT: tile.add %{{.*}}, %{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_mul_1_f32
func @fold_mul_1_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = tile.constant(1.0 : f32) : tensor<f32>
  %0 = tile.mul %arg0, %cst : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_mul_1_i32
func @fold_mul_1_i32(%arg0: tensor<si32>) -> tensor<si32> {
  %cst = tile.constant(1 : i32) : tensor<si32>
  %0 = tile.mul %arg0, %cst : (tensor<si32>, tensor<si32>) -> tensor<si32>
  return %0 : tensor<si32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_add_0_f32
func @fold_add_0_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = tile.constant(0.0 : f32) : tensor<f32>
  %0 = tile.add %arg0, %cst : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_add_0_i32
func @fold_add_0_i32(%arg0: tensor<si32>) -> tensor<si32> {
  %cst = tile.constant(0 : i32) : tensor<si32>
  %0 = tile.add %arg0, %cst : (tensor<si32>, tensor<si32>) -> tensor<si32>
  return %0 : tensor<si32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_add_f32_f32
func @fold_add_f32_f32() -> tensor<f32> {
  %cst_0 = tile.constant(1.0 : f32) : tensor<f32>
  %cst_1 = tile.constant(3.0 : f32) : tensor<f32>
  %0 = tile.add %cst_0, %cst_1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(4.000000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_add_f32_i32
func @fold_add_f32_i32() -> tensor<f32> {
  %cst_0 = tile.constant(1.0 : f32) : tensor<f32>
  %cst_1 = tile.constant(3 : i32) : tensor<si32>
  %0 = tile.add %cst_0, %cst_1 : (tensor<f32>, tensor<si32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(4.000000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}} : tensor<f32>
}

// CHECK-LABEL: @fold_add_i32_i32
func @fold_add_i32_i32() -> tensor<si32> {
  %cst_0 = tile.constant(1 : i32) : tensor<si32>
  %cst_1 = tile.constant(3 : i32) : tensor<si32>
  %0 = tile.add %cst_0, %cst_1 : (tensor<si32>, tensor<si32>) -> tensor<si32>
  return %0 : tensor<si32>
  // CHECK-NEXT: tile.constant(4 : i32) : tensor<si32>
  // CHECK-NEXT: return %{{.*}} : tensor<si32>
}

// CHECK-LABEL: @fold_sub_f32_f32
func @fold_sub_f32_f32() -> tensor<f32> {
  %cst_0 = tile.constant(1.0 : f32) : tensor<f32>
  %cst_1 = tile.constant(3.0 : f32) : tensor<f32>
  %0 = tile.sub %cst_0, %cst_1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(-2.000000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_sub_f32_i32
func @fold_sub_f32_i32() -> tensor<f32> {
  %cst_0 = tile.constant(1.0 : f32) : tensor<f32>
  %cst_1 = tile.constant(3 : i32) : tensor<si32>
  %0 = tile.sub %cst_0, %cst_1 : (tensor<f32>, tensor<si32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(-2.000000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}} : tensor<f32>
}

// CHECK-LABEL: @fold_sub_i32_f32
func @fold_sub_i32_f32() -> tensor<f32> {
  %cst_0 = tile.constant(1 : i32) : tensor<si32>
  %cst_1 = tile.constant(3.0 : f32) : tensor<f32>
  %0 = tile.sub %cst_0, %cst_1 : (tensor<si32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(-2.000000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}} : tensor<f32>
}

// CHECK-LABEL: @fold_sub_i32_i32
func @fold_sub_i32_i32() -> tensor<si32> {
  %cst_0 = tile.constant(1 : i32) : tensor<si32>
  %cst_1 = tile.constant(3 : i32) : tensor<si32>
  %0 = tile.sub %cst_0, %cst_1 : (tensor<si32>, tensor<si32>) -> tensor<si32>
  return %0 : tensor<si32>
  // CHECK-NEXT: tile.constant(-2 : i32) : tensor<si32>
  // CHECK-NEXT: return %{{.*}} : tensor<si32>
}

// CHECK-LABEL: @fold_sub_i32_0
func @fold_sub_i32_0(%arg0: tensor<si32>) -> tensor<si32> {
  %cst = tile.constant(0 : i32) : tensor<si32>
  %0 = tile.sub %arg0, %cst : (tensor<si32>, tensor<si32>) -> tensor<si32>
  return %0 : tensor<si32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_sub_f32_0
func @fold_sub_f32_0(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = tile.constant(0.0 : f32) : tensor<f32>
  %0 = tile.sub %arg0, %cst : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_div_f32_f32
func @fold_div_f32_f32() -> tensor<f32> {
  %cst_0 = tile.constant(3.0 : f32) : tensor<f32>
  %cst_1 = tile.constant(2.0 : f32) : tensor<f32>
  %0 = tile.div %cst_0, %cst_1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(1.500000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_div_f32_i32
func @fold_div_f32_i32() -> tensor<f32> {
  %cst_0 = tile.constant(3.0 : f32) : tensor<f32>
  %cst_1 = tile.constant(2 : i32) : tensor<si32>
  %0 = tile.div %cst_0, %cst_1 : (tensor<f32>, tensor<si32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(1.500000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}} : tensor<f32>
}

// CHECK-LABEL: @fold_div_i32_f32
func @fold_div_i32_f32() -> tensor<f32> {
  %cst_0 = tile.constant(3 : i32) : tensor<si32>
  %cst_1 = tile.constant(2.0 : f32) : tensor<f32>
  %0 = tile.div %cst_0, %cst_1 : (tensor<si32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(1.500000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}} : tensor<f32>
}

// CHECK-LABEL: @fold_div_i32_i32
func @fold_div_i32_i32() -> tensor<si32> {
  %cst_0 = tile.constant(3 : i32) : tensor<si32>
  %cst_1 = tile.constant(2 : i32) : tensor<si32>
  %0 = tile.div %cst_0, %cst_1 : (tensor<si32>, tensor<si32>) -> tensor<si32>
  return %0 : tensor<si32>
  // CHECK-NEXT: tile.constant(1 : i32) : tensor<si32>
  // CHECK-NEXT: return %{{.*}} : tensor<si32>
}

// CHECK-LABEL: @fold_div_i32_1
func @fold_div_i32_1(%arg0: tensor<si32>) -> tensor<si32> {
  %cst = tile.constant(1 : i32) : tensor<si32>
  %0 = tile.div %arg0, %cst : (tensor<si32>, tensor<si32>) -> tensor<si32>
  return %0 : tensor<si32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_div_f32_1
func @fold_div_f32_1(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = tile.constant(1.0 : f32) : tensor<f32>
  %0 = tile.div %arg0, %cst : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: return %{{.*}}
}

// CHECK-LABEL: @fold_div_0_i32
func @fold_div_0_i32(%arg0: tensor<si32>) -> tensor<si32> {
  %cst = tile.constant(0 : i32) : tensor<si32>
  %0 = tile.div %cst, %arg0 : (tensor<si32>, tensor<si32>) -> tensor<si32>
  return %0 : tensor<si32>
  // CHECK-NEXT: tile.constant(0 : i32) : tensor<si32>
  // CHECK-NEXT: return %{{.*}} : tensor<si32>
}

// CHECK-LABEL: @fold_div_0_f32
func @fold_div_0_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = tile.constant(0.0 : f32) : tensor<f32>
  %0 = tile.div %cst, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(0.000000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: return %{{.*}} : tensor<f32>
}

// Expected behavior of div by 0 is to not fold
// CHECK-LABEL: @fold_div_f32_0
func @fold_div_f32_0(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = tile.constant(0.0 : f32) : tensor<f32>
  %0 = tile.div %arg0, %cst : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
  // CHECK-NEXT: tile.constant(0.000000e+00 : f32) : tensor<f32>
  // CHECK-NEXT: tile.div %{{.*}}, %{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return %{{.*}} : tensor<f32>
}

// Expected behavior in this case of cast is to fold
// CHECK-LABEL: @fold_cast_si32
func @fold_cast_si32(%arg0: tensor<3xsi32>) -> tensor<3xsi32> {
  %0 = tile.cast %arg0 : (tensor<3xsi32>) -> tensor<3xsi32>
  return %0 : tensor<3xsi32>
  // CHECK-NEXT: return %{{.*}} : tensor<3xsi32>
}

// Expected behavior in this case of cast is to fold
// CHECK-LABEL: @fold_cast_f32
func @fold_cast_f32(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = tile.cast %arg0 : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
  // CHECK-NEXT: return %{{.*}} : tensor<3xf32>
}

// Expected behavior in this case of cast is NOT to fold
// CHECK-LABEL: @cast_si32_to_f32
func @cast_si32_to_f32(%arg0: tensor<3xsi32>) -> tensor<3xf32> {
  %0 = tile.cast %arg0 : (tensor<3xsi32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
  // CHECK-NEXT: %[[cast:.*]] = tile.cast %{{.*}} : (tensor<3xsi32>) -> tensor<3xf32>
  // CHECK-NEXT: return %[[cast]] : tensor<3xf32>
}

// CHECK-LABEL: @mixed_binary
func @mixed_binary() -> tensor<si64> {
    %c1_six = tile.constant(1 : i64) : tensor<!tile.six>
    %c1_uix = tile.constant(1 : i64) : tensor<!tile.uix>
    %0 = tile.mul %c1_six, %c1_uix : (tensor<!tile.six>, tensor<!tile.uix>) -> tensor<si64>
    // CHECK-NEXT: tile.constant(1 : i64) : tensor<si64>
    return %0 : tensor<si64>
}
