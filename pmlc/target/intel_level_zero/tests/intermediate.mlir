// RUN: pmlc-opt --target-intel_level_zero %s | FileCheck %s

!tensor = type tensor<2x4xf32>

func @eltwise_add(%A: !tensor, %B: !tensor, %C: !tensor) -> !tensor {
  %0 = tile.add %A, %B : (!tensor, !tensor) -> !tensor
  %1 = tile.add %0, %C : (!tensor, !tensor) -> !tensor
  return %1 : !tensor
}

// CHECK: llvm.call @level_zero_schedule_compute
