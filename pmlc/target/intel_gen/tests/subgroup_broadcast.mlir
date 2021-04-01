// RUN: pmlc-opt -stdx-subgroup-broadcast -cse %s | FileCheck %s

// CHECK-LABEL: @subgroup_read_extract_bcast_write
func @subgroup_read_extract_bcast_write(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c64 = constant 64 : index
  %c1_i32 = constant 1 : i32
  // CHECK: scf.parallel (%{{.*}}) = (%{{.*}})
  // CHECK: scf.parallel (%[[SID:.*]], %[[I:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) {
  // CHECK-NEXT:   %[[IDX:.*]] = addi %[[I]], %[[SID]] : index
  // CHECK-NEXT:   %[[LOAD:.*]] = load %{{.*}}[%[[IDX]]] : memref<64xf32>
  // CHECK-NEXT:   index_cast
  // CHECK-NEXT:   %[[BROADCAST:.*]] = stdx.subgroup_broadcast(%[[LOAD]], %{{.*}}) : f32
  // CHECK-NEXT:   store %[[BROADCAST]], %{{.*}}[%[[IDX]]] : memref<64xf32>
  scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
    scf.parallel (%i) = (%c0) to (%c64) step (%c8) {
      %0 = vector.load %arg0[%i] : memref<64xf32>, vector<8xf32>
      %1 = vector.extractelement %0[%c1_i32:i32] : vector<8xf32>
      %2 = vector.broadcast %1 : f32 to vector<8xf32>
      vector.store %2, %arg1[%i] : memref<64xf32>, vector<8xf32>
      scf.yield
    } {tags = {gpuThread, subgroupSize=8}}
  } {tags = {gpuBlock, subgroupSize=8}}
  return
}

// CHECK-LABEL: @subgroup_read_write
func @subgroup_read_write(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c64 = constant 64 : index
  // CHECK: scf.parallel (%{{.*}}) = (%{{.*}})
  // CHECK: scf.parallel (%[[SID:.*]], %[[I:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) {
  // CHECK-NEXT:   %[[IDX:.*]] = addi %[[I]], %[[SID]] : index
  // CHECK-NEXT:   %[[LOAD:.*]] = load %{{.*}}[%[[IDX]]] : memref<64xf32>
  // CHECK-NEXT:   store %[[LOAD]], %{{.*}}[%[[IDX]]] : memref<64xf32>
  scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
    scf.parallel (%i) = (%c0) to (%c64) step (%c8) {
      %0 = vector.load %arg0[%i] : memref<64xf32>, vector<8xf32>
      vector.store %0, %arg1[%i] : memref<64xf32>, vector<8xf32>
      scf.yield
    } {tags = {gpuThread, subgroupSize=8}}
  } {tags = {gpuBlock, subgroupSize=8}}
  return
}

// CHECK-LABEL: @subgroup_bcast_write
func @subgroup_bcast_write(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c64 = constant 64 : index
  %cst = constant 0.000000e+00 : f32
  // CHECK: scf.parallel (%{{.*}}) = (%{{.*}})
  // CHECK: scf.parallel (%[[SID:.*]], %[[I:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) {
  // CHECK-NEXT:   %[[IDX:.*]] = addi %[[I]], %[[SID]] : index
  // CHECK-NEXT:   store %{{.*}}, %{{.*}}[%[[IDX]]] : memref<64xf32>
  scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
    scf.parallel (%i) = (%c0) to (%c64) step (%c8) {
      %2 = vector.broadcast %cst : f32 to vector<8xf32>
      vector.store %2, %arg1[%i] : memref<64xf32>, vector<8xf32>
      scf.yield
    } {tags = {gpuThread, subgroupSize=8}}
  } {tags = {gpuBlock, subgroupSize=8}}
  return
}

// CHECK-LABEL: @subgroup_test_read_extract_broadcast_mulf_write
func @subgroup_test_read_extract_broadcast_mulf_write(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c64 = constant 64 : index
  %c1_i32 = constant 1 : i32
  // CHECK: scf.parallel (%{{.*}}) = (%{{.*}})
  // CHECK: scf.parallel (%[[SID:.*]], %[[I:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) {
  // CHECK-NEXT:   %[[IDX:.*]] = addi %[[I]], %[[SID]] : index
  // CHECK-NEXT:   %[[LOAD:.*]] = load %{{.*}}[%[[IDX]]] : memref<64xf32>
  // CHECK-NEXT:   index_cast
  // CHECK-NEXT:   %[[BROADCAST:.*]] = stdx.subgroup_broadcast(%[[LOAD]], %{{.*}}) : f32
  // CHECK-NEXT:   %[[MUL:.*]] = mulf %[[BROADCAST]], %[[LOAD]]
  // CHECK-NEXT:   store %[[MUL]], %{{.*}}[%[[IDX]]] : memref<64xf32>
  scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
    scf.parallel (%i) = (%c0) to (%c64) step (%c8) {
      %0 = vector.load %arg0[%i] : memref<64xf32>, vector<8xf32>
      %1 = vector.extractelement %0[%c1_i32:i32] : vector<8xf32>
      %2 = vector.broadcast %1 : f32 to vector<8xf32>
      %mul = mulf %2, %0 : vector<8xf32>
      vector.store %mul, %arg1[%i] : memref<64xf32>, vector<8xf32>
      scf.yield
    } {tags = {gpuThread, subgroupSize=8}}
  } {tags = {gpuBlock, subgroupSize=8}}
  return
}

// CHECK-LABEL: @subgroup_test_multiple_dim_inner_loop
func @subgroup_test_multiple_dim_inner_loop(%arg0: memref<1x1x8x16xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c8 = constant 8 : index
  %c64 = constant 64 : index
  // CHECK: scf.parallel
  scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c64, %c64, %c8) step (%c1, %c1, %c1) {
    // CHECK-NEXT: alloc()
    %3 = alloc() : memref<1x1x8x16xf32>
    // CHECK: scf.parallel (%[[SID:.*]], %{{.*}}, %[[I:.*]]) = (%{{.*}}, %{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}, %{{.*}}) {
    // CHECK-NEXT:   %[[IDX:.*]] = addi %[[I]], %[[SID]] : index
    // CHECK-NEXT:   %[[LOAD:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %[[IDX]]] : memref<1x1x8x16xf32>
    // CHECK-NEXT:   index_cast
    // CHECK-NEXT:   index_cast
    // CHECK-NEXT:   %[[BROADCAST:.*]] = stdx.subgroup_broadcast(%[[LOAD]], %{{.*}}) : f32
    // CHECK-NEXT:   %[[MUL:.*]] = mulf %[[BROADCAST]], %[[LOAD]]
    // CHECK-NEXT:   store %[[MUL]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %[[IDX]]]  : memref<1x1x8x16xf32>
    scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c8, %c2) step (%c1, %c8) {
      %4 = vector.load %3[%c0, %c0, %arg5, %arg6] : memref<1x1x8x16xf32>, vector<8xf32>
      %5 = index_cast %arg4:index to i32
      %6 = vector.extractelement %4[%5:i32] : vector<8xf32>
      %7 = vector.broadcast %6 : f32 to vector<8xf32>
      %mul = mulf %7, %4 : vector<8xf32>
      vector.store %mul, %3[%c0, %c0, %arg5, %arg6] : memref<1x1x8x16xf32>, vector<8xf32>
      scf.yield
    } {tags = {gpuThread, subgroupSize=8}}
  } {tags = {gpuBlock, subgroupSize=8}}
  return
}

// CHECK-LABEL: @devectorize_constant
func @devectorize_constant() {
  %cst = constant dense<0.0> : vector<8xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %c8 = constant 8 : index
  // CHECK: scf.parallel (%{{.*}}) = (%{{.*}})
  scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
    // CHECK: scf.parallel
    scf.parallel (%arg4) = (%c0) to (%c4) step (%c1) {
      // CHECK-NEXT: alloc() : memref<8x1xf32>
      // CHECK-NEXT: scf.for
      // CHECK-NEXT: constant 0.000000e+00 : f32
      // CHECK-NEXT: store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x1xf32>
      %0 = alloc() : memref<8x1xvector<8xf32>>
      scf.for %arg5 = %c0 to %c8 step %c1 {
        store %cst, %0[%arg5, %c0] : memref<8x1xvector<8xf32>>
      }
    } {tags = {gpuThread, subgroupSize = 8}}
  } {tags = {gpuBlock, subgroupSize=8}}
  return
}
