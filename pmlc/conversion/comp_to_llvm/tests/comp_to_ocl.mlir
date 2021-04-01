// RUN: pmlc-opt -pmlc-convert-comp-to-llvm="comp-llvm-prefix=ocl_" --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func @create_destroy
  //  CHECK-SAME:     %[[DEV:.*]]: !llvm.ptr<i8>
  //       CHECK:   %[[ENV:.*]] = llvm.call @ocl_create_execenv(%[[DEV]])
  //       CHECK:   llvm.call @ocl_destroy_execenv(%[[ENV]])
  func @create_destroy(%dev: !comp.device) {
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
    comp.destroy_execenv %env : !comp.execenv<ocl:0, [11]>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @no_gpu_func
  //  CHECK-SAME:     %[[DEV:.*]]: !llvm.ptr<i8>
  //       CHECK:   %[[ENV:.*]] = llvm.call @ocl_create_execenv(%[[DEV]])
  //       CHECK:   %[[CST0:.*]] = llvm.mlir.constant(0
  //       CHECK:   %[[EV:.*]] = llvm.call @ocl_schedule_barrier(%[[ENV]], %[[CST0]])
  //       CHECK:   llvm.call @ocl_submit(%[[ENV]])
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   llvm.call @ocl_wait(%[[CST1]], %[[EV]])
  //       CHECK:   llvm.call @ocl_destroy_execenv(%[[ENV]])
  func @no_gpu_func(%dev: !comp.device) {
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
    %ev = comp.schedule_barrier %env : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
    comp.submit %env : !comp.execenv<ocl:0, [11]>
    comp.wait %ev : !comp.event<ocl>
    comp.destroy_execenv %env : !comp.execenv<ocl:0, [11]>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @alloc_dealloc
  //  CHECK-SAME:     %[[DEV:.*]]: !llvm.ptr<i8>
  //       CHECK:   %[[ENV:.*]] = llvm.call @ocl_create_execenv(%[[DEV]])
  //       CHECK:   %[[MEM:.*]] = llvm.call @ocl_alloc(%[[ENV]], %{{.*}})
  //       CHECK:   llvm.call @ocl_dealloc(%[[ENV]], %{{.*}})
  //       CHECK:   llvm.call @ocl_destroy_execenv(%[[ENV]])
  func @alloc_dealloc(%dev: !comp.device) {
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
    %mem = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<8x32xf32, 11>
    comp.dealloc %env %mem : (!comp.execenv<ocl:0, [11]>, memref<8x32xf32, 11>) -> ()
    comp.destroy_execenv %env : !comp.execenv<ocl:0, [11]>
    return
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL: func @one_gpu_func
  //  CHECK-SAME:     %[[DEV:[a-zA-Z0-9]*]]: !llvm.ptr<i8>
  //       CHECK:   %[[ENV:.*]] = llvm.call @ocl_create_execenv(%[[DEV]])
  //       CHECK:   %[[MEM:.*]] = llvm.call @ocl_alloc(%[[ENV]], %{{.*}})
  //       CHECK:   llvm.extractvalue
  //       CHECK:   %[[CST0:.*]] = llvm.mlir.constant(0
  //       CHECK:   %[[WEV:.*]] = llvm.call @ocl_schedule_write(%{{.*}}, %{{.*}}, %[[ENV]], %[[CST0]])
  //       CHECK:   %[[KRNL:.*]] = llvm.call @ocl_create_kernel(%[[ENV]]
  //       CHECK:   %[[FEV:.*]] = llvm.call @ocl_schedule_compute(%[[ENV]], %[[KRNL]]
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   %[[REV:.*]] = llvm.call @ocl_schedule_read(%{{.*}}, %{{.*}}, %[[ENV]], %[[CST1]], %[[FEV]])
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   llvm.call @ocl_wait(%[[CST1]], %[[REV]])
  //       CHECK:   llvm.call @ocl_dealloc(%[[ENV]], %{{.*}})
  //       CHECK:   llvm.call @ocl_destroy_execenv(%[[ENV]])
  func @one_gpu_func(%dev: !comp.device, %arg0: memref<8x32xf32>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
    %mem0 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<8x32xf32, 11>
    %wev = comp.schedule_write %arg0 to %mem0 on %env : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
    %ker = comp.create_kernel on %env {kernelFunc = @gpu_module::@zero} : (!comp.execenv<ocl:0, [11]>) -> !comp.kernel
    %fev = comp.schedule_compute %ker grid %c8, %c1, %c1 block %c32, %c1, %c1 args %mem0 on %env wait for %wev : (!comp.execenv<ocl:0, [11]>, !comp.kernel, index, index, index, index, index, index, memref<8x32xf32, 11>, !comp.event<ocl>) -> !comp.event<ocl>
    %rev = comp.schedule_read %arg0 from %mem0 on %env wait for %fev : (memref<8x32xf32>, memref<8x32xf32, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %rev : !comp.event<ocl>
    comp.dealloc %env %mem0 : (!comp.execenv<ocl:0, [11]>, memref<8x32xf32, 11>) -> ()
    comp.destroy_kernel %ker on %env : (!comp.execenv<ocl:0, [11]>, !comp.kernel) -> ()
    comp.destroy_execenv %env : !comp.execenv<ocl:0, [11]>
    return
  }
  // CHECK-NOT: spv.module
  // CHECK-NOT: gpu.module
  spv.module Physical64 OpenCL requires #spv.vce<v1.0, [Addresses, Kernel], []> {
    spv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
    spv.func @zero(%arg0: !spv.ptr<!spv.struct<(!spv.array<256 x f32, stride=4> [0])>, CrossWorkgroup>) "None" attributes {
      spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>},
      workgroup_attributions = 0 : i64
    } {
      spv.Return
    }
  }
  gpu.module @gpu_module {
    gpu.func @zero(%arg0: memref<8x32xf32, 11>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %cst = constant 0.000000e+00 : f32
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<8x32xf32, 11>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL: func @one_gpu_func
  //  CHECK-SAME:     %[[DEV:[a-zA-Z0-9]*]]: !llvm.ptr<i8>
  //       CHECK:   %[[ENV:.*]] = llvm.call @ocl_create_execenv(%[[DEV]])
  //       CHECK:   %[[MEM:.*]] = llvm.call @ocl_alloc(%[[ENV]], %{{.*}})
  //       CHECK:   llvm.extractvalue
  //       CHECK:   %[[CST0:.*]] = llvm.mlir.constant(0
  //       CHECK:   %[[WEV:.*]] = llvm.call @ocl_schedule_write(%{{.*}}, %{{.*}}, %[[ENV]], %[[CST0]])
  //       CHECK:   %[[KRNL:.*]] = llvm.call @ocl_create_kernel(%[[ENV]]
  //       CHECK:   %[[FEV:.*]] = llvm.call @ocl_schedule_compute(%[[ENV]], %[[KRNL]]
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   %[[REV:.*]] = llvm.call @ocl_schedule_read(%{{.*}}, %{{.*}}, %[[ENV]], %[[CST1]], %[[FEV]])
  //       CHECK:   %[[CST1:.*]] = llvm.mlir.constant(1
  //       CHECK:   llvm.call @ocl_wait(%[[CST1]], %[[REV]])
  //       CHECK:   llvm.call @ocl_dealloc(%[[ENV]], %{{.*}})
  //       CHECK:   llvm.call @ocl_destroy_execenv(%[[ENV]])
  func @one_gpu_func(%dev: !comp.device, %arg0: memref<8x32xf16>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
    %mem0 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<8x32xf16, 11>
    %wev = comp.schedule_write %arg0 to %mem0 on %env : (memref<8x32xf16>, memref<8x32xf16, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
    %ker = comp.create_kernel on %env {kernelFunc = @gpu_module::@zero} : (!comp.execenv<ocl:0, [11]>) -> !comp.kernel
    %fev = comp.schedule_compute %ker grid %c8, %c1, %c1 block %c32, %c1, %c1 args %mem0 on %env wait for %wev : (!comp.execenv<ocl:0, [11]>, !comp.kernel, index, index, index, index, index, index, memref<8x32xf16, 11>, !comp.event<ocl>) -> !comp.event<ocl>
    %rev = comp.schedule_read %arg0 from %mem0 on %env wait for %fev : (memref<8x32xf16>, memref<8x32xf16, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>) -> !comp.event<ocl>
    comp.wait %rev : !comp.event<ocl>
    comp.dealloc %env %mem0 : (!comp.execenv<ocl:0, [11]>, memref<8x32xf16, 11>) -> ()
    comp.destroy_kernel %ker on %env : (!comp.execenv<ocl:0, [11]>, !comp.kernel) -> ()
    comp.destroy_execenv %env : !comp.execenv<ocl:0, [11]>
    return
  }
  // CHECK-NOT: spv.module
  // CHECK-NOT: gpu.module
  spv.module Physical64 OpenCL requires #spv.vce<v1.0, [Addresses, Kernel], []> {
    spv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
    spv.func @zero(%arg0: !spv.ptr<!spv.struct<(!spv.array<256 x f16, stride=4> [0])>, CrossWorkgroup>) "None" attributes {
      spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>},
      workgroup_attributions = 0 : i64
    } {
      spv.Return
    }
  }
  gpu.module @gpu_module {
    gpu.func @zero(%arg0: memref<8x32xf16, 11>) kernel attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
      %cst = constant 0.000000e+00 : f16
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      store %cst, %arg0[%0, %1] : memref<8x32xf16, 11>
      gpu.return
    }
  }
}
