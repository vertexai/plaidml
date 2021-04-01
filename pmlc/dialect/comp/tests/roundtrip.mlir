// RUN: pmlc-opt --split-input-file %s | pmlc-opt | FileCheck %s

func @runtime_number_vk(%arg0 : !comp.device) {
  // CHECK-LABEL: func @runtime_number_vk
  //       CHECK:   comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<vk:0, [0]>
  %env = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<0:0, [0]>
  return
}

func @runtime_string_vk(%arg0: !comp.device) {
  // CHECK-LABEL: func @runtime_string_vk
  //       CHECK:   comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<vk:0, [1]>
  %env = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<vk:0, [1]>
  return
}

func @runtime_number_ocl(%arg0: !comp.device) {
  // CHECK-LABEL: func @runtime_number_ocl
  //       CHECK:   comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %env = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<1:0, [11]>
  return
}

func @runtime_string_ocl(%arg0: !comp.device) {
  // CHECK-LABEL: func @runtime_string_ocl
  //       CHECK:   comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:1, [11]>
  %env = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<ocl:1, [11]>
  return
}

func @runtime_number_custom(%arg0: !comp.device) {
  // CHECK-LABEL: @runtime_number_custom
  //       CHECK:   comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<1000:0, [2]>
  %env = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<1000:0, [2]>
  return
}

func @runtime_memory_spaces(%arg0: !comp.device) {
  // CHECK-LABEL: @runtime_memory_spaces
  //       CHECK:   comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<1000:0, [2, 11, 3]>
  %env = comp.create_execenv %arg0 : (!comp.device) -> !comp.execenv<1000:0, [2, 11, 3]>
  return
}

// -----
// Test parsing and printing of dependencies

// CHECK-LABEL: func @read_no_dependencies
//  CHECK-SAME:     %[[ENV:.*]]: !comp.execenv<ocl:0, [11]>
//  CHECK-SAME:     %[[HOST:.*]]: memref<2x3xf32>
//  CHECK-SAME:     %[[DEVICE:.*]]: memref<2x3xf32, 11>
//       CHECK:   comp.schedule_read %[[HOST]] from %[[DEVICE]] on %[[ENV]]
func @read_no_dependencies(%env : !comp.execenv<ocl:0, [11]>,
                           %host : memref<2x3xf32>,
                           %device : memref<2x3xf32, 11>) {
  %ev = comp.schedule_read %host from %device on %env
      : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>)
      -> (!comp.event<ocl>)
  return
}

// CHECK-LABEL: func @read_one_dependency
//  CHECK-SAME:     %[[ENV:.*]]: !comp.execenv<ocl:0, [11]>
//  CHECK-SAME:     %[[HOST:.*]]: memref<2x3xf32>
//  CHECK-SAME:     %[[DEVICE:.*]]: memref<2x3xf32, 11>
//  CHECK-SAME:     %[[DEP:.*]]: !comp.event<ocl>
//       CHECK:   comp.schedule_read %[[HOST]] from %[[DEVICE]] on %[[ENV]] wait for %[[DEP]]
func @read_one_dependency(%env : !comp.execenv<ocl:0, [11]>,
                          %host : memref<2x3xf32>,
                          %device : memref<2x3xf32, 11>,
                          %dep : !comp.event<ocl>) {
  %ev = comp.schedule_read %host from %device on %env wait for %dep
      : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>)
      -> (!comp.event<ocl>)
  return
}

// CHECK-LABEL: func @read_many_dependencies
//  CHECK-SAME:     %[[ENV:.*]]: !comp.execenv<ocl:0, [11]>
//  CHECK-SAME:     %[[HOST:.*]]: memref<2x3xf32>
//  CHECK-SAME:     %[[DEVICE:.*]]: memref<2x3xf32, 11>
//  CHECK-SAME:     %[[DEP0:[0-9a-zA-Z]*]]: !comp.event<ocl>
//  CHECK-SAME:     %[[DEP1:[0-9a-zA-Z]*]]: !comp.event<ocl>
//  CHECK-SAME:     %[[DEP2:[0-9a-zA-Z]*]]: !comp.event<ocl>
//       CHECK:   comp.schedule_read %[[HOST]] from %[[DEVICE]] on %[[ENV]] wait for %[[DEP0]], %[[DEP1]], %[[DEP2]]
func @read_many_dependencies(%env : !comp.execenv<ocl:0, [11]>,
                             %host : memref<2x3xf32>,
                             %device : memref<2x3xf32, 11>,
                             %dep0 : !comp.event<ocl>,
                             %dep1 : !comp.event<ocl>,
                             %dep2 : !comp.event<ocl>) {
  %ev = comp.schedule_read %host from %device on %env wait for %dep0, %dep1, %dep2
      : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>, !comp.event<ocl>, !comp.event<ocl>)
      -> (!comp.event<ocl>)
  return
}
