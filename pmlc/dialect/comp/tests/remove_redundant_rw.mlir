// RUN: pmlc-opt --comp-remove-redundant-rw --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @write_read
//  CHECK-NEXT:   comp.schedule_write
//  CHECK-NEXT:   return
func @write_read(%env: !comp.execenv<ocl:0, [11]>, %mem: memref<2x3xf32, 11>, %host: memref<2x3xf32>) {
  %ev1 = comp.schedule_write %host to %mem on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  %ev2 = comp.schedule_read %host from %mem on %env wait for %ev1 : (memref<2x3xf32>,  memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>) -> !comp.event<ocl>
  return
}

// CHECK-LABEL: func @write_read_wait
//  CHECK-NEXT:   %[[EV1:.*]] = comp.schedule_write
//  CHECK-NEXT:   comp.wait %[[EV1]]
//  CHECK-NEXT:   return
func @write_read_wait(%env: !comp.execenv<ocl:0, [11]>, %mem: memref<2x3xf32, 11>, %host: memref<2x3xf32>) {
  %ev1 = comp.schedule_write %host to %mem on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  %ev2 = comp.schedule_read %host from %mem on %env wait for %ev1 : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>) -> !comp.event<ocl>
  comp.wait %ev2 : !comp.event<ocl>
  return
}

// CHECK-LABEL: func @alloc_read_wait
//  CHECK-NEXT:   comp.alloc
//  CHECK-NEXT:   comp.schedule_write
//  CHECK-NEXT:   comp.wait
//  CHECK-NEXT:   return
func @alloc_read_wait(%env: !comp.execenv<ocl:0, [11]>, %host: memref<2x3xf32>) {
  %mem = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  %w = comp.schedule_write %host to %mem on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.wait %w : !comp.event<ocl>
  %ev = comp.schedule_read %host from %mem on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.wait %ev : !comp.event<ocl>
  return
}

// CHECK-LABEL: func @write_read_barrier
//  CHECK-NEXT:   %[[EV1:.*]] = comp.schedule_write
//  CHECK-NEXT:   comp.schedule_barrier {{.*}} wait for %[[EV1]]
//  CHECK-NEXT:   return
func @write_read_barrier(%env: !comp.execenv<ocl:0, [11]>, %mem: memref<2x3xf32, 11>, %host: memref<2x3xf32>) {
  %ev1 = comp.schedule_write %host to %mem on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  %ev2 = comp.schedule_read %host from %mem on %env wait for %ev1 : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>) -> !comp.event<ocl>
  %ev3 = comp.schedule_barrier %env wait for %ev2 : ( !comp.execenv<ocl:0, [11]>, !comp.event<ocl>) -> !comp.event<ocl>
  return
}

// CHECK-LABEL: func @return_event
//       CHECK:   comp.schedule_read
func @return_event(%env: !comp.execenv<ocl:0, [11]>, %host: memref<2x3xf32>) -> !comp.event<ocl> {
  %mem = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  %w = comp.schedule_write %host to %mem on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.wait %w : !comp.event<ocl>
  // expected-remark@+2 {{could not remove redundant operation - unknown replacement semantic}}
  // expected-note@+2 {{see user: return}}
  %ev = comp.schedule_read %host from %mem on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  return %ev : !comp.event<ocl>
}

// CHECK-LABEL: func @return_event_replace
//  CHECK-SAME:     %[[EV1:[a-zA-Z0-9]*]]: !comp.event
//  CHECK-NEXT:   comp.alloc
//  CHECK-NEXT:   comp.schedule_write
//  CHECK-NEXT:   comp.wait
//  CHECK-NEXT:   return %[[EV1]]
func @return_event_replace(%env: !comp.execenv<ocl:0, [11]>, %host: memref<2x3xf32>, %ev1: !comp.event<ocl>) -> !comp.event<ocl> {
  %mem = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  %w = comp.schedule_write %host to %mem on %env wait for %ev1: (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>) -> !comp.event<ocl>
  comp.wait %w : !comp.event<ocl>
  %ev2 = comp.schedule_read %host from %mem on %env wait for %ev1 : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>, !comp.event<ocl>) -> !comp.event<ocl>
  return %ev2 : !comp.event<ocl>
}
