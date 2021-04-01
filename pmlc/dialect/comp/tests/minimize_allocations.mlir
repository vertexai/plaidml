// RUN: pmlc-opt --comp-minimize-allocations --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func @two_wo_host
//  CHECK-SAME:     %[[ENV:[a-zA-Z0-9]*]]: !comp.execenv
//       CHECK:   %[[MEM:.*]] = comp.alloc %[[ENV]]
//  CHECK-NEXT:   "op1"(%[[MEM]])
//  CHECK-NEXT:   "op2"(%[[MEM]])
//  CHECK-NEXT:   comp.dealloc %[[ENV]] %[[MEM]]
func @two_wo_host(%env: !comp.execenv<ocl:0, [11]>) {
  %mem1 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  "op1"(%mem1) : (memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem1 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()

  %mem2 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  "op2"(%mem2) : (memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem2 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()
  return
}

// CHECK-LABEL: func @two_with_host
//  CHECK-SAME:     %[[ENV:[a-zA-Z0-9]*]]: !comp.execenv
//  CHECK-SAME:     %[[ARG:[a-zA-Z0-9]*]]: memref
//       CHECK:   %[[MEM:.*]] = comp.alloc %[[ENV]]
//       CHECK:   "op1"(%[[MEM]])
//  CHECK-NEXT:   %[[WEV:.*]] = comp.schedule_write %[[ARG]] to %[[MEM]]
//  CHECK-NEXT:   comp.wait %[[WEV]]
//  CHECK-NEXT:   "op2"(%[[MEM]])
//  CHECK-NEXT:   comp.dealloc %[[ENV]] %[[MEM]]
func @two_with_host(%env: !comp.execenv<ocl:0, [11]>, %arg: memref<2x3xf32>) {
  %mem1 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  %w1 = comp.schedule_write %arg to %mem1 on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.wait %w1 : !comp.event<ocl>
  "op1"(%mem1) : (memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem1 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()

  %mem2 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  %w2 = comp.schedule_write %arg to %mem2 on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.wait %w2 : !comp.event<ocl>
  "op2"(%mem2) : (memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem2 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()
  return
}

// CHECK-LABEL: func @three_overlaping
//  CHECK-SAME:     %[[ENV:[a-zA-Z0-9]*]]: !comp.execenv
//       CHECK:   %[[MEM1:.*]] = comp.alloc %[[ENV]]
//       CHECK:   %[[MEM2:.*]] = comp.alloc %[[ENV]]
//       CHECK:   "op1"(%[[MEM1]], %[[MEM2]])
//       CHECK:   "op2"(%[[MEM2]], %[[MEM1]])
//   CHECK-DAG:   comp.dealloc %[[ENV]] %[[MEM1]]
//   CHECK-DAG:   comp.dealloc %[[ENV]] %[[MEM2]]
func @three_overlaping(%env: !comp.execenv<ocl:0, [11]>) {
  %mem1 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  %mem2 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  "op1"(%mem1, %mem2) :  (memref<2x3xf32, 11>, memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem1 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()
  %mem3 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  "op2"(%mem2, %mem3) :  (memref<2x3xf32, 11>, memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem2 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem3 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()
  return
}

// CHECK-LABEL: func @prioritize_in_sync
//  CHECK-SAME:     %[[ENV:[a-zA-Z0-9]*]]: !comp.execenv
//       CHECK:   %[[MEM1:.*]] = comp.alloc %[[ENV]]
//       CHECK:   "op1"(%[[MEM1]])
//       CHECK:   %[[MEM2:.*]] = comp.alloc %[[ENV]]
//   CHECK-NOT:   comp.schedule_write
//       CHECK:   "op2"(%[[MEM1]], %[[MEM2]])
//       CHECK:   comp.dealloc %[[ENV]] %[[MEM1]]
//       CHECK:   comp.dealloc %[[ENV]] %[[MEM2]]
func @prioritize_in_sync(%env: !comp.execenv<ocl:0, [11]>) {
  %host1 = alloc() : memref<2x3xf32>
  %host2 = alloc() : memref<2x3xf32>
  %mem1 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  "op1"(%mem1) : (memref<2x3xf32, 11>) -> ()
  %ev = comp.schedule_read %host1 from %mem1 on %env : (memref<2x3xf32>, memref<2x3xf32, 11>, !comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.wait %ev : !comp.event<ocl>
  comp.dealloc %env %mem1 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()
  %mem2 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  %mem3 = comp.alloc %env : (!comp.execenv<ocl:0, [11]>) -> memref<2x3xf32, 11>
  "op2"(%mem2, %mem3) : (memref<2x3xf32, 11>, memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem2 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()
  comp.dealloc %env %mem3 : (!comp.execenv<ocl:0, [11]>, memref<2x3xf32, 11>) -> ()
  return
}
