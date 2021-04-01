// RUN: pmlc-opt --comp-execenv-coalescing %s | FileCheck %s

// CHECK-LABEL: func @one_type
//  CHECK-SAME:     %[[DEV:.*]]: !comp.device
//  CHECK-NEXT:   %[[ENV:.*]] = comp.create_execenv %[[DEV]] : (!comp.device) -> !comp.execenv<ocl:0, [11]>
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV]]
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV]]
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV]]
//  CHECK-NEXT:   comp.destroy_execenv %[[ENV]]
//  CHECK-NEXT:   return
func @one_type(%dev: !comp.device) {
  %env0 = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %ev0 = comp.schedule_barrier %env0 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env0 : !comp.execenv<ocl:0, [11]>
  %env1 = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %ev1 = comp.schedule_barrier %env1 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env1 : !comp.execenv<ocl:0, [11]>
  %env2 = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %ev2 = comp.schedule_barrier %env2 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env2 : !comp.execenv<ocl:0, [11]>
  return
}

// CHECK-LABEL: func @two_types_in_seq
//  CHECK-SAME:     %[[DEV_OCL:[a-zA-Z0-9]*]]: !comp.device
//  CHECK-SAME:     %[[DEV_VK:[a-zA-Z0-9]*]]: !comp.device
//  CHECK-NEXT:   %[[ENV_OCL:.*]] = comp.create_execenv %[[DEV_OCL]] : (!comp.device) -> !comp.execenv<ocl:0, [11]>
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV_OCL]]
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV_OCL]]
//  CHECK-NEXT:   comp.destroy_execenv %[[ENV_OCL]]
//  CHECK-NEXT:   %[[ENV_VK:.*]] = comp.create_execenv %[[DEV_VK]] : (!comp.device) -> !comp.execenv<vk:0, [0]>
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV_VK]]
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV_VK]]
//  CHECK-NEXT:   comp.destroy_execenv %[[ENV_VK]]
//  CHECK-NEXT:   return
func @two_types_in_seq(%dev_ocl: !comp.device, %dev_vk: !comp.device) {
  %env0 = comp.create_execenv %dev_ocl : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %ev0 = comp.schedule_barrier %env0 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  %env1 = comp.create_execenv %dev_ocl : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %ev1 = comp.schedule_barrier %env1 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env1 : !comp.execenv<ocl:0, [11]>
  comp.destroy_execenv %env0 : !comp.execenv<ocl:0, [11]>

  %env2 = comp.create_execenv %dev_vk : (!comp.device) -> !comp.execenv<vk:0, [0]>
  %ev2 = comp.schedule_barrier %env2 : (!comp.execenv<vk:0, [0]>) -> !comp.event<vk>
  %env3 = comp.create_execenv %dev_vk : (!comp.device) -> !comp.execenv<vk:0, [0]>
  %ev3 = comp.schedule_barrier %env3 : (!comp.execenv<vk:0, [0]>) -> !comp.event<vk>
  comp.destroy_execenv %env3 : !comp.execenv<vk:0, [0]>
  comp.destroy_execenv %env2 : !comp.execenv<vk:0, [0]>
  return
}

// CHECK-LABEL: func @two_types_mixed
//  CHECK-SAME:     %[[DEV_OCL:[a-zA-Z0-9]*]]: !comp.device
//  CHECK-SAME:     %[[DEV_VK:[a-zA-Z0-9]*]]: !comp.device
//  CHECK-NEXT:   %[[ENV_OCL:.*]] = comp.create_execenv %[[DEV_OCL]] : (!comp.device) -> !comp.execenv<ocl:0, [11]>
//  CHECK-NEXT:   %[[ENV_VK:.*]] = comp.create_execenv %[[DEV_VK]] : (!comp.device) -> !comp.execenv<vk:0, [0]>
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV_OCL]]
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV_VK]]
//  CHECK-NEXT:   comp.schedule_barrier %[[ENV_OCL]]
//  CHECK-NEXT:   comp.destroy_execenv %[[ENV_OCL]]
//  CHECK-NEXT:   comp.destroy_execenv %[[ENV_VK]]
//  CHECK-NEXT:   return
func @two_types_mixed(%dev_ocl: !comp.device, %dev_vk: !comp.device) {
  %env0 = comp.create_execenv %dev_ocl : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %env1 = comp.create_execenv %dev_vk : (!comp.device) -> !comp.execenv<vk:0, [0]>
  %ev0 = comp.schedule_barrier %env0 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  %env2 = comp.create_execenv %dev_ocl : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  comp.destroy_execenv %env0 : !comp.execenv<ocl:0, [11]>
  %ev1 = comp.schedule_barrier %env1 : (!comp.execenv<vk:0, [0]>) -> !comp.event<vk>
  %ev2 = comp.schedule_barrier %env2 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env2 : !comp.execenv<ocl:0, [11]>
  comp.destroy_execenv %env1 : !comp.execenv<vk:0, [0]>
  return
}

// CHECK-LABEL: func @same_device_different_type
//  CHECK-SAME:     %[[DEV:.*]]: !comp.device
//  CHECK-NEXT: %[[ENV0:.*]] = comp.create_execenv %[[DEV]] : (!comp.device) -> !comp.execenv<ocl:0, [11]>
//  CHECK-NEXT: comp.schedule_barrier %[[ENV0]]
//  CHECK-NEXT: comp.destroy_execenv %[[ENV0]]
//  CHECK-NEXT: %[[ENV1:.*]] = comp.create_execenv %[[DEV]] : (!comp.device) -> !comp.execenv<ocl:1, [11]>
//  CHECK-NEXT: comp.schedule_barrier %[[ENV1]]
//  CHECK-NEXT: comp.destroy_execenv %[[ENV1]]
//  CHECK-NEXT: return
func @same_device_different_type(%dev: !comp.device) {
  %env0 = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %ev0 = comp.schedule_barrier %env0 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env0 : !comp.execenv<ocl:0, [11]>
  %env1 = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:1, [11]>
  %ev1 = comp.schedule_barrier %env1 : (!comp.execenv<ocl:1, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env1 : !comp.execenv<ocl:1, [11]>
  return
}

// CHECK-LABEL: func @different_devices_same_type
//  CHECK-SAME:     %[[DEV0:[a-zA-Z0-9]*]]: !comp.device
//  CHECK-SAME:     %[[DEV1:[a-zA-Z0-9]*]]: !comp.device
//  CHECK-NEXT: %[[ENV0:.*]] = comp.create_execenv %[[DEV0]] : (!comp.device) -> !comp.execenv<ocl:0, [11]>
//  CHECK-NEXT: comp.schedule_barrier %[[ENV0]]
//  CHECK-NEXT: comp.destroy_execenv %[[ENV0]]
//  CHECK-NEXT: %[[ENV1:.*]] = comp.create_execenv %[[DEV1]] : (!comp.device) -> !comp.execenv<ocl:0, [11]>
//  CHECK-NEXT: comp.schedule_barrier %[[ENV1]]
//  CHECK-NEXT: comp.destroy_execenv %[[ENV1]]
//  CHECK-NEXT: return
func @different_devices_same_type(%dev0: !comp.device, %dev1: !comp.device) {
  %env0 = comp.create_execenv %dev0 : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %ev0 = comp.schedule_barrier %env0 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env0 : !comp.execenv<ocl:0, [11]>
  %env1 = comp.create_execenv %dev1 : (!comp.device) -> !comp.execenv<ocl:0, [11]>
  %ev1 = comp.schedule_barrier %env1 : (!comp.execenv<ocl:0, [11]>) -> !comp.event<ocl>
  comp.destroy_execenv %env1 : !comp.execenv<ocl:0, [11]>
  return
}
