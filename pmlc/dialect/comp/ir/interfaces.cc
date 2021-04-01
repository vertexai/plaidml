// Copyright 2020 Intel Corporation
#include "pmlc/dialect/comp/ir/interfaces.h"
#include "pmlc/dialect/comp/ir/types.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::comp; // NOLINT

mlir::LogicalResult pmlc::dialect::comp::verifyScheduleOp(mlir::Operation *op) {
  auto scheduleOp = mlir::cast<ScheduleOpInterface>(op);
  auto execEnvOp = mlir::cast<ExecEnvOpInterface>(op);

  Value execEnv = execEnvOp.getExecEnv();
  auto execEnvType = execEnv.getType().cast<ExecEnvType>();
  ExecEnvRuntime runtime = execEnvType.getRuntime();

  // Check that resulting event runtime matches execenv runtime.
  Value resultEvent = scheduleOp.getResultingEvent();
  auto resultEventType = resultEvent.getType().cast<EventType>();

  if (resultEventType.getRuntime() != runtime)
    return scheduleOp.emitOpError(
        "mismatch between execenv runtime and resulting event runtime");

  // Check that dependent events come from same runtime.
  for (Value dep : scheduleOp.getDependencies()) {
    auto depType = dep.getType().cast<EventType>();

    if (depType.getRuntime() != runtime)
      return scheduleOp.emitOpError(
          "mismatch between execenv runtime and dependant event runtime");
  }

  return success();
}

static LogicalResult verifyMemorySpace(OpState &op, ExecEnvType execEnvType,
                                       Type memoryType) {
  auto memRefType = memoryType.cast<MemRefType>();
  Attribute requestedSpace = memRefType.getMemorySpace();

  if (!execEnvType.supportsMemorySpace(requestedSpace))
    return op.emitOpError("memory space is not supported by execenv");

  return success();
}

mlir::LogicalResult
pmlc::dialect::comp::verifyMemoryTransferOp(mlir::Operation *op) {
  auto memoryOp = mlir::cast<MemoryTransferOpInterface>(op);
  if (memoryOp.sourceHasExecEnv()) {
    Value execEnv = memoryOp.getSourceExecEnv();
    auto execEnvType = execEnv.getType().cast<ExecEnvType>();
    Type sourceType = memoryOp.getSource().getType();
    if (failed(verifyMemorySpace(memoryOp, execEnvType, sourceType)))
      return failure();
  }

  if (memoryOp.destinationHasExecEnv()) {
    Value execEnv = memoryOp.getDestinationExecEnv();
    auto execEnvType = execEnv.getType().cast<ExecEnvType>();
    Type destinationType = memoryOp.getDestination().getType();
    if (failed(verifyMemorySpace(memoryOp, execEnvType, destinationType)))
      return failure();
  }

  return success();
}

#include "pmlc/dialect/comp/ir/interfaces.cc.inc"
