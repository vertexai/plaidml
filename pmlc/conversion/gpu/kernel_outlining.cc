//===- KernelOutlining.cpp - Implementation of GPU kernel outlining -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GPU dialect kernel outlining pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/GPU/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/RegionUtils.h"
#include "pmlc/conversion/gpu/pass_detail.h"
#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/memuse.h"
#include "pmlc/util/tags.h"
#include "llvm/ADT/MapVector.h"

namespace pmlc::conversion::gpu {

using namespace mlir; // NOLINT[build/namespaces]
namespace gpu = mlir::gpu;
namespace comp = pmlc::dialect::comp;
namespace stdx = pmlc::dialect::stdx;

template <typename OpTy>
static void createForAllDimensions(OpBuilder &builder, Location loc,
                                   SmallVectorImpl<Value> &values) {
  for (StringRef dim : {"x", "y", "z"}) {
    Value v = builder.create<OpTy>(loc, builder.getIndexType(),
                                   builder.getStringAttr(dim));
    values.push_back(v);
  }
}

// Add operations generating block/thread ids and grid/block dimensions at the
// beginning of the `launchFuncOpBody` region. Add mapping from argument in
// entry block of `launchOpBody`, to the corresponding result value of the added
// operations.
static void injectGpuIndexOperations(Location loc, Region &launchFuncOpBody,
                                     Region &launchOpBody,
                                     BlockAndValueMapping &map) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = launchOpBody.front();
  builder.setInsertionPointToStart(&launchFuncOpBody.front());
  SmallVector<Value, 12> indexOps;
  createForAllDimensions<gpu::BlockIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::ThreadIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::GridDimOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::BlockDimOp>(builder, loc, indexOps);
  // Replace the leading 12 function args with the respective thread/block index
  // operations. Iterate backwards since args are erased and indices change.
  for (auto indexOp : enumerate(indexOps))
    map.map(firstBlock.getArgument(indexOp.index()), indexOp.value());
}

// Outline the `gpu.launch` operation body into a kernel function. Replace
// `gpu.terminator` operations by `gpu.return` in the generated function.
static gpu::GPUFuncOp outlineKernelFuncImpl(gpu::LaunchOp launchOp,
                                            Attribute memorySpace,
                                            StringRef kernelFnName,
                                            llvm::SetVector<Value> &operands) {
  Location loc = launchOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp.getContext());
  Region &launchOpBody = launchOp.body();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  getUsedValuesDefinedAbove(launchOpBody, operands);

  // Create the gpu.func operation.
  SmallVector<Type, 4> kernelOperandTypes;
  kernelOperandTypes.reserve(operands.size());
  for (Value operand : operands) {
    Type type = operand.getType();
    if (auto memRefType = type.dyn_cast<MemRefType>()) {
      type = MemRefType::Builder(memRefType).setMemorySpace(memorySpace);
    }
    kernelOperandTypes.push_back(type);
  }
  FunctionType type =
      FunctionType::get(launchOp.getContext(), kernelOperandTypes, {});
  auto outlinedFunc = builder.create<gpu::GPUFuncOp>(loc, kernelFnName, type);
  outlinedFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());
  BlockAndValueMapping map;

  // Map the arguments corresponding to the launch parameters like blockIdx,
  // threadIdx, etc.
  Region &outlinedFuncBody = outlinedFunc.body();
  injectGpuIndexOperations(loc, outlinedFuncBody, launchOpBody, map);

  // Map arguments from gpu.launch region to the arguments of the gpu.func
  // operation.
  Block &entryBlock = outlinedFuncBody.front();
  for (auto operand : enumerate(operands))
    map.map(operand.value(), entryBlock.getArgument(operand.index()));

  // Clone the region of the gpu.launch operation into the gpu.func operation.
  // TODO(ravishankarm): If cloneInto can be modified such that if a mapping for
  // a block exists, that block will be used to clone operations into (at the
  // end of the block), instead of creating a new block, this would be much
  // cleaner.
  launchOpBody.cloneInto(&outlinedFuncBody, map);

  // Branch from entry of the gpu.func operation to the block that is cloned
  // from the entry block of the gpu.launch operation.
  Block &launchOpEntry = launchOpBody.front();
  Block *clonedLaunchOpEntry = map.lookup(&launchOpEntry);
  builder.setInsertionPointToEnd(&entryBlock);
  builder.create<BranchOp>(loc, clonedLaunchOpEntry);

  outlinedFunc.walk([](gpu::TerminatorOp op) {
    OpBuilder replacer(op);
    replacer.create<gpu::ReturnOp>(op.getLoc());
    op.erase();
  });
  outlinedFunc.walk([](AllocOp op) {
    auto newType = MemRefType::Builder(op.getType()).setMemorySpace(6);
    op.memref().setType(newType);
  });
  return outlinedFunc;
}

// A class holding state during gpu->comp + outlining of a single function
class OutlineToComp {
public:
  // Begin rewrite of the funciton op for comp support, initalize state
  OutlineToComp(FuncOp func, comp::ExecEnvRuntime runtime,
                Attribute memorySpace)
      : func(func), memorySpace(memorySpace) {
    // Precompute the various types
    execEnvType = comp::ExecEnvType::get(func.getContext(), runtime, /*tag=*/0,
                                         {memorySpace});
    eventType = execEnvType.getEventType();

    // Make a builder pointer to the begining of the function
    OpBuilder builder(func.getBody());

    // If there isn't a 'pack' parameter, add a device operand and construct the
    // environment.  Otherwise, extract environment from pack
    hasPack = func.getNumArguments() > 0 &&
              func.getArgument(0).getType().isa<stdx::ArgpackType>();
    if (!hasPack) {
      // Add a device parameter to the function
      auto oldFuncTy = func.getType();
      auto deviceTy = builder.getType<comp::DeviceType>();
      SmallVector<Type, 4> inputs{deviceTy};
      inputs.insert(inputs.end(), oldFuncTy.getInputs().begin(),
                    oldFuncTy.getInputs().end());
      auto newFuncTy = builder.getFunctionType(inputs, oldFuncTy.getResults());
      func.front().insertArgument(0u, deviceTy);
      func.setType(newFuncTy);

      // Insert CreateExecEnv
      auto device = func.getArgument(0);
      execEnv = builder.create<comp::CreateExecEnv>(func.getLoc(), execEnvType,
                                                    device);
    } else {
      // Find the unpack
      auto argpack = func.getArgument(0);
      assert(argpack.hasOneUse());
      auto unpack = cast<stdx::UnpackOp>(*argpack.user_begin());
      // Make a new unpack that also unpacks the exec env
      builder.setInsertionPointToStart(&func.getBody().front());
      SmallVector<Type, 4> newUnpackTypes(unpack.getResultTypes().begin(),
                                          unpack.getResultTypes().end());
      newUnpackTypes.insert(newUnpackTypes.begin(), execEnvType);
      auto newUnpack = builder.create<stdx::UnpackOp>(unpack.getLoc(),
                                                      newUnpackTypes, argpack);
      // Reconnect uses + delete old unpack
      execEnv = newUnpack.getResult(0);
      for (size_t i = 0; i < newUnpackTypes.size() - 1; i++) {
        unpack.getResult(i).replaceAllUsesWith(newUnpack.getResult(i + 1));
      }
      unpack.erase();
    }
  }

  void convertLaunch(gpu::LaunchOp launchOp, gpu::GPUFuncOp kernelFunc,
                     ValueRange operands) {
    assert(operands.size() == kernelFunc.getNumArguments());

    // Compute how arguments are used (RO/WO/RW)
    SmallVector<util::MemUse, 4> useTypes;
    for (size_t i = 0; i < operands.size(); i++) {
      useTypes.push_back(util::getMemoryUses(kernelFunc.getArgument(i)));
    }

    // Make a builder before the op we are replacing
    OpBuilder builder(launchOp);
    auto loc = launchOp.getLoc();

    // Convert operands to GPU memory, allocating as needed
    SmallVector<Value, 4> kernelArgs;
    SmallVector<Value, 4> memoryArgs;
    DenseMap<Value, util::MemUse> memoryUse;
    for (size_t i = 0; i < operands.size(); i++) {
      Value operand = operands[i];
      if (auto memrefTy = operand.getType().dyn_cast<MemRefType>()) {
        if (!gpuMemRefs.count(operand)) {
          MemRefType newType =
              MemRefType::Builder(memrefTy).setMemorySpace(memorySpace);
          Value out = builder.create<comp::Alloc>(loc, newType, execEnv);
          gpuMemRefs[operand] = out;
        }
        Value gpuMem = gpuMemRefs[operand];
        kernelArgs.push_back(gpuMem);
        memoryArgs.push_back(operand);
        memoryUse[operand] = useTypes[i];
      } else {
        kernelArgs.push_back(operand);
      }
    }

    // Do host -> GPU transfers
    SmallVector<Value, 4> events;
    for (auto mem : memoryArgs) {
      if (doesRead(memoryUse[mem])) {
        events.push_back(builder.create<comp::ScheduleWrite>(
            loc, eventType, mem, gpuMemRefs[mem], execEnv, ValueRange()));
      }
    }

    // Create the kernel
    auto kernelModule = kernelFunc->getParentOfType<gpu::GPUModuleOp>();
    auto kernelSymbol = builder.getSymbolRefAttr(
        kernelModule.getName(),
        {builder.getSymbolRefAttr(kernelFunc.getName())});
    auto kernel = builder.create<comp::CreateKernel>(
        loc, builder.getType<comp::KernelType>(), execEnv, kernelSymbol);
    kernelsToDestroy.push_back(kernel);

    // Call the kernel
    auto gridValues = launchOp.getGridSizeOperandValues();
    auto blockSizeValues = launchOp.getBlockSizeOperandValues();
    auto callKernel =
        builder
            .create<comp::ScheduleCompute>(
                loc, eventType, execEnv, kernel,                         //
                gridValues.x, gridValues.y, gridValues.z,                //
                blockSizeValues.x, blockSizeValues.y, blockSizeValues.z, //
                kernelArgs, events)
            .getResult();

    // Update memory state post kernel
    events.clear();
    for (auto mem : memoryArgs) {
      if (doesWrite(memoryUse[mem])) {
        // Make copy back to host
        events.push_back(builder.create<comp::ScheduleRead>(
            loc, eventType, mem, gpuMemRefs[mem], execEnv,
            ValueRange(callKernel)));
      }
    }
    if (events.size()) {
      // Wait until all data is copied back
      builder.create<comp::Wait>(loc, events);
    }
    launchOp.erase();
  }

  void finalize() {
    // Make a builder at the end of the function entry block
    auto builder = OpBuilder::atBlockTerminator(&func.back());
    Location loc = func.getLoc();
    bool destroyEnv = true;
    if (func.getNumResults() == 1 &&
        func.getType().getResult(0).isa<stdx::ArgpackType>()) {
      // Send env out rather than destroying it
      destroyEnv = false;
      // Replace argpack, and put insertion point befor it
      auto retOp = cast<ReturnOp>(func.front().getTerminator());
      auto pack = cast<stdx::PackOp>(retOp.getOperand(0).getDefiningOp());
      SmallVector<Value, 4> args(pack.getOperands().begin(),
                                 pack.getOperands().end());
      args.insert(args.begin(), execEnv);
      builder.setInsertionPoint(pack.getOperation());
      auto newPack = builder.create<stdx::PackOp>(
          pack.getLoc(), builder.getType<stdx::ArgpackType>(), args);
      builder.setInsertionPoint(newPack);
      pack.getResult().replaceAllUsesWith(newPack.getResult());
      pack.erase();
    }
    if (hasPack && func.getName() != "fini") {
      // If we got env in, and we are not fini, don't destroy
      destroyEnv = false;
      builder.create<comp::DumpProfiling>(loc, execEnv);
    }

    // Destroy all the kernels
    for (auto kernel : kernelsToDestroy) {
      builder.create<comp::DestroyKernel>(loc, execEnv, kernel);
    }
    // Destroy all the buffers
    for (auto &kvp : gpuMemRefs) {
      builder.create<comp::Dealloc>(loc, execEnv, kvp.second);
    }
    // Insert DestroyExecEnv
    if (destroyEnv) {
      builder.create<comp::DestroyExecEnv>(loc, execEnv);
    }
  }

private:
  // The host side function being rewritten into comp
  FuncOp func;
  // THe memory space for GPU buffers
  Attribute memorySpace;
  // The type of the execution environement
  comp::ExecEnvType execEnvType;
  // The type of events
  comp::EventType eventType;
  // Did we get env from an argpack
  bool hasPack;
  // The comp execution environment
  Value execEnv;
  // A map from device memory reference to their GPU eqivilant.  Using
  // MapVector to make ordering of deletions deterministic
  llvm::MapVector<Value, Value> gpuMemRefs;
  // A list of all the kernels to be destoryed.  Not a 'SmallVector' since # of
  // kernels is more O(N) than O(1)
  std::vector<Value> kernelsToDestroy;
};

/// Pass that moves the kernel of each LaunchOp into its separate nested module.
///
/// This pass moves the kernel code of each LaunchOp into a function created
/// inside a nested module. It also creates an external function of the same
/// name in the parent module.
///
/// The gpu.modules are intended to be compiled to a cubin blob independently in
/// a separate pass. The external functions can then be annotated with the
/// symbol of the cubin accessor function.
class GpuKernelOutliningPass
    : public GpuKernelOutliningPassBase<GpuKernelOutliningPass> {
public:
  GpuKernelOutliningPass() = default;
  GpuKernelOutliningPass(comp::ExecEnvRuntime runtime, unsigned memorySpace) {
    this->execEnvRuntime = static_cast<unsigned>(runtime);
    this->execEnvMemorySpace = memorySpace;
  }

  void runOnOperation() override {
    // set spv.target_env to moduleOp
    auto target_env = getOperation()->getAttrOfType<spirv::TargetEnvAttr>(
        spirv::getTargetEnvAttrName());
    if (!target_env) {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_5,
          {spirv::Capability::Shader, spirv::Capability::GroupNonUniformBallot,
           spirv::Capability::Int64, spirv::Capability::Int16,
           spirv::Capability::Int8, spirv::Capability::Float64,
           spirv::Capability::Float16,
           spirv::Capability::StorageBuffer16BitAccess},
          ArrayRef<spirv::Extension>(
              {spirv::Extension::SPV_KHR_storage_buffer_storage_class,
               spirv::Extension::SPV_KHR_16bit_storage}),
          &getContext());
      getOperation()->setAttr(
          spirv::getTargetEnvAttrName(),
          spirv::TargetEnvAttr::get(
              triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
              spirv::TargetEnvAttr::kUnknownDeviceID,
              spirv::getDefaultResourceLimits(&getContext())));
    }

    SymbolTable symbolTable(getOperation());
    bool modified = false;
    for (auto func : getOperation().getOps<FuncOp>()) {
      // Skip external functions
      if (func.isExternal()) {
        continue;
      }
      // Insert just after the function.
      Block::iterator insertPt(func.getOperation()->getNextNode());
      // Prep for comp conversion
      auto runtime =
          static_cast<comp::ExecEnvRuntime>(execEnvRuntime.getValue());
      auto i64Type = IntegerType::get(&getContext(), 64);
      auto memorySpace =
          IntegerAttr::get(i64Type, this->execEnvMemorySpace.getValue());
      OutlineToComp toComp(func, runtime, memorySpace);

      auto funcWalkResult = func.walk([&](gpu::LaunchOp op) {
        llvm::SetVector<Value> operands;
        std::string kernelFnName =
            Twine(op->getParentOfType<FuncOp>().getName(), "_kernel").str();

        // Pull in instructions that can be sunk
        if (failed(sinkOperationsIntoLaunchOp(op)))
          return WalkResult::interrupt();
        gpu::GPUFuncOp outlinedFunc =
            outlineKernelFuncImpl(op, memorySpace, kernelFnName, operands);

        // Create nested module and insert outlinedFunc. The module will
        // originally get the same name as the function, but may be renamed on
        // insertion into the parent module.

        gpu::KernelDim3 blockSize = op.getBlockSizeOperandValues();

        auto kernelModule =
            createKernelModule(outlinedFunc, symbolTable, blockSize);
        symbolTable.insert(kernelModule, insertPt);

        // Convert to comp
        toComp.convertLaunch(op, outlinedFunc, operands.getArrayRef());
        modified = true;
        return WalkResult::advance();
      });

      // Finalize comp conversion
      toComp.finalize();

      if (funcWalkResult.wasInterrupted())
        return signalPassFailure();
    }

    // If any new module was inserted in this module, annotate this module as
    // a container module.
    if (modified)
      getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                              UnitAttr::get(&getContext()));
  }

private:
  // Returns a gpu.module containing kernelFunc and all callees (recursive).
  gpu::GPUModuleOp createKernelModule(gpu::GPUFuncOp kernelFunc,
                                      const SymbolTable &parentSymbolTable,
                                      gpu::KernelDim3 &blockSize) {
    // TODO: This code cannot use an OpBuilder because it must be inserted into
    // a SymbolTable by the caller. SymbolTable needs to be refactored to
    // prevent manual building of Ops with symbols in code using SymbolTables
    // and then this needs to use the OpBuilder.
    auto context = getOperation().getContext();
    OpBuilder builder(context);

    auto entry_point_abi = kernelFunc->getAttrOfType<spirv::EntryPointABIAttr>(
        spirv::getEntryPointABIAttrName());
    if (!entry_point_abi) {
      int x = blockSize.x.getDefiningOp()
                  ->getAttrOfType<IntegerAttr>("value")
                  .getInt();
      int y = blockSize.y.getDefiningOp()
                  ->getAttrOfType<IntegerAttr>("value")
                  .getInt();
      int z = blockSize.z.getDefiningOp()
                  ->getAttrOfType<IntegerAttr>("value")
                  .getInt();
      auto entryPointAbiAttr =
          mlir::spirv::getEntryPointABIAttr({x, y, z}, kernelFunc.getContext());

      kernelFunc->setAttr(spirv::getEntryPointABIAttrName(), entryPointAbiAttr);
    }

    OperationState state(kernelFunc.getLoc(),
                         gpu::GPUModuleOp::getOperationName());
    gpu::GPUModuleOp::build(builder, state, kernelFunc.getName());
    auto kernelModule = cast<gpu::GPUModuleOp>(Operation::create(state));
    SymbolTable symbolTable(kernelModule);
    symbolTable.insert(kernelFunc);

    SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
    while (!symbolDefWorklist.empty()) {
      if (Optional<SymbolTable::UseRange> symbolUses =
              SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
        for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
          StringRef symbolName =
              symbolUse.getSymbolRef().cast<FlatSymbolRefAttr>().getValue();
          if (symbolTable.lookup(symbolName))
            continue;

          Operation *symbolDefClone =
              parentSymbolTable.lookup(symbolName)->clone();
          symbolDefWorklist.push_back(symbolDefClone);
          symbolTable.insert(symbolDefClone);
        }
      }
    }

    return kernelModule;
  }
};

std::unique_ptr<mlir::Pass> createGpuKernelOutliningPass() {
  return std::make_unique<GpuKernelOutliningPass>();
}

std::unique_ptr<mlir::Pass>
createGpuKernelOutliningPass(comp::ExecEnvRuntime runtime,
                             unsigned memorySpace) {
  return std::make_unique<GpuKernelOutliningPass>(runtime, memorySpace);
}

} // namespace pmlc::conversion::gpu
