// Copyright 2020, Intel Corporation

#include "pmlc/target/intel_gen_ocl_spirv/pipeline.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/comp_to_llvm/passes.h"
#include "pmlc/conversion/gpu/passes.h"
#include "pmlc/conversion/gpu_to_spirv/passes.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/affinex/transforms/passes.h"
#include "pmlc/dialect/comp/ir/types.h"
#include "pmlc/dialect/comp/transforms/passes.h"
#include "pmlc/dialect/layer/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/target/intel_gen/passes.h"
#include "pmlc/target/intel_gen_ocl_spirv/passes.h"
#include "pmlc/transforms/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::intel_gen_ocl_spirv {

namespace comp = dialect::comp;
namespace layer = dialect::layer;
namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

struct OclPipelineOptions : public PassPipelineOptions<OclPipelineOptions> {
  Option<unsigned> spirvVersion{*this, "spirv-version",
                                llvm::cl::desc("SPIR-V Version"),
                                llvm::cl::initializer(150)};
};

void pipelineBuilder(OpPassManager &pm,
                     const OclPipelineOptions &oclPipelineOptions) {
  // Bound + pad initial tile code
  pm.addNestedPass<FuncOp>(layer::createInlineLayersPass());
  pm.addNestedPass<FuncOp>(tile::createComputeBoundsPass());
  pm.addNestedPass<FuncOp>(tile::createPadConstraintsPass());
  pm.addPass(tile::createSplitMainPass());
  pm.addPass(transforms::createHoistingPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Lower to PXA
  pm.addPass(conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do subgroup or accumulation
  pm.addNestedPass<FuncOp>(pxa::createSubgroupsPass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do tiled fusion
  pm.addNestedPass<FuncOp>(pxa::createFusionPass(/*memoryActivityThreshold=*/0,
                                                 /*exactlyMatch=*/false,
                                                 /*tiledFusion=*/true,
                                                 /*loopDepth=*/3));
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(pxa::createSimplifyArithmeticPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(
      pxa::createMemRefDataFlowOptPass(/*onlyParallelNested=*/true));
  pm.addPass(createCanonicalizerPass());
  // TODO: parametrize localize pass depending on memory size and HW caps
  pm.addNestedPass<FuncOp>(pxa::createLocalizePass());
  pm.addNestedPass<FuncOp>(
      pxa::createResizeTmpsPass(/*onlyParallelNested=*/true));
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Assign GPU blocks + threads to outermost loop
  pm.addNestedPass<FuncOp>(pxa::createGPUThreadPass(/*maxThreads=*/64));
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Data layout optimization.
  pm.addNestedPass<FuncOp>(
      createIntelGenOclReorderLayoutsPass(/*maxThreads=*/64,
                                          /*allowReorder=*/true));
  pm.addNestedPass<FuncOp>(pxa::createSimplifyWithConstraintsPass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(pxa::createVectorizeMemPass());
  pm.addNestedPass<FuncOp>(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Lower out of PXA memory semantics
  pm.addPass(pmlc::target::intel_gen::createLowerPXAToAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createAffineLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Unroll affine.for loops.
  pm.addNestedPass<FuncOp>(pmlc::dialect::affinex::createAffinexLoopUnroll(
      /*operationLimit=*/2048));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Block level MemRef dataflow optimization
  // WARNING: Assumes no aliasing
  // (try disabling this pass in case of correctness errors)
  pm.addNestedPass<FuncOp>(
      pmlc::dialect::affinex::createAffinexMemRefDataFlowOpt());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(
      pmlc::dialect::affinex::createAffinexDeadMemRefElimination());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Pack dims
  pm.addNestedPass<FuncOp>(
      pmlc::target::intel_gen::createAffineIndexPackPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do a custom version of lower-affine which also set GPU mappings
  pm.addNestedPass<FuncOp>(
      pmlc::target::intel_gen::createIntelGenLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Fix booleans
  pm.addNestedPass<FuncOp>(stdx::createI1StorageToI32Pass());

  // Devectorize
  pm.addNestedPass<FuncOp>(pmlc::target::intel_gen::createSubgroupBroadcastPass(
      /*useBlockOps=*/true));
  pm.addPass(createCSEPass());

  // Lower mapped scf.parallel's to GPU
  pm.addNestedPass<FuncOp>(
      pmlc::target::intel_gen::createParallelLoopToGpuPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // GPU transforms
  pm.addPass(
      createAddSpirvTargetPass(oclPipelineOptions.spirvVersion.getValue()));
  pm.addPass(conversion::gpu::createGpuKernelOutliningPass(
      comp::ExecEnvRuntime::OpenCL, /*memorySpace=*/11));

  // Hoist GPU ops
  pm.addPass(transforms::createHoistingPass());
  // pm.addPass(conversion::gpu::createGatherGpuLaunchFuncsPass());
  // pm.addPass(comp::createMinimizeBufferTransfersPass());
  // pm.addPass(comp::createExecEnvCoalescingPass());
  // pm.addPass(comp::createMinimizeAllocationsPass());
  pm.addNestedPass<FuncOp>(comp::createRemoveRedundantRWPass());
  // pm.addPass(comp::createRecalculateEventDepsPass(/*safeDealloc=*/false));

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  bool nonUniformBroadcast = false;
  if (oclPipelineOptions.spirvVersion.getValue() >= 150) {
    nonUniformBroadcast = true;
  }
  pm.addPass(conversion::gpu_to_spirv::createGPUToSPIRVCustomPass(
      nonUniformBroadcast));

  // SPIR-V passes for lowering attributes.
  pm.addNestedPass<mlir::spirv::ModuleOp>(createSetSubgroupSizePass());
  pm.addNestedPass<mlir::spirv::ModuleOp>(createSetAccessQualifiersPass());
  pm.addNestedPass<mlir::spirv::ModuleOp>(createLegalizeSpirvPass());
  pm.addNestedPass<mlir::spirv::ModuleOp>(
      spirv::createLowerABIAttributesPass());
  pm.addNestedPass<mlir::spirv::ModuleOp>(
      spirv::createUpdateVersionCapabilityExtensionPass());

  // Unbox wrapped argsort and lower remaining affine loops.
  pm.addNestedPass<FuncOp>(layer::createInlineLayersPass());
  pm.addPass(pmlc::target::intel_gen::createLowerPXAToAffinePass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createLowerToCFGPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Comp to LLVM - OpenCL function calls.
  pm.addPass(
      pmlc::conversion::comp_to_llvm::createConvertCompToLLVMPass("ocl_"));

  // Lower SCF to Standard before converting to LLVM
  pm.addPass(createLowerToCFGPass());

  // Convert to LLVM code.
  pm.addPass(pmlc::target::intel_gen::createConvertStandardToLLVM());
}

static constexpr const char *kTargetName = "intel_gen_ocl_spirv";
static constexpr const char *kPassPipelineTargetName =
    "target-intel_gen_ocl_spirv";

static PassPipelineRegistration<OclPipelineOptions>
    passPipelineReg(kPassPipelineTargetName,
                    "Target pipeline for Intel GEN iGPUs", pipelineBuilder);

class Target : public compiler::Target {
public:
  void buildPipeline(mlir::OpPassManager &pm, llvm::StringRef targetOptions) {
    auto oclPipelineOptions =
        OclPipelineOptions::createFromString(targetOptions);
    pipelineBuilder(pm, *oclPipelineOptions);
  }

  util::BufferPtr
  save(compiler::Program &program,
       const std::unordered_map<std::string, std::string> &config) {
    throw std::runtime_error(
        llvm::formatv("Target '{0}' does not have 'save' support.", kTargetName)
            .str());
  }
};

void registerTarget() {
  pmlc::compiler::registerTarget(kTargetName, std::make_shared<Target>());
}

} // namespace pmlc::target::intel_gen_ocl_spirv
