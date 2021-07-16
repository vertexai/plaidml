#ifndef __PMLC_DIALECT_PXA_PASSES__
#define __PMLC_DIALECT_PXA_PASSES__

include "mlir/Pass/PassBase.td"

def AffineNormalize : FunctionPass<"pxa-normalize"> {
  let summary = "Normalize affine ops";
  let constructor = "pmlc::dialect::pxa::createAffineNormalizePass()";
  let options = [
    Option<"promote", "promote", "bool",
           /*default=*/"true",
           "Promote the body of 1 trip loops">,
    Option<"denest", "denest", "bool",
           /*default=*/"false",
           "Combine perfectly nested loops">
  ];
}

def AutoTileExample : FunctionPass<"pxa-autotile-example"> {
  let summary = "Tile all dimensions by 10";
  let constructor = "pmlc::dialect::pxa::createAutoTileExamplePass()";
}

def DeallocPlacement : Pass<"pxa-dealloc-placement", "mlir::ModuleOp"> {
  let summary = "Adds deallocs for memref allocs";
  let constructor = "pmlc::dialect::pxa::createDeallocPlacementPass()";
}

def Cache : FunctionPass<"pxa-cache"> {
  let summary = "Cache global memory into local memory";
  let constructor = "pmlc::dialect::pxa::createCachePass()";
  let options = [
    Option<"cacheTag", "cache-tag", "std::string",
           /*default=*/"\"cache\"",
           "Cache loop tag to be set">,
    Option<"inputTag", "input-tag", "std::string",
           /*default=*/"\"input\"",
           "Input cache loop tag">,
    Option<"outputTag", "output-tag", "std::string",
           /*default=*/"\"output\"",
           "Output cache loop tag">,
    Option<"outerTag", "outer-tag", "std::string",
           /*default=*/"\"outer\"",
           "Outer loop tag">,
    Option<"middleTag", "middle-tag", "std::string",
           /*default=*/"\"middle\"",
           "Middle loop tag">,
    Option<"innerTag", "inner-tag", "std::string",
           /*default=*/"\"inner\"",
           "Inner loop tag">,
    Option<"wholeBlock", "whole-block", "bool",
           /*default=*/"false",
           "Whether cache the whole memory block">
  ];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def ConvertMemOp : FunctionPass<"pxa-convert-mem-op"> {
  let summary = "Convert standard memory operations to pxa if possible";
  let constructor = "pmlc::dialect::pxa::createConvertMemOpPass()";
}

def Fusion : FunctionPass<"pxa-fusion"> {
  let summary = "Fuse blocks that are compatible";
  let constructor = "pmlc::dialect::pxa::createFusionPass()";
  let options = [
    Option<"memoryActivityThreshold", "mem-threshold", "int64_t",
           /*default=*/"0",
           "Prevent over-fusion by specifying the maximum memory activity "
           "(alloc, read, write) allowed">,
    Option<"exactlyMatch", "exactly-match", "bool",
           /*default=*/"false",
           "Whether fuse the loops with exactly matched indices">,
    Option<"tiledFusion", "tiled-fusion", "bool",
           /*default=*/"false",
           "Perform tiled fusions">,
    Option<"loopDepth", "loop-depth", "int64_t",
           /*default=*/"0",
           "How deep the fusion should perform">,
    Option<"singleOutput", "single-output", "bool",
           /*default=*/"false",
           "Allow single output only">,
  ];
}

def GPUThread : FunctionPass<"pxa-gpu-thread"> {
  let summary = "Tile outmost loops for GPU threading";
  let constructor = "pmlc::dialect::pxa::createGPUThreadPass()";
  let options = [
    Option<"maxThreads", "max-threads", "unsigned", /*default=*/"64",
           "Set maximum/goal inner threads">
  ];
}

def CPUThread : FunctionPass<"pxa-cpu-thread"> {
  let summary = "Tile outmost loops for CPU threading";
  let constructor = "pmlc::dialect::pxa::createCPUThreadPass()";
  let options = [
    Option<"threads", "threads", "unsigned",
           /*default=*/"std::thread::hardware_concurrency()",
           "Set maximum/goal inner threads">
  ];
}

def Localize : FunctionPass<"pxa-localize"> {
  let summary = "Push memref allocs to innermost scope";
  let constructor = "pmlc::dialect::pxa::createLocalizePass()";
}

def MemRefDataFlowOpt : FunctionPass<"pxa-dataflow-opt"> {
  let summary = "Perform reduce/load forwarding for memrefs";
  let constructor = "pmlc::dialect::pxa::createMemRefDataFlowOptPass()";
    let options = [
    Option<"onlyParallelNested", "only-parallel-nested", "bool",
           /*default=*/"false",
           "Perform memref optimizations only under AffineParallel's scope.">,
  ];
}

def NestLoops : FunctionPass<"pxa-nest-loops"> {
  let summary = "Nests outermost loops to a minimum IV count";
  let constructor = "pmlc::dialect::pxa::createNestLoopsPass()";
  let options = [
    Option<"minLoopIVs", "ivs", "unsigned", /*default=*/"2",
           "Set minimum required IV's">
  ];
}

def ResizeTmps : FunctionPass<"pxa-resize-tmps"> {
  let summary = "Resize temporary buffers to minimal sizes";
  let constructor = "pmlc::dialect::pxa::createResizeTmpsPass()";
  let options = [
  Option<"onlyParallelNested", "only-parallel-nested", "bool",
         /*default=*/"false",
         "Perform resize tmp optimizations only under AffineParallel's scope.">,
  ];
}

def Subgroups : FunctionPass<"pxa-subgroups"> {
  let summary = "Regorganize top level block to maximum use of subgroups on gen9";
  let constructor = "pmlc::dialect::pxa::createSubgroupsPass()";
  let dependentDialects = ["mlir::vector::VectorDialect"];
}

def TestIndirectUsesIterator : Pass<"pxa-test-indirect-uses-iterator"> {
  let summary = "Trace the IndirectUsesIterator";
  let constructor = "pmlc::dialect::pxa::createTestIndirectUsesIteratorPass()";
}

def TestIndirectValuesIterator : Pass<"pxa-test-indirect-values-iterator"> {
  let summary = "Trace the IndirectValuesIterator for unit tests";
  let constructor = "pmlc::dialect::pxa::createTestIndirectValuesIteratorPass()";
}

def TestStrideInfo : Pass<"pxa-stride-info"> {
  let summary = "Report stride data for all loads/stores for unit tests";
  let constructor = "pmlc::dialect::pxa::createTestStrideInfoPass()";
}

def TileAccumulate : FunctionPass<"pxa-tile-accumulate"> {
  let summary = "Tile accumulation indexes into an inner block to avoid "
    "possible race conditions";
  let constructor = "pmlc::dialect::pxa::createTileAccumulatePass()";
}

def Vectorize : FunctionPass<"pxa-vectorize"> {
  let summary = "Perform vectorization based on specified strategy";
  let constructor = "pmlc::dialect::pxa::createVectorizePass()";
  let dependentDialects = [
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"strategy", "strategy", "std::string",
           "\"simple\"", "Strategy to use for vectorization">,
    Option<"vectorWidth", "width", "unsigned", /*default=*/"8",
           "Vector register size in elements">,
    Option<"vectorizeMathOp", "vectorize-math-op", "bool", /*default=*/"true",
           "Vectorize math dialect op or create inner loop for it">
  ];
}

def SimplifyArithmetic : FunctionPass<"pxa-simplify-arithmetic"> {
  let summary = "Simplifies arithmetic";
  let constructor = "pmlc::dialect::pxa::createSimplifyArithmeticPass()";
}

def SimplifyWithConstraints : FunctionPass<"pxa-simplify-with-constraints"> {
  let summary = "Simplfies affine maps using constraints on input dimensions";
  let constructor = "pmlc::dialect::pxa::createSimplifyWithConstraintsPass()";
}

def ReorderLayouts : FunctionPass<"pxa-reorder-layouts"> {
  let summary = "Optimize data layouts for efficient reading";
  let description = [{
    Pass that performs optimization of data layouts in order to align order
    of elements in memory with order of iteration over these elements.
    Aim of this optimization is to decrease distance between accesses to
    same memory, which should result in lowering number of cache misses
    and allow to leverage implicit prefetching of consecutive cachelines.
    Overall flow of the algorithm is:
    1. Gather global memory information, such as shape, read operation
       affine maps, vectorization, etc.
    2. Select new layout by analyzing access patterns: affine maps of read
       operations, loop order, vectorization, etc.
    3. Try to modify producer(s) of data to store into optimized layout
       or create separate "reorder" operation.

    Selecting of new layout is implemented by analyzing affine maps of read
    operations, with constraints on loop variables, expanding number of
    dimensions to reduce number of variables used in each dimension and
    permuting those dimensions to better align with order of loops.

    As an example given input with following structure:
    ```mlir
      %1 = memref.alloc() : memref<1x58x58x64xf32>
      %2 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 64) {
        %7 = pxa.reduce assign %6, %1[0, %arg3 + 1, %arg4 + 1, %arg5]
            : memref<1x58x58x64xf32>
        // (d0, d1, d2) -> (0, d0 + 1, d1 + 1, d2)
      }
      %4 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 7, 8) {
        %7 = affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (3, 3, 4) {
          %12 = affine.parallel (%arg9, %arg10) = (0, 0) to (8, 2) {
            %14 = pxa.vector_load %2[
                0, %arg3 + %arg6, %arg4 * 8 + %arg7 + %arg9,
                %arg8 * 16 + %arg10 * 8] : memref<1x58x58x64xf32>, vector<8xf32>
            // (d0, d1, d2, d3, d4, d5, d6)
            //     -> (0, d0 + d2, d1 * 8 + d3 + d4, d5 * 16 + d6 * 8)
          }
        }
      }
    ```
    Expected output of the pass would be similar to:
    ```mlir
      %1 = memref.alloc() : memref<1x58x4x58x2x8xf32>
      %2 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 56, 64) {
        %7 = pxa.reduce assign %6, %1[
            0, %arg3 + 1, %arg5 floordiv 16, %arg4 + 1, %arg5 floordiv 8 % 2,
            %arg5 % 8] : memref<1x58x4x58x2x8xf32>
        // (d0, d1, d2) ->
        //     (0, d0 + 1, d2 floordiv 16, d1 + 1, d2 floordiv 8 % 2, d2 % 8)
      }
      %4 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (56, 7, 8) {
        %7 = affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (3, 3, 4) {
          %12 = affine.parallel (%arg9, %arg10) = (0, 0) to (8, 2) {
            %14 = pxa.vector_load %2[
                0, %arg3 + %arg6, %arg8, %arg4 * 8 + %arg7 + %arg9, %arg10, 0]
                : memref<1x58x4x58x2x8xf32>, vector<8xf32>
            // (d0, d1, d2, d3, d4, d5, d6) ->
            //     (0, d0 + d2, d5, d1 * 8 + d3 + d4, d6, 0)
          }
        }
      }
    ```
    Please note above snippets are not valid IR, they have been heavily
    formated for readability, uninteresting portions have been erased and
    a lot of syntactic noise has been stripped.
  }];
  let constructor = "pmlc::dialect::pxa::createReorderLayoutsPass()";
  let options = [
    Option<"allowReorder", "allow-reorder", "bool", /*default=*/"false",
           "Allow for creating separate reorder">,
    Option<"userLayouts", "user-layouts", "bool", /*default=*/"false",
           "Pattern match ops e.g., convs and introduce user provided layouts">,
    Option<"datatileSize", "datatile-size", "int64_t",
           /*default=*/"16",
           "Data tile size">
  ];
}

def VectorizeMem : FunctionPass<"pxa-vectorize-mem"> {
  let summary = "Vectorize load and reduce assign ops";
  let constructor = "pmlc::dialect::pxa::createVectorizeMemPass()";
  let dependentDialects = ["mlir::vector::VectorDialect"];
}

#endif // __PMLC_DIALECT_PXA_PASSES__
