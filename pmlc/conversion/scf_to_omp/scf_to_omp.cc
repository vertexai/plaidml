// Copyright 2020, Intel Corporation

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/scf_to_omp/pass_detail.h"
#include "pmlc/conversion/scf_to_omp/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

#include "mlir/Support/DebugStringHelper.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::scf_to_omp {

namespace {

const char *kThreadFuncName = "plaidml_rt_thread_num";

LogicalResult convertSCPParallel(scf::ParallelOp op) {
  IVLOG(2, "convertSCPParallel");
  OpBuilder builder(op);
  auto loc = op.getLoc();

  // Extract the range of each parallel index, requiring that all indexes are
  // normalized (lower bound of zero, step size of 1).
  int64_t numThreads = 1;
  SmallVector<int64_t, 4> ranges;
  for (size_t i = 0; i < op.lowerBound().size(); i++) {
    APInt lowerAP, upperAP, stepAP;
    if (!matchPattern(op.lowerBound()[i], m_ConstantInt(&lowerAP)) ||
        !matchPattern(op.upperBound()[i], m_ConstantInt(&upperAP)) ||
        !matchPattern(op.step()[i], m_ConstantInt(&stepAP))) {
      IVLOG(2, "Unable to get constant bounds in scf::ParallelOp");
      op.emitError("scf.parallel bounds must be constant");
      return failure();
    }
    // Verify everthing is normalized.
    uint64_t lower = lowerAP.getLimitedValue();
    uint64_t upper = upperAP.getLimitedValue();
    uint64_t step = stepAP.getLimitedValue();
    if (lower != 0 || step != 1) {
      op.emitError("Non-normalized loops not supported");
      return failure();
    }
    // Add range to array + update number of threads
    ranges.push_back(upper);
    numThreads *= upper;
  }

  // Emit a constant of the number of threads
  auto numThreadsVal = builder.create<ConstantIndexOp>(loc, numThreads);

  auto clauseAttr =
      StringAttr::get(builder.getContext(),
                      stringifyClauseDefault(omp::ClauseDefault::defshared));

  // Create the omp parallel fork/join region
  auto parallelOp =
      builder.create<omp::ParallelOp>(loc,
                                      /*if_expr_va=*/Value(),             //
                                      /*num_threads_var=*/numThreadsVal,  //
                                      /*default_val=*/clauseAttr,         //
                                      /*private_vars=*/ValueRange(),      //
                                      /*firstprivate_vars=*/ValueRange(), //
                                      /*shared_vars=*/ValueRange(),       //
                                      /*copyin_vars=*/ValueRange(),       //
                                      /*allocate_vars=*/ValueRange(),     //
                                      /*allocators_vars=*/ValueRange(),   //
                                      /*proc_bind_val=*/StringAttr());

  // create a terminator (representing the join operation)
  builder.createBlock(&parallelOp.getRegion());
  auto *block = &parallelOp.getRegion().back();
  builder.setInsertionPointToStart(block);
  auto termOp = builder.create<omp::TerminatorOp>(loc);

  // copy the body of the scf loop into the parallel region
  auto &insts = op.getBody()->getOperations();
  block->getOperations().splice(Block::iterator(termOp), insts, insts.begin(),
                                std::prev(insts.end()));

  // generate a call to omp_get_thread_num() at the start of the block
  builder.setInsertionPointToStart(block);
  SmallVector<Value, 0> emptyOperands;
  auto indexType = builder.getIndexType();
  auto callOp = builder.create<CallOp>(
      loc, ArrayRef<Type>{indexType}, builder.getSymbolRefAttr(kThreadFuncName),
      emptyOperands);
  Value tid = callOp.getResult(0);

  // Extract the components of the threadID and replace indexes
  for (size_t i = 0; i < ranges.size(); i++) {
    Value range = builder.create<ConstantIndexOp>(loc, ranges[i]);
    Value local = tid;
    if (i != ranges.size() - 1) {
      local = builder.create<UnsignedRemIOp>(loc, tid, range);
      tid = builder.create<UnsignedDivIOp>(loc, tid, range);
    }
    Value iv = op.getInductionVars()[i];
    iv.replaceAllUsesWith(local);
  }

  // make sure the module contains a declaration for omp_get_thread_num
  auto module = op->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol(kThreadFuncName)) {
    OpBuilder subBuilder(module.getBody()->getTerminator());
    subBuilder
        .create<FuncOp>(module.getLoc(), kThreadFuncName,
                        FunctionType::get(subBuilder.getContext(), {},
                                          ArrayRef<Type>{indexType}))
        .setPrivate();
  }

  // Delete old loop + return
  op.erase();
  return success();
}

/// A pass converting SCF parallel loop into the OpenMP dialect.
struct LowerSCFToOpenMPPass
    : public LowerSCFToOpenMPBase<LowerSCFToOpenMPPass> {
  // Run the dialect converter on the module.
  void runOnOperation() final {
    Operation *op = getOperation();
    op->walk([&](scf::ParallelOp op) {
      if (failed(convertSCPParallel(op))) {
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createLowerSCFToOpenMPPass() {
  return std::make_unique<LowerSCFToOpenMPPass>();
}

} // namespace pmlc::conversion::scf_to_omp
