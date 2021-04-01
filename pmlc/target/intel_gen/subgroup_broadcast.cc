// Copyright 2020 Intel Corporation

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/target/intel_gen/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::intel_gen {

namespace {

static bool isBlockOpTypeSupported(Type elementType) {
  if (auto vectorType = elementType.dyn_cast<VectorType>())
    elementType = vectorType.getElementType();
  return elementType.isInteger(16) || elementType.isInteger(32) ||
         elementType.isF16() || elementType.isF32();
}

class DevectorizeImpl {
private:
  scf::ParallelOp loop;
  unsigned vectorSize;
  Value sid;
  bool useBlockOps;

public:
  DevectorizeImpl(scf::ParallelOp loop, unsigned vectorSize, bool useBlockOps)
      : loop(loop), vectorSize(vectorSize), useBlockOps(useBlockOps) {}

  bool isVectorTypeValid(Type type) {
    if (auto vectorType = type.dyn_cast<VectorType>())
      return vectorType.getRank() == 1 &&
             vectorType.getDimSize(0) % vectorSize == 0;
    return false;
  }

  static bool isAllocOfVector(Operation *op) {
    if (auto allocOp = dyn_cast_or_null<AllocOp>(op)) {
      return allocOp.getType().getElementType().isa<VectorType>();
    }
    return false;
  }

  LogicalResult devectorizeVectorOp(Operation *op) {
    IVLOG(3, "devectorizeVectorOp: " << debugString(*op));

    OpBuilder builder(op);
    // Skip on std.extractelement as it will be removed with broadcast op
    // transformation
    if (isa<vector::ExtractElementOp>(op))
      return success();

    // Replace operands if they are still vectorized. This can happen for
    // constant ops.
    for (OpOperand &operand : op->getOpOperands()) {
      if (auto vectorType = operand.get().getType().dyn_cast<VectorType>()) {
        if (!isVectorTypeValid(vectorType))
          return failure();
        DenseElementsAttr attr;
        if (!matchPattern(operand.get(), m_Constant(&attr)) && !attr.isSplat())
          return failure();
        auto constantOp = builder.create<ConstantOp>(
            op->getLoc(), vectorType.getElementType(), attr.getSplatValue());
        operand.set(constantOp);
      }
    }

    // Update the result type if it is a vector, operands types should have
    // already been updated before.
    for (auto result : op->getResults()) {
      if (auto vectorType = result.getType().dyn_cast<VectorType>()) {
        if (!isVectorTypeValid(vectorType))
          return failure();
        result.setType(vectorType.getElementType());
      }
    }
    return success();
  }

  LogicalResult devectorizeVectorLoad(vector::LoadOp op) {
    IVLOG(3, "devectorizeVectorLoad: " << debugString(op));

    // Note: it's possible that during this pass, some values might temporarily
    // be scalar types, so avoid using getVectorType(), which causes an
    // assertion.
    if (!isVectorTypeValid(op.result().getType()) || op.indices().size() < 1)
      return failure();

    OpBuilder builder(op);
    // Add sid to lowest index
    SmallVector<Value, 4> idxs = op.indices();
    // TODO: Current HW supports only block read\write on global mem scope.
    // This can change in the future so probably need to be better handled with
    // HW specific parameters
    scf::ParallelOp actualBlock = loop->getParentOfType<scf::ParallelOp>();
    bool invalidMemScope = !actualBlock.isDefinedOutsideOfLoop(op.base());
    VectorType vectorType = op.getVectorType();

    // TODO: Based on the HW caps we should accept these for certain data types
    // Right now we accept i16/fp16 and i32/fp32 for the block read extensions
    // TODO: add additional requirements like mem alignment
    if (useBlockOps && !invalidMemScope && isBlockOpTypeSupported(vectorType)) {
      // Case1: vector size is a factor of subgroup size, in this case
      // we use vector size divided by subgroup size as output from
      // SubgroupBlockReadINTELOp
      if (vectorType.getDimSize(0) > vectorSize) {
        // Change vector dimension, we already checked earlier that dimsize %
        // vecsize == 0
        auto newVectorSize = vectorType.getDimSize(0) / vectorSize;
        SmallVector<int64_t, 1> newShape;
        newShape.push_back(newVectorSize);
        auto newMemrefType =
            VectorType::get(newShape, vectorType.getElementType());
        auto newBlockReadOp =
            builder.create<dialect::stdx::SubgroupBlockReadINTELOp>(
                op.getLoc(), newMemrefType, op.base(), idxs);

        IVLOG(3, "Load Op: " << debugString(*newBlockReadOp));
        op.replaceAllUsesWith(newBlockReadOp.getResult());
        // Case2: output from SubgroupBlockReadINTELOp is scalar,
        // In this case it is also needed to devectorize the operands afterwards
      } else {
        auto newBlockReadOp =
            builder.create<dialect::stdx::SubgroupBlockReadINTELOp>(
                op.getLoc(), op.base(), idxs);
        if (failed(devectorizeVectorOp(newBlockReadOp)))
          return failure();
        op.replaceAllUsesWith(newBlockReadOp.getResult());
        IVLOG(3, "Load Op: " << debugString(*newBlockReadOp));
      }
      // Case3: Used buffer was allocated inside the kernel, no block reads
      // allowed. Check if element type of the allocated memory is vector, if so
      // then it is needed to use LoadOp with orginal indices
    } else if (isAllocOfVector(op.base().getDefiningOp()) &&
               vectorType.getDimSize(0) != vectorSize) {
      auto newLoadOp = builder.create<LoadOp>(op.getLoc(), op.base(), idxs);
      IVLOG(3, "Block read Op: " << debugString(*newLoadOp));
      op.replaceAllUsesWith(newLoadOp.getResult());
      // Case4: No block reads or vectors, use default devectorization
    } else {
      idxs.back() = builder.create<AddIOp>(op.getLoc(), idxs.back(), sid);
      auto newLoadOp = builder.create<LoadOp>(op.getLoc(), op.base(), idxs);
      if (failed(devectorizeVectorOp(newLoadOp)))
        return failure();
      op.replaceAllUsesWith(newLoadOp.getResult());
      IVLOG(3, "Load Op: " << debugString(*newLoadOp));
    }
    op.erase();
    return success();
  }

  LogicalResult devectorizeVectorStore(vector::StoreOp op) {
    IVLOG(3, "devectorizeVectorStore: " << debugString(op));

    if (op.indices().size() < 1)
      return failure();

    OpBuilder builder(op);
    // Add sid to lowest index
    SmallVector<Value, 4> idxs = op.indices();
    // TODO: Current HW supports only block read\write on global mem scope.
    // This can change in the future so probably need to be better handled with
    // HW specific parameters
    scf::ParallelOp actualBlock = loop->getParentOfType<scf::ParallelOp>();
    bool invalidMemScope = !actualBlock.isDefinedOutsideOfLoop(op.base());

    // TODO: Based on the HW caps we should accept these for certain data types
    // Right now we accept i16/fp16 and i32/fp32 for the block read extensions
    // TODO: add additional requirements like mem alignment
    // Case1: Use block write
    if (useBlockOps && !invalidMemScope &&
        isBlockOpTypeSupported(op.valueToStore().getType())) {
      auto newBlockWriteOp =
          builder.create<dialect::stdx::SubgroupBlockWriteINTELOp>(
              op.getLoc(), op.valueToStore(), op.base(), idxs);
      if (failed(devectorizeVectorOp(newBlockWriteOp)))
        return failure();
      IVLOG(3, "Block Write Op: " << debugString(*newBlockWriteOp));
      op.erase();
      // Case2: No block writes or vectors, use default devectorization
    } else {
      idxs.back() = builder.create<AddIOp>(op.getLoc(), idxs.back(), sid);
      auto newStoreOp = builder.create<StoreOp>(op.getLoc(), op.valueToStore(),
                                                op.base(), idxs);
      if (failed(devectorizeVectorOp(newStoreOp)))
        return failure();
      IVLOG(3, "Block Write Op: " << debugString(*newStoreOp));
      op.erase();
    }
    return success();
  }

  LogicalResult devectorizeAlloc(AllocOp op) {
    IVLOG(3, "devectorizeAlloc: " << debugString(op));

    MemRefType oldMemRefType = op.getType();
    auto vecType = oldMemRefType.getElementType().dyn_cast<VectorType>();
    if (!vecType) {
      // Ignore non-vector allocs
      return success();
    }
    ArrayRef<int64_t> shape = vecType.getShape();
    if (shape.size() != 1) {
      // Vector allocations must match the subgroup size
      return failure();
    }

    // Check if element type is vector, it should be of size multiplication
    // of subgroup size or equal. If it is multiplication then divide it by
    // subgroup size to create new Alloc
    Type newElementType = vecType.getElementType();
    if (shape[0] > vectorSize) {
      auto newVectorSize = shape[0] / vectorSize;
      newElementType =
          VectorType::get({newVectorSize}, vecType.getElementType());
    } else if (shape[0] != vectorSize) {
      return failure();
    }

    MemRefType newMemRefType =
        MemRefType::Builder(oldMemRefType).setElementType(newElementType);
    op.getResult().setType(newMemRefType);
    IVLOG(3, "Alloc Op: " << debugString(*op));
    return success();
  }

  LogicalResult devectorizeExtractMap(vector::ExtractMapOp op) {
    IVLOG(3, "devectorizeExtractMap: " << debugString(op));

    OpBuilder builder(op);
    Value idVal = op.ids().front();

    // It is needed to use i32 type for extract element index. Use cast in case
    // it comes from index
    if (idVal.getType().isa<IndexType>()) {
      auto indexCast = builder.create<IndexCastOp>(op.getLoc(), idVal,
                                                   builder.getIntegerType(32));
      idVal = indexCast.getResult();
    }

    auto newExtractOp = builder.create<vector::ExtractElementOp>(
        op.getLoc(), op.vector(), idVal);
    op.replaceAllUsesWith(newExtractOp.getResult());
    op.erase();

    return success();
  }

  LogicalResult devectorizeInsertMap(vector::InsertMapOp op) {
    IVLOG(3, "devectorizeInsertMap: " << debugString(op));

    OpBuilder builder(op);
    assert(op.ids().size() == 1);
    Value idVal = op.ids().front();

    // It is needed to use i32 type for extract element index. Use cast in case
    // it comes from index
    if (idVal.getType().isa<IndexType>()) {
      auto indexCast = builder.create<IndexCastOp>(op.getLoc(), idVal,
                                                   builder.getIntegerType(32));
      idVal = indexCast.getResult();
    }

    // Assume that the user of insert_map is vector::StoreOp,
    // so the whole structure was modelled in the vectorizeMemPass
    auto &insertMapUser = *op.getResult().use_begin();
    auto vectorStoreOp = dyn_cast<vector::StoreOp>(insertMapUser.getOwner());
    if (!vectorStoreOp)
      return failure();
    auto vecType = op.getResultType();
    auto newVectorSize = vecType.getShape()[0] / vectorSize;
    vecType = VectorType::get({newVectorSize}, vecType.getElementType());

    vectorStoreOp.base().setType(MemRefType::get({1}, vecType));

    // Read the vector first from temporary memory, and then do an
    // element insert.
    auto newReadOp = builder.create<LoadOp>(op.getLoc(), vectorStoreOp.base(),
                                            vectorStoreOp.indices());

    auto newInsertOp = builder.create<vector::InsertElementOp>(
        op.getLoc(), op.vector(), newReadOp.getResult(), idVal);
    op.replaceAllUsesWith(newInsertOp.getResult());
    op.erase();

    return success();
  }

  LogicalResult devectorizeBroadcast(vector::BroadcastOp op) {
    IVLOG(3, "devectorizeBroadcast: " << debugString(op));

    // Note: it's possible that during this pass, some values might temporarily
    // be scalar types, so avoid using getVectorType(), which causes an
    // assertion.
    if (!isVectorTypeValid(op.vector().getType()))
      return failure();

    // If the broadcast result comes from extractelement then replace it with
    // stdx.subgroup_broadcast, otherwise it is removed.
    if (auto extractElementOp = dyn_cast_or_null<vector::ExtractElementOp>(
            op.source().getDefiningOp())) {
      Value vector = extractElementOp.vector();
      OpBuilder builder(op);
      auto newIdx = builder.create<IndexCastOp>(
          op.getLoc(), extractElementOp.position(), builder.getIndexType());
      auto newBroadcast = builder.create<dialect::stdx::SubgroupBroadcastOp>(
          op.getLoc(), vector.getType(), vector, newIdx);
      op.replaceAllUsesWith(newBroadcast.getResult());
      extractElementOp.replaceAllUsesWith(newBroadcast.getResult());
      extractElementOp.erase();
      op.erase();
    } else {
      op.replaceAllUsesWith(op.source());
      op.erase();
    }
    return success();
  }

  LogicalResult devectorizeOperation(Operation *op) {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<vector::LoadOp>(
            [this](vector::LoadOp op) { return devectorizeVectorLoad(op); })
        .Case<vector::StoreOp>(
            [this](vector::StoreOp op) { return devectorizeVectorStore(op); })
        .Case<vector::BroadcastOp>(
            [this](vector::BroadcastOp op) { return devectorizeBroadcast(op); })
        .Case<vector::ExtractMapOp>([this](vector::ExtractMapOp op) {
          return devectorizeExtractMap(op);
        })
        .Case<vector::InsertMapOp>(
            [this](vector::InsertMapOp op) { return devectorizeInsertMap(op); })
        .Case<scf::ForOp>([this](scf::ForOp op) {
          for (auto &innerOp : llvm::make_early_inc_range(op.getOps())) {
            if (failed(devectorizeOperation(&innerOp))) {
              return failure();
            }
          }
          return success();
        })
        .Case<AllocOp>([this](AllocOp op) { return devectorizeAlloc(op); })
        .Default([this](Operation *op) { return devectorizeVectorOp(op); });
  }

  LogicalResult devectorize() {
    Block *body = loop.getBody();

    // Rewrite the loop with new dimension added
    OpBuilder builder(loop);

    // Create proper lower, upper bounds and steps for the updated
    // scf.parallel op and create the actual op
    SmallVector<Value, 4> newLowerBounds = loop.lowerBound();
    SmallVector<Value, 4> newUpperBounds = loop.upperBound();
    SmallVector<Value, 4> newStepsBounds = loop.step();
    auto loc = loop.getLoc();

    // Add an initial index for the sid, with a range of vectorSize.
    newLowerBounds.insert(newLowerBounds.begin(),
                          builder.create<ConstantIndexOp>(loc, 0));
    newUpperBounds.insert(newUpperBounds.begin(),
                          builder.create<ConstantIndexOp>(loc, vectorSize));
    newStepsBounds.insert(newStepsBounds.begin(),
                          builder.create<ConstantIndexOp>(loc, 1));

    auto newLoop = builder.create<scf::ParallelOp>(
        loop.getLoc(), newLowerBounds, newUpperBounds, newStepsBounds);
    Block *newBody = newLoop.getBody();

    // Splice across interior + erase originals
    auto &oldBodyOps = body->getOperations();
    auto &newBodyOps = newBody->getOperations();
    newBodyOps.splice(std::prev(newBodyOps.end()), oldBodyOps,
                      oldBodyOps.begin(), std::prev(oldBodyOps.end()));

    // Replace block argsA
    for (size_t i = 0; i < body->getNumArguments(); i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i + 1));
    }
    sid = newBody->getArgument(0);

    // Do the Ops transform
    for (auto &op : llvm::make_early_inc_range(newBody->getOperations())) {
      if (failed(devectorizeOperation(&op))) {
        return failure();
      }
    }

    // Reset GPU thread mappings
    SmallVector<gpu::ParallelLoopDimMapping, 8> mappings;
    auto proc = gpu::Processor::ThreadX;
    for (unsigned i = 0; i < newBody->getNumArguments(); i++) {
      mappings.push_back(getParallelLoopDimMappingAttr(
          proc, builder.getDimIdentityMap(), builder.getDimIdentityMap()));
      proc = static_cast<gpu::Processor>(static_cast<int>(proc) + 1);
    }
    if (failed(setMappingAttr(newLoop, mappings)))
      return failure();
    loop.erase();
    return success();
  }
};

struct SubgroupBroadcastPass
    : public SubgroupBroadcastBase<SubgroupBroadcastPass> {
  SubgroupBroadcastPass() = default;
  explicit SubgroupBroadcastPass(bool useBlockOps) {
    this->useBlockOps = useBlockOps;
  }
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](scf::ParallelOp op) {
      int64_t subgroupSize = getIntegerTag(op, subgroupSizeTag(), 1);
      if (hasUnitTag(op, gpuThreadTag())) {
        // Make sure that gpuBlock loop is present
        auto gpuBlockOp = op->getParentOfType<scf::ParallelOp>();
        if (gpuBlockOp && hasUnitTag(gpuBlockOp, gpuBlockTag())) {
          if (useBlockOps.getValue()) {
            IVLOG(3, "SubgroupBroadcastPass: Block ops enabled")
          } else {
            IVLOG(3, "SubgroupBroadcastPass: Block ops disabled")
          }
          DevectorizeImpl impl(op, subgroupSize, useBlockOps.getValue());
          if (failed(impl.devectorize())) {
            signalPassFailure();
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createSubgroupBroadcastPass() {
  return std::make_unique<SubgroupBroadcastPass>();
}

std::unique_ptr<Pass> createSubgroupBroadcastPass(bool useBlockOps) {
  return std::make_unique<SubgroupBroadcastPass>(useBlockOps);
}

} // namespace pmlc::target::intel_gen
