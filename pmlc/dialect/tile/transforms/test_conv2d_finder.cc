// Copyright 2021, Intel Corporation

#include "pmlc/dialect/tile/analysis/conv2d_finder.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"

namespace pmlc::dialect::tile {

using namespace mlir; // NOLINT

namespace {

struct TestConv2dFinderPass : public Conv2dFinderBase<TestConv2dFinderPass> {
  void runOnFunction() final;
};

void TestConv2dFinderPass::runOnFunction() {
  auto func = getFunction();
  llvm::errs() << "Testing : " << getFunction().getName() << "\n";
  func.walk([&](ContractionOp op) {
    Conv2dFinder conv2dFinder(op);
    if (conv2dFinder.isContractOpConv2d()) {
      llvm::errs() << "You say you want a convolution.\n";

      llvm::errs() << "paddings =";
      auto paddings = conv2dFinder.getPaddings();
      for (size_t i = 0; i < paddings.size(); ++i) {
        llvm::errs() << " " << paddings[i];
      }
      llvm::errs() << "\n";

      llvm::errs() << "strides =";
      auto strides = conv2dFinder.getStrides();
      for (size_t i = 0; i < strides.size(); ++i) {
        llvm::errs() << " " << strides[i];
      }
      llvm::errs() << "\n";

      llvm::errs() << "dilations =";
      auto dilations = conv2dFinder.getDilations();
      for (size_t i = 0; i < dilations.size(); ++i) {
        llvm::errs() << " " << dilations[i];
      }
      llvm::errs() << "\n";

    } else {
      llvm::errs() << "Well, you know, we all want to change the world.\n";
      llvm::errs() << conv2dFinder.getReason() << "\n";
    }
  });
}

} // namespace

std::unique_ptr<Pass> createTestConv2dFinderPass() {
  return std::make_unique<TestConv2dFinderPass>();
}

} // namespace pmlc::dialect::tile
