// Copyright 2020, Intel Corporation
#include "pmlc/dialect/pxa/analysis/affine_constraints.h"

#include <algorithm>
#include <functional>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/dialect/pxa/ir/interfaces.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::dialect::pxa {

mlir::LogicalResult
addAffineParallelIVDomain(mlir::AffineParallelOp parallelOp, unsigned idx,
                          mlir::FlatAffineConstraints &constraints) {
  unsigned pos;
  if (!constraints.findId(parallelOp.getIVs()[idx], &pos)) {
    assert(false && "Value not found");
    return mlir::failure();
  }

  mlir::AffineValueMap lowerMap = parallelOp.getLowerBoundsValueMap();
  mlir::AffineValueMap upperMap = parallelOp.getUpperBoundsValueMap();
  if (auto constLower =
          lowerMap.getResult(idx).dyn_cast<mlir::AffineConstantExpr>())
    constraints.addConstantLowerBound(pos, constLower.getValue());
  if (auto constUpper =
          upperMap.getResult(idx).dyn_cast<mlir::AffineConstantExpr>())
    constraints.addConstantUpperBound(pos, constUpper.getValue() - 1);
  return mlir::success();
}

mlir::Optional<int64_t>
getLowerBound(mlir::AffineExpr expr, mlir::FlatAffineConstraints &constraints) {
  return getLowerUpperBounds(expr, constraints).first;
}

mlir::Optional<int64_t>
getUpperBound(mlir::AffineExpr expr, mlir::FlatAffineConstraints &constraints) {
  return getLowerUpperBounds(expr, constraints).second;
}

namespace {

template <typename T, typename BinOp>
auto applyOptional(const mlir::Optional<T> &a, const mlir::Optional<T> &b,
                   BinOp &&op)
    -> mlir::Optional<decltype(op(a.getValue(), b.getValue()))> {
  if (a.hasValue() && b.hasValue())
    return op(a.getValue(), b.getValue());
  return llvm::None;
}

template <typename T>
struct minimum {
  T operator()(const T &lhs, const T &rhs) { return std::min(lhs, rhs); }
};

template <typename T>
struct maximum {
  T operator()(const T &lhs, const T &rhs) { return std::max(lhs, rhs); }
};

} // namespace

std::pair<mlir::Optional<int64_t>, mlir::Optional<int64_t>>
getLowerUpperBounds(mlir::AffineExpr expr,
                    mlir::FlatAffineConstraints &constraints) {
  if (auto constExpr = expr.dyn_cast<mlir::AffineConstantExpr>())
    return std::pair<mlir::Optional<int64_t>, mlir::Optional<int64_t>>(
        constExpr.getValue(), constExpr.getValue());
  if (auto dimExpr = expr.dyn_cast<mlir::AffineDimExpr>()) {
    unsigned pos = dimExpr.getPosition();
    mlir::Optional<int64_t> lower = constraints.getConstantLowerBound(pos);
    mlir::Optional<int64_t> upper = constraints.getConstantUpperBound(pos);
    return std::make_pair(lower, upper);
  }
  if (auto binExpr = expr.dyn_cast<mlir::AffineBinaryOpExpr>()) {
    auto crossEvaluate = [&](mlir::AffineExpr lhs, mlir::AffineExpr rhs,
                             auto operation) {
      auto lhsBounds = getLowerUpperBounds(binExpr.getLHS(), constraints);
      auto rhsBounds = getLowerUpperBounds(binExpr.getRHS(), constraints);

      mlir::Optional<int64_t> lowerLower =
          applyOptional(lhsBounds.first, rhsBounds.first, operation);
      mlir::Optional<int64_t> lowerUpper =
          applyOptional(lhsBounds.first, rhsBounds.second, operation);
      mlir::Optional<int64_t> upperLower =
          applyOptional(lhsBounds.second, rhsBounds.first, operation);
      mlir::Optional<int64_t> upperUpper =
          applyOptional(lhsBounds.second, rhsBounds.second, operation);

      mlir::Optional<int64_t> lower, upper;
      lower = applyOptional(lowerLower, lowerUpper, minimum<int64_t>());
      lower = applyOptional(lower, upperLower, minimum<int64_t>());
      lower = applyOptional(lower, upperUpper, minimum<int64_t>());
      upper = applyOptional(lowerLower, lowerUpper, maximum<int64_t>());
      upper = applyOptional(upper, upperLower, maximum<int64_t>());
      upper = applyOptional(upper, upperUpper, maximum<int64_t>());

      return std::make_pair(lower, upper);
    };

    switch (binExpr.getKind()) {
    case mlir::AffineExprKind::Add: {
      auto lhsBounds = getLowerUpperBounds(binExpr.getLHS(), constraints);
      auto rhsBounds = getLowerUpperBounds(binExpr.getRHS(), constraints);
      mlir::Optional<int64_t> lower =
          applyOptional(lhsBounds.first, rhsBounds.first, std::plus<int64_t>());
      mlir::Optional<int64_t> upper = applyOptional(
          lhsBounds.second, rhsBounds.second, std::plus<int64_t>());
      return std::make_pair(lower, upper);
    }
    case mlir::AffineExprKind::Mul: {
      return crossEvaluate(binExpr.getLHS(), binExpr.getRHS(),
                           std::multiplies<int64_t>());
    }
    case mlir::AffineExprKind::Mod: {
      auto rhsBounds = getLowerUpperBounds(binExpr.getRHS(), constraints);
      // TODO: Take lhs bounds into account.
      mlir::Optional<int64_t> lower = 0;
      mlir::Optional<int64_t> upper =
          rhsBounds.second.map([](int64_t val) { return val - 1; });
      return std::make_pair(lower, upper);
    }
    case mlir::AffineExprKind::FloorDiv: {
      return crossEvaluate(binExpr.getLHS(), binExpr.getRHS(),
                           std::divides<int64_t>());
    }
    case mlir::AffineExprKind::CeilDiv: {
      auto ceilDivides = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
      return crossEvaluate(binExpr.getLHS(), binExpr.getRHS(), ceilDivides);
    }
    default:
      return std::pair<mlir::Optional<int64_t>, mlir::Optional<int64_t>>(
          llvm::None, llvm::None);
    }
  }
  return std::pair<mlir::Optional<int64_t>, mlir::Optional<int64_t>>(
      llvm::None, llvm::None);
}

mlir::AffineExpr
simplifyExprWithConstraints(mlir::AffineExpr expr,
                            mlir::FlatAffineConstraints &constraints) {
  auto binExpr = expr.dyn_cast<mlir::AffineBinaryOpExpr>();
  if (!binExpr)
    return expr;
  mlir::AffineExpr lhs = binExpr.getLHS();
  mlir::AffineExpr rhs = binExpr.getRHS();
  auto lhsLUBounds = getLowerUpperBounds(lhs, constraints);
  auto rhsLUBounds = getLowerUpperBounds(rhs, constraints);
  mlir::Optional<bool> lhsLessThanRhs = applyOptional(
      lhsLUBounds.second, rhsLUBounds.first, std::less<int64_t>());
  if (mlir::AffineExprKind::Mod == expr.getKind() &&
      lhsLessThanRhs.getValueOr(false))
    return simplifyExprWithConstraints(lhs, constraints);
  if (mlir::AffineExprKind::FloorDiv == expr.getKind() &&
      lhsLessThanRhs.getValueOr(false))
    return mlir::getAffineConstantExpr(0, expr.getContext());
  mlir::AffineExpr lhsSimplified =
      simplifyExprWithConstraints(lhs, constraints);
  mlir::AffineExpr rhsSimplified =
      simplifyExprWithConstraints(rhs, constraints);
  return mlir::getAffineBinaryOpExpr(expr.getKind(), lhsSimplified,
                                     rhsSimplified);
}

mlir::AffineMap
simplifyMapWithConstraints(mlir::AffineMap map,
                           mlir::FlatAffineConstraints &constraints) {
  mlir::SmallVector<mlir::AffineExpr, 6> simplifiedExprs;
  for (mlir::AffineExpr expr : map.getResults())
    simplifiedExprs.push_back(simplifyExprWithConstraints(expr, constraints));
  return mlir::AffineMap::get(map.getNumInputs(), map.getNumSymbols(),
                              simplifiedExprs, map.getContext());
}

mlir::AffineValueMap simplifyMapWithConstraints(mlir::AffineValueMap valueMap) {
  mlir::AffineMap affineMap = valueMap.getAffineMap();
  mlir::ArrayRef<mlir::Value> operandsRef = valueMap.getOperands();
  mlir::SmallVector<mlir::Value, 8> operands(operandsRef.begin(),
                                             operandsRef.end());
  mlir::fullyComposeAffineMapAndOperands(&affineMap, &operands);
  mlir::FlatAffineConstraints constraints =
      gatherAffineMapConstraints(mlir::AffineValueMap(affineMap, operands));
  affineMap = simplifyMapWithConstraints(affineMap, constraints);
  return mlir::AffineValueMap(affineMap, operands);
}

mlir::FlatAffineConstraints
gatherAffineMapConstraints(mlir::AffineValueMap map) {
  auto constraints = mlir::FlatAffineConstraints::getUniverse(
      map.getNumDims(), map.getNumSymbols());
  constraints.setIdValues(0, map.getOperands().size(), map.getOperands());
  for (mlir::Value operandVal : map.getOperands()) {
    auto arg = operandVal.dyn_cast<mlir::BlockArgument>();
    mlir::Operation *parent = arg.getOwner()->getParentOp();
    if (auto parallelOp = mlir::dyn_cast<mlir::AffineParallelOp>(parent)) {
      // TODO: check LogicalResult
      (void)addAffineParallelIVDomain(parallelOp, arg.getArgNumber(),
                                      constraints);
    }
  }
  return constraints;
}

} // namespace pmlc::dialect::pxa
