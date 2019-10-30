// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/builder.h"

#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"

#include "base/util/env.h"
#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/dialect.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/tile/dialect.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/dialect/tile/program.h"
#include "pmlc/util/slice.h"
#include "tile/base/shape.h"

namespace pmlc::dialect::tile {

using eltwise::ScalarConstantOp;
using eltwise::ScalarType;
using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::StringRef;
using mlir::Block;
using mlir::BlockAndValueMapping;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::UnknownLoc;
using mlir::Value;

struct DomainInfo {
  BlockAndValueMapping mapping;
};

struct UniqueNamer {
  std::set<std::string> names;

  std::string get(StringRef name) {
    auto next = name.str();
    auto [it, isUnique] = names.insert(next);  // NOLINT(whitespace/braces)
    for (unsigned i = 0; !isUnique; i++) {
      next = llvm::formatv("{0}_{1}", name, i).str();
      std::tie(it, isUnique) = names.insert(next);
    }
    return next;
  }
};

struct TileBuilder::Impl {
  MLIRContext context;
  ModuleOp module;
  OpBuilder builder;
  std::map<AffineDomainOp, DomainInfo> domains;

  Impl()
      : module(ModuleOp::create(UnknownLoc::get(&context))),  //
        builder(module.getBody()) {
    builder.setInsertionPointToStart(module.getBody());
  }

  mlir::Type ComputeElementType(llvm::ArrayRef<mlir::Type> types) {
    DataType ret = DataType::INVALID;
    for (auto type : types) {
      auto tensorType = type.cast<ShapedType>();
      auto dtype = tensorType.getElementType().cast<ScalarType>().type();
      ret = CommonSupertype(ret, dtype);
    }
    return builder.getType<ScalarType>(ret);
  }

  const mlir::AbstractOperation* lookupOperation(StringRef op) {
    auto opName = eltwise::Dialect::getCanonicalOpName(op);
    auto abstractOp = mlir::AbstractOperation::lookup(opName, &context);
    if (!abstractOp) {
      opName = tile::Dialect::getCanonicalOpName(op);
      abstractOp = mlir::AbstractOperation::lookup(opName, &context);
      if (!abstractOp) {
        throw std::runtime_error("Unknown op: " + op.str());
      }
    }
    return abstractOp;
  }

  using CreateOpFunc = std::function<void(OpBuilder, BlockAndValueMapping*)>;

  Operation* MakeContraction(             //
      llvm::ArrayRef<mlir::Value*> srcs,  //
      mlir::Value* sink,                  //
      mlir::Value* sizes,                 //
      CreateOpFunc fn) {
    IVLOG(5, "TileBuilder::Impl::MakeContraction>");
    IVLOG(5, mlir::debugString(module));
    // Compute the sink shape of the contraction
    llvm::SmallVector<mlir::Type, 3> types;
    for (auto src : srcs) {
      IVLOG(6, "  src: " << mlir::debugString(*src));
      auto map_op = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
      types.push_back(map_op.tensor()->getType());
    }
    IVLOG(6, "  sink: " << mlir::debugString(*sink));
    IVLOG(6, "  sizes: " << mlir::debugString(*sizes));
    auto elementType = ComputeElementType(types);
    auto size_map_op = llvm::cast<AffineSizeMapOp>(sizes->getDefiningOp());
    llvm::SmallVector<Value*, 4> size_map_sizes(size_map_op.sizes());
    auto shape = eltwise::ComputeShape(size_map_sizes);
    auto tensorType = builder.getTensorType(shape, elementType);
    auto domain = builder.create<AffineDomainOp>(builder.getUnknownLoc(), tensorType);
    auto& info = domains[domain];
    auto body = new Block();
    domain.body().push_back(body);
    llvm::SetVector<mlir::Value*> values;
    values.insert(srcs.begin(), srcs.end());
    values.insert(sink);
    values.insert(sizes);
    auto slice = util::getBackwardSlice(values, false, [](Value* value) {  //
      return value->getType().isa<IndexType>();
    });
    // Find and replace each AffineIndexOp with a BlockArgument of the domain op
    SmallVector<Attribute, 8> idxNames;
    std::queue<mlir::Value*> worklist;
    for (auto value : slice) {
      auto op = value->getDefiningOp();
      if (auto idx_op = llvm::dyn_cast<AffineIndexOp>(op)) {
        auto arg = body->addArgument(idx_op.getType());
        info.mapping.map(value, arg);
        worklist.push(value);
        if (auto attr = idx_op.getAttrOfType<StringAttr>("name")) {
          idxNames.emplace_back(attr);
        } else {
          auto name = llvm::formatv("x{0}", arg->getArgNumber());
          idxNames.emplace_back(builder.getStringAttr(name.str()));
        }
      }
    }
    domain.setAttr("idx_names", mlir::ArrayAttr::get(idxNames, &context));
    // Move across only the values/ops that depend on AffineIndexOps
    // First determine the transitive users of AffineIndexOps
    std::set<Value*> belong;
    while (worklist.size()) {
      auto value = worklist.front();
      worklist.pop();
      for (auto user : value->getUsers()) {
        auto user_value = user->getResult(0);
        if (!belong.count(user_value)) {
          belong.insert(user_value);
          worklist.push(user_value);
        }
      }
    }
    // Now move across ops but do so in topologically sorted order
    OpBuilder domain_builder(body);
    for (auto value : slice) {
      auto op = value->getDefiningOp();
      if (belong.count(value) ||                    //
          llvm::isa<AffineSourceIndexMapOp>(op) ||  //
          llvm::isa<AffineSinkIndexMapOp>(op) ||    //
          llvm::isa<AffineSizeMapOp>(op)) {
        auto new_value = domain_builder.clone(*op, info.mapping)->getResult(0);
        info.mapping.map(value, new_value);
      }
    }
    fn(domain_builder, &info.mapping);
    IVLOG(5, mlir::debugString(domain));
    return domain.getOperation();
  }

  template <typename ConOp>
  mlir::Value* MakeUnaryContraction(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes) {
    if (srcs.size() != 1) {
      throw std::runtime_error("Unary contraction op requires 1 operand");
    }
    auto domain = MakeContraction({srcs[0]}, sink, sizes, [&](OpBuilder domain_builder, BlockAndValueMapping* mapping) {
      auto new_src = mapping->lookup(srcs[0]);
      auto new_sink = mapping->lookup(sink);
      auto new_sizes = mapping->lookup(sizes);
      domain_builder.create<ConOp>(builder.getUnknownLoc(), new_sizes, new_src, new_sink);
    });
    return domain->getResult(0);
  }

  template <typename ConOp>
  mlir::Value* MakeBinaryContraction(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes) {
    if (srcs.size() != 2) {
      throw std::runtime_error("Binary contraction op requires 2 operands");
    }
    auto domain =
        MakeContraction({srcs[0], srcs[1]}, sink, sizes, [&](OpBuilder domain_builder, BlockAndValueMapping* mapping) {
          auto new_src1 = mapping->lookup(srcs[0]);
          auto new_src2 = mapping->lookup(srcs[1]);
          auto new_sink = mapping->lookup(sink);
          auto new_sizes = mapping->lookup(sizes);
          domain_builder.create<ConOp>(builder.getUnknownLoc(), new_sizes, new_src1, new_src2, new_sink);
        });
    return domain->getResult(0);
  }

  template <typename ConOp>
  mlir::Value* MakeTernaryContraction(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes) {
    if (srcs.size() != 3) {
      throw std::runtime_error("Ternary contraction op requires 3 operands");
    }
    auto domain = MakeContraction(
        {srcs[0], srcs[1], srcs[2]}, sink, sizes, [&](OpBuilder domain_builder, BlockAndValueMapping* mapping) {
          auto new_src1 = mapping->lookup(srcs[0]);
          auto new_src2 = mapping->lookup(srcs[1]);
          auto new_src3 = mapping->lookup(srcs[2]);
          auto new_sink = mapping->lookup(sink);
          auto new_sizes = mapping->lookup(sizes);
          domain_builder.create<ConOp>(builder.getUnknownLoc(), new_sizes, new_src1, new_src2, new_src3, new_sink);
        });
    return domain->getResult(0);
  }
};

TileBuilder::TileBuilder() : impl(new Impl) {}

TileBuilder::~TileBuilder() = default;

void TileBuilder::Destroy(Value* value) {
  IVLOG(5, "TileBuilder::Destroy> value");
  // TODO: fix memory mgmt issues, once purely MLIR path is complete
  // if (value && value->use_empty()) {
  //   auto op = value->getDefiningOp();
  //   if (op && op->use_empty()) {
  //     op->erase();
  //   }
  // }
}

void TileBuilder::BindTensorDim(unsigned dim, mlir::Value* from, mlir::Value** into) {
  if (!from) {
    throw std::runtime_error("BindTensorDim: from == nullptr");
  }
  IVLOG(5, "TileBuilder::BindTensorDim> from: " << mlir::debugString(*from));
  if (!into) {
    throw std::runtime_error("BindTensorDim: into == nullptr");
  }
  if (*into) {
    IVLOG(6, "into: " << mlir::debugString(**into));
    auto fromType = from->getType().dyn_cast<mlir::RankedTensorType>();
    if (!fromType) {
      throw std::runtime_error("Unexpected type");
    }
    auto fromSize = fromType.getDimSize(dim);
    if (!mlir::ShapedType::isDynamic(fromSize)) {
      auto op = (*into)->getDefiningOp();
      if (!op) {
        throw std::runtime_error("No defining op");
      }
      if (auto const_op = llvm::dyn_cast<AffineConstantOp>(op)) {
        auto attr = const_op.getValue().dyn_cast<IntegerAttr>();
        if (!attr) {
          throw std::runtime_error("Expected IntegerAttr for value of AffineConstantOp");
        }
        IVLOG(6, "dim: " << dim << ", from: " << fromSize << ", into: " << attr.getInt());
        if (fromSize != attr.getInt()) {
          std::string str;
          llvm::raw_string_ostream os(str);
          os << llvm::formatv("bind_dims() mismatch on dim {0}. from: {1}, into: {2}", dim, fromSize, attr.getInt());
          throw std::runtime_error(os.str());
        }
      }
    }
  }
  *into = MakeDimOp(from, dim);
}

Shape TileBuilder::GetShape(mlir::Value* tensor) {
  IVLOG(5, "TileBuilder::GetShape>");
  auto type = tensor->getType().dyn_cast<mlir::RankedTensorType>();
  if (!type) {
    throw std::runtime_error("Only tensor types are supported");
  }
  auto elementType = type.getElementType().dyn_cast<ScalarType>();
  if (!elementType) {
    throw std::runtime_error("Only scalar element types are supported");
  }
  return Shape{elementType.type(), type.getShape()};
}

Value* TileBuilder::MakePrimitiveOp(StringRef fn, ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakePrimitiveOp> " << fn.str());
  for (auto arg : args) {
    IVLOG(6, "  arg: " << mlir::debugString(*arg));
  }
  auto abstractOp = impl->lookupOperation(fn);
  auto genericBuilder = abstractOp->getInterface<util::GenericBuilder>();
  if (!genericBuilder) {
    throw std::runtime_error("Unknown intrinsic: " + fn.str());
  }
  auto type = impl->builder.getType<ScalarType>(DataType::FLOAT32);  // TODO
  auto op = genericBuilder->create(&impl->builder, impl->builder.getUnknownLoc(), type, args);
  return op->getResult(0);
}

Value* TileBuilder::Clone(Value* value) {
  IVLOG(5, "TileBuilder::Clone> " << mlir::debugString(*value));
  return impl->builder.clone(*value->getDefiningOp())->getResult(0);
}

Value* TileBuilder::MakeNoneOp() {
  IVLOG(5, "TileBuilder::MakeNoneOp>");
  auto type = impl->builder.getNoneType();
  return impl->builder.create<NoneOp>(impl->builder.getUnknownLoc(), type).result();
}

Value* TileBuilder::MakeStringOp(StringRef value) {
  IVLOG(5, "TileBuilder::MakeStringOp> " << value.str());
  auto type = StringType::get(&impl->context);
  auto attr = impl->builder.getStringAttr(value);
  return impl->builder.create<StringOp>(impl->builder.getUnknownLoc(), type, attr).result();
}

Value* TileBuilder::MakeTupleOp(ArrayRef<Value*> elts) {
  IVLOG(5, "TileBuilder::MakeTupleOp> elts: " << elts.size());
  std::vector<Type> types;
  for (auto elt : elts) {
    types.push_back(elt->getType());
  }
  auto tupleType = impl->builder.getTupleType(types);
  return impl->builder.create<TupleOp>(impl->builder.getUnknownLoc(), tupleType, elts).result();
}

std::vector<Value*> TileBuilder::GetTupleElements(Value* value) {
  IVLOG(5, "TileBuilder::GetTupleElements> " << mlir::debugString(*value));
  if (auto op = llvm::dyn_cast_or_null<TupleOp>(value->getDefiningOp())) {
    return std::vector<Value*>(op.elts().begin(), op.elts().end());
  }
  throw std::runtime_error("Expected TupleOp");
}

Value* TileBuilder::MakeScalarConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  auto type = impl->builder.getType<ScalarType>(DataType::INT32);
  return impl->builder.create<ScalarConstantOp>(impl->builder.getUnknownLoc(), type, value).result();
}

Value* TileBuilder::MakeScalarConstantOp(double value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  auto type = impl->builder.getType<ScalarType>(DataType::FLOAT32);
  return impl->builder.create<ScalarConstantOp>(impl->builder.getUnknownLoc(), type, value).result();
}

Value* TileBuilder::MakeDimOp(Value* tensor, unsigned dim) {
  IVLOG(5, "TileBuilder::MakeDimOp> tensor: " << mlir::debugString(*tensor) << ", dim: " << dim);
  return impl->builder.create<DimOp>(impl->builder.getUnknownLoc(), tensor, dim).result();
}

mlir::Value* TileBuilder::MakePlaceholderOp(DataType dtype, llvm::ArrayRef<int64_t> dims) {
  IVLOG(5, "TileBuilder::MakePlaceholderOp> " << to_string(dtype));
  auto elt_type = impl->builder.getType<ScalarType>(dtype);
  // Convert dims: PlaidML semantics use 0 for unknown size, MLIR uses -1.
  llvm::SmallVector<int64_t, 4> mlir_dims(dims.begin(), dims.end());
  for (unsigned i = 0; i < mlir_dims.size(); i++) {
    if (mlir_dims[i] == 0) {
      mlir_dims[i] = -1;
    }
  }
  auto shape = RankedTensorType::get(mlir_dims, elt_type);
  return impl->builder.create<PlaceholderOp>(impl->builder.getUnknownLoc(), shape).result();
}

mlir::Value* TileBuilder::MakeAffineConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeAffineConstantOp> " << value);
  return impl->builder.create<AffineConstantOp>(impl->builder.getUnknownLoc(), value).result();
}

Value* TileBuilder::MakeAffineIndexOp(StringRef name) {
  IVLOG(5, "TileBuilder::MakeAffineIndexOp> " << name.str());
  auto op = impl->builder.create<AffineIndexOp>(impl->builder.getUnknownLoc());
  if (!name.empty()) {
    op.setAttr("name", impl->builder.getStringAttr(name));
  }
  return op.result();
}

Value* TileBuilder::MakeAffineAddOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineAddOp>");
  return impl->builder.create<AffineAddOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineSubOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineSubOp>");
  return impl->builder.create<AffineSubOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineMulOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMulOp>");
  return impl->builder.create<AffineMulOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineDivOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineDivOp>");
  return impl->builder.create<AffineDivOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineNegOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineNegOp>");
  return impl->builder.create<AffineNegOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineMaxOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMaxOp>");
  return impl->builder.create<AffineMaxOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineMinOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMinOp>");
  return impl->builder.create<AffineMinOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineSourceIndexMapOp(Value* tensor, ArrayRef<Value*> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineSourceIndexMapOp>");
  return impl->builder.create<AffineSourceIndexMapOp>(impl->builder.getUnknownLoc(), tensor, idxs).result();
}

Value* TileBuilder::MakeAffineSinkIndexMapOp(ArrayRef<Value*> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineSinkIndexMapOp>");
  return impl->builder.create<AffineSinkIndexMapOp>(impl->builder.getUnknownLoc(), idxs).result();
}

Value* TileBuilder::MakeAffineSizeMapOp(ArrayRef<Value*> sizes) {
  IVLOG(5, "TileBuilder::MakeAffineSizeMapOp>");
  return impl->builder.create<AffineSizeMapOp>(impl->builder.getUnknownLoc(), sizes).result();
}

void TileBuilder::AddConstraint(Value* cion, Value* lhs, Value* rhs) {
  IVLOG(5, "TileBuilder::AddConstraint>");
  auto op = cion->getDefiningOp();
  auto domainOp = llvm::dyn_cast_or_null<AffineDomainOp>(op);
  if (!domainOp) {
    throw std::runtime_error("add_constraint can only be specified on a contraction.");
  }

  auto& region = domainOp.body();
  auto src = &region.front();
  OpBuilder builder(src->getTerminator());

  // Get a backward slice to trace the transitive defs of the lhs and rhs.
  auto& info = impl->domains[domainOp];
  llvm::SetVector<Value*> values;
  values.insert(lhs);
  values.insert(rhs);
  auto slice = util::getBackwardSlice(values, false, [](Value* value) {  //
    return value->getType().isa<IndexType>();
  });

  // Previously, some values will have already been cloned into the AffineDomainOp
  // However, there might be other ops that this constraint introduced that needs
  // to be cloned into the AffineDomainOp.
  for (auto value : slice) {
    if (!info.mapping.contains(value)) {
      IVLOG(5, "clone: " << mlir::debugString(*value));
      auto op = value->getDefiningOp();
      auto newValue = builder.clone(*op, info.mapping)->getResult(0);
      info.mapping.map(value, newValue);
    }
  }

  // Create the ConstraintOp as a parent of the existing terminator.
  auto constraintOp = builder.create<ConstraintOp>(op->getLoc(), info.mapping.lookup(lhs), info.mapping.lookup(rhs));
  auto it = std::prev(src->end(), 1);
  auto block = builder.createBlock(&constraintOp.body());
  auto& dst = block->getOperations();
  dst.splice(dst.end(), src->getOperations(), it, src->end());
}

void TileBuilder::SetUseDefault(Value* cion, Value* defaultValue) {
  IVLOG(2, "TileBuilder::SetUseDefault>");
  auto op = cion->getDefiningOp();
  auto domainOp = llvm::dyn_cast_or_null<AffineDomainOp>(op);
  if (!domainOp) {
    throw std::runtime_error("use_default can only be specified on a contraction.");
  }
  auto terminator = domainOp.body().front().getTerminator();
  while (!llvm::isa<ContractionOp>(terminator)) {
    terminator = terminator->getRegion(0).front().getTerminator();
  }
  SmallVector<Value*, 6> operands{terminator->getOperands()};
  operands.emplace_back(defaultValue);
  terminator->setOperands(operands);
}

#define DEFINE_CONTRACTION_OPS(_agg_op_)                                                                    \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##Op(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink,     \
                                                  mlir::Value* sizes) {                                     \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##Op>");                                                        \
    return impl->MakeUnaryContraction<Con##_agg_op_##Op>(srcs, sink, sizes);                                \
  }                                                                                                         \
                                                                                                            \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##AddOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink,  \
                                                     mlir::Value* sizes) {                                  \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##AddOp>");                                                     \
    return impl->MakeBinaryContraction<Con##_agg_op_##AddOp>(srcs, sink, sizes);                            \
  }                                                                                                         \
                                                                                                            \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##CondOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, \
                                                      mlir::Value* sizes) {                                 \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##CondOp>");                                                    \
    return impl->MakeTernaryContraction<Con##_agg_op_##CondOp>(srcs, sink, sizes);                          \
  }                                                                                                         \
                                                                                                            \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##EqOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink,   \
                                                    mlir::Value* sizes) {                                   \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##EqOp>");                                                      \
    return impl->MakeBinaryContraction<Con##_agg_op_##EqOp>(srcs, sink, sizes);                             \
  }                                                                                                         \
                                                                                                            \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##MulOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink,  \
                                                     mlir::Value* sizes) {                                  \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##MulOp>");                                                     \
    return impl->MakeBinaryContraction<Con##_agg_op_##MulOp>(srcs, sink, sizes);                            \
  }

DEFINE_CONTRACTION_OPS(Assign);
DEFINE_CONTRACTION_OPS(Max);
DEFINE_CONTRACTION_OPS(Min);
DEFINE_CONTRACTION_OPS(Prod);
DEFINE_CONTRACTION_OPS(Sum);

std::shared_ptr<TileProgram> TileBuilder::MakeProgram(  //
    llvm::StringRef name,                               //
    llvm::ArrayRef<mlir::Value*> outputs,               //
    llvm::MutableArrayRef<mlir::Value*> new_outputs) {
  IVLOG(5, "TileBuilder::MakeProgram> " << name.str());
  IVLOG(6, mlir::debugString(impl->module));
  // Compute the result types
  std::vector<Type> resultTypes(outputs.size());
  llvm::SetVector<Value*> values;
  for (unsigned i = 0; i < outputs.size(); i++) {
    if (!outputs[i]) {
      throw std::runtime_error("Invalid output");
    }
    resultTypes[i] = outputs[i]->getType();
    if (values.count(outputs[i]) || llvm::isa<PlaceholderOp>(outputs[i]->getDefiningOp())) {
      values.insert(MakePrimitiveOp("ident", {outputs[i]}));
    } else {
      values.insert(outputs[i]);
    }
  }
  auto slice = util::getBackwardSlice(values, true);
  // Compute the input types
  std::vector<Type> inputTypes;
  for (auto value : slice) {
    auto op = value->getDefiningOp();
    if (auto placeholderOp = llvm::dyn_cast_or_null<PlaceholderOp>(op)) {
      inputTypes.push_back(placeholderOp.result()->getType());
    }
  }
  // Construct a module
  auto loc = mlir::UnknownLoc::get(&impl->context);
  auto module = ModuleOp::create(loc);
  auto program = std::make_shared<TileProgram>(module);
  // Construct a function to represent the entire program
  auto funcType = mlir::FunctionType::get(inputTypes, resultTypes, &impl->context);
  auto funcOp = mlir::FuncOp::create(loc, name, funcType, {});
  funcOp.addEntryBlock();
  OpBuilder builder(funcOp.getBody());
  UniqueNamer namer;
  auto attrName = Dialect::getDialectAttrName("name");
  unsigned argcnt = 0;
  for (auto value : slice) {
    auto op = value->getDefiningOp();
    // Only copy over top-level ops (those owned by the workspace module)
    if (op && op->getBlock() == impl->module.getBody()) {
      if (auto placeholderOp = llvm::dyn_cast<PlaceholderOp>(op)) {
        // Replace placeholders with block arguments
        auto new_value = funcOp.getArgument(argcnt++);
        if (auto attr = placeholderOp.getAttrOfType<StringAttr>("name")) {
          auto uniqueName = namer.get(attr.getValue());
          auto uniqueAttr = builder.getStringAttr(uniqueName);
          funcOp.setArgAttr(new_value->getArgNumber(), attrName, uniqueAttr);
        }
        IVLOG(5, "BlockArgument mapping: " << value << " -> " << new_value);
        program->mapper.map(value, new_value);
      } else {
        auto new_value = builder.clone(*op, program->mapper)->getResult(0);
        IVLOG(5, "mapping: " << value << " -> " << new_value);
        program->mapper.map(value, new_value);
      }
    }
  }
  // Add a final ReturnOp
  std::vector<Value*> rets;
  for (unsigned i = 0; i < values.size(); i++) {
    auto new_value = program->mapper.lookup(values[i]);
    new_outputs[i] = new_value;
    rets.push_back(new_value);
  }
  builder.create<mlir::ReturnOp>(loc, rets);
  // Attach the function to the module
  module.push_back(funcOp);
  IVLOG(5, mlir::debugString(module));
  if (failed(mlir::verify(module))) {
    throw std::runtime_error("Module verification error");
  }
  // Do some optimization passes
  mlir::PassManager pm(&impl->context);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  auto result = pm.run(module);
  if (failed(result)) {
    throw std::runtime_error("Optimization passes failure");
  }
  IVLOG(2, mlir::debugString(module));
  return program;
}

std::vector<Value*> TileBuilder::ComputeGradients(ArrayRef<Value*> wrt, Value* loss) {
  // TODO
  return wrt;
}

}  // namespace pmlc::dialect::tile
