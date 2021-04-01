// Copyright 2020 Intel Corporation
#pragma once

#include <tuple>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {

class DialectAsmParser;
class DialectAsmPrinter;

} // namespace mlir

namespace pmlc::dialect::comp {
// ============================================================================
// RuntimeType
// ============================================================================
/// Runtime identifies low level API that is a target for comp dialect
/// operations.
/// This differentiation is provided at type level, as in general types created
/// and managed by one runtime are incompatible with other runtimes and this
/// allows to catch such errors during compilation.
/// Because runtime only serves as identifier, it is implemented as unsigned
/// integer with ExecEnvRuntime providing predefined target API's.
/// Custom runtimes that are not included in predefined list should start
/// at ExecEnvRuntime::FirstCustom.
enum ExecEnvRuntime : unsigned { Vulkan, OpenCL, FirstCustom = 1000 };

struct RuntimeTypeStorage : public mlir::TypeStorage {
  using KeyTy = ExecEnvRuntime;
  explicit RuntimeTypeStorage(ExecEnvRuntime runtime) : runtime(runtime) {}

  bool operator==(const KeyTy &key) const { return key == runtime; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static RuntimeTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<RuntimeTypeStorage>())
        RuntimeTypeStorage(key); // NOLINT
  }

  ExecEnvRuntime getRuntime() const { return runtime; }

  ExecEnvRuntime runtime;
};

/// Base class for types connected to a runtime.
class RuntimeType : public mlir::Type {
public:
  using ImplType = RuntimeTypeStorage;
  using mlir::Type::Type;

  static bool classof(mlir::Type type);

  /// Returns runtime of this type.
  ExecEnvRuntime getRuntime() const {
    return static_cast<const ImplType *>(impl)->runtime;
  }
};

// ============================================================================
// DeviceType
// ============================================================================

/// Devices represent a device capable of evaluating a Comp program.
class DeviceType : public mlir::Type::TypeBase<DeviceType, mlir::Type,
                                               mlir::DefaultTypeStorage> {
public:
  using Base::Base;
  using Base::getChecked;
};

// ============================================================================
// ExecEnvType
// ============================================================================
class EventType;

using ExecEnvTag = unsigned;

struct ExecEnvStorage : public RuntimeTypeStorage {
  ExecEnvStorage(ExecEnvRuntime runtime, ExecEnvTag tag,
                 mlir::ArrayRef<mlir::Attribute> memorySpaces)
      : RuntimeTypeStorage(runtime), tag(tag),
        memorySpaces(memorySpaces.begin(), memorySpaces.end()) {}

  using KeyTy = std::tuple<ExecEnvRuntime, ExecEnvTag,
                           mlir::SmallVector<mlir::Attribute, 1>>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getRuntime(), tag, memorySpaces);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(
        std::get<0>(key), std::get<1>(key),
        mlir::ArrayRef<mlir::Attribute>(std::get<2>(key)));
  }

  static KeyTy getKey(ExecEnvRuntime runtime, ExecEnvTag tag,
                      mlir::ArrayRef<mlir::Attribute> memorySpaces) {
    return KeyTy(runtime, tag,
                 mlir::SmallVector<mlir::Attribute, 1>(memorySpaces.begin(),
                                                       memorySpaces.end()));
  }

  static ExecEnvStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<ExecEnvStorage>())
        ExecEnvStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  ExecEnvTag tag;
  mlir::SmallVector<mlir::Attribute, 1> memorySpaces;
};

/// Execution environment represents target API's interface for managing memory
/// and executing operations on device.
/// Its type is represented as a triple of runtime, tag and memory spaces.
/// Runtime identifies low level target API and determines other runtime types
/// this execution environment can interface with.
/// Tag is an unsigned integer, that serves as further customizable abstraction.
/// For example this could be represent different devices,
/// sub-device partitions, or OpenCL command queues.
/// Last element is a list of memory spaces this execution environment can
/// access, first of which is used as default one. This determines what space
/// memory allocated on device may reside in.
class ExecEnvType
    : public mlir::Type::TypeBase<ExecEnvType, RuntimeType, ExecEnvStorage> {
public:
  using Base::Base;

  static ExecEnvType get(mlir::MLIRContext *context, ExecEnvRuntime runtime,
                         ExecEnvTag tag,
                         mlir::ArrayRef<mlir::Attribute> memorySpaces) {
    return Base::get(context, runtime, tag, memorySpaces);
  }

  static ExecEnvType
  getChecked(mlir::function_ref<mlir::InFlightDiagnostic()> emitError,
             mlir::MLIRContext *context, ExecEnvRuntime runtime, ExecEnvTag tag,
             mlir::ArrayRef<mlir::Attribute> memorySpaces) {
    return Base::getChecked(emitError, context, runtime, tag, memorySpaces);
  }

  /// Returns tag of this execution environment.
  ExecEnvTag getTag() const { return getImpl()->tag; }

  /// Returns reference to list of supported memory spaces.
  mlir::ArrayRef<mlir::Attribute> getMemorySpaces() const {
    return getImpl()->memorySpaces;
  }

  /// Returns default memory space, first in the list of all memory spaces.
  mlir::Attribute getDefaultMemorySpace() const { return getMemorySpaces()[0]; }

  /// Returns whether `requestedSpace` is supported memory space.
  bool supportsMemorySpace(mlir::Attribute requestedSpace) const;

  /// Returns EventType with matching runtime.
  EventType getEventType() const;
};

// ============================================================================
// EventType
// ============================================================================
struct EventTypeStorage : public RuntimeTypeStorage {
  using RuntimeTypeStorage::RuntimeTypeStorage;

  using KeyTy = ExecEnvRuntime;

  bool operator==(const KeyTy &key) const { return key == KeyTy(getRuntime()); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(ExecEnvRuntime runtime) { return KeyTy(runtime); }

  static EventTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<EventTypeStorage>()) EventTypeStorage(key);
  }
};

/// Events represent status of operations, which allows for specifying
/// dependencies between operations or explicitly waiting for completion.
///
/// This mechanism has three main advantages over using SSA dependencies.
/// Firstly, some dependencies may be indirect, aliased or crossing
/// block/function boundaries.
/// Analysing such dependencies may be complex and time consuming,
/// in which case events allow to calculate dependencies once and
/// update only when necessary.
/// Secondly, events provide more fine grained control over order of execution.
/// For example when there are no actual data dependencies, but executing
/// in certain order will have other benefits like better performance due to
/// favourable parallelization by target API.
/// Finally, events allow for passing dependencies from or to external code.
class EventType
    : public mlir::Type::TypeBase<EventType, RuntimeType, EventTypeStorage> {
public:
  using Base::Base;
  using Base::getChecked;

  static EventType get(mlir::MLIRContext *context, ExecEnvRuntime runtime) {
    return Base::get(context, runtime);
  }

  static EventType
  getChecked(mlir::function_ref<mlir::InFlightDiagnostic()> emitError,
             mlir::MLIRContext *context, ExecEnvRuntime runtime) {
    return Base::getChecked(emitError, context, runtime);
  }

  static EventType get(mlir::MLIRContext *context, ExecEnvType envType) {
    return EventType::get(context, envType.getRuntime());
  }

  static EventType
  getChecked(mlir::function_ref<mlir::InFlightDiagnostic()> emitError,
             ExecEnvType envType) {
    return EventType::getChecked(emitError, envType.getContext(),
                                 envType.getRuntime());
  }
};

// ============================================================================
// KernelType
// ============================================================================

/// Kernels represent programs that run on devices.
class KernelType : public mlir::Type::TypeBase<KernelType, mlir::Type,
                                               mlir::DefaultTypeStorage> {
public:
  using Base::Base;
};

// ============================================================================
// Printing and parsing
// ============================================================================
namespace detail {

void printType(mlir::Type type, mlir::DialectAsmPrinter &printer);

mlir::Type parseType(mlir::DialectAsmParser &parser);

} // namespace detail
} // namespace pmlc::dialect::comp
