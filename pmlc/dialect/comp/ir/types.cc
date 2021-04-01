// Copyright 2020 Intel Corporation

#include "pmlc/dialect/comp/ir/types.h"

#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::comp; // NOLINT

bool RuntimeType::classof(Type type) {
  return type.isa<ExecEnvType, EventType>();
}

bool ExecEnvType::supportsMemorySpace(Attribute requestedSpace) const {
  if (!requestedSpace) {
    return llvm::any_of(getMemorySpaces(), [&](Attribute execEnvSpace) {
      if (auto intAttr = execEnvSpace.dyn_cast<IntegerAttr>()) {
        return intAttr.getInt() == 0;
      }
      return false;
    });
  }
  return llvm::any_of(getMemorySpaces(), [&](Attribute execEnvSpace) {
    return execEnvSpace == requestedSpace;
  });
}

EventType ExecEnvType::getEventType() const {
  return EventType::get(getContext(), *this);
}

static std::string runtimeToString(ExecEnvRuntime runtime) {
  switch (runtime) {
  case ExecEnvRuntime::Vulkan:
    return "vk";
  case ExecEnvRuntime::OpenCL:
    return "ocl";
  default: {
    unsigned customNr = static_cast<unsigned>(runtime);
    return std::to_string(customNr);
  }
  }
}

static ParseResult parseRuntime(DialectAsmParser &parser,
                                ExecEnvRuntime &runtime) {
  OptionalParseResult parseResult =
      parser.parseOptionalInteger(reinterpret_cast<unsigned &>(runtime));
  if (parseResult.hasValue())
    return parseResult.getValue();

  StringRef runtimeKeyword;
  if (failed(parser.parseKeyword(&runtimeKeyword)))
    return failure();

  auto failOrRuntime =
      llvm::StringSwitch<llvm::function_ref<FailureOr<ExecEnvRuntime>()>>(
          runtimeKeyword)
          .Case("vk", [] { return ExecEnvRuntime::Vulkan; })
          .Case("ocl", [] { return ExecEnvRuntime::OpenCL; })
          .Default([&] {
            auto loc = parser.getCurrentLocation();
            parser.emitError(loc, "unrecognized runtime string name: ")
                .append(runtimeKeyword)
                .attachNote()
                .append("available runtime names are: vk, ocl");
            return failure();
          })();

  if (failed(failOrRuntime))
    return failure();

  runtime = failOrRuntime.getValue();
  return success();
}

static void printDeviceType(DeviceType type, DialectAsmPrinter &printer) {
  printer << "device";
}

static Type parseDeviceType(DialectAsmParser &parser) {
  return DeviceType::get(parser.getBuilder().getContext());
}

static void printEventType(EventType type, DialectAsmPrinter &printer) {
  printer << "event<" << runtimeToString(type.getRuntime()) << ">";
}

static Type parseEventType(DialectAsmParser &parser) {
  ExecEnvRuntime runtime;
  if (parser.parseLess() ||            //
      parseRuntime(parser, runtime) || //
      parser.parseGreater())
    return Type();

  MLIRContext *context = parser.getBuilder().getContext();
  return EventType::get(context, runtime);
}

static void printExecEnvType(ExecEnvType type, DialectAsmPrinter &printer) {
  ArrayAttr array = ArrayAttr::get(type.getContext(), type.getMemorySpaces());
  printer << "execenv<" << runtimeToString(type.getRuntime()) << ":"
          << type.getTag() << ", " << array << ">";
}

static Type parseExecEnvType(DialectAsmParser &parser) {
  ExecEnvRuntime runtime;
  unsigned tag;
  ArrayAttr memorySpaces;
  if (parser.parseLess() ||                  //
      parseRuntime(parser, runtime) ||       //
      parser.parseColon() ||                 //
      parser.parseInteger(tag) ||            //
      parser.parseComma() ||                 //
      parser.parseAttribute(memorySpaces) || //
      parser.parseGreater())
    return Type();

  MLIRContext *context = parser.getBuilder().getContext();
  return ExecEnvType::get(context, runtime, tag, memorySpaces.getValue());
}

static void printKernelType(KernelType type, DialectAsmPrinter &printer) {
  printer << "kernel";
}

static Type parseKernelType(DialectAsmParser &parser) {
  return KernelType::get(parser.getBuilder().getContext());
}

void pmlc::dialect::comp::detail::printType(mlir::Type type,
                                            mlir::DialectAsmPrinter &printer) {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<DeviceType>(
          [&](DeviceType deviceType) { printDeviceType(deviceType, printer); })
      .Case<EventType>(
          [&](EventType eventType) { printEventType(eventType, printer); })
      .Case<ExecEnvType>([&](ExecEnvType execEnvType) {
        printExecEnvType(execEnvType, printer);
      })
      .Case<KernelType>(
          [&](KernelType kernelType) { printKernelType(kernelType, printer); })
      .Default([](Type) { llvm_unreachable("Unhandled 'comp' type"); });
}

Type pmlc::dialect::comp::detail::parseType(mlir::DialectAsmParser &parser) {
  StringRef typeKeyword;
  if (failed(parser.parseKeyword(&typeKeyword)))
    return Type();

  return llvm::StringSwitch<function_ref<Type()>>(typeKeyword)
      .Case("device", [&] { return parseDeviceType(parser); })
      .Case("event", [&] { return parseEventType(parser); })
      .Case("execenv", [&] { return parseExecEnvType(parser); })
      .Case("kernel", [&] { return parseKernelType(parser); })
      .Default([&] {
        parser.emitError(parser.getNameLoc(),
                         "unknown comp type " + typeKeyword);
        return Type();
      })();
}
