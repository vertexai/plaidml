// Copyright 2020 Intel Corporation

#include "pmlc/rt/llvm/register.h"
#include "pmlc/rt/register.h"
#include "pmlc/rt/runtime_registry.h"

namespace pmlc::rt {

void registerRuntimes() {
  registerRuntime();
  llvm::registerRuntime();
}

} // namespace pmlc::rt
