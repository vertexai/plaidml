// Copyright 2020 Intel Corporation

#include "pmlc/compiler/program.h"
#include "pmlc/target/x86/pipeline.h"

namespace pmlc::compiler {

void registerTargets() { pmlc::target::x86::registerTarget(); }

} // namespace pmlc::compiler
