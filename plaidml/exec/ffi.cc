// Copyright 2019 Intel Corporation.

#include "plaidml/exec/ffi.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml/core/internal.h"
#include "plaidml/core/settings.h"
#include "pmlc/rt/device_id.h"
#include "pmlc/rt/executable.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using plaidml::core::ffi_vector;
using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using pmlc::compiler::Program;
using pmlc::rt::Device;
using pmlc::rt::Executable;
using pmlc::rt::getDeviceIDs;
using pmlc::util::BufferPtr;
using pmlc::util::TensorShape;
using namespace mlir;  // NOLINT[build/namespaces]

extern "C" {

struct plaidml_executable {
  std::unique_ptr<Executable> exec;
  std::shared_ptr<Program> program;
};

void plaidml_exec_init(  //
    plaidml_error* err) {
  static std::once_flag once;
  ffi_wrap_void(err, [&] {  //
    IVLOG(1, "plaidml_exec_init");
    std::call_once(once, []() {
      pmlc::rt::registerRuntimes();
      pmlc::rt::initRuntimes();
    });
  });
}

plaidml_strings* plaidml_devices_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {  //
    return ffi_vector<plaidml_strings, plaidml_string>(getDeviceIDs());
  });
}

plaidml_executable* plaidml_jit(  //
    plaidml_error* err,           //
    plaidml_program* program,     //
    const char* raw_device) {
  return ffi_wrap<plaidml_executable*>(err, nullptr, [&] {
    std::string device(raw_device);
    if (device.empty()) {
      device = plaidml::core::Settings::Instance()->get("PLAIDML_DEVICE");
    }
    IVLOG(1, "JITing for device: " << device);
    return new plaidml_executable{Executable::fromProgram(program->program, device), program->program};
  });
}

void plaidml_executable_free(  //
    plaidml_error* err,        //
    plaidml_executable* exec) {
  ffi_wrap_void(err, [&] {  //
    delete exec;
  });
}

double plaidml_executable_run(  //
    plaidml_error* err,         //
    plaidml_executable* exec,   //
    size_t ninputs,             //
    plaidml_buffer** inputs,    //
    size_t noutputs,            //
    plaidml_buffer** outputs) {
  return ffi_wrap<double>(err, 0.0, [&] {  //
    llvm::SmallVector<BufferPtr, 8> inputBuffers;
    IVLOG(0, "pml exec run: started");
    if (exec->program->inputs.size() != ninputs) {
      throw std::runtime_error(
          llvm::formatv("Program expects {0} inputs, but {1} were specified", exec->program->inputs.size(), ninputs));
    }
    if (exec->program->outputs.size() != noutputs) {
      throw std::runtime_error(llvm::formatv("Program expects {0} outputs, but {1} were specified",
                                             exec->program->outputs.size(), noutputs));
    }
    IVLOG(0, "ping 1");
    for (unsigned i = 0; i < ninputs; i++) {
      TensorShape actual = inputs[i]->buffer->shape();
      TensorShape expected = TensorShape::fromType(exec->program->inputs[i]);
      if (actual != expected) {
        throw std::runtime_error(
            llvm::formatv("Shape mismatch for input buffer #{0}, expected '{1}' but '{2}' was specified", i,
                          expected.str(), actual.str()));
      }
      inputBuffers.push_back(inputs[i]->buffer);
    }
    
    IVLOG(0, "pml exec run: ping 2");
    llvm::SmallVector<BufferPtr, 8> outputBuffers;
    for (unsigned i = 0; i < noutputs; i++) {
      TensorShape actual = outputs[i]->buffer->shape();
      TensorShape expected = TensorShape::fromType(exec->program->outputs[i]);
      if (actual != expected) {
        throw std::runtime_error(
            llvm::formatv("Shape mismatch for output buffer #{0}, expected '{1}' but '{2}' was specified", i,
                          expected.str(), actual.str()));
      }
      outputBuffers.push_back(outputs[i]->buffer);
    }
    IVLOG(0, "pml exec run: ping 3");
    auto ret = exec->exec->invoke(inputBuffers, outputBuffers);
    IVLOG(0, "pml exec run: done");
    
    return ret;
  });
}

}  // extern "C"
