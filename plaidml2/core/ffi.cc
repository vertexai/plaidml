// Copyright 2019 Intel Corporation.

#include "plaidml2/core/ffi.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>

#include "base/config/config.h"
#include "base/context/context.h"
#include "base/context/eventlog.h"
#include "base/util/any_factory_map.h"
#include "base/util/env.h"
#include "plaidml2/core/internal.h"
#include "plaidml2/core/settings.h"
#include "tile/platform/local_machine/platform.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GetPlatform;
using plaidml::core::GlobalContext;
using plaidml::core::Settings;
using vertexai::context::Context;
using vertexai::tile::DataType;
using vertexai::tile::TensorDimension;
using vertexai::tile::TensorShape;
using LocalPlatform = vertexai::tile::local_machine::Platform;

extern const char* PLAIDML_VERSION;

namespace plaidml {
namespace core {

PlatformHolder::PlatformHolder() : platform(new LocalPlatform) {}

PlatformHolder& GetPlatform() {
  static PlatformHolder holder;
  return holder;
}

}  // namespace core
}  // namespace plaidml

extern "C" {

void plaidml_init(plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      vertexai::env::Set("PLAIDML_CLEANUP_NAMES", "1");
      auto level_str = vertexai::env::Get("PLAIDML_VERBOSE");
      if (level_str.size()) {
        auto level = std::atoi(level_str.c_str());
        if (level) {
          el::Loggers::setVerboseLevel(level);
        }
      }
      auto ctx = GlobalContext::getContext();
      auto eventlog_str = vertexai::env::Get("PLAIDML_EVENTLOG_FILENAME");
      if (ctx && eventlog_str.size()) {
        IVLOG(1, "Logging events to " << eventlog_str);
        auto config = "{\"@type\": \"type.vertex.ai/vertexai.eventing.file.proto.EventLog\", \"filename\": \"" +
                      eventlog_str + "\"}";
        auto pconfig = vertexai::ParseConfig<google::protobuf::Any>(config);
        auto eventlog = vertexai::AnyFactoryMap<vertexai::context::EventLog>::Instance()->MakeInstance(*ctx, pconfig);
        ctx->set_eventlog(std::move(eventlog));
        ctx->set_is_logging_events(true);
      } else {
        ctx->set_is_logging_events(false);
        ctx->set_eventlog(std::shared_ptr<vertexai::context::EventLog>());
      }
      IVLOG(1, "plaidml_init");
      Settings::Instance()->load();
      GetPlatform();
    });
  });
}

void plaidml_shutdown(plaidml_error* err) {
  ffi_wrap_void(err, [&] {
    IVLOG(1, "plaidml_shutdown");
    GetPlatform().platform.reset(nullptr);
  });
}

const char* plaidml_version(  //
    plaidml_error* err) {
  return ffi_wrap<const char*>(err, nullptr, [&] {  //
    return PLAIDML_VERSION;
  });
}

size_t plaidml_settings_list_count(  //
    plaidml_error* err) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return Settings::Instance()->all().size();
  });
}

void plaidml_settings_list(  //
    plaidml_error* err,      //
    size_t nitems,           //
    plaidml_string** keys,   //
    plaidml_string** values) {
  ffi_wrap_void(err, [&] {
    size_t i = 0;
    const auto& settings = Settings::Instance()->all();
    for (auto it = settings.begin(); it != settings.end() && i < nitems; it++, i++) {
      keys[i] = new plaidml_string{it->first};
      values[i] = new plaidml_string{it->second};
    }
  });
}

void plaidml_settings_load(  //
    plaidml_error* err) {
  ffi_wrap_void(err, [&] {  //
    Settings::Instance()->load();
  });
}

void plaidml_settings_save(  //
    plaidml_error* err) {
  ffi_wrap_void(err, [&] {  //
    Settings::Instance()->save();
  });
}

plaidml_string* plaidml_settings_get(  //
    plaidml_error* err,                //
    const char* key) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&]() -> plaidml_string* {  //
    return new plaidml_string{Settings::Instance()->get(key)};
  });
}

void plaidml_settings_set(  //
    plaidml_error* err,     //
    const char* key,        //
    const char* value) {
  ffi_wrap_void(err, [&] {  //
    Settings::Instance()->set(key, value);
  });
}

const char* plaidml_string_ptr(plaidml_string* str) { return str->str.c_str(); }

void plaidml_string_free(plaidml_string* str) {
  plaidml_error err;
  ffi_wrap_void(&err, [&] {  //
    delete str;
  });
}

void plaidml_shape_free(  //
    plaidml_error* err,   //
    plaidml_shape* shape) {
  ffi_wrap_void(err, [&] {  //
    delete shape;
  });
}

plaidml_shape* plaidml_shape_alloc(  //
    plaidml_error* err,              //
    plaidml_datatype dtype,          //
    size_t ndims,                    //
    const int64_t* sizes,            //
    const int64_t* strides) {
  return ffi_wrap<plaidml_shape*>(err, nullptr, [&] {
    std::vector<TensorDimension> dims(ndims);
    for (size_t i = 0; i < ndims; i++) {
      dims[i] = TensorDimension{strides[i], static_cast<uint64_t>(sizes[i])};
    }
    return new plaidml_shape{TensorShape{static_cast<DataType>(dtype), dims}};
  });
}

plaidml_string* plaidml_shape_repr(  //
    plaidml_error* err,              //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    std::stringstream ss;
    ss << shape->shape;
    return new plaidml_string{ss.str()};
  });
}

size_t plaidml_shape_get_ndims(  //
    plaidml_error* err,          //
    plaidml_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return shape->shape.dims.size();
  });
}

plaidml_datatype plaidml_shape_get_dtype(  //
    plaidml_error* err,                    //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {  //
    return static_cast<plaidml_datatype>(shape->shape.type);
  });
}

int64_t plaidml_shape_get_dim_size(  //
    plaidml_error* err,              //
    plaidml_shape* shape,            //
    size_t dim) {
  return ffi_wrap<int64_t>(err, 0, [&] {  //
    return shape->shape.dims.at(dim).size;
  });
}

int64_t plaidml_shape_get_dim_stride(  //
    plaidml_error* err,                //
    plaidml_shape* shape,              //
    size_t dim) {
  return ffi_wrap<int64_t>(err, 0, [&] {  //
    return shape->shape.dims.at(dim).stride;
  });
}

uint64_t plaidml_shape_get_nbytes(  //
    plaidml_error* err,             //
    plaidml_shape* shape) {
  return ffi_wrap<int64_t>(err, 0, [&] {  //
    return shape->shape.byte_size();
  });
}

void plaidml_buffer_free(  //
    plaidml_error* err,    //
    plaidml_buffer* buffer) {
  ffi_wrap_void(err, [&] {  //
    delete buffer;
  });
}

plaidml_buffer* plaidml_buffer_alloc(  //
    plaidml_error* err,                //
    const char* device_id,             //
    size_t size) {
  return ffi_wrap<plaidml_buffer*>(err, nullptr, [&] {
    Context* ctx = GlobalContext::getContext();
    auto buffer = GetPlatform()->MakeBuffer(*ctx, device_id, size);
    return new plaidml_buffer{buffer};
  });
}

plaidml_view* plaidml_buffer_mmap_current(  //
    plaidml_error* err,                     //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_view*>(err, nullptr, [&] {  //
    Context* ctx = GlobalContext::getContext();
    return new plaidml_view{buffer->buffer->MapCurrent(*ctx).get()};
  });
}

plaidml_view* plaidml_buffer_mmap_discard(  //
    plaidml_error* err,                     //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_view*>(err, nullptr, [&] {  //
    Context* ctx = GlobalContext::getContext();
    return new plaidml_view{buffer->buffer->MapDiscard(*ctx)};
  });
}

void plaidml_view_free(  //
    plaidml_error* err,  //
    plaidml_view* view) {
  ffi_wrap_void(err, [&] {  //
    delete view;
  });
}

char* plaidml_view_data(  //
    plaidml_error* err,   //
    plaidml_view* view) {
  return ffi_wrap<char*>(err, nullptr, [&] {  //
    return view->view->data();
  });
}

size_t plaidml_view_size(  //
    plaidml_error* err,    //
    plaidml_view* view) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return view->view->size();
  });
}

void plaidml_view_writeback(  //
    plaidml_error* err,       //
    plaidml_view* view) {
  ffi_wrap_void(err, [&] {
    Context* ctx = GlobalContext::getContext();
    view->view->WriteBack(*ctx);
  });
}

}  // extern "C"
