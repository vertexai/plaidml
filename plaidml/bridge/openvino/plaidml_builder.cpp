// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_builder.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/opsets/opset1.hpp"

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

namespace PlaidMLPlugin {

namespace {

plaidml::Program nodeProgramBuilder(const std::shared_ptr<ngraph::Node>& _node) {
  IE_ASSERT(_node);  // PlaidML requires that the nGraph-based API be used

  plaidml::edsl::TensorVec edsl_inputs;
  plaidml::edsl::TensorVec edsl_outputs;

  Context ctx{_node.get()};

  for (const auto& input : _node->inputs()) {
    std::vector<int64_t> dims{input.get_shape().begin(), input.get_shape().end()};
    auto type = to_plaidml(input.get_element_type());
    std::string tensor_name;
    if (input.get_tensor().get_names().empty()) {
      tensor_name = "";
    } else {
      // PlaidML only has one name per tensor, and it isn't semantic, just pick one arbitrarily
      tensor_name = *input.get_tensor().get_names().begin();
    }
    auto tensor = plaidml::edsl::Placeholder(type, dims, tensor_name);
    edsl_inputs.push_back(tensor);
    ctx.operands.push_back(tensor);
  }

  const Op op = OpsRegistry::instance()->resolve(_node->description());
  if (!op) {
    THROW_IE_EXCEPTION << "Unsupported operation: " << _node->description();
  }

  auto value = op(ctx);
  auto tuple = value.as_tuple();
  IE_ASSERT(tuple.size() == _node->get_output_size());
  for (unsigned i = 0; i < tuple.size(); i++) {
    auto tensor = tuple.at(i).as_tensor();
    edsl_outputs.push_back(tensor);
  }

  return plaidml::edsl::buildProgram(_node->get_name(), edsl_inputs, edsl_outputs);
}

class ProgramBuilder {
 public:
  ProgramBuilder(const std::shared_ptr<const ngraph::Function>& func, const std::string& netName,
                 const InferenceEngine::InputsDataMap& inputsInfo, const InferenceEngine::OutputsDataMap& outputsInfo);

  plaidml::Program build();

 private:
  void handleConstant(const std::shared_ptr<ngraph::Node>& node);
  void handleParameter(const std::shared_ptr<ngraph::Node>& node);
  void handleOutput(const std::shared_ptr<ngraph::Node>& node);
  void handleOp(const std::shared_ptr<ngraph::Node>& node);

  std::shared_ptr<const ngraph::Function> func;
  const std::string& netName;
  const InferenceEngine::InputsDataMap& inputsInfo;
  const InferenceEngine::OutputsDataMap& outputsInfo;

  // Lets us look up the PlaidML tensor by the name of the node that produces it and the index of which output it is
  std::map<std::pair<std::string, size_t>, plaidml::edsl::Tensor> tensorMap;

  // Go from the names OV uses for a networks inputs and outputs to the corresponding PlaidML Tensor
  std::map<std::string, plaidml::edsl::Tensor> tensorIONameMap;
};

ProgramBuilder::ProgramBuilder(const std::shared_ptr<const ngraph::Function>& func, const std::string& netName,
                               const InferenceEngine::InputsDataMap& inputsInfo,
                               const InferenceEngine::OutputsDataMap& outputsInfo)
    : func(func), netName(netName), inputsInfo(inputsInfo), outputsInfo(outputsInfo) {}

plaidml::Program ProgramBuilder::build() {
  IE_ASSERT(func);  // PlaidML requires that the nGraph-based API be used
  for (const std::shared_ptr<ngraph::Node>& node : func->get_ordered_ops()) {
    if (node->description() == "Constant") {
      handleConstant(node);
    } else if (node->description() == "Parameter") {
      handleParameter(node);
    } else if (node->description() == "Result") {
      handleOutput(node);
    } else {
      handleOp(node);
    }
  }

  std::vector<plaidml::edsl::Tensor> inputs;
  for (const auto& kvp : inputsInfo) {
    inputs.push_back(tensorIONameMap.at(kvp.first));
  }

  std::vector<plaidml::edsl::Tensor> outputs;
  for (const auto& kvp : outputsInfo) {
    outputs.push_back(tensorIONameMap.at(kvp.first));
  }

  return plaidml::edsl::buildProgram(netName, inputs, outputs);
}

void ProgramBuilder::handleConstant(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  IE_ASSERT(node->description() == "Constant");
  plaidml::DType type = to_plaidml(node->get_element_type());
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  plaidml::TensorShape shape(type, dims);
  plaidml::Buffer buffer(shape);
  Context ctx{node.get()};
  auto* layer = dynamic_cast<ngraph::opset1::Constant*>(ctx.layer);
  buffer.copy_from(layer->get_data_ptr());
  plaidml::edsl::Tensor tensor = plaidml::edsl::Constant(buffer, node->get_friendly_name());
  tensorMap[std::make_pair(node->get_name(), 0)] = tensor;
}

void ProgramBuilder::handleParameter(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  const InferenceEngine::TensorDesc& inputDesc = inputsInfo.at(node->get_friendly_name())->getTensorDesc();
  std::vector<int64_t> dims{inputDesc.getDims().begin(), inputDesc.getDims().end()};
  plaidml::DType nodeType = to_plaidml(node->get_element_type());
  plaidml::DType inputType = to_plaidml(inputDesc.getPrecision());
  plaidml::edsl::Tensor tensor = plaidml::edsl::Placeholder(inputType, dims, node->get_friendly_name());
  tensorIONameMap[node->get_friendly_name()] = tensor;
  if (nodeType != inputType) {
    tensor = plaidml::edsl::cast(tensor, nodeType);
  }
  tensorMap[std::make_pair(node->get_name(), 0)] = tensor;
}

void ProgramBuilder::handleOutput(const std::shared_ptr<ngraph::Node>& node) {
  // The OV output name is the name of the node _prior_ to the result
  // When there are multiple outputs, it has .# appended, where # is the output index
  const ngraph::Output<ngraph::Node>& src_output = node->input(0).get_source_output();
  const ngraph::Node* src_node = src_output.get_node();
  std::string name = src_node->get_friendly_name();
  if (src_node->get_output_size() > 1) {
    name += "." + std::to_string(src_output.get_index());
  }
  plaidml::edsl::Tensor tensor = tensorMap.at(std::make_pair(src_node->get_name(), src_output.get_index()));
  plaidml::DType outputType = to_plaidml(outputsInfo.at(name)->getTensorDesc().getPrecision());
  if (outputType != tensor.dtype()) {
    tensor = plaidml::edsl::cast(tensor, outputType);
  }
  tensorIONameMap[name] = tensor;
}

struct PlaidMLAttributeVisitor : public ngraph::AttributeVisitor {
  plaidml::edsl::Dictionary attrs;

  void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) final {
    THROW_IE_EXCEPTION << "Unsupported 'void' attribute: " << name;
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) final {}

  void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) final {
    attrs[name] = plaidml::edsl::Value(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) final {
    attrs[name] = plaidml::edsl::Value(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) final {
    attrs[name] = plaidml::edsl::Value(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<double>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int8_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int16_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint8_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint16_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint32_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<void*>& adapter) final {
    THROW_IE_EXCEPTION << "Unsupported 'void*' attribute: " << name;
  }
};

void ProgramBuilder::handleOp(const std::shared_ptr<ngraph::Node>& node) {
  const Op op = OpsRegistry::instance()->resolve(node->description());
  if (!op) {
    THROW_IE_EXCEPTION << "Unsupported operation: " << node->description();
  }

  Context ctx{node.get()};
  for (const auto& input : node->inputs()) {
    const ngraph::Output<ngraph::Node>& src_output = input.get_source_output();
    const std::string& name = src_output.get_node()->get_name();
    size_t index = src_output.get_index();
    plaidml::edsl::Tensor tensor = tensorMap.at(std::make_pair(name, index));
    ctx.operands.push_back(tensor);
  }
  PlaidMLAttributeVisitor visitor;
  node->visit_attributes(visitor);
  plaidml::edsl::TensorVec tuple =
      plaidml::edsl::layer("ng." + node->description(), ctx.operands, visitor.attrs, [&]() {
        plaidml::edsl::Value value = op(ctx);
        std::vector<plaidml::edsl::Value> tuple = value.as_tuple();
        plaidml::edsl::TensorVec outputs;
        outputs.reserve(tuple.size());
        for (plaidml::edsl::Value output : tuple) {
          outputs.push_back(output.as_tensor());
        }
        return outputs;
      });
  IE_ASSERT(tuple.size() == node->get_output_size());
  for (unsigned i = 0; i < tuple.size(); i++) {
    plaidml::edsl::Tensor tensor = tuple.at(i);
    tensorMap[std::make_pair(node->get_name(), i)] = tensor;
  }
}

}  // namespace

static std::once_flag once;

plaidml::Program buildProgram(const std::shared_ptr<const ngraph::Function>& func, const std::string& netName,
                              const InferenceEngine::InputsDataMap& inputsInfo,
                              const InferenceEngine::OutputsDataMap& outputsInfo) {
  std::call_once(once, []() { registerOps(); });
  plaidml::init();
  return ProgramBuilder(func, netName, inputsInfo, outputsInfo).build();
}

plaidml::Program buildNodeProgram(const std::shared_ptr<ngraph::Node>& node) {
  std::call_once(once, []() { registerOps(); });
  plaidml::init();
  return nodeProgramBuilder(node);
}

}  // namespace PlaidMLPlugin
