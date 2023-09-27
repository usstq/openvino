// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reference.h"
#include <ie_ngraph_utils.hpp>
#include <shape_util.hpp>
#include <dnnl_extension_utils.h>
#include "openvino/runtime/tensor.hpp"
#include "common/blocked_desc_creator.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"


#include <shape_inference/shape_inference_ngraph.hpp>
#include "utils/profiler.hpp"

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

template <typename T>
class ScalarAttributeCollector : public ngraph::AttributeVisitor {
public:
    ScalarAttributeCollector() {}
    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<int>>(&adapter)) {
            m_attrs[name] = static_cast<T>(a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<float>>(&adapter)) {
            m_attrs[name] = static_cast<T>(a->get());
        } else {
            //IE_ASSERT(false) << "ScalarAttributeCollector: unsupported data type for attribute: " << name;
        }
    }

    T& get(std::string v) {
        return m_attrs[v];
    }

private:
    std::map<std::string, T> m_attrs;
};

class LLM_experimental_shapeInfer : public ShapeInferEmptyPads {
public:
    LLM_experimental_shapeInfer(const std::shared_ptr<ov::Node>& op) {
        ScalarAttributeCollector<int> op_attr;
        op->visit_attributes(op_attr);
        const auto& type_info = op->get_type_info();
        auto version_info = std::string(type_info.get_version());
        if (std::string("FC") == type_info.name) {
            m_type = 1;
            m_outputShape.resize(3);
            m_outputShape[2] = op_attr.get("N");
            return;
        }
        IE_THROW() << "Unsupported experimental op : " << type_info.name;
    }

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        if (m_type == 1) {
            auto i0 = input_shapes[0].get();
            m_outputShape[0] = i0[0];  // B
            m_outputShape[1] = i0[1];  // M
        }
        //std::cout << "=======================" << m_outputShape[0] << "," << m_outputShape[1] << "," << m_outputShape[2] << std::endl;
        return {{m_outputShape}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return PortMask(0);
    }

private:
    int m_type = 0;
    VectorDims m_outputShape;
};

class ReferenceShapeInferFactory : public ShapeInferFactory {
public:
    ReferenceShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        const auto& type_info = m_op->get_type_info();
        auto version_info = std::string(type_info.get_version());
        if (version_info == "llm::experimental")
            return std::make_shared<LLM_experimental_shapeInfer>(m_op);
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), FULL_PORT_MASK);
    }

private:
    const std::shared_ptr<ov::Node> m_op;
    bool is_experimental_FC;
};

Reference::Reference(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context,
                                         const std::string& errorMessage) :
        Node(op, context, ReferenceShapeInferFactory(op)), ngraphOp(op), additionalErrorMessage(errorMessage) {
    if (!op->has_evaluate()) {
        IE_THROW(NotImplemented) << "Cannot fallback on ngraph reference implementation (Ngraph::Node::evaluate() is not implemented)";
    }
    setType(Type::Reference);
    setTypeStr("Reference");

    // RandomUniform should generate new sequence each run even if all inputs are constants. So that method Node::IsConstant()
    // doesn't return 'True' for RandomUniform with all constant inputs and the node generates new values for each inference,
    // we set 'NoConst' value for 'ConstantType' in ctor
    if (ov::is_type<ngraph::op::v8::RandomUniform>(ngraphOp)) {
        constant = ConstantType::NoConst;
    }
}

void Reference::getSupportedDescriptors() {}

void Reference::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inputConfigurators;
    inputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); i++) {
        inputConfigurators.emplace_back(LayoutType::ncsp, convertPrecision(ngraphOp->get_input_element_type(i)), inputShapes[i]);
    }

    std::vector<PortConfigurator> outputConfigurators;
    outputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); i++) {
        outputConfigurators.emplace_back(LayoutType::ncsp, convertPrecision(ngraphOp->get_output_element_type(i)), outputShapes[i]);
    }

    addSupportedPrimDesc(inputConfigurators, outputConfigurators, impl_desc_type::ref);
}

void Reference::createPrimitive() {}

void Reference::execute(dnnl::stream strm) {
    auto inputs = prepareInputs();
    auto outputs = prepareOutputs();
    if (!ngraphOp->evaluate(outputs, inputs)) {
        IE_THROW() << "Evaluation failed on node of type: " << std::string(ngraphOp->get_type_name()) << " name: " << getName();
    }
}

void Reference::executeDynamicImpl(dnnl::stream strm) {
    auto inputs = prepareInputs();
    ov::TensorVector outputs;
    auto result = Node::shapeInfer();
    if (ShapeInferStatus::success == result.status) {
        Node::redefineOutputMemory(result.dims);
        outputs = prepareOutputs();
    } else if (ShapeInferStatus::skip == result.status) {
        outputs.reserve(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); ++i) {
            auto mem_desc = getBaseMemDescAtOutputPort(i);
            if (mem_desc->isDefined()) {
                outputs.emplace_back(ngraphOp->get_output_element_type(i), mem_desc->getShape().getStaticDims());
            } else {
                outputs.emplace_back(ngraphOp->get_output_element_type(i), ov::util::make_dynamic_shape());
            }
        }
    } else {
         IE_THROW(Unexpected) <<
            "Unexpected shape infer result status during the inference of a node with type " <<
            getTypeStr() << " and name " << getName();
    }
    {
        PROFILE(prof, "evaluate");
        if (!ngraphOp->evaluate(outputs, inputs)) {
            IE_THROW() << "Evaluation failed on node of type: " << std::string(ngraphOp->get_type_name()) << " name: " << getName();
        }
    }
    if (ShapeInferStatus::skip == result.status) {
        std::vector<VectorDims> newOutputDims;
        newOutputDims.reserve(outputs.size());
        for (auto& tensor : outputs) {
            newOutputDims.emplace_back(tensor.get_shape());
        }
        Node::redefineOutputMemory(newOutputDims);
        for (size_t i = 0; i < outputShapes.size(); ++i) {
            auto memory = getChildEdgesAtPort(i)[0]->getMemoryPtr();
            auto& tensor = outputs[i];
            if (memory->getSize() != tensor.get_byte_size()) {
                IE_THROW(Unexpected) << "Output tensor data size mismatch occurred during the inference of a node with type " <<
                getTypeStr() << " and name " << getName() << " on output port number " << i;
            }
            cpu_memcpy(memory->getData(), tensor.data(), tensor.get_byte_size());
        }
    }
}

bool Reference::created() const {
    return getType() == Type::Reference;
}

bool Reference::needShapeInfer() const {
    return false;
}

ov::TensorVector Reference::prepareInputs() const {
    ov::TensorVector inputs;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        void *srcDataPtr = getParentEdgesAtPort(i)[0]->getMemory().getData();
        inputs.push_back(ov::Tensor(ngraphOp->get_input_element_type(i),
                                             getParentEdgesAtPort(i)[0]->getMemory().getStaticDims(), srcDataPtr));
    }
    return inputs;
}

ov::TensorVector Reference::prepareOutputs() const {
    ov::TensorVector outputs;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        void *dstDataPtr = getChildEdgesAtPort(i)[0]->getMemory().getData();
        outputs.push_back(ov::Tensor(ngraphOp->get_output_element_type(i),
                                              getChildEdgesAtPort(i)[0]->getMemory().getStaticDims(), dstDataPtr));
    }
    return outputs;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
