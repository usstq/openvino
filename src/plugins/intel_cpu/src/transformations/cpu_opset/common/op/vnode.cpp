// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnode.hpp"

#include "openvino/core/graph_util.hpp"
#include "transformations/itt.hpp"

void dump_subgraph(const ngraph::OutputVector& inputs, const ngraph::OutputVector& outputs, std::string model_name) {
    // replace inputs
    ov::ParameterVector params;
    for (auto& in : inputs) {
        auto p = std::make_shared<ov::op::v0::Parameter>(in.get_element_type(), in.get_partial_shape());
        p->set_friendly_name(in.get_node()->get_friendly_name());
        params.push_back(p);

        ov::replace_node(in.get_node_shared_ptr(), p);
    }

    // build model and serialize
    {
        ov::Model model(outputs, params);
        ov::serialize(model.shared_from_this(), model_name + ".xml", "/dev/null");
        std::cout << " VNode: " << model_name << " is dumpped into " << model_name << ".xml" << std::endl;
    }

    // recover inputs
    for (size_t i = 0; i < params.size(); i++) {
        ov::replace_node(params[i], inputs[i].get_node_shared_ptr());
    }
}

ov::intel_cpu::VNode::VNode(const ngraph::OutputVector& args,
                            const ngraph::OutputVector& org_outputs,
                            const std::string& vtype)
    : Op({args}),
      m_org_outputs(org_outputs),
      m_vtype(vtype) {
    dump_subgraph(args, m_org_outputs, m_vtype);
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::VNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(DimOfNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::VNode>(new_args, m_org_outputs, m_vtype);
}

void ov::intel_cpu::VNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(VNode_validate_and_infer_types);

    for (int i = 0; i < m_org_outputs.size(); i++) {
        auto node = m_org_outputs[i].get_node_shared_ptr();
        auto port = m_org_outputs[i].get_index();
        node->validate_and_infer_types();

        auto output_type = m_org_outputs[i].get_element_type();
        auto output_pshape = m_org_outputs[i].get_partial_shape();
        set_output_type(i, output_type, output_pshape);
    }
}

bool ov::intel_cpu::VNode::visit_attributes(ngraph::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(FullyConnectedNode_visit_attributes);
    visitor.on_attribute("vtype", m_vtype);
    return true;
}

void ov::intel_cpu::VNode::get_internal_vnodes(ov::NodeVector& nv, ngraph::Output<Node> value) {
    for (int i = 0; i < get_input_size(); i++) {
        if (value == input_value(i))
            return;
    }

    auto node = value.get_node_shared_ptr();
    if (dynamic_cast<VNode*>(node.get())) {
        nv.push_back(node);
    }

    // recursively find dependent input
    for (int i = 0; i < node->get_input_size(); i++) {
        get_internal_vnodes(nv, node->input_value(i));
    }
}

void ov::intel_cpu::VNode::clear_org() {
    m_org_outputs.clear();
}
