// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_prim.hpp"

// RNN primitive in oneDNN
//  https://oneapi-src.github.io/oneDNN/dev_guide_rnn.html
ov::intel_cpu::RNNPrim::RNNPrim(const std::string cell_type,
                                  const std::string dir,
                                  const bool batch_first,
                                  const ngraph::OutputVector& args)
    : Op(args),
    cell_type(cell_type),
    dir(dir),
    batch_first(batch_first) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::RNNPrim::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::RNNPrim>(cell_type, dir, batch_first, new_args);
}

/*
const Output<Node>& src,
const Output<Node>& src_iter,
const Output<Node>& scr_iter_c,
const Output<Node>& weight,
const Output<Node>& weight_iter,
const Output<Node>& weight_peehole,
const Output<Node>& weight_projection,
const Output<Node>& bias
*/
void ov::intel_cpu::RNNPrim::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 8,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected: 8.");

    auto exist = [](const ov::PartialShape & pshape) {
        return pshape.rank().get_length() > 1;
    };

    // shape inference
    if (cell_type == "lstm") {
        
        NODE_VALIDATION_CHECK(this, dir == "left2right" ||
                                    dir == "right2left" ||
                                    dir == "bidirectional_concat" ||
                                    dir == "bidirectional_sum");

        const auto src = get_input_partial_shape(0);
        const auto src_iter = get_input_partial_shape(1);
        const auto scr_iter_c = get_input_partial_shape(2);
        const auto weight = get_input_partial_shape(3);
        const auto weight_iter = get_input_partial_shape(4);
        const auto weight_peehole = get_input_partial_shape(5);
        const auto weight_projection = get_input_partial_shape(6);
        const auto bias = get_input_partial_shape(7);

        // For LSTM cells, the gates order is input, forget, candidate and output gate.

        // src/dst:
        //   batch_first:  dnnl_ntc:   (batch, seq_length, input channels)
        //  !batch_first:  dnnl_tnc:   (seq_length, batch, input channels)

        // weight:
        //  dnnl_ldigo: (num_layers, num_directions, input_channels, num_gates, output_channels)
        
        // weight_peehole/bias:
        //  dnnl_ldgo: (num_layers, num_directions, num_gates, output_channels)

        // weight_projection:
        //  dnnl_ldio:  (num_layers, num_directions, num_channels_in_hidden_state, num_channels_in_recurrent_projection)

        // src/dst_iter/scr/dst_iter_c:
        // dnnl_ldnc:  (num_layers, num_directions, batch, state channels)
        auto num_layers = weight[0];
        auto num_directions = weight[1];
        auto input_channels = weight[2];
        auto num_gates = weight[3];
        auto output_channels = weight[4];

        auto batch = batch_first ? src[0]:src[1];
        auto state_channels = exist(weight_projection) ? weight_projection[2]: output_channels;
        
        if (exist(weight_peehole)) {
            decltype(weight_peehole) expect{num_layers, num_directions, num_gates, output_channels};
            NODE_VALIDATION_CHECK(this, weight_peehole ==  expect);
        }

        if (exist(weight_projection)) {
            decltype(weight_projection) expect{num_layers, num_directions, num_gates, num_layers};
            NODE_VALIDATION_CHECK(this, weight_projection == expect);
        }
        
        if (exist(bias)) {
            decltype(bias) expect{num_layers, num_directions, num_gates, output_channels};
            NODE_VALIDATION_CHECK(this, bias == expect);
        }
        
        ov::PartialShape dst{num_layers, num_directions, batch, state_channels};

        set_output_size(3);
        auto output_type = get_input_element_type(0);
        set_output_type(0, output_type, dst);
        set_output_type(0, output_type, dst);
        set_output_type(0, output_type, dst);
    }else{
        NODE_VALIDATION_CHECK(this, false, "Unsurpported cell_type: ", cell_type);
    }
}

bool ov::intel_cpu::RNNPrim::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("cell_type", cell_type);
    visitor.on_attribute("direction", dir);
    visitor.on_attribute("batch_first", batch_first);
    return true;
}
