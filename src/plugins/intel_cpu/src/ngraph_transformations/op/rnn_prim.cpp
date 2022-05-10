// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_prim.hpp"


ov::intel_cpu::RNNPrim::RNNPrim(const std::string cell_type,
                                  const std::string dir,
                                  const bool input_batch_first,
                                  const bool output_batch_first,
                                  const bool extra_direction_dim,
                                  const ngraph::OutputVector& args)
    : Op(args),
    cell_type(cell_type),
    dir(dir),
    input_batch_first(input_batch_first),
    output_batch_first(output_batch_first),
    extra_direction_dim(extra_direction_dim) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::RNNPrim::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::RNNPrim>(cell_type, dir, input_batch_first, output_batch_first, extra_direction_dim, new_args);
}

void ov::intel_cpu::RNNPrim::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 6,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected: 6.");

    auto exist = [](const ov::PartialShape & pshape) {
        return pshape.rank().get_length() > 1;
    };

    // shape inference
    if (cell_type == "lstm") {

        if (dir == "left2right" || dir == "right2left") {
            m_num_directions = 1;
        } else if (dir == "bidirectional_concat") {
            m_num_directions = 2;
        } else {
            NODE_VALIDATION_CHECK(this, false, "Unsupported dir:", dir);
        }
        /*
        input 
            0 src,              X           [N, T, C] batch_first / [T, N, C]
            1 src_iter,         Hi          [N, L*D, H] 
            2 scr_iter_c,       Ci          [N, L*D, H]
            3 weight,           W           [L*D, 4*H, C]
            4 weight_iter,      R           [L*D, 4*H, H]
            5 bias              B           [L*D, 4*H]

        output
            0 dst:              Y           [N, D, T, C] batch_first / [T, D, N, C]
            1 dst_iter:         Ho          [N, L*D, H]
            2 dst_iter_c:       Co          [N, L*D, H]
        */
        const auto X = get_input_partial_shape(0);
        const auto Hi = get_input_partial_shape(1);
        const auto Ci = get_input_partial_shape(2);
        const auto W = get_input_partial_shape(3);
        const auto R = get_input_partial_shape(4);
        const auto B = get_input_partial_shape(5);

        //std::cout << this->get_friendly_name() << std::endl;
        //for(int i=0;i<get_input_size();i++)
        //    std::cout << "input[" << i << "]" << get_input_partial_shape(i) << std::endl;

        const auto dim_LD = W[0];
        NODE_VALIDATION_CHECK(this, R[0] == dim_LD && B[0] == dim_LD);

        const auto dim_N = input_batch_first ? X[0] : X[1];
        const auto dim_T = input_batch_first ? X[1] : X[0];
        const auto dim_C = X[2];
        const auto dim_H = Hi[2];

        ov::PartialShape state_shape{dim_N, dim_LD, dim_H};
        NODE_VALIDATION_CHECK(this, (Hi == state_shape));
        NODE_VALIDATION_CHECK(this, (Ci == state_shape));

        NODE_VALIDATION_CHECK(this, (W == ov::PartialShape{dim_LD, dim_H*4, dim_C}), "W:", W, " expect:",
                    ov::PartialShape{dim_LD, dim_H*4, dim_C});
        NODE_VALIDATION_CHECK(this, (R == ov::PartialShape{dim_LD, dim_H*4, dim_H}));
        NODE_VALIDATION_CHECK(this, (B == ov::PartialShape{dim_LD, dim_H*4}));
        
        m_batch = dim_N.is_dynamic() ? -1: dim_N.get_length();
        m_num_layers = dim_LD.get_length() / m_num_directions;
        m_input_size = dim_C.get_length();  // this must be static dimension
        m_hidden_size = dim_H.get_length(); // this must be static dimension
        m_num_gates = 4;

        set_output_size(3);
        auto output_type = get_input_element_type(0);
        
        if (extra_direction_dim) {
            if (output_batch_first) {
                set_output_type(0, output_type,
                                ov::PartialShape{dim_N, m_num_directions, dim_T, dim_H});
            } else {
                set_output_type(0, output_type,
                                ov::PartialShape{dim_T, m_num_directions, dim_N, dim_H});
            }
        } else {
            if (output_batch_first) {
                set_output_type(0, output_type,
                                ov::PartialShape{dim_N, dim_T, ov::Dimension(m_num_directions) * dim_H});
            } else {
                set_output_type(0, output_type,
                                ov::PartialShape{dim_T, dim_N, ov::Dimension(m_num_directions) * dim_H});
            }
        }

        set_output_type(1, output_type, state_shape);
        set_output_type(2, output_type, state_shape);

        for(int i=0;i<get_output_size();i++)
            std::cout << "out[" << i << "]" << get_output_partial_shape(i) << std::endl;

    }else{
        NODE_VALIDATION_CHECK(this, false, "Unsurpported cell_type: ", cell_type);
    }
}

bool ov::intel_cpu::RNNPrim::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("cell_type", cell_type);
    visitor.on_attribute("direction", dir);
    visitor.on_attribute("input_batch_first", input_batch_first);
    visitor.on_attribute("output_batch_first", output_batch_first);
    visitor.on_attribute("extra_direction_dim", extra_direction_dim);
    return true;
}
