// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_sequences_fusion.hpp"
#include "op/rnn_prim.hpp"
#include <numeric>
#include <climits>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "transformations/utils/utils.hpp"

ov::intel_cpu::MultilayerSequenceFusion::MultilayerSequenceFusion() {
    auto RNNP = atoi(std::getenv("RNNP")?:"0");
    if ((RNNP & 2) == 0)
        return;
    auto X1 = ngraph::pattern::any_input();
    auto Hi1 = ngraph::pattern::any_input();
    auto Ci1 = ngraph::pattern::any_input();
    auto W1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto R1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto B1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto RNN1 = ngraph::pattern::wrap_type<ov::intel_cpu::RNNPrim>({ X1, Hi1, Ci1, W1, R1, B1}, [](ngraph::Output<ngraph::Node> output) {
        return ngraph::pattern::consumers_count(1)(output);
    });

    auto Hi2 = ngraph::pattern::any_input();
    auto Ci2 = ngraph::pattern::any_input();
    auto W2 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto R2 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto B2 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto RNN2 = ngraph::pattern::wrap_type<ov::intel_cpu::RNNPrim>({ RNN1, Hi2, Ci2, W2, R2, B2});
    
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        auto rnn1 = std::dynamic_pointer_cast<ov::intel_cpu::RNNPrim>(pattern_to_output[RNN1].get_node_shared_ptr());
        auto rnn2 = std::dynamic_pointer_cast<ov::intel_cpu::RNNPrim>(pattern_to_output[RNN2].get_node_shared_ptr());
        
        auto hi1 = pattern_to_output[Hi1].get_node_shared_ptr();
        auto ci1 = pattern_to_output[Ci1].get_node_shared_ptr();
        auto w1 = pattern_to_output[W1].get_node_shared_ptr();
        auto r1 = pattern_to_output[R1].get_node_shared_ptr();
        auto b1 = pattern_to_output[B1].get_node_shared_ptr();

        auto hi2 = pattern_to_output[Hi2].get_node_shared_ptr();
        auto ci2 = pattern_to_output[Ci2].get_node_shared_ptr();
        auto w2 = pattern_to_output[W2].get_node_shared_ptr();
        auto r2 = pattern_to_output[R2].get_node_shared_ptr();
        auto b2 = pattern_to_output[B2].get_node_shared_ptr();


        // RNN primitive can only stack multi layers when input/output channels are consistent
        if (rnn1->m_hidden_size * rnn1->m_num_directions == rnn1->m_input_size 
          && rnn2->m_hidden_size * rnn2->m_num_directions == rnn2->m_input_size
          && rnn1->m_input_size == rnn2->m_input_size
          && rnn1->m_num_directions == rnn2->m_num_directions
          && rnn1->cell_type == rnn2->cell_type
          && rnn1->dir == rnn2->dir) {

            std::cout << "========================= MultilayerSequenceFusion" << std::endl;
            std::cout << pattern_to_output[RNN1].get_node_shared_ptr()->get_friendly_name() << std::endl;
            std::cout << pattern_to_output[RNN2].get_node_shared_ptr()->get_friendly_name() << std::endl;

            // concat w/r/b along dimension 0
            // weight      [L*D, 4*H, C]   [L*D, 4*H, C]                    {L*D, 4*H, C} acb   {L, D, C, 4, H},abcde=ldigo.
            // weight_iter [L*D, 4*H, H]   [L*D, 4*H, H]                    {L*D, 4*H, H} acb   {L, D, H, 4, H},abcde=ldigo.
            // bias        [L*D, 4*H]      [L*D, 4*H]                       {L*D, 4*H} ab       {L, D, 4, H},abcd=ldgo
            auto w = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{w1, w2}, 0);
            auto r = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{r1, r2}, 0);
            auto b = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{b1, b2}, 0);
            
            // concat Hi1&Hi2 Ci1&Ci2
            // state [N, L*D, H]      [N, L*D, H]                           {N, L*D, H}         {L, D, N, H} ldnc(optimal layout)
            auto hi = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{hi1, hi2}, 1);
            auto ci = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{ci1, ci2}, 1);

            // create stacked rnn
            auto rnn_prim = std::make_shared<ov::intel_cpu::RNNPrim>(
                                        rnn1->cell_type,
                                        rnn1->dir,
                                        rnn1->input_batch_first,
                                        rnn2->output_batch_first,
                                        rnn2->extra_direction_dim,
                                        ngraph::OutputVector{
                                            rnn1->input_value(0),
                                            hi->output(0),
                                            ci->output(0),
                                            w->output(0),
                                            r->output(0),
                                            b->output(0)
                                        });
            //
            auto ho1 = std::make_shared<ngraph::opset8::Slice>(
                rnn_prim->output(1),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {0}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {-rnn_prim->m_num_directions}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {1}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {1})
            );

            auto ho2 = std::make_shared<ngraph::opset8::Slice>(
                rnn_prim->output(1),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {-rnn_prim->m_num_directions}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {INT_MAX}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {1}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {1})
            );

            auto co1 = std::make_shared<ngraph::opset8::Slice>(
                rnn_prim->output(2),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {0}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {-rnn_prim->m_num_directions}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {1}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {1})
            );
            auto co2 = std::make_shared<ngraph::opset8::Slice>(
                rnn_prim->output(2),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {-rnn_prim->m_num_directions}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {INT_MAX}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {1}),
                ngraph::opset1::Constant::create(ngraph::element::i32, {1}, {1})
            );

            rnn_prim->set_friendly_name(rnn1->get_friendly_name() + "_plus_" + rnn2->get_friendly_name());
            ngraph::copy_runtime_info({rnn1, rnn2}, rnn_prim);
            ngraph::replace_node(rnn1, ngraph::OutputVector{rnn2->output(0), ho1->output(0), co1->output(0)});
            ngraph::replace_node(rnn2, ngraph::OutputVector{rnn_prim->output(0), ho2->output(0), co2->output(0)});
            return true;
        }
        
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(RNN2, "MultilayerSequenceFusion");
    this->register_matcher(m, callback);
}
