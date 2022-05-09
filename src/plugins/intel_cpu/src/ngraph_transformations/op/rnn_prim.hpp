// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/op/util/attr_types.hpp>

/*******************************
 RNN primitive in oneDNN
  https://oneapi-src.github.io/oneDNN/dev_guide_rnn.html
 LSTMSequence in ngraph 
  https://docs.openvino.ai/latest/openvino_docs_ops_sequence_LSTMSequence_1.html

 *  Why?
 *    LSTMSequence makes stacking very difficult, because it's output cannot be feed
 *  to input of next layer LSTMSequence directly due to in-compatible shape.
 *  
 *    notice following fact about the differences between the mem-descriptions in ngraph and oneDNN :
 *  
 *             ngraph  mem-desc            |         oneDNN mem-desc 
 *      --------------------------------------------------------------------------
 *      dimensions are in described order  |    dimensions are in canonical order (so we use {} enclose shape)
 *       lack of layout, or can be layout  |    (which is defined by oneDNN)
 *       in any order                      |      layout must be specified
 *       -------------------------------------------------------------------------
 *       (192,3,240) batch_first = false   |    {192,3,240} abc=tnc    <============ (A) consistent shape
 *       (3,192,240) batch_first = true    |    {192,3,240} bac=ntc    <============ (B) in-consistent shape
 *  
 *    when ngraph node producing (3,192,240) is transfomed to oneDNN node,
 *   it's output will have shape (3,192,240) instead of canonical order,
 *   a transpose is required to generate a oneDNN canonical shape (192,3,240)
 *   
 *   notice oneDNN can actually handle case (B) w/o transpose if we can give it correct mem-desc,
 *   so that transpose can be saved, that's why we need to differentiate CPU-plugin mem-desc
 *   with oneDNN mem-desc:
 *   
 *     CPU-plugin mem-desc is consistent with ngraph, layout is not exact: (3,192,240) abc
 *     oneDNN mem-desc is 
 *   
 * {192,3,240} bac=ntc   ===== if we can ignore the "canonical order" ===>   {3,192,240} abc
 * 
 *   CPU plugin node implements function defined by ngraph OP, so it must ensure the shape-interface
 *   is in consistent with ngraph, otherwise shape/inference/propagation would have problem.
 *   
 *   Specifically, each CPU plugin node initializes it's PrimitiveDescriptors with
 *   the shape described by ngraph OP's input/output shape plus a few favourite layouts.
 *   In this design, the ngraph shape is assumed to be of "canonical" shape, with
 *   physical meaning of each dimension predetermined by ngraph OP, that also means ngraph OP
 *   must ensure the input shape have exactly same canonical shape as oneDNN primitive.
 * 
 * if we invent a RNNPrim CPU plugin node accepting {3,192,240} as canonical shape which is in-consistent
 * with oneDNN RNN primitive, then we have to do the reshape  re-interpretation internally inside CPU plugin.
 * 
 * Specifically, suppose [C] is RNNPrim CPU plugin node, [P] is oneDNN RNN primitive, then
 *   [C] must accept 2 different shape: 
 *         batch_first version :  (3,192,240)
 *         time first version:    (192,3,240)
 *   
 *   [P] only support 1 canonical shape which is time first version {192,3,240}, thus for batch first
 *   input [C] needs to re-interpret it to {192,3,240} with layout bac=ntc, for calling [P] to accomplish
 *   the inference task.
 * 
 *   consider concat/stacking of [C] with batch first shape:
 * 
 *           C1<output:batch_first> == `(3,192,240)` ==> C2<input:batch_first>
 *           
 *   (Note: the `batch first` is not attribute of memory, but a way how ngraph/CPU_plugin node would interpret
 *     the input memory, so to outside world, the tensor is (3,192,240) with layout abc=tnc which seems to be 
 *     very wired, but here we must know that the layout abc means nothing w/o considering the consumer, here
 *     the consumer is [C] rather than [P], and [C] is interpreting (3,192,240) as batch first shape rather than
 *     oneDNN-RNN primitive's canoninal shape)
 *   
 *   C1 internally calls oneDNN with output mem-desc `{192,3,240}bac`, and C2 will do similar things by
 *   calling oneDNN with input mem-desc `{192,3,240}bac`.
 * 
 * Note that usually [C] dosen't have to do re-interpretation since:
 *   - many ngraph OP is defined to have same canonical shape as oneDNN:
 *       - Convolution:  oneDNN: NCDHW      ngraph: [N, C_IN, Z, Y, X]/[N, C_OUT, Z, Y, X]
 * 
 *   - many OP don't need a physical meanning for each dimension:
 *       - Softmax only needs softmax_axis which is set by attr;
 *       - Reduction compares src/dst dimension to know the reduction dimension space;
 * 
 * So with this RNNPrim OP, we can remove the transpose before LSTMSequence by simply setting correct input shape
 * convention `batch_first` or not `batch_first`. and we can also remove the transpose after LSTMSequence by setting
 * output shape convension.
 * 
 * 
 * LSTMSequence                RNNPrim                             RNN2             RNN primitive
 * (ngraph OP)           (CPU-spec ngraph OP)                  (CPU plugin)           (oneDNN)
 * -------------------------------------------------------------------------------------------------------------------------------
 * D: num_directions=1
 * N: batch_size
 * T: seq_len
 * H: hidden_size/output_size
 * I/C: input_size
 * 
 * 
 * w/o pre-transpose
 *  in   [N, T, C]        [N, T, C] batch_first = true          {N, T, C} abc       {T, N, C} bac=ntc(optimal layoutA)
 * with pre-transpose(in is to the pre-transpose node)
 *  in   [T, N, C]        [T, N, C] batch_first = false         {T, N, C} abc       {T, N, C} abc=tnc(optimal layoutB)

 * state [N, L*D, H]      [N, L*D, H]                           {N, L*D, H}         {L, D, N, H} ldnc(optimal layout)
                                                                 abc if N=1
                                                                 bac otherwise (reorder will be inserted)

 * output w/o post-transpose
 *  out  [N, D, T, H]     [N, D, T, H] batch_first = true       {N, D, T, H}        {T, N, D*H} bac=ntc(optimal layoutA)
 *                                                                abcd if D=1
 *                                                                acbd otherwise (reorder will be inserted)
 * output with post-transpose(out is from the post-transpose node)
 *  out  [T, D, N, H]     [T, D, N, H] batch_first = false      {T, D, N, H}        {T, N, D*H} abc=tnc(optimal layoutB)
 *                                                                abcd if D=1
 *                                                                acbd otherwise (reorder will be inserted)
 * 
 * weight and bias (W & R)
 *   - acb layout at CPU plugin level ensure a reorder node will be inserted automatically
 *   - gates order needs to be re-arranged (at some level)
 * 
 * weight      [L*D, 4*H, C]   [L*D, 4*H, C]                         {L*D, 4*H, C} acb   {L, D, C, 4, H},abcde=ldigo.
 * weight_iter [L*D, 4*H, H]   [L*D, 4*H, H]                         {L*D, 4*H, H} acb   {L, D, H, 4, H},abcde=ldigo.
 * bias        [L*D, 4*H]        [L*D, 4*H]                            {L*D, 4*H} ab       {L, D, 4, H},abcd=ldgo
 * 
 */

namespace ov {
namespace intel_cpu {

class RNNPrim : public ngraph::op::Op {
public:
    OPENVINO_OP("RNNPrim", "cpu_plugin_opset");

    using direction = op::RecurrentSequenceDirection;
    RNNPrim() = default;

    //  cell_type: rnn, lstm, gru, lbr_gru
    //  dir:       left2right, right2left, bidirectional_concat, bidirectional_sum
    RNNPrim(const std::string cell_type,
            const std::string dir,
            const bool input_batch_first,
            const bool output_batch_first,
            const ngraph::OutputVector& args);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    int64_t get_hidden_size();
    int64_t get_input_channels();
    int64_t get_num_layers();

    std::string cell_type;
    std::string dir;
    bool input_batch_first;
    bool output_batch_first;

    int64_t m_num_layers;
    int64_t m_batch;            // -1 when dynamic
    int64_t m_num_directions;
    int64_t m_input_size;       // input channels
    int64_t m_hidden_size;      // hidden state
    int64_t m_num_gates;
private:
    
};



}   // namespace intel_cpu
}   // namespace ov
