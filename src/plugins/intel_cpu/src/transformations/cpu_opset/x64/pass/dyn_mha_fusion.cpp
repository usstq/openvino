// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dyn_mha_fusion.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "simplify_fakequantize.hpp"
#include "transformations/cpu_opset/common/op/vnode.hpp"
#include "transformations/cpu_opset/x64/op/dyn_mha.hpp"
#include "utils/pattern_node.hpp"

namespace ov {
namespace intel_cpu {

static bool labels_eq_or_eq_static_dims(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    bool labels_exist_and_equal = false;

    auto lhs_label = ov::DimensionTracker::get_label(lhs);
    auto rhs_label = ov::DimensionTracker::get_label(rhs);
    auto table_l = ov::DimensionTracker::get_table_of_equivalence(lhs);
    auto table_r = ov::DimensionTracker::get_table_of_equivalence(rhs);
    if (table_l)
        labels_exist_and_equal = lhs_label != 0 && table_l->are_equal(lhs, rhs);
    else if (table_r)
        labels_exist_and_equal = lhs_label != 0 && table_r->are_equal(lhs, rhs);
    else
        labels_exist_and_equal = lhs_label != 0 && lhs_label == rhs_label;
    bool dims_are_static_and_equal = lhs.is_static() && lhs == rhs;
    return labels_exist_and_equal || dims_are_static_and_equal;
}

template <typename T>
bool is_constant_expected(Node* node, std::initializer_list<T> v) {
    auto constb = reinterpret_cast<opset1::Constant*>(node);
    if (constb) {
        if (std::is_integral<T>::value) {
            if (!constb->get_element_type().is_integral_number())
                return false;
        }
        if (std::is_floating_point<T>::value) {
            if (!constb->get_element_type().is_real())
                return false;
        }
        if (ov::shape_size(constb->get_shape()) != v.size())
            return false;
        return constb->cast_vector<T>() == std::vector<T>(v);
    }
    return false;
}

MHADynamicFloatFusionWhisper::MHADynamicFloatFusionWhisper() {
    MATCHER_SCOPE(MHADynamicFloatFusionWhisper);

    if (!std::getenv("DYNMHA") || atoi(std::getenv("DYNMHA")) == 0)
        return;

    auto alternatives = [](const OutputVector& outputs) {
        return std::make_shared<ov::pass::pattern::op::Or>(outputs);
    };

    auto maybe_reshaped = [&](const Output<Node>& out, const Output<Node>& target_shape) {
        auto reshaped = GenPattern<opset1::Reshape>({out, target_shape}, nullptr, {{"special_zero", 1}});
        return alternatives({reshaped, out});
    };

    auto q_input = GenPattern(ov::Rank(4));
    auto k_input = GenPattern(ov::Rank(4));
    auto v_input = GenPattern(ov::Rank(4));

    // maybe transposed [B,L,H,S] => [B,H,L,S]
    // (transposed when k/v is derived from fullyconnected layer directly
    //  no transpose in cross-attention when k/v is derived from parameters
    //  or concatenation of current k/v with past k/v)
    auto q_trans0213 = GenPattern<opset1::Transpose>({q_input, {0, 2, 1, 3}});
    auto q4d = alternatives({q_trans0213, q_input});

    auto k_trans0213 = GenPattern<opset1::Transpose>({k_input, {0, 2, 1, 3}});
    auto k4d = alternatives({k_trans0213, k_input});

    auto v_trans0213 = GenPattern<opset1::Transpose>({v_input, {0, 2, 1, 3}});
    auto v4d = alternatives({v_trans0213, v_input});

    // runtime shape of q4d/k4d/v4d must be : [B, H, L, S]

    auto q_shape2 = GenPattern("i32[3]");  // need validation at runtime: [B*H, L(-1), S]
    auto k_shape2 = GenPattern("i32[3]");  // need validation at runtime: [B*H, L(-1), S]
    auto v_shape2 = GenPattern("i32[3]");  // need validation at runtime: [B*H, L(-1), S]

    // q/k/v is reshaped from 4D to 3D
    // the reshape is expected to be [B,H,L,S] => [B*H,L,S]
    auto query = GenPattern<opset1::Reshape>({q4d, q_shape2}, nullptr, {{"special_zero", 1}});
    auto key = GenPattern<opset1::Reshape>({k4d, k_shape2}, nullptr, {{"special_zero", 1}});
    auto value = GenPattern<opset1::Reshape>({v4d, v_shape2}, nullptr, {{"special_zero", 1}});

    // query/key/value are all 3D in [BH, L, S] shape
    auto mm1 = GenPattern<opset1::MatMul>({query, key}, nullptr, {{"transpose_a", 0}, {"transpose_b", 1}});

    // optionally reshaped to 4D and add attention_mask
    auto attention_mask = GenPattern(ov::Rank(4));
    auto attn_shape4d = GenPattern("i32[4]");  // need validation at runtime: [B, H, qL, kL]
    auto attn_shape3d = GenPattern("i32[3]");  // need validation at runtime: [B*H, qL, kL]
    auto attn_4d = GenPattern<opset1::Reshape>({mm1, attn_shape4d}, nullptr, {{"special_zero", 1}});
    auto attn_addmask = GenPattern<opset1::Add>({attn_4d, attention_mask}, nullptr, {{"auto_broadcast", "numpy"}});
    auto attn_3d = GenPattern<opset1::Reshape>({attn_addmask, attn_shape3d}, nullptr, {{"special_zero", 1}});

    auto attn_score = alternatives({attn_3d, mm1});

    auto softmax = GenPattern<opset1::Softmax>({attn_score}, nullptr, {{"axis", 2}});
    auto mm2 = GenPattern<opset1::MatMul>({softmax, value}, nullptr, {{"transpose_a", 0}, {"transpose_b", 0}});

    auto result_pattern = mm2;

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto result_node = m.get_match_root();
        auto& pattern_to_value = m.get_pattern_value_map();
        auto mm2_v = pattern_to_value.at(mm2);
        auto q_input_v = pattern_to_value.at(q_input);
        OutputVector args;
        MHADynamic::Config cfg;

        auto q_is_transposed = pattern_to_value.count(q_trans0213);
        auto q_input_pshape = q_input_v.get_partial_shape();
        // q_is_transposed ? [B, L, H, S] : [B, H, L, S]
        auto head_count = q_is_transposed ? q_input_pshape[2] : q_input_pshape[1];
        auto head_size = q_input_pshape[3];
        if (head_count.is_dynamic())
            return false;
        if (head_size.is_dynamic())
            return false;

        cfg.head_count = head_count.get_length();
        cfg.head_size = head_size.get_length();

        args.push_back(pattern_to_value.at(q_input));
        args.push_back(pattern_to_value.at(k_input));
        args.push_back(pattern_to_value.at(v_input));

        // query pre-process chain
        if (pattern_to_value.count(q_trans0213)) {
            cfg.relayout_query.emplace_back(MHADynamic::ReLayout::transpose({0, 2, 1, 3}));
        }
        args.push_back(pattern_to_value.at(q_shape2));
        cfg.q_reshape_to_3d = true;

        // key pre-process chain
        if (pattern_to_value.count(k_trans0213)) {
            cfg.relayout_key.emplace_back(MHADynamic::ReLayout::transpose({0, 2, 1, 3}));
        }
        args.push_back(pattern_to_value.at(k_shape2));
        cfg.k_reshape_to_3d = true;

        // value pre-process chain
        if (pattern_to_value.count(v_trans0213)) {
            cfg.relayout_value.emplace_back(MHADynamic::ReLayout::transpose({0, 2, 1, 3}));
        }
        args.push_back(pattern_to_value.at(v_shape2));
        cfg.v_reshape_to_3d = true;

        if (pattern_to_value.count(attention_mask)) {
            args.push_back(pattern_to_value.at(attention_mask));
            cfg.with_attn_mask = true;

            args.push_back(pattern_to_value.at(attn_shape4d));
            args.push_back(pattern_to_value.at(attn_shape3d));
            cfg.with_attn_mask_reshapes = true;
        }

        //=============================================================
        // extend pattern down-stream (this has to be done inside callback
        // because these optional pattern-tails will always be omitted
        // in the top-down comparing process)
        cfg.result_is_3d = true;

        auto next_ops = mm2_v.get_target_inputs();
        if (next_ops.size() == 1) {
            auto it = next_ops.begin();
            auto reshape = reinterpret_cast<opset1::Reshape*>(it->get_node());
            if (reshape && it->get_index() == 0) {
                args.push_back(reshape->input_value(1));  // check at runtime : [B, H, qL, S]
                cfg.result_is_3d = false;
                next_ops = reshape->output(0).get_target_inputs();
            }
        }
        if (next_ops.size() == 1) {
            auto trans = reinterpret_cast<opset1::Transpose*>(next_ops.begin()->get_node());
            if (trans) {
                auto order_index = 1 - next_ops.begin()->get_index();
                if (is_constant_expected(trans->input_value(order_index).get_node(), {0, 2, 1, 3})) {
                    result_node = trans->shared_from_this();
                    cfg.trans_result_to_BLHS = true;
                }
            }
        }

        auto vnode = std::make_shared<MHADynamic>(args, cfg);

        ngraph::replace_node(result_node, vnode);
        std::cout << " MHADynamicFloatFusion found >>>>>>>>>>>>> " << result_node << std::endl;
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result_pattern, matcher_name);
    this->register_matcher(m, callback);
}

MHADynamicFloatFusion4D::MHADynamicFloatFusion4D() {
    MATCHER_SCOPE(MHADynamicFloatFusion4D);

    if (!std::getenv("DYNMHA") || atoi(std::getenv("DYNMHA")) == 0)
        return;

    auto with_only_one_element = [](const Output<Node>& value) -> bool {
        const auto& pshape = value.get_partial_shape();
        if (!pshape.is_static())
            return false;
        return shape_size(pshape.get_shape()) == 1;
    };

    auto alternatives = [](const OutputVector& outputs) {
        return std::make_shared<ov::pass::pattern::op::Or>(outputs);
    };

    // [B, L, H*3*S] ==reshape(0,0,H,3,S)==> [B, L, H, 3, S] ==gather==> [B, L, H, S] ==transpose==> [B,H,L,S]
    // [B, L, 3*H*S] ==reshape(0,0,3,H,S)==> [B, L, 3, H, S] ==gather==> [B, L, H, S] ==transpose==> [B,H,L,S]
    // [B, L, H*S] ==reshape==>                               [B, L, H, S] ==transpose==> [B,H,L,S]
    auto H = Symbol("H");
    auto S = Symbol("S");

    auto q_BLH3S_3d = GenPattern(ov::Rank(3));
    auto q_BLHS_4d = GenPattern(ov::Rank(4));
    auto q_BHLS_4d = GenPattern(ov::Rank(4));

    auto q_BLH3S_3d_Reshape =
        GenPattern<opset1::Reshape>({q_BLH3S_3d, {0, 0, H, 3, S}}, nullptr, {{"special_zero", 1}});
    auto q_BLH3S_3d_Gather_2 = GenPattern<opset8::Gather>({q_BLH3S_3d_Reshape, 0, 3}, nullptr, {{"batch_dims", 0}});
    auto q_BLH3S_3d_Transpose =
        GenPattern<opset1::Transpose>({alternatives({q_BLH3S_3d_Gather_2, q_BLHS_4d}), {0, 2, 1, 3}});

    auto query = alternatives({q_BLH3S_3d_Transpose, q_BHLS_4d});

    auto k_4d = GenPattern(ov::Rank(4));

    auto key = alternatives({k_4d});

    auto v_4d = GenPattern(ov::Rank(4));

    // [batch_size, head_num, k_len, head_size]
    auto value = alternatives({v_4d});

    auto qk_scale = ov::pass::pattern::any_input(with_only_one_element);

    auto maybe_qk_scaled = [&](const Output<Node>& out) {
        auto out_scaled = GenPattern<opset1::Multiply>({out, qk_scale});
        return alternatives({out_scaled, out});
    };

    // optionally, key is multiplied with 1/sqrt(head_size)
    // [batch_size, head_num, q_len, k_len]
    auto matmul1 = GenPattern<opset1::MatMul>({query, maybe_qk_scaled(key)}, nullptr, {{"transpose_a", 0}});

    // some LLM apply scalar scale to result of Q*K' (and it maybe not a constant)
    auto opt0 = maybe_qk_scaled(matmul1);

    // optional mask steps are assumed to be of fixed order
    //      matmul1=>[+alibi_mask] => [select] => [+attention_mask] =>attn_score
    // these mask must be rank4 and broadcast-able to matmul1
    auto alibi_mask = GenPattern(ov::Rank(4));
    auto causal_mask = GenPattern(ov::Rank(4));
    auto attention_mask = GenPattern(ov::Rank(4));

    auto add1 = GenPattern<opset1::Add>({opt0, alibi_mask});
    auto opt1 = alternatives({add1, opt0});

    // causal_mask may be sliced from a constant

    auto causal_mask_slice1 = GenPattern<opset8::Slice>({causal_mask,
                                                         ov::pass::pattern::any_input(with_only_one_element),
                                                         ov::pass::pattern::any_input(with_only_one_element),
                                                         ov::pass::pattern::any_input(with_only_one_element),
                                                         ov::pass::pattern::any_input(with_only_one_element)});
    auto causal_mask_slice2 = GenPattern<opset8::Slice>({causal_mask_slice1,
                                                         ov::pass::pattern::any_input(with_only_one_element),
                                                         ov::pass::pattern::any_input(with_only_one_element),
                                                         ov::pass::pattern::any_input(with_only_one_element),
                                                         ov::pass::pattern::any_input(with_only_one_element)});

    auto causal_mask_stridedslice1 = GenPattern<opset1::StridedSlice>(
        {causal_mask, values_info("i32[3]"), values_info("i32[3]"), values_info("i32[3]")},
        nullptr,
        {{"begin_mask", {1, 1, 0}},
         {"end_mask", {1, 1, 0}},
         {"new_axis_mask", {}},
         {"shrink_axis_mask", {}},
         {"ellipsis_mask", {}}});

    auto causal_mask_stridedslice2 = GenPattern<opset1::StridedSlice>(
        {
            causal_mask_stridedslice1,
            values_info("i32[4]"),
            values_info("i32[4]"),
            values_info("i32[4]"),
        },
        nullptr,
        {{"begin_mask", {1, 1, 1, 0}},
         {"end_mask", {1, 1, 1, 0}},
         {"new_axis_mask", {}},
         {"shrink_axis_mask", {}},
         {"ellipsis_mask", {}}});

    auto causal_mask_opt = alternatives(
        {causal_mask_slice2, causal_mask_slice1, causal_mask_stridedslice2, causal_mask_stridedslice1, causal_mask});

    auto select1 = GenPattern<opset1::Select>({causal_mask_opt, -FLT_MAX, opt1});
    auto select2 = GenPattern<opset1::Select>({causal_mask_opt, opt1, -FLT_MAX});
    auto opt2 = alternatives({select1, select2, opt1});

    auto add2 = GenPattern<opset1::Add>({maybe_qk_scaled(opt2), attention_mask});

    // [batch_size, head_num, q_len, head_size]
    auto opt3 = alternatives({add2, opt2});

    // some model has this maximum that changes nothing
    auto maximum = GenPattern<opset1::Maximum>({opt3, {-FLT_MAX}}, nullptr, {{"auto_broadcast", "numpy"}});

    auto attn_score = alternatives({maximum, opt3});

    // [batch_size, head_num, q_len, k_len]
    // softmax along last axis
    auto attn_weight = GenPattern<opset1::Softmax>({attn_score}, nullptr, {{"axis", 3}});

    // no transpose in matmul2 which is assumed to be [B,H,qL,kL] x [B,H,kL,S] =>
    // [batch_size, head_num, q_len, head_size]
    auto matmul2 = GenPattern<opset1::MatMul>({attn_weight, value}, nullptr, {{"transpose_a", 0}, {"transpose_b", 0}});

    auto result_pattern = matmul2;
    // strictly in MHA, value's batch_size & head_num must be the same as attn_weight
    // and no broadcast should happen, this can be done with the help of symbolic shape
    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto result_node = m.get_match_root();
        auto& pattern_to_value = m.get_pattern_value_map();
        auto matmul1_v = pattern_to_value.at(matmul1);
        auto attn_weight_v = pattern_to_value.at(attn_weight);
        auto matmul2_v = pattern_to_value.at(matmul2);

        /*
        auto value_v = pattern_to_value.at(value);
        bool is_batch_size_same =
            labels_eq_or_eq_static_dims(matmul1_v.get_partial_shape()[0], value_v.get_partial_shape()[0]);
        if (!is_batch_size_same)
            return false;

        bool is_head_num_same =
            labels_eq_or_eq_static_dims(matmul1_v.get_partial_shape()[1], value_v.get_partial_shape()[1]);
        if (!is_head_num_same)
            return false;
        */
        auto head_count = matmul2_v.get_partial_shape()[1];
        auto head_size = matmul2_v.get_partial_shape()[3];
        if (head_count.is_dynamic())
            return false;
        if (head_size.is_dynamic())
            return false;

        auto matmul1_node = std::dynamic_pointer_cast<opset1::MatMul>(matmul1_v.get_node_shared_ptr());
        auto matmul2_node = std::dynamic_pointer_cast<opset1::MatMul>(matmul2_v.get_node_shared_ptr());

        // key into matmul1 can be [B,H,S,qL]
        auto trans_key = matmul1_node->get_transpose_b();

        std::map<std::string, double> symbol_name2value;
        if (!validate_matched_symbols(m, symbol_name2value)) {
            return false;
        }

        OutputVector args;
        MHADynamic::Config cfg;

        cfg.head_count = head_count.get_length();
        cfg.head_size = head_size.get_length();

        //=============================================================
        if (pattern_to_value.count(q_BLH3S_3d)) {
            args.push_back(pattern_to_value.at(q_BLH3S_3d));
            cfg.relayout_query = {MHADynamic::ReLayout::reshape({0, 0, cfg.head_count, 3, cfg.head_size}),
                                  MHADynamic::ReLayout::gather(0, 3),
                                  MHADynamic::ReLayout::transpose({0, 2, 1, 3})};
        } else if (pattern_to_value.count(q_BLHS_4d)) {
            args.push_back(pattern_to_value.at(q_BLHS_4d));
            cfg.relayout_query = {MHADynamic::ReLayout::transpose({0, 2, 1, 3})};
        } else if (pattern_to_value.count(q_BHLS_4d)) {
            args.push_back(pattern_to_value.at(q_BHLS_4d));
            cfg.relayout_query = {};
        }

        //=============================================================
        if (pattern_to_value.count(k_4d)) {
            args.push_back(pattern_to_value.at(k_4d));
            // key input to matmul1 has normalized shape of [B, H, kL, S], thus trans_key==true
            cfg.relayout_key = {};
            // when trans_key==false(in bloom-like topology), need transpose to normalize the shape
            if (!trans_key)
                cfg.relayout_key.push_back(MHADynamic::ReLayout::transpose({0, 1, 3, 2}));
        }

        //=============================================================
        if (pattern_to_value.count(v_4d)) {
            args.push_back(pattern_to_value.at(v_4d));
            cfg.relayout_value = {};
        }

        //=============================================================
        // extend pattern down-stream
        auto matmul2_to = matmul2_v.get_target_inputs();
        if (matmul2_to.size() == 1) {
            auto trans = reinterpret_cast<opset1::Transpose*>(matmul2_to.begin()->get_node());
            if (trans) {
                auto order_index = 1 - matmul2_to.begin()->get_index();
                if (is_constant_expected(trans->input_value(order_index).get_node(), {0, 2, 1, 3})) {
                    result_node = trans->shared_from_this();
                    cfg.trans_result_to_BLHS = true;
                }
            }
        }

        if (pattern_to_value.count(qk_scale)) {
            cfg.with_qk_scale = true;
            args.push_back(pattern_to_value.at(qk_scale));
        }

        // mask_alibi, mask_causal, mask_attention
        if (pattern_to_value.count(alibi_mask)) {
            // with alibi_mask
            cfg.with_alibi_mask = true;
            args.push_back(pattern_to_value.at(alibi_mask));
        }

        if (pattern_to_value.count(select1)) {
            args.push_back(pattern_to_value.at(causal_mask));
            cfg.select_nfltmax_at_0 = false;
            cfg.with_causal_mask = true;
        } else if (pattern_to_value.count(select2)) {
            args.push_back(pattern_to_value.at(causal_mask));
            cfg.select_nfltmax_at_0 = true;
            cfg.with_causal_mask = true;
        }

        // absorb causal mask slice (all slice is down on 1 axis)
        if (cfg.with_causal_mask) {
            if (pattern_to_value.count(causal_mask_slice1)) {
                cfg.with_causal_mask_slice1 = true;
                auto slice = pattern_to_value.at(causal_mask_slice1);
                args.push_back(slice.get_node()->input_value(1));
                args.push_back(slice.get_node()->input_value(2));
                args.push_back(slice.get_node()->input_value(3));
                args.push_back(slice.get_node()->input_value(4));
            }

            if (pattern_to_value.count(causal_mask_slice2)) {
                cfg.with_causal_mask_slice2 = true;
                auto slice = pattern_to_value.at(causal_mask_slice2);
                args.push_back(slice.get_node()->input_value(1));
                args.push_back(slice.get_node()->input_value(2));
                args.push_back(slice.get_node()->input_value(3));
                args.push_back(slice.get_node()->input_value(4));
            }

            if (pattern_to_value.count(causal_mask_stridedslice1)) {
                cfg.with_causal_mask_stridedslice1 = true;
                auto slice = pattern_to_value.at(causal_mask_stridedslice1);
                args.push_back(slice.get_node()->input_value(1));
                args.push_back(slice.get_node()->input_value(2));
                args.push_back(slice.get_node()->input_value(3));
            }

            if (pattern_to_value.count(causal_mask_stridedslice2)) {
                cfg.with_causal_mask_stridedslice2 = true;
                auto slice = pattern_to_value.at(causal_mask_stridedslice2);
                args.push_back(slice.get_node()->input_value(1));
                args.push_back(slice.get_node()->input_value(2));
                args.push_back(slice.get_node()->input_value(3));
            }
        }

        if (pattern_to_value.count(attention_mask)) {
            cfg.with_attn_mask = true;
            args.push_back(pattern_to_value.at(attention_mask));
        }

        auto vnode = std::make_shared<MHADynamic>(args, cfg);

        ngraph::replace_node(result_node, vnode);
        std::cout << " MHADynamicFloatFusion found >>>>>>>>>>>>> " << attn_weight_v << std::endl;
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result_pattern, matcher_name);
    this->register_matcher(m, callback);
}

class VNodeIn : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("VNodeIn", "0");

    int get_all_labels(const Output<Node>& value, OutputVector& all_labels) {
        auto node = value.get_node_shared_ptr();
        if (dynamic_cast<ov::pass::pattern::op::Label*>(node.get())) {
            all_labels.push_back(node);
            return 1;
        }
        int ret = 0;
        for (size_t i = 0; i < node->get_input_size(); i++) {
            ret += get_all_labels(node->input_value(i), all_labels);
        }
        return ret;
    }

    template <typename F>
    VNodeIn(const char* vtype, F func, std::function<bool(OutputVector&)> pred = {}) {
        MATCHER_SCOPE(VNodeIn);

        if (auto* wlist = std::getenv("VNODE_WLIST")) {
            std::string vnode_whitelist(wlist);
            if (vnode_whitelist.find(std::string(vtype) + ",") == std::string::npos) {
                return;
            }
        }

        OutputVector fake_inputs;
        for (int i = 0; i < 32; i++) {
            fake_inputs.push_back(GenInput());
        }
        auto pattern_values = func(fake_inputs);

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            auto& pvmap = m.get_pattern_value_map();
            // auto node_src = pvmap.at(select.node).get_node_shared_ptr();
            auto root_value = m.get_match_value();
            std::map<std::string, double> symbol_name2value;
            std::cout << "VNodeIn::callback " << root_value << std::endl;
            if (!validate_matched_symbols(m, symbol_name2value)) {
                std::cout << "VNodeIn: symbol validation failed!" << std::endl;
                return false;
            }

            OutputVector real_inputs;
            for (auto& in : fake_inputs) {
                auto it = pvmap.find(in.get_node_shared_ptr());
                if (it == pvmap.end())
                    break;
                real_inputs.push_back(it->second);
            }
            OutputVector real_outputs;
            for (size_t i = 0; i < pattern_values.size(); i++) {
                real_outputs.push_back(pvmap[pattern_values[i].get_node_shared_ptr()]);
            }

            if (pred && !pred(real_inputs))
                return false;

            auto vnode = std::make_shared<VNode>(real_inputs, real_outputs, vtype);

            vnode->get_rt_info()["symbol_name2value"] = symbol_name2value;

            for (size_t i = 0; i < pattern_values.size(); i++) {
                auto out = pvmap[pattern_values[i].get_node_shared_ptr()];
                ngraph::replace_node(out.get_node_shared_ptr(), {vnode->output(i)});
                auto name = out.get_node_shared_ptr()->get_friendly_name();
                vnode->output(i).set_names({name});
            }
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(pattern_values[0], matcher_name);
        this->register_matcher(m, callback);
    }
};

#include "vnode_bloom.txt"
#include "vnode_gpt2.txt"
#include "vnode_gptneox_attn.txt"
#include "vnode_llama.txt"
#include "vnode_opt.txt"
#include "vnode_whisper.txt"

MHADynamicVNodeIn::MHADynamicVNodeIn() {
    /*
    add_matcher<VNodeIn>("gptneox_attention", vnode_gptneox_attn);
    add_matcher<VNodeIn>("gpt2_attention", vnode_gpt2);
    add_matcher<VNodeIn>("opt_attention", vnode_opt_attn);
    add_matcher<VNodeIn>("open_llama_attention", vnode_llama_attn);
    add_matcher<VNodeIn>("bloom_attention", vnode_bloom_attn);
    add_matcher<VNodeIn>("bloom2_attention", vnode_bloom2_attn);
    */
    add_matcher<VNodeIn>("whisper_enc_attention", vnode_whisper_enc_attention);
    add_matcher<VNodeIn>("whisper_dec_self_attn", vnode_whisper_dec_self_attn);
    add_matcher<VNodeIn>("whisper_dec_enc_attn", vnode_whisper_dec_enc_attn);
    add_matcher<VNodeIn>("whisper_dec2_self_attn", vnode_whisper_dec2_self_attn);
    add_matcher<VNodeIn>("whisper_dec2_enc_attn", vnode_whisper_dec2_enc_attn);
}

#if 0
MHADynamicVNodeOut::MHADynamicVNodeOut() {
    MATCHER_SCOPE(MHADynamicVNodeOut);
    auto vnode = ov::pass::pattern::wrap_type<VNode>();
    std::string vnode_whitelist = std::getenv("VNODE_WLIST") ? std::getenv("VNODE_WLIST") : "gpt2_attention,";

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        // auto& pattern_to_output = m.get_pattern_value_map();
        auto root_value = m.get_match_value();
        auto vnode = std::dynamic_pointer_cast<VNode>(root_value.get_node_shared_ptr());

        std::cout << "MHADynamicVNodeOut found : " << root_value.get_node_shared_ptr()->get_friendly_name()
                  << std::endl;

        if (vnode_whitelist.find(vnode->get_vtype() + ",") != std::string::npos) {
            // leave this VNode since it's in the white-list, clear it's internal references to original subgraph
            vnode->clear_org();
            return false;
        }

        // all nodes inside vnode may contain vnodes
        OutputVector org_outputs = vnode->get_org();
        ov::NodeVector nv;
        vnode->get_internal_vnodes(nv, org_outputs[0]);
        for (auto& n : nv) {
            register_new_node(n);
        }

        ngraph::replace_node(vnode, org_outputs);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(vnode, matcher_name);
    this->register_matcher(m, callback);
}
#endif

}  // namespace intel_cpu
}  // namespace ov