// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "identity.hpp"

#include<openvino/core/model.hpp>
#include<openvino/core/node.hpp>
#include<openvino/core/version.hpp>
#include"openvino/opsets/opset.hpp"

//#include"omp.h"
#include "tbb/parallel_for.h"

#include <immintrin.h>

std::string get_version() {
    return"0.1";
}

class Initor {
public:
    Initor() {
        const char * p_add_to_opset = std::getenv("add_RnntUpdate_opset8");
        if (!p_add_to_opset)
            p_add_to_opset = "0";
        if (atoi(p_add_to_opset) > 0) {
            auto &ops = const_cast<ov::OpSet&>(ov::get_opset8());
            ops.insert<TemplateExtension::RnntUpdate>();
        }
    }
};

Initor g_initor;

using namespace TemplateExtension;

//! [op:ctor]
Identity::Identity(const ov::Output<ov::Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void Identity::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> Identity::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<Identity>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool Identity::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool Identity::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto in = inputs[0];
    auto out = outputs[0];
    out.set_shape(in.get_shape());
    memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool Identity::has_evaluate() const {
    return true;
}
//! [op:evaluate]

/*

self.rnnt_update =  opset.ops._get_node_factory_opset8().create("RnntUpdate",
                        [
                            current_iter,
                            features,
                            logits,
                            Ho1,
                            Co1,
                            Ho2,
                            Co2
                        ],
                        attributes)

self.next_condition = self.rnnt_update.output(0)
self.next_f = self.rnnt_update.output(1)
self.next_last_symbols = self.rnnt_update.output(2)
self.next_Hi1 = self.rnnt_update.output(3)
self.next_Ci1 = self.rnnt_update.output(4)
self.next_Hi2 = self.rnnt_update.output(5)
self.next_Ci2 = self.rnnt_update.output(6)
self.num_symbols_generated = self.rnnt_update.output(7)
self.state_time_idx = self.rnnt_update.output(8)
self.all_predictions = self.rnnt_update.output(9)
self.all_length = self.rnnt_update.output(10)
*/
RnntUpdate::RnntUpdate(const ov::OutputVector& arguments, bool eager_mode) : Op(arguments), eager_mode(eager_mode) {
    constructor_validate_and_infer_types();
}

void RnntUpdate::validate_and_infer_types() {
    if (eager_mode)
        return;

    // features: N,T,1024
    ov::Dimension N = get_input_partial_shape(1)[0];
    ov::Dimension C = get_input_partial_shape(1)[2];

    // current_iter
    enum InputNames {
        current_iter = 0,
        features,
        logits,
        Ho1,
        Co1,
        Ho2,
        Co2
    };

    NGRAPH_CHECK(get_input_element_type(current_iter) == ov::element::Type_t::i32);
    NGRAPH_CHECK(get_input_partial_shape(current_iter) == ov::PartialShape({1}));

    NGRAPH_CHECK(get_input_element_type(features) == ov::element::Type_t::f32);
    NGRAPH_CHECK(get_input_partial_shape(features)[2] == 1024);

    NGRAPH_CHECK(get_input_element_type(logits) == ov::element::Type_t::f32);
    NGRAPH_CHECK(get_input_partial_shape(logits) == ov::PartialShape({N, ov::Dimension(29)}));

    NGRAPH_CHECK(get_input_element_type(Ho1) == ov::element::Type_t::f32);
    NGRAPH_CHECK(get_input_partial_shape(Ho1) == ov::PartialShape({N, ov::Dimension(320)}));

    NGRAPH_CHECK(get_input_element_type(Co1) == ov::element::Type_t::f32);
    NGRAPH_CHECK(get_input_partial_shape(Co1) == ov::PartialShape({N, ov::Dimension(320)}));

    NGRAPH_CHECK(get_input_element_type(Ho2) == ov::element::Type_t::f32);
    NGRAPH_CHECK(get_input_partial_shape(Ho2) == ov::PartialShape({N, ov::Dimension(320)}));

    NGRAPH_CHECK(get_input_element_type(Co2) == ov::element::Type_t::f32);
    NGRAPH_CHECK(get_input_partial_shape(Co2) == ov::PartialShape({N, ov::Dimension(320)}));

    int i = 0;
    set_output_type(i++, ov::element::Type_t::u8, {1});
    set_output_type(i++, get_input_element_type(1), {N, C});
    set_output_type(i++, ov::element::Type_t::i32, {N});

    set_output_type(i++, get_input_element_type(3), get_input_partial_shape(3));
    set_output_type(i++, get_input_element_type(4), get_input_partial_shape(4));
    set_output_type(i++, get_input_element_type(5), get_input_partial_shape(5));
    set_output_type(i++, get_input_element_type(6), get_input_partial_shape(6));

    set_output_type(i++, ov::element::Type_t::i32, {N});
    set_output_type(i++, ov::element::Type_t::u32, {N});

    set_output_type(i++, ov::element::Type_t::u8, {N, 1024});
    set_output_type(i++, ov::element::Type_t::i32, {N});    
}

std::shared_ptr<ov::Node> RnntUpdate::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    auto ret = std::make_shared<RnntUpdate>(new_args, eager_mode);
    ret->bf16 = bf16;
    return ret;
}

bool RnntUpdate::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("bf16", bf16);
    visitor.on_attribute("eager_mode", eager_mode);
    return true;
}

template<typename T>
class BTensor {
public:
    BTensor(const ov::Tensor & tensor) {
        shape = tensor.get_shape();
        strides = tensor.get_strides();
        ptr = reinterpret_cast<uint8_t*>(tensor.data());
    }

    T& at(int i0) {
        return *reinterpret_cast<T*>(ptr + i0 * strides[0]);
    }
    T& at(int i0, int i1) {
        return *reinterpret_cast<T*>(ptr + i0 * strides[0] + i1 * strides[1]);
    }

    uint8_t * ptr;
    ov::Shape shape;
    ov::Strides strides;
};

bool RnntUpdate::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (bf16)
        return evaluate_T<float, float>(outputs, inputs);
    return evaluate_T<float, float>(outputs, inputs);
}

// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
bool all_ge_avx512f(BTensor<uint32_t> v, uint32_t t) {
    auto thr = _mm512_set1_epi32(t);
    int N = v.shape[0];
    int i;
    auto * p = &v.at(0);
    for(i = 0; i+16 < N; i += 16) {
        auto a = _mm512_loadu_si512(p + i);
        auto k = _mm512_cmp_epi32_mask(a, thr, _MM_CMPINT_LT);
        if (k)
            return false;
    }
    // last compare only contains (N-i) valid mask bits (potentially read overflow)
    auto a = _mm512_loadu_si512(p + i);
    auto k = _mm512_cmp_epi32_mask(a, thr, _MM_CMPINT_LT);
    if (k & ((1<<(N - i)) - 1))
        return false;
    return true;
}

template<typename T, typename V>
bool RnntUpdate::evaluate_T(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    int idx = 0;
    BTensor<int32_t> current_iter(inputs[idx++]);   // [1]
    BTensor<T> all_f(inputs[idx++]);                // N,T,1024
    BTensor<V> logits(inputs[idx++]);
    BTensor<T> o_hs1(inputs[idx++]);
    BTensor<T> o_cs1(inputs[idx++]);
    BTensor<T> o_hs2(inputs[idx++]);
    BTensor<T> o_cs2(inputs[idx++]);

    // memory has been allocated and data is there, just need to update according 
    idx = 0; 
    BTensor<uint8_t> next_cond(outputs[idx++]);   // [1]
    BTensor<T>       f(outputs[idx++]);           // N,1024
    BTensor<int32_t> last_symbol(outputs[idx++]); // 
    BTensor<T> hs1(outputs[idx++]);
    BTensor<T> cs1(outputs[idx++]);
    BTensor<T> hs2(outputs[idx++]);
    BTensor<T> cs2(outputs[idx++]);
    BTensor<int32_t> num_symbols_generated(outputs[idx++]);
    BTensor<uint32_t> time_idxs(outputs[idx++]);
    BTensor<uint8_t> all_predictions(outputs[idx++]);
    BTensor<int32_t> all_length(outputs[idx++]);

    int N = logits.shape[0];
    int C = logits.shape[1];

#define BLANK 28

    int nthreads = tbb::this_task_arena::max_concurrency();
    tbb::parallel_for(0, nthreads, [&](int th_id) {

        // revert Thread didn't introduce perf drop
        // th_id = nthreads - 1 - th_id;

        int n = N/nthreads;
        int n_left = N % nthreads;
        int i_start, i_end;
        if (th_id < n_left) {
            n += 1;
            i_start = th_id * n;
            i_end = i_start + n;
        }else{
            i_start = n_left*(n + 1) + (th_id - n_left) * n;
            i_end = i_start + n;
        }
        
        //std::stringstream ss;
        //ss << "========= th_id " << th_id << "/" << nthreads << "        " << i_start << "+" << n;
        //std::cout << ss.str() << std::endl;

        if (current_iter.at(0) == 0) {
            // initialize states
            for(int i = i_start; i < i_end; i++) {
                memset(&hs1.at(i, 0), 0, hs1.shape[1]*sizeof(T));
                memset(&cs1.at(i, 0), 0, cs1.shape[1]*sizeof(T));
                memset(&hs2.at(i, 0), 0, hs2.shape[1]*sizeof(T));
                memset(&cs2.at(i, 0), 0, cs2.shape[1]*sizeof(T));
                
                num_symbols_generated.at(i) = 0;
                last_symbol.at(i) = BLANK;
                time_idxs.at(i) = 0;

                // f only update partially, initialize is needed
                memcpy(&f.at(i), &all_f.at(i, 0), f.shape[1]*sizeof(T));

                all_length.at(i) = 0;
            }
        }

        for(int i = i_start; i < i_end; i++) {
            //auto& k = kargmax.at(i);

            auto * p_logits = &logits.at(i);
            int k = C-1;
            auto max = p_logits[k];
            for (int c = 0; c < C-1; c++) {
                if (max < p_logits[c]) {
                    max = p_logits[c];
                    k = c;
                }
            }

            auto & num = num_symbols_generated.at(i);
            //auto & f_end = flag_end.at(i);
            if (k != BLANK && num < 30) {
                
                auto & cur_len = all_length.at(i);
                auto & pred = all_predictions.at(i, cur_len);
                pred = k;
                cur_len ++;
                
                num ++;
                last_symbol.at(i) = k;
                memcpy(&hs1.at(i, 0), &o_hs1.at(i, 0), hs1.shape[1]*sizeof(T));
                memcpy(&cs1.at(i, 0), &o_cs1.at(i, 0), cs1.shape[1]*sizeof(T));
                memcpy(&hs2.at(i, 0), &o_hs2.at(i, 0), hs2.shape[1]*sizeof(T));
                memcpy(&cs2.at(i, 0), &o_cs2.at(i, 0), cs2.shape[1]*sizeof(T));
            } else {
                auto & t = time_idxs.at(i);
                num = 0;
                t ++;
                if (t < all_f.shape[1]) {
                    // update i'th item in feature batch given new time_idx
                    memcpy(&f.at(i), &all_f.at(i, t), f.shape[1]*sizeof(T));
                }
            }
        }
    });

    //std::cout << "RnntUpdate: logits:" << logits.shape << ", "<< logits.strides << std::endl;
    //std::cout << "RnntUpdate: k:" << k.shape << ", "<< k.strides << std::endl;

#if 0
    for(int i = 0; i < N; i++) {
        if (time_idxs.at(i) < all_f.shape[1])
            // continue execute next time
            return true;
    }
#else
    if (all_ge_avx512f(time_idxs, all_f.shape[1]))
        next_cond.at(0) = 0;
    else
        next_cond.at(0) = 1;
#endif

    return true;
}

bool RnntUpdate::has_evaluate() const {
    return true;
}