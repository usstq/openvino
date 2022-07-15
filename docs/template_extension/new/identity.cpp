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

std::string get_version() {
    return"0.1";
}

class Initor {
public:
    Initor() {
        auto &ops = const_cast<ov::OpSet&>(ov::get_opset8());
        ops.insert<TemplateExtension::RnntUpdate>();
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


void RnntUpdate::validate_and_infer_types() {
}

std::shared_ptr<ov::Node> RnntUpdate::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<RnntUpdate>();
}

bool RnntUpdate::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("bf16", bf16);
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



template<typename T, typename V>
bool RnntUpdate::evaluate_T(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    BTensor<V> logits(inputs[0]);
    BTensor<T> o_hs1(inputs[1]);
    BTensor<T> o_cs1(inputs[2]);
    BTensor<T> o_hs2(inputs[3]);
    BTensor<T> o_cs2(inputs[4]);
    BTensor<T> all_f(inputs[5]);    // N,T,1024

    // memory has been allocated and data is there, just need to update according 
    BTensor<int32_t> kargmax(outputs[0]);             // argmax(logits axis=1)
    BTensor<int32_t> last_symbol(outputs[1]);   // 
    BTensor<T> hs1(outputs[2]);
    BTensor<T> cs1(outputs[3]);
    BTensor<T> hs2(outputs[4]);
    BTensor<T> cs2(outputs[5]);
    BTensor<int32_t> num_symbols_generated(outputs[6]);
    BTensor<T> f(outputs[7]);           // N,1024
    BTensor<uint32_t> time_idxs(outputs[8]);
    BTensor<uint8_t> all_predictions(outputs[9]);
    BTensor<int32_t> all_length(outputs[10]);
    
    //argmax_2D_axis1(logits, kargmax);

    int N = kargmax.shape[0];
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

        for(int i = i_start; i < i_end; i++) {
            auto& k = kargmax.at(i);

            auto * p_logits = &logits.at(i);
            k = C-1;
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
    
    for(int i = 0; i < N; i++) {
        if (time_idxs.at(i) < all_f.shape[1])
            // continue execute next time
            return true;
    }

    return false;
}

bool RnntUpdate::has_evaluate() const {
    return true;
}