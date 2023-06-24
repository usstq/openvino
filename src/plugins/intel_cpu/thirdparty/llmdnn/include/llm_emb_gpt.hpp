// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "llm_types.hpp"

namespace llmdnn {

class emb_gpt {
public:
    struct create_param {
        size_t num_heads;
        size_t head_size;
        size_t head_size_aligned;       // better to aligned to 64 bytes for best performance, apply for qkv
        size_t max_seq_len;             // max seq length for computing the size of matmul tmp result
        // supported (qkv, dst): (bf16, bf16)
        data_type_t qkv_precision;
        data_type_t dst_precision;
        size_t rotary_emb_base;
        float rotary_pct;
        bool use_position2d;
    };
    struct exec_param {
        size_t batch;
        size_t query_seq_len;
        size_t past_seq_len;
        uint8_t* qkv;
        uint8_t* query_dst;
        uint8_t** layer_past_key_padded;
        uint8_t** layer_past_value_padded;
        int* position2d_ids;        // shape: [batch, 2, query_seq_len]
    };

    emb_gpt();
    bool create(const create_param& param);
    void exec(const exec_param& param);

    struct impl {
        virtual bool create(const create_param& param) = 0;
        virtual void exec(const exec_param& param) = 0;
    };
protected:
    std::shared_ptr<impl> _impl;
};

}
