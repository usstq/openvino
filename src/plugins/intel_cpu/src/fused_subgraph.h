// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <list>
#include <limits>

#include "onednn/dnnl.h"

namespace ov {
namespace intel_cpu {

struct brdvector {
    std::vector<float> values;

    size_t size() const {
        return values.size();
    }

    size_t get_broadcast_compatible_size(const brdvector& other) const {
        if (other.size() > 1 && values.size() > 1) {
            assert(other.size() == values.size());
            return values.size();
        }
        return std::max(other.values.size(), values.size());
    }

    // broadcast access (read-only)
    float operator[](int i) const {
        if (values.size() == 1)
            return values[0];
        assert(i < values.size());
        return values[i];
    }

    brdvector operator+(const brdvector& other) const {
        brdvector result;
        size_t sz = get_broadcast_compatible_size(other);
        result.values.resize(sz);
        for (int i = 0; i < sz; i++) {
            result.values[i] = (*this)[i] + other[i];
        }
        return result;
    }
    brdvector operator*(const brdvector& other) const {
        brdvector result;
        size_t sz = get_broadcast_compatible_size(other);
        result.values.resize(sz);
        for (int i = 0; i < sz; i++) {
            result.values[i] = (*this)[i] * other[i];
        }
        return result;
    }
    bool operator==(const float& rhs) const {
        for (auto& v : values) {
            if (v != rhs)
                return false;
        }
        return true;
    }

    bool operator>=(const float& rhs) const {
        assert(values.size());
        for (auto& v : values) {
            if (!(v >= rhs))
                return false;
        }
        return true;
    }

    bool operator<=(const float& rhs) const {
        assert(values.size());
        for (auto& v : values) {
            if (!(v <= rhs))
                return false;
        }
        return true;
    }

    // broadcast assign
    void operator=(const float& rhs) {
        if (values.size() == 0) {
            values.emplace_back(rhs);
            return;
        }

        for (auto& v : values) {
            v = rhs;
        }
    }
    void operator=(const std::vector<float>& rhs) {
        values = rhs;
    }
    brdvector min(const brdvector& other) const {
        brdvector result;
        size_t sz = get_broadcast_compatible_size(other);
        result.values.resize(sz);
        for (int i = 0; i < sz; i++) {
            result.values[i] = std::min((*this)[i], other[i]);
        }
        return result;
    }
    brdvector max(const brdvector& other) const {
        brdvector result;
        size_t sz = get_broadcast_compatible_size(other);
        result.values.resize(sz);
        for (int i = 0; i < sz; i++) {
            result.values[i] = std::max((*this)[i], other[i]);
        }
        return result;
    }

    template <typename P>
    brdvector mul_if(const brdvector& other, P pred) const {
        brdvector result;
        size_t sz = get_broadcast_compatible_size(other);
        result.values.resize(sz);
        for (int i = 0; i < sz; i++) {
            auto v = (*this)[i];
            result.values[i] = pred(v) ? (v * other[i]) : v;
        }
        return result;
    }

    brdvector round() const {
        brdvector result;
        result.values.resize(size());
        for (int i = 0; i < size(); i++) {
            result.values[i] = std::round((*this)[i]);
        }
        return result;
    }

    bool is_integer() const {
        for (auto& v : values) {
            if (abs(std::round(v) - v) > std::numeric_limits<float>::min()) {
                return false;
            }
        }
        return true;
    }

    bool recover_scalar(float zero_thr = std::numeric_limits<float>::min()) {
        if (values.size() <= 1)
            return false;

        auto v0 = values[0];
        for (auto& v : values) {
            if (abs(v - v0) > zero_thr) {
                // it's not a scalar, return
                return false;
            }
        }

        // resize to scalar
        values.resize(1);
        return true;
    }
};

class Node;

struct FusedOP {
    enum class types { relu = 0, linear = 1, clip = 3, round = 4, unknown = 99 };
    types type;

    FusedOP(types type = types::unknown) : type(type), fusedIndex(-1) {}
    ~FusedOP();

    bool is_clip() const {
        return type == types::clip;
    }
    bool is_relu() const {
        return type == types::relu;
    }
    bool is_linear() const {
        return type == types::linear;
    }
    bool is_round() const {
        return type == types::round;
    }
    bool is_unknown() const {
        return type == types::unknown;
    }
    // relu:
    //    y =         x if x >= 0
    //         a[0] * x if x < 0
    // linear (a and b are broadcasted):
    //    y = x * a + b
    // clip (a is low, b is high, broadcasted):
    //    y = clip(x, a, b)
    // round
    //    y = round(x)

    brdvector a;
    brdvector b;

    // unknown node
    std::shared_ptr<Node> node;
    int fusedIndex;
};

struct FusedSubGraph {
    std::string name;
    std::list<FusedOP> ops;
    size_t common_dim_length = 1;

    void append(const FusedOP& op);
    int move_clip_backward();
    int fuse_linear();
    int move_round_backward();
    int try_move_linear_to_begin();
    int recover_scalar();
    int strip_tail_clip_x8(bool isSigned);
    int strip_tail_round();

    void optimize(dnnl::memory::data_type outputDataType);
};

std::ostream& operator<<(std::ostream& os, const brdvector& v);
std::ostream& operator<<(std::ostream& os, const FusedOP& op);
std::ostream& operator<<(std::ostream& os, const FusedSubGraph& fsg);

}   // namespace intel_cpu
}   // namespace ov
