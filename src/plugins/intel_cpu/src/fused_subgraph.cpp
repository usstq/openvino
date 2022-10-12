// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <iostream>
#include <sstream>

#include "fused_subgraph.h"
#include "node.h"

namespace ov {
namespace intel_cpu {

FusedOP::~FusedOP() {}

void FusedSubGraph::append(const FusedOP& op) {
    // check whether a/b size is broadcast-able to common_dim_length
    if (op.is_unknown())
        return;

    if (common_dim_length == 1 && op.a.size() > 1)
        common_dim_length = op.a.size();
    if (common_dim_length == 1 && op.b.size() > 1)
        common_dim_length = op.b.size();

    if (common_dim_length > 1) {
        assert(op.a.size() <= 1 || op.a.size() == common_dim_length);
        assert(op.b.size() <= 1 || op.b.size() == common_dim_length);
    }

    ops.push_back(op);
}

int FusedSubGraph::move_clip_backward() {
    // move each clip to end and fuse clips along the way
    FusedOP clip;
    bool is_moving = false;
    int changed = 0;
    auto it = ops.begin();

    auto ensure_clip_b_bigger_than_a = [&]() {
        assert(clip.a.size() == clip.b.size());
        size_t sz = std::max(clip.a.size(), clip.b.size());
        for (int i = 0; i < sz; i++) {
            auto& va = clip.a.values[i];
            auto& vb = clip.b.values[i];
            if (va > vb)
                std::swap(va, vb);
        }
    };

    while (it != ops.end()) {
        // copy the clip and then erase it
        if (!is_moving) {
            if (!it->is_clip()) {
                ++it;
                continue;
            }
            is_moving = true;
            clip = *it;
            it = ops.erase(it);
            continue;
        }
        ensure_clip_b_bigger_than_a();
        // moving clip
        if (it->is_relu()) {
            clip.a = clip.a.mul_if(it->a, [](float v) {
                return v < 0.0f;
            });
            clip.b = clip.b.mul_if(it->a, [](float v) {
                return v < 0.0f;
            });
            ++it;
            changed++;
        } else if (it->is_linear()) {
            clip.a = clip.a * it->a + it->b;
            clip.b = clip.b * it->a + it->b;
            ++it;
            changed++;
        } else if (it->is_clip()) {
            // fuse clips
            clip.a = clip.a.max(it->a);
            clip.b = clip.b.min(it->b);
            // drop this one
            it = ops.erase(it);
            changed++;
        } else if (it->is_round()) {
            clip.a = clip.a.round();
            clip.b = clip.b.round();
            ++it;
            changed++;
            // put clip right after round helps to ensure
            // most clip to be per-tensor
            // (some accumulative calculation error produces
            //  false per-OC clip, after round node the error
            //  is gone).
            break;
        } else {
            // Note that we can push clip through any monotonic function in theory.
            break;
        }
    }

    // insert the clip back right before where we stop
    if (is_moving) {
        ensure_clip_b_bigger_than_a();
        ops.insert(it, clip);
    }
    return changed;
}

int FusedSubGraph::fuse_linear() {
    // fuse consecutive linear
    FusedOP linear;
    int changed = 0;
    auto it = ops.begin();
    auto prev = it;
    ++it;
    while (it != ops.end()) {
        if (prev->is_linear() && it->is_linear()) {
            // (x*a1 + b1)*a2 + b2 = x*(a1*a2) + (b1*a2 + b2)
            prev->a = prev->a * it->a;
            prev->b = prev->b * it->a + it->b;
            // drop it, prev is un-changed
            it = ops.erase(it);
            changed++;
            continue;
        }
        prev = it;
        ++it;
    }
    return changed;
}

int FusedSubGraph::move_round_backward() {
    // round can move backward if it's followed by adding an integer
    // moving round backward helps mapping to oneDNN's default saturation/rounding steps.
    auto it = ops.begin();
    auto prev = it;
    int changed = 0;
    ++it;
    while (it != ops.end()) {
        if (prev->is_round() && it->is_linear()) {
            if (it->a == 1.0f && it->b.is_integer()) {
                std::swap(*prev, *it);
                changed++;
            }
        }
        prev = it;
        ++it;
    }
    return changed;
}

int FusedSubGraph::try_move_linear_to_begin() {
    // try to move linear to the begin (to map to oneDNN's output scales)
    int changed = 0;
    auto it = ops.begin();
    if (it->is_linear())
        return changed;

    auto prev = it;
    ++it;
    if (prev->is_relu() && it->is_linear()) {
        //    relu(x,alpha)*a + b  ===>  relu(x*a,alpha) + b
        // since we only move the multiplication rather than whole linear
        // we will need to create a new OP
        FusedOP mul_a = *it;
        mul_a.b = 0.0f;
        it->a = 1.0f;  // now it only contains +b
        ops.insert(prev, mul_a);
        if (it->b == 0.0f) {
            // remove +0
            it = ops.erase(it);
        }
        changed++;
    }

    return changed;
}

int FusedSubGraph::recover_scalar() {
    // recover scalar parameters
    int changed = 0;
    for (auto & op : ops) {
        if (op.is_linear() || op.is_clip()) {
            changed += op.a.recover_scalar();
            changed += op.b.recover_scalar();
        }
    }
    return changed;
}

int FusedSubGraph::strip_tail_clip_x8(bool isSigned) {
    int changed = 0;
    auto it = ops.end();
    --it;
    float lo = isSigned ? -128.0f : 0.0f;
    float hi = isSigned ? 127.0f : 255.0f;

    // clip over valid range is also not usefull
    if (it->is_clip() && (it->a <= lo) && (it->b >= hi)) {
        ops.erase(it);
        changed++;
    }
    return changed;
}

int FusedSubGraph::strip_tail_round() {
    int changed = 0;
    auto it = ops.end();
    --it;
    if (it->is_round()) {
        ops.erase(it);
        changed++;
    }
    return changed;
}

void FusedSubGraph::optimize(dnnl::memory::data_type outputDataType) {
    // optimizations are done based on what we saw in practice
    // rather than on imaginary use-case, so we are not trying to
    // cover every possible optimizations, just add what we need.
    std::cout << "before FusedSubGraph::optimize ==========" << std::endl;
    std::cout << *this;

    int changed = 0;
    changed += move_clip_backward();
    changed += move_round_backward();
    changed += fuse_linear();
    changed += try_move_linear_to_begin();
    changed += recover_scalar();

    // optimize the round+clip sequence at tail because oneDNN does that by default
    if (outputDataType == dnnl::memory::data_type::u8 || outputDataType == dnnl::memory::data_type::s8) {
        changed += strip_tail_clip_x8(outputDataType == dnnl::memory::data_type::s8);
    }
    if (outputDataType == dnnl::memory::data_type::u8 || outputDataType == dnnl::memory::data_type::s8 ||
        outputDataType == dnnl::memory::data_type::s32) {
        changed += strip_tail_round();
    }

    if (changed) {
        std::cout << "after FusedSubGraph::optimize ========== changed " << changed << " times" << std::endl;
        std::cout << *this;
    } else {
        std::cout << "after FusedSubGraph::optimize ========== no changes " << std::endl;
    }
}

std::ostream& operator<<(std::ostream& os, const brdvector& v) {
    if (v.size() == 0) {
        os << "[]";
    } else if (v.size() == 1) {
        os << v[0];
    } else {
        std::ostringstream ss;
        ss << v.values[0];
        for (int i = 0; i < v.size(); ++i) {
            if (ss.tellp() > 40) {
                ss << ",...";
                break;
            }
            ss << "," << v.values[i];
        }
        os << "[" << ss.str() << "]";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const FusedOP& op) {
    if (op.is_clip()) {
        os << " x = clip(x, " << op.a << ", " << op.b << ");";
    } else if (op.is_relu()) {
        os << " x = relu(x, " << op.a << ");";
    } else if (op.is_linear()) {
        os << " x = (x * " << op.a << " + " << op.b << ");";
    } else if (op.is_round()) {
        os << " x = round(x);";
    } else if (op.is_unknown()) {
        os << " x = " << op.node->getTypeStr() << "(x);";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const FusedSubGraph& fsg) {
    os << fsg.name << "(x) {" << std::endl;
    for (auto& op : fsg.ops) {
        os << "\t" << op << std::endl;
    }
    os << "}" << std::endl;
    return os;
}

}   // namespace intel_cpu
}   // namespace ov
