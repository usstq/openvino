// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace intel_cpu {

class MHADynamic : public ngraph::op::Op {
public:
    OPENVINO_OP("MHADynamic", "cpu_plugin_opset");

    MHADynamic() = default;

    struct ReLayout {
        bool do_reshape = false;
        bool do_transpose = false;
        bool do_gather = false;
        std::vector<size_t> param;

        std::string repr() const {
            std::stringstream ss;
            ss << do_reshape << "," << do_transpose << "," << do_gather;
            for (auto& p : param)
                ss << ":" << p;
            return ss.str();
        }

        static ReLayout reshape(const std::vector<size_t>& p) {
            ReLayout r;
            r.do_reshape = true;
            r.param = p;
            return r;
        }
        static ReLayout transpose(const std::vector<size_t>& p) {
            ReLayout r;
            r.do_transpose = true;
            r.param = p;
            return r;
        }
        static ReLayout gather(size_t index, size_t axis) {
            ReLayout r;
            r.do_gather = true;
            r.param.resize(2);
            r.param[0] = index;
            r.param[1] = axis;
            return r;
        }
        template <typename T>
        std::vector<T> apply(const std::vector<T>& src) {
            std::vector<T> dst;
            // empty param means param is dynamic, only available at runtime
            if (param.size() == 0)
                return dst;

            if (do_reshape) {
                assert(src.size() == param.size());
                auto it = src.begin();
                for (size_t i = 0; i < param.size(); i++) {
                    auto p = param[i];
                    if (p == 0) {
                        dst.push_back(*it++);
                        continue;
                    }
                    dst.push_back(p);
                }
                return dst;
            }
            if (do_transpose) {
                assert(src.size() == param.size());
                for (size_t i = 0; i < param.size(); i++) {
                    auto p = param[i];
                    dst.push_back(src[p]);
                }
                return dst;
            }
            if (do_gather) {
                assert(param.size() == 2);
                auto index = param[0];
                auto axis = param[1];
                for (size_t i = 0; i < src.size(); i++) {
                    if (i != axis)
                        dst.push_back(src[i]);
                }
                return dst;
            }
            assert(false);
            return dst;
        }
    };

    struct Config {
        size_t head_count = 0;
        size_t head_size = 0;
        bool with_qk_scale = false;
        bool with_alibi_mask = false;
        bool with_causal_mask = false;
        bool with_causal_mask_slice1 = false;
        bool with_causal_mask_slice2 = false;
        bool with_causal_mask_stridedslice1 = false;
        bool with_causal_mask_stridedslice2 = false;
        bool select_nfltmax_at_0 = false;
        bool with_attn_mask = false;
        bool with_attn_mask_reshapes = false;
        bool trans_result_to_BLHS = false;
        bool q_reshape_to_3d = false; // [B,H,qL,S] => [B*H, qL, S]
        bool k_reshape_to_3d = false; // [B,H,kL,S] => [B*H, kL, S]
        bool v_reshape_to_3d = false; // [B,H,vL,S] => [B*H, vL, S]
        bool result_is_3d = false; // [B*H,qL,S] => [B, H, vL, S]

        // followig preprocess is done in 4D domain
        std::vector<ReLayout> relayout_query;
        std::vector<ReLayout> relayout_key;
        std::vector<ReLayout> relayout_value;
    };
    //
    // arguments: [xxx] means xxx is optional
    //      {query, key, value, [mask_alibi], [mask_causal], [mask_attention]}
    //
    // query, key, value can come from same node, with shape [B,L,3*H*S] or [B,L,H*3*S]
    //
    // mask_alibi    : [batch_size or 1, head_count,     1,     kv_len]
    // mask_causal   : [batch_size or 1, head_count, query_len, kv_len]
    // mask_attention: [batch_size, head_count, query_len, kv_len]
    MHADynamic(const OutputVector& arguments, const Config& cfg);

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    const Config& get_config() {
        return m_cfg;
    }

private:
    Config m_cfg;
};

}  // namespace intel_cpu
}  // namespace ov
