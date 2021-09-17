// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn/primitives/non_max_suppression.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<non_max_suppression> : public typed_program_node_base<non_max_suppression> {
    using parent = typed_program_node_base<non_max_suppression>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog)
    {}

    program_node& input() const { return get_dependency(0); }

    program_node& input_boxes() const {
        return get_dependency(0);
    }

    program_node& input_scores() const {
        return get_dependency(1);
    }

    bool has_num_select_per_class() const { return !get_primitive()->num_select_per_class.empty(); }
    program_node& num_select_per_class_node() const {
        return get_dependency(2);
    }

    bool has_iou_threshold() const { return !get_primitive()->iou_threshold.empty(); }
    program_node& iou_threshold_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        return get_dependency(offset);
    }

    bool has_score_threshold() const { return !get_primitive()->score_threshold.empty(); }
    program_node& score_threshold_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        return get_dependency(offset);
    }

    bool has_soft_nms_sigma() const { return !get_primitive()->soft_nms_sigma.empty(); }
    program_node& soft_nms_sigma_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        return get_dependency(offset);
    }

    bool has_second_output() const { return !get_primitive()->second_output.empty(); }
    program_node& second_output_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        offset += has_soft_nms_sigma();
        return get_dependency(offset);
    }

    bool has_third_output() const { return !get_primitive()->third_output.empty(); }
    program_node& third_output_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        offset += has_soft_nms_sigma();
        offset += has_second_output();
        return get_dependency(offset);
    }
};

using non_max_suppression_node = typed_program_node<non_max_suppression>;

template <>
class typed_primitive_inst<non_max_suppression> : public typed_primitive_inst_base<non_max_suppression> {
    using parent = typed_primitive_inst_base<non_max_suppression>;

public:
    typed_primitive_inst(network& network, non_max_suppression_node const& node)
        : parent(network, node)
    {}

    static layout calc_output_layout(non_max_suppression_node const& node);
    static std::string to_string(non_max_suppression_node const& node);

    memory::ptr input_boxes_mem() const {
        return dep_memory_ptr(0);
    }

    memory::ptr input_scores_mem() const {
        return dep_memory_ptr(1);
    }

    bool has_num_select_per_class() const { return node.has_num_select_per_class(); }
    memory::ptr num_select_per_class_mem() const {
        return dep_memory_ptr(2);
    }

    bool has_iou_threshold() const { return node.has_iou_threshold(); }
    memory::ptr iou_threshold_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        return dep_memory_ptr(offset);
    }

    bool has_score_threshold() const { return node.has_score_threshold(); }
    memory::ptr score_threshold_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        return dep_memory_ptr(offset);
    }

    bool has_soft_nms_sigma() const { return node.has_soft_nms_sigma(); }
    memory::ptr soft_nms_sigma_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        return dep_memory_ptr(offset);
    }

    bool has_second_output() const { return node.has_second_output(); }
    memory::ptr second_output_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        offset += has_soft_nms_sigma();
        return dep_memory_ptr(offset);
    }

    bool has_third_output() const { return node.has_third_output(); }
    memory::ptr third_output_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        offset += has_soft_nms_sigma();
        offset += has_second_output();
        return dep_memory_ptr(offset);
    }
};

using non_max_suppression_inst = typed_primitive_inst<non_max_suppression>;

}  // namespace cldnn
