// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/region_yolo.hpp"

#include "intel_gpu/primitives/region_yolo.hpp"

namespace ov {
namespace intel_gpu {

static void CreateRegionYoloOp(Program& p, const std::shared_ptr<ngraph::op::v0::RegionYolo>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t coords = op->get_num_coords();
    uint32_t classes = op->get_num_classes();
    uint32_t num = op->get_num_regions();
    bool do_softmax = op->get_do_softmax();
    uint32_t mask_size = op->get_mask().size();

    auto regionPrim = cldnn::region_yolo(layerName,
                                         inputs[0],
                                         coords,
                                         classes,
                                         num,
                                         mask_size,
                                         do_softmax);

    p.add_primitive(*op, regionPrim);
}

REGISTER_FACTORY_IMPL(v0, RegionYolo);

}  // namespace intel_gpu
}  // namespace ov
