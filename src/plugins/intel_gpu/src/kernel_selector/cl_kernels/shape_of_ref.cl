// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"

KERNEL(shape_of_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global OUTPUT_TYPE* output
    )
{
    const unsigned int i = (uint)get_global_id(2);

#if IS_DYNAMIC
    output[i] = TO_OUTPUT_TYPE(shape_info[i]);
#else
    output[i] = TO_OUTPUT_TYPE(INPUT_DIMS[i]);
#endif
}
