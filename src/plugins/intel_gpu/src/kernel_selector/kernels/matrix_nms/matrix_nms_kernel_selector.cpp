// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matrix_nms_kernel_selector.h"

#include "matrix_nms_kernel_ref.h"

namespace kernel_selector {

matrix_nms_kernel_selector::matrix_nms_kernel_selector() {
    Attach<MatrixNmsKernelRef>();
}

KernelsData matrix_nms_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::MATRIX_NMS);
}
}  // namespace kernel_selector
