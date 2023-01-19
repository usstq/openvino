// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_sequence_kernel_selector.h"
#include "reverse_sequence_kernel_ref.h"

namespace kernel_selector {

reverse_sequence_kernel_selector::reverse_sequence_kernel_selector() { Attach<ReverseSequenceKernelRef>(); }

KernelsData reverse_sequence_kernel_selector::GetBestKernels(const Params& params,
                                                             const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::REVERSE_SEQUENCE);
}
}  // namespace kernel_selector
