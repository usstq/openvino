// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mutable_data_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"

namespace cldnn {
namespace ocl {

struct mutable_data_impl : public typed_primitive_impl_ocl<mutable_data> {
    using parent = typed_primitive_impl_ocl<mutable_data>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mutable_data_impl>(*this);
    }

public:
    static std::unique_ptr<primitive_impl> create(mutable_data_node const& arg, const kernel_impl_params&) {
        return make_unique<mutable_data_impl>(arg, kernel_selector::kernel_data{});
    }
};

namespace detail {

attach_mutable_data_impl::attach_mutable_data_impl() {
    implementation_map<mutable_data>::add(impl_types::ocl, mutable_data_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::mutable_data_impl, cldnn::object_type::MUTABLE_DATA_IMPL)
