// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief FakeQuantizeTransformation fuses dequantization operations into FakeQuantize operation.
 *
 * For more details about the transformation, refer to
 * [FakeQuantizeTransformation](@ref openvino_docs_IE_DG_lpt_FakeQuantizeTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API FakeQuantizeTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    FakeQuantizeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

    static bool checkElementwise(const std::shared_ptr<Node>& eltwise);

    static std::shared_ptr<opset1::FakeQuantize> fuseElementwise(
            TransformationContext& context,
            MatcherPass* matcherPass,
            const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
