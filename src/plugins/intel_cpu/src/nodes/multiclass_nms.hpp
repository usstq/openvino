// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

#include <string>

namespace ov {
namespace intel_cpu {
namespace node {

enum class MulticlassNmsSortResultType {
    CLASSID,  // sort selected boxes by class id (ascending) in each batch element
    SCORE,    // sort selected boxes by score (descending) in each batch element
    NONE      // do not guarantee the order in each batch element
};

class MultiClassNms : public Node {
public:
    MultiClassNms(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr& cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool isExecutable() const override;
    void executeDynamicImpl(mkldnn::stream strm) override;

    bool needShapeInfer() const override { return false; }
    void prepareParams() override;

private:
    // input (port Num)
    const size_t NMS_BOXES = 0;
    const size_t NMS_SCORES = 1;

    // output (port Num)
    const size_t NMS_SELECTEDOUTPUTS = 0;
    const size_t NMS_SELECTEDINDICES = 1;
    const size_t NMS_SELECTEDNUM = 2;

    bool m_sortResultAcrossBatch = false;
    MulticlassNmsSortResultType m_sortResultType = MulticlassNmsSortResultType::NONE;

    size_t m_numBatches = 0;
    size_t m_numBoxes = 0;
    size_t m_numClasses = 0;
    size_t m_maxBoxesPerBatch = 0;

    int m_nmsRealTopk = 0;
    int m_nmsTopK = 0;
    float m_iouThreshold = 0.0f;
    float m_scoreThreshold = 0.0f;

    int32_t m_backgroundClass = 0;
    int32_t m_keepTopK = 0;
    float m_nmsEta = 0.0f;
    bool m_normalized = true;

    std::string m_errorPrefix;

    std::vector<std::vector<size_t>> m_numFiltBox;
    std::vector<size_t> m_numBoxOffset;
    const std::string m_inType = "input", m_outType = "output";

    struct filteredBoxes {
        float score;
        int batch_index;
        int class_index;
        int box_index;
        filteredBoxes() = default;
        filteredBoxes(float _score, int _batch_index, int _class_index, int _box_index)
            : score(_score), batch_index(_batch_index), class_index(_class_index), box_index(_box_index) {}
    };

    struct boxInfo {
        float score;
        int idx;
        int suppress_begin_index;
    };

    std::vector<filteredBoxes> m_filtBoxes;

    void checkPrecision(const InferenceEngine::Precision prec, const std::vector<InferenceEngine::Precision> precList, const std::string name,
                        const std::string type);

    float intersectionOverUnion(const float* boxesI, const float* boxesJ, const bool normalized);

    void nmsWithEta(const float* boxes, const float* scores, const InferenceEngine::SizeVector& boxesStrides, const InferenceEngine::SizeVector& scoresStrides);

    void nmsWithoutEta(const float* boxes, const float* scores, const InferenceEngine::SizeVector& boxesStrides,
                       const InferenceEngine::SizeVector& scoresStrides);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
