// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"

#include <string>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class RNN : public Node {
public:
    RNN(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    void initDescriptor(const NodeConfig& config) override {}
    std::shared_ptr<MemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator& primitive_desc_it, size_t idx) override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(mkldnn::primitive_desc_iterator& primitive_desc_it, size_t idx) override;
    bool created() const override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;

    void execute(mkldnn::stream strm) override;

    inline bool hasNativeOrder() const {
        return nativeOrder;
    }

    void cleanup() override;

protected:
    std::vector<VectorDims> shapeInfer() const override;
    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    void initCell();
    void initSequence();
    void fillCellDesc();
    void fillSequenceDesc();
    bool verifyWeightsPrecision(const InferenceEngine::Precision& layerPrec,
                                const InferenceEngine::Precision& weightsPrec);

    template <typename Prec>
    void fillWeights(const int* gate_map, const size_t wIdx, const size_t rIdx);
    template <InferenceEngine::Precision::ePrecision Prec>
    void fillBiases(const int* gate_map);

    void copyWeightsData();

    /** Specify mode Cell or Seq. true - Cell, false - Seq */
    bool is_cell = false;

    /** Native order if [batch, seq, data], other case is [seq, batch, data] */
    bool nativeOrder = true;

    /** Direction of iteration through sequence dimension */
    mkldnn::rnn_direction direction = mkldnn::rnn_direction::unidirectional;

    /** RNN Cell type (type/activation_alg/clip)*/
    mkldnn::algorithm cell_type = mkldnn::algorithm::undef;

    /** activation type for vanilla RNN cell */
    mkldnn::algorithm cell_act = mkldnn::algorithm::undef;

    /** Weights data and state memory format: ldigo or any */
    mkldnn::memory::format_tag wFormat = mkldnn::memory::format_tag::any;

    size_t DC = 0;  /**< Input data channel size */
    size_t SC = 0;  /**< State channel size value */
    size_t G = 0;   /**< Gate size. LSTM - 4, GRU - 3, RNN - 1 */
    size_t Gb = 0;  /**< Gate size for biases. Gb = GRU_lbr ? G+1 : G */
    size_t S = 2;   /**< Num of state. LSTM - 2, GRU & RNN - 1 */
    const size_t L = 1;   /**< What is it??. Constant for mkldnn impl */
    const size_t D = 1;   /**< Num of direction. 1 or 2 */

    std::vector<DnnlBlockedMemoryDescPtr> inDataDescs;
    std::vector<DnnlBlockedMemoryDescPtr> outDataDescs;
    std::vector<mkldnn::memory::desc> wDescs;

    enum RNNInOutKind {
        Layer       = 0,
        HiddenState = 1,
        CellState   = 2
    };

    size_t wIdx = 0;
    size_t rIdx = 0;
    size_t bIdx = 0;

    static const std::map<InferenceEngine::Precision, InferenceEngine::Precision> weightsByLayerPrec;

    static constexpr size_t optimalBatchSize = 16lu;
    static constexpr size_t batchDimDummyValue = 64lu;

    bool wasMemoryPrepared = false;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov