// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn2.h"
#include <utils/general_utils.h>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "utils/bfloat16.hpp"
#include "input.h"
#include <dnnl_extension_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <common/primitive_hashing_utils.hpp>

#include <ngraph/node.hpp>

#include <string>
#include <utility>

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

using namespace mkldnn;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

inline size_t gatesCount(const algorithm& alg) {
    switch (alg) {
        case algorithm::vanilla_rnn:     return 1;
        case algorithm::vanilla_gru:
        case algorithm::lbr_gru:         return 3;
        case algorithm::vanilla_lstm:    return 4;
        default:
            IE_THROW() << "Unsupported cell type";
            return 0;
    }
}

inline size_t statesCount(const mkldnn::algorithm& alg) {
    switch (alg) {
        case mkldnn::algorithm::vanilla_rnn:
        case mkldnn::algorithm::vanilla_gru:
        case mkldnn::algorithm::lbr_gru:         return 1;
        case mkldnn::algorithm::vanilla_lstm:    return 2;
        default:
            IE_THROW() << "Unsupported cell type";
            return 0;
    }
}

inline bool haveCellState(const mkldnn::algorithm& alg) {
    return alg == mkldnn::algorithm::vanilla_lstm;
}

const std::map<Precision, Precision> RNN2::weightsByLayerPrec {
    // layer precision,                weights precision
    {Precision::FP32, Precision::FP32},
    {Precision::BF16, Precision::BF16},
    // FP16 and U8 are not supported yet
    // {Precision::FP16, Precision::FP16},
    // {Precision::U8,   Precision::I8},
};

struct RNN2Key {
    const std::vector<DnnlBlockedMemoryDescPtr> inDataDescs;
    const std::vector<DnnlBlockedMemoryDescPtr> outDataDescs;
    const std::vector<mkldnn::memory::desc> wDescs;
    mkldnn::algorithm cellType;
    mkldnn::algorithm cellAct;
    mkldnn::rnn_direction direction;

    size_t hash() const;
    bool operator==(const RNN2Key& rhs) const;
};

size_t RNN2Key::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0lu;

    for (auto& desc : inDataDescs) {
        if (desc != nullptr)
            seed = hash_combine(seed, get_md_hash(desc->getDnnlDesc().data));
    }
    for (auto& desc : outDataDescs) {
        if (desc != nullptr)
            seed = hash_combine(seed, get_md_hash(desc->getDnnlDesc().data));
    }
    for (auto& desc : wDescs) {
        seed = hash_combine(seed, get_md_hash(desc.data));
    }
    seed = hash_combine(seed, cellType);
    seed = hash_combine(seed, cellAct);
    seed = hash_combine(seed, direction);
    return seed;
}

bool RNN2Key::operator==(const RNN2Key& rhs) const {
    if (inDataDescs.size() != rhs.inDataDescs.size() || outDataDescs.size() != rhs.outDataDescs.size() || wDescs.size() != rhs.wDescs.size() ||
            cellType != rhs.cellType || cellAct != rhs.cellAct || direction != rhs.direction) {
        return false;
    }

    for (size_t i = 0lu; i < inDataDescs.size(); i++) {
        if (inDataDescs[i] != rhs.inDataDescs[i] && (inDataDescs[i] == nullptr || rhs.inDataDescs[i] == nullptr ||
                inDataDescs[i]->getDnnlDesc() != rhs.inDataDescs[i]->getDnnlDesc()))
            return false;
    }
    for (size_t i = 0lu; i < outDataDescs.size(); i++) {
        if (outDataDescs[i] != rhs.outDataDescs[i] && (outDataDescs[i] == nullptr || rhs.outDataDescs[i] == nullptr ||
                outDataDescs[i]->getDnnlDesc() != rhs.outDataDescs[i]->getDnnlDesc()))
            return false;
    }
    for (size_t i = 0lu; i < wDescs.size(); i++) {
        if (wDescs[i] != rhs.wDescs[i])
            return false;
    }

    return true;
}

RNN2::RNN2(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache) :
        Node(op, eng, cache) {
    if (op->get_type_info() != RNNPrim::get_type_info_static()) {
        IE_THROW(NotImplemented) << "Only RNNPrim is supported";
    }

    m_op = std::dynamic_pointer_cast<RNNPrim>(op);

    //direction = ieDirection2dnnl(op);
    if (m_op->dir == "left2right")
        direction = rnn_direction::unidirectional_left2right;
    else if (m_op->dir == "right2left")
        direction = rnn_direction::unidirectional_right2left;
    else if (m_op->dir == "bidirectional_concat")
        direction = rnn_direction::bidirectional_concat;
    else
        //rnn_direction::unidirectional
        IE_THROW(NotImplemented) << "unknown direction:" << m_op->dir;

    if (m_op->cell_type == "rnn")
        cell_type = mkldnn::algorithm::vanilla_rnn;
    else if (m_op->cell_type == "lbr_gru")
        cell_type = mkldnn::algorithm::lbr_gru;
    else if (m_op->cell_type == "lstm")
        cell_type = mkldnn::algorithm::vanilla_lstm;
    else if (m_op->cell_type == "gru")
        cell_type = mkldnn::algorithm::vanilla_gru;
    else
        IE_THROW(NotImplemented) << "unknown cell_type:" << m_op->cell_type;

    // nativeOrder
    batch_first = m_op->input_batch_first;

    // input port
    wIdx = 3;   // weight
    rIdx = 4;   // weight_iter
    bIdx = 7;   // bias

    internalBlobDesc.emplace_back([&](primitive_desc_iterator& primitive_desc_it, size_t idx) -> DnnlMemoryDescPtr {
        return DnnlExtensionUtils::makeDescriptor(primitive_desc_it.weights_desc(0));
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator& primitive_desc_it, size_t idx) -> DnnlMemoryDescPtr {
        return DnnlExtensionUtils::makeDescriptor(primitive_desc_it.weights_desc(1));
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator& primitive_desc_it, size_t idx) -> DnnlMemoryDescPtr {
        return DnnlExtensionUtils::makeDescriptor(primitive_desc_it.weights_desc(2));
    });

    G = gatesCount(cell_type);
    Gb = (cell_type != mkldnn::algorithm::lbr_gru) ? G : G + 1;
    S = statesCount(cell_type);
    L = m_op->m_num_layers;
    SC = m_op->m_hidden_size;
    DC = m_op->m_input_size;

    // src/dst:
    //   batch_first:  dnnl_ntc:   (batch, seq_length, input channels)
    //  !batch_first:  dnnl_tnc:   (seq_length, batch, input channels)
    const auto& inDataShape = getInputShapeAtPort(0);
    const auto& outDataShape = getOutputShapeAtPort(0);

    if (inDataShape.getRank() != 3lu || outDataShape.getRank() != 3lu)
        THROW_ERROR << "has incorrect input/output shapes. Input data shape: " << inDataShape.toString() <<
                " Output shape: " << outDataShape.toString();

    /*
    if (!one_of(getOriginalInputsNumber(), 6, 7))
        THROW_ERROR << "has incorrect number of input ports: " << getOriginalInputsNumber();
    if (!one_of(getOriginalOutputsNumber(), 2, 3))
        THROW_ERROR << "has incorrect number of output ports: " << getOriginalOutputsNumber();


    // weight dnnl_ldigo: (num_layers, num_directions, input_channels, num_gates, output_channels) 
    if (cell_type == algorithm::vanilla_lstm)
        DC = getInputShapeAtPort(wIdx).getStaticDims()[2];
    else
        DC = getInputShapeAtPort(3).getStaticDims()[2];
    */
}

bool RNN2::created() const {
    return getType() == Type::RNN2;
}

void RNN2::getSupportedDescriptors() {
    const auto dataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(0));
    copyWeightsData();

    std::vector<MemoryDescPtr> inCandidate;
    std::vector<MemoryDescPtr> outCandidate;
    inCandidate.reserve(7);
    outCandidate.reserve(3);

    auto add_src = [&](const Shape & shape, memory::format_tag fmt_tag) {
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shape, dataType, fmt_tag));
    };

    auto add_dst = [&](const Shape & shape, memory::format_tag fmt_tag) {
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shape, dataType, fmt_tag));   
    };

    if (cell_type == mkldnn::algorithm::vanilla_lstm) {
        // X:      since oneDNN RNN primitive support both ntc&tnc input, no external reorder is needed for input
        add_src(getInputShapeAtPort(0), memory::format_tag::abc);

        // Hi,Ci:  {N, L*D, H} bac  ===   {L, D, N, H} abcd
        add_src(getInputShapeAtPort(1), m_op->m_batch == 1 ? memory::format_tag::abc : memory::format_tag::bac);
        add_src(getInputShapeAtPort(2), m_op->m_batch == 1 ? memory::format_tag::abc : memory::format_tag::bac);

        // W,R:    {L*D, 4*H, C} acb  ==== {L, D, C, 4, H},abcde=ldigo.
        add_src(getInputShapeAtPort(3), memory::format_tag::acb);
        add_src(getInputShapeAtPort(4), memory::format_tag::acb);

        // B:      {L*D, 4*H} ab  ===  {L, D, 4, H},abcd=ldgo
        add_src(getInputShapeAtPort(5), memory::format_tag::ab);


        if (m_op->output_batch_first) {
            // Y:      {N, D, T, C} ===   {T, N, D*C} bac=ntc(optimal layoutA)
            //           abcd if D=1
            //       acbd otherwise (reorder will be inserted)
            add_dst(getOutputShapeAtPort(0), m_op->m_num_directions == 1 ? memory::format_tag::abcd:memory::format_tag::acbd);    // dst
        } else {
            // Y with post-transpose(out is from the post-transpose node)
            // out     {T, D, N, C}        {T, N, D*C} abc=tnc(optimal layoutB)
            //         abcd if D=1
            //         acbd otherwise (reorder will be inserted)
            add_dst(getOutputShapeAtPort(0), m_op->m_num_directions == 1 ? memory::format_tag::abcd:memory::format_tag::acbd);    // dst
        }
        
        // Ho,Co:  {N, L*D, H} bac  ===   {L, D, N, H} abcd
        add_dst(getOutputShapeAtPort(1), m_op->m_batch == 1 ? memory::format_tag::abc : memory::format_tag::bac);    // dst_iter
        add_dst(getOutputShapeAtPort(2), m_op->m_batch == 1 ? memory::format_tag::abc : memory::format_tag::bac);    // dst_iter_c
    } else {
        THROW_ERROR << "not implemented";
    }

    createDescriptor(inCandidate, outCandidate);
}

bool RNN2::verifyWeightsPrecision(const Precision &layerPrec, const Precision &weightsPrec) {
    if (!weightsByLayerPrec.count(layerPrec))
        THROW_ERROR << "has unsupported layer precision " << layerPrec;
    return weightsPrec == weightsByLayerPrec.at(layerPrec);
}

template <typename Prec>
void RNN2::fillWeights(const int *gate_map, const size_t wIdx, const size_t rIdx) {
    const auto& dataPrecision = getOriginalInputPrecisionAtPort(0);
    const auto& weightPrec = getOriginalInputPrecisionAtPort(wIdx);
    if (!verifyWeightsPrecision(dataPrecision, weightPrec) && dataPrecision != Precision::BF16 && weightPrec != Precision::FP32) {
        THROW_ERROR << "doesn't support combination of weights precision: " << weightPrec << " and runtime precision: " << dataPrecision;
    }
    // create weight blobs (data and state part)
    const VectorDims dims_w = { L, D, DC, G, SC };
    TensorDesc w_data_desc(dataPrecision, dims_w, getWeightsLayoutByDims(dims_w, false));
    Blob::Ptr w_data_mem = make_shared_blob<Prec>(w_data_desc);
    w_data_mem->allocate();
    auto w_ptr = static_cast<Prec*>(w_data_mem->buffer());
    if (w_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    const VectorDims dims_s = { L, D, SC, G, SC };
    TensorDesc w_state_desc(dataPrecision, dims_s, getWeightsLayoutByDims(dims_s, false));
    Blob::Ptr w_state_mem = make_shared_blob<Prec>(w_state_desc);
    w_state_mem->allocate();
    auto r_ptr = static_cast<Prec*>(w_state_mem->buffer());
    if (r_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    const size_t ie_w_vec_size = getInputShapeAtPort(wIdx).getElementsCount();
    const size_t ie_r_vec_size = getInputShapeAtPort(rIdx).getElementsCount();

    auto *wInputNode = dynamic_cast<Input *>(getParentEdgesAtPort(wIdx)[0]->getParent().get());
    auto wConstBlob = wInputNode->getMemoryPtr();

    auto *rInputNode = dynamic_cast<Input *>(getParentEdgesAtPort(rIdx)[0]->getParent().get());
    auto rConstBlob = rInputNode->getMemoryPtr();

    std::vector<Prec> ie_w_vec(ie_w_vec_size), ie_r_vec(ie_r_vec_size);

    auto ie_w_ptr = ie_w_vec.data();
    auto ie_r_ptr = ie_r_vec.data();
    cpu_convert(wConstBlob->GetPtr(), ie_w_ptr, weightPrec, dataPrecision, ie_w_vec_size);
    cpu_convert(rConstBlob->GetPtr(), ie_r_ptr, weightPrec, dataPrecision, ie_r_vec_size);

    const int step = SC * G;

    for (int g = 0; g < G; g++) {
        for (int out_i = 0; out_i < SC; out_i++) {
            Prec *l_w_ptr = w_ptr + gate_map[g] * SC + out_i;
            for (int in_i = 0; in_i < DC; in_i++) {
                *l_w_ptr = *ie_w_ptr;
                ie_w_ptr++;
                l_w_ptr += step;
            }

            Prec *l_r_ptr = r_ptr + gate_map[g] * SC + out_i;
            for (int in_i = 0; in_i < SC; in_i++) {
                *l_r_ptr = *ie_r_ptr;
                ie_r_ptr++;
                l_r_ptr += step;
            }
        }
    }

    internalBlobs.push_back(w_data_mem);
    internalBlobs.push_back(w_state_mem);
}

template <Precision::ePrecision Prec>
void RNN2::fillBiases(const int *gate_map) {
    using dataType = typename PrecisionTrait<Prec>::value_type;

    if (getOriginalInputPrecisionAtPort(bIdx) != Precision::FP32) {
        THROW_ERROR << "doesn't support bias precision: " << getOriginalInputPrecisionAtPort(bIdx);
    }

    VectorDims dims_b = { L, D, Gb, SC };
    TensorDesc w_bias_data_desc(Prec, dims_b, getWeightsLayoutByDims(dims_b, false));
    Blob::Ptr w_bias_data_mem = make_shared_blob<dataType>(w_bias_data_desc);
    w_bias_data_mem->allocate();
    auto b_ptr = static_cast<dataType*>(w_bias_data_mem->buffer());
    if (b_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    auto *constInputNode = dynamic_cast<Input *>(getParentEdgesAtPort(bIdx)[0]->getParent().get());
    auto constBlob = constInputNode->getMemoryPtr();
    auto const elementsCount = constBlob->GetSize() / constBlob->getDesc().getPrecision().size();

    std::vector<dataType> ie_b_vec(elementsCount);
    cpu_convert(constBlob->GetPtr(),
                &ie_b_vec[0],
                DnnlExtensionUtils::DataTypeToIEPrecision(constBlob->GetDataType()),
                Prec,
                elementsCount);

    for (int g = 0; g < Gb; g++) {
        dataType *l_b_ptr = b_ptr + gate_map[g] * SC;
        const dataType *l_ie_b_ptr = &ie_b_vec[g * SC];
        cpu_memcpy(l_b_ptr, l_ie_b_ptr, SC * sizeof(typename PrecisionTrait<Prec>::value_type));
    }
    internalBlobs.push_back(w_bias_data_mem);
}

void RNN2::copyWeightsData() {
    /* Copy Weight data
     * IE format:
     *   W - [gates, out_state_size, in_data_size]
     *   R - [gates, out_state_size, in_state_size]
     *   B - [gates, out_state_size]
     *
     * DNNL format:
     *   W - [1, 1, in_date_size,  gates, out_state_size]
     *   R - [1, 1, in_state_size, gates, out_state_size]
     *   B - [gates, out_state_size]
     *
     *   Gate order
     *   ====== LSTM ======
     *   Caffe - IFOC, ONNX   - IOFC
     *   IE    - FICO, mkldnn - IFCO
     *
     *   ====== GRU ======
     *   IE - URO, mkldnn - URO
     */
    const int gate_map_lstm[] = {1, 0, 2, 3};  // FICO -> IFCO
    const int gate_map_gru[]  = {0, 1, 2, 3};
    const int gate_map_rnn[]  = {0};
    const int *gate_map;
    const int gate_map_lstm_size = sizeof(gate_map_lstm) / sizeof(int);
    const int gate_map_gru_size = sizeof(gate_map_gru) / sizeof(int);
    const int gate_map_rnn_size = sizeof(gate_map_rnn) / sizeof(int);
    if (cell_type == mkldnn::algorithm::vanilla_lstm) {
        gate_map = gate_map_lstm;
        if (G > gate_map_lstm_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    } else if (cell_type == mkldnn::algorithm::vanilla_gru) {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map";
        }
    } else if (cell_type == mkldnn::algorithm::lbr_gru) {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    } else if (cell_type == mkldnn::algorithm::vanilla_rnn) {
        gate_map = gate_map_rnn;
        if (G > gate_map_rnn_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    } else {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    }

    const auto& dataPrecision = getOriginalInputPrecisionAtPort(0);
    if (dataPrecision == Precision::BF16) {
        fillWeights<uint16_t>(gate_map, wIdx, rIdx);
    } else if (dataPrecision == Precision::FP32) {
        fillWeights<float>(gate_map, wIdx, rIdx);
    } else {// TODO FP16 and INT8 support
        THROW_ERROR << "has unsupported data type: " << dataPrecision;
    }

    if (dataPrecision == Precision::BF16 || dataPrecision == Precision::FP32)
        fillBiases<Precision::FP32>(gate_map);
}

namespace {

DnnlDesriptor makePrimDesc(const RNN2Key& key) {
    switch (key.cellType) {
        case mkldnn::algorithm::vanilla_rnn: {
            DnnlDesriptor desc(std::make_shared<vanilla_rnn_forward::desc>(
                                        prop_kind::forward_scoring,
                                        key.cellAct,
                                        key.direction,
                    /* In Data       */ key.inDataDescs[0]->getDnnlDesc(),
                    /* In State      */ key.inDataDescs[1]->getDnnlDesc(),
                    /* Weights data  */ key.wDescs[0],
                    /* Weights state */ key.wDescs[1],
                    /* Bias          */ key.wDescs[2],
                    /* Out Data      */ key.outDataDescs[0]->getDnnlDesc(),
                    /* Out State     */ key.outDataDescs[1]->getDnnlDesc()));
            return desc;
        } break;
        case mkldnn::algorithm::vanilla_gru: {
            DnnlDesriptor desc(std::make_shared<gru_forward::desc>(
                                        prop_kind::forward_scoring,
                                        key.direction,
                    /* In Data       */ key.inDataDescs[0]->getDnnlDesc(),
                    /* In State      */ key.inDataDescs[1]->getDnnlDesc(),
                    /* Weights data  */ key.wDescs[0],
                    /* Weights state */ key.wDescs[1],
                    /* Bias          */ key.wDescs[2],
                    /* Out Data      */ key.outDataDescs[0]->getDnnlDesc(),
                    /* Out State     */ key.outDataDescs[1]->getDnnlDesc()));
            return desc;
        } break;
        case mkldnn::algorithm::lbr_gru: {
            DnnlDesriptor desc(std::make_shared<lbr_gru_forward::desc>(
                                        prop_kind::forward_scoring,
                                        key.direction,
                    /* In Data       */ key.inDataDescs[0]->getDnnlDesc(),
                    /* In State      */ key.inDataDescs[1]->getDnnlDesc(),
                    /* Weights data  */ key.wDescs[0],
                    /* Weights state */ key.wDescs[1],
                    /* Bias          */ key.wDescs[2],
                    /* Out Data      */ key.outDataDescs[0]->getDnnlDesc(),
                    /* Out State     */ key.outDataDescs[1]->getDnnlDesc()));
            return desc;
        } break;
        case mkldnn::algorithm::vanilla_lstm: {
            DnnlDesriptor desc(std::make_shared<lstm_forward::desc>(
                                        prop_kind::forward_scoring,
                                        key.direction,
                    /* In Data       */ key.inDataDescs[0]->getDnnlDesc(),
                    /* In State      */ key.inDataDescs[1]->getDnnlDesc(),
                    /* In State C    */ key.inDataDescs[2]->getDnnlDesc(),
                    /* Weights data  */ key.wDescs[0],
                    /* Weights state */ key.wDescs[1],
                    /* Bias          */ key.wDescs[2],
                    /* Out Data      */ key.outDataDescs[0]->getDnnlDesc(),
                    /* Out State     */ key.outDataDescs[1]->getDnnlDesc(),
                    /* Out State C   */ key.outDataDescs[2]->getDnnlDesc()));
            return desc;
        } break;
        default: {
            IE_THROW() << "Can't make RNN desc from unknown cell type.";
        }
    }
}

}; // namespace

void RNN2::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                           const std::vector<MemoryDescPtr> &outputDesc) {
    // Fill supported config
    NodeConfig config;
    config.dynBatchSupport = false;
    for (size_t i = 0; i < inputDesc.size(); i++) {
        PortConfig dataConfig;
        dataConfig.inPlace(-1);
        dataConfig.constant(false);
        dataConfig.setMemDesc(inputDesc[i]);
        config.inConfs.push_back(dataConfig);
    }

    for (size_t i = 0; i < outputDesc.size(); i++) {
        PortConfig dataConfig;
        dataConfig.inPlace(-1);
        dataConfig.constant(false);
        dataConfig.setMemDesc(outputDesc[i]);
        config.outConfs.push_back(dataConfig);
    }

    supportedPrimitiveDescriptors.emplace_back(config, ref_any);
}

void RNN2::createPrimitive() {
    wDescs.resize(3);
    auto biasDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{ L, D, Gb, SC });
    wDescs[2] = mkldnn::memory::desc(biasDims, memory::data_type::f32, memory::format_tag::ldgo);

    inDataDescs.resize(S + 1);
    outDataDescs.resize(S + 1);

    Node::createPrimitive();
}

struct RnnCacheEntry {
    RnnCacheEntry(std::shared_ptr<mkldnn::primitive> prim, DnnlDesriptor desc) : prim(prim), desc(desc) {}
    std::shared_ptr<mkldnn::primitive> prim;
    DnnlDesriptor desc;
};

void RNN2::prepareParams() {
    for (size_t i = 0; i < wIdx; i++) {
        auto memPtr = getParentEdgesAtPort(i).front()->getMemoryPtr();
        if (!memPtr || !memPtr->isAllocated())
            THROW_ERROR << "has uninitialized memory at port " << i;
    }

    const auto& dataPrecision = getOriginalInputPrecisionAtPort(0);
    const auto dataType = DnnlExtensionUtils::IEPrecisionToDataType(dataPrecision);

    auto dataMemPtr = getParentEdgesAtPort(0).front()->getMemoryPtr();
    auto Xdims = dataMemPtr->GetShape().getStaticDims();

    const size_t N = m_op->input_batch_first ? Xdims[0] : Xdims[1];
    const size_t T = m_op->input_batch_first ? Xdims[1] : Xdims[0];
    const size_t C = Xdims[2];
    const size_t D = m_op->m_num_directions;
    const size_t L = m_op->m_num_layers;
    const size_t H = m_op->m_hidden_size;

    if (m_op->input_batch_first) {
        // {N, T, C} abc       {T, N, C} bac=ntc(optimal layoutA)
        inDataDescs[0] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{T, N, C}, dataType, memory::format_tag::ntc);
    } else {
        // {T, N, C} abc       {T, N, C} abc=tnc(optimal layoutB)
        inDataDescs[0] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{T, N, C}, dataType, memory::format_tag::tnc);
    }

    if (m_op->output_batch_first) {
        // {N, D, T, H}        {T, N, D*H} bac=ntc(optimal layoutA)
        outDataDescs[0] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{T, N, D*H}, dataType, memory::format_tag::ntc);
    } else {
        // {T, D, N, H}        {T, N, D*H} abc=tnc(optimal layoutB)
        outDataDescs[0] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{T, N, D*H}, dataType, memory::format_tag::tnc);
    }

    // {N, L*D, H}         {L, D, N, H} ldnc(optimal layout)
    const Shape state_shape {L, D, N, H};
    inDataDescs[1] = std::make_shared<DnnlBlockedMemoryDesc>(state_shape, dataType, memory::format_tag::ldnc);
    if (haveCellState(cell_type))
        inDataDescs[2] = std::make_shared<DnnlBlockedMemoryDesc>(state_shape, memory::data_type::f32, memory::format_tag::ldnc);

/*
    const size_t B = dataMemPtr->GetShape().getStaticDims()[0];
    const size_t SL = is_cell ? 1lu : dataMemPtr->GetShape().getStaticDims()[1];
    const Shape shapeS_4D{L, D, B, SC};

    inDataDescs[0] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{SL, B, DC}, dataType, memory::format_tag::tnc);
    outDataDescs[0] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{SL, B, SC}, dataType, memory::format_tag::tnc);

    inDataDescs[1] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, dataType, memory::format_tag::ldnc);
    outDataDescs[1] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, dataType, memory::format_tag::ldnc);

    if (haveCellState(cell_type)) {
        inDataDescs[2] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, memory::data_type::f32, memory::format_tag::ldnc);
        outDataDescs[2] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, memory::data_type::f32, memory::format_tag::ldnc);
    }
*/
    bool wFormatWasChanged = false;
    // WA To avoid different weights layer and iter formats in FP32 case.
    if ((T > 1 || N < optimalBatchSize) && false) {
        if (wFormat != mkldnn::memory::format_tag::ldigo) {
            wFormat = mkldnn::memory::format_tag::ldigo;
            wFormatWasChanged = true;
        }
    } else if (wFormat != mkldnn::memory::format_tag::any || !wasMemoryPrepared) {
        wFormat = mkldnn::memory::format_tag::any;
        wFormatWasChanged = true;
    }
    if (wFormatWasChanged) {
        // {L, D, C, 4, H}
        auto weightsDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{ L, D, DC, G, SC });
        wDescs[0] = mkldnn::memory::desc(weightsDims, dataType, wFormat);
        auto statesDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{ L, D, SC, G, SC });
        wDescs[1] = mkldnn::memory::desc(statesDims, dataType, wFormat);
    }

    RNN2Key key = { inDataDescs, outDataDescs, wDescs, cell_type, cell_act, direction };
    auto engine = getEngine();
    auto builder = [&engine, this](const RNN2Key& key) -> std::shared_ptr<RnnCacheEntry> {
        DnnlDesriptor dnnlDesc = makePrimDesc(key);
        if (key.cellType == mkldnn::algorithm::vanilla_rnn) {
            std::shared_ptr<vanilla_rnn_forward::desc> desc = dnnlDesc;
            return std::make_shared<RnnCacheEntry>(std::make_shared<vanilla_rnn_forward>(vanilla_rnn_forward::primitive_desc(*desc, engine)), dnnlDesc);
        } else if (key.cellType == mkldnn::algorithm::vanilla_gru) {
            std::shared_ptr<gru_forward::desc> desc = dnnlDesc;
            return std::make_shared<RnnCacheEntry>(std::make_shared<gru_forward>(gru_forward::primitive_desc(*desc, engine)), dnnlDesc);
        } else if (key.cellType == mkldnn::algorithm::lbr_gru) {
            std::shared_ptr<lbr_gru_forward::desc> desc = dnnlDesc;
            return std::make_shared<RnnCacheEntry>(std::make_shared<lbr_gru_forward>(lbr_gru_forward::primitive_desc(*desc, engine)), dnnlDesc);
        } else if (key.cellType == mkldnn::algorithm::vanilla_lstm) {
            std::shared_ptr<lstm_forward::desc> desc = dnnlDesc;
            auto pd = lstm_forward::primitive_desc(*desc, engine);
            return std::make_shared<RnnCacheEntry>(std::make_shared<lstm_forward>(pd), dnnlDesc);
        } else {
            IE_THROW() << "Can't create primitive descriptor for RNN node";
        }
    };

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first->prim) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    prim = result.first->prim;
    if (!wasMemoryPrepared || wFormatWasChanged) {
        auto itpd = result.first->desc.createPrimitiveDescriptorIterator(getEngine(), mkldnn::primitive_attr());
        prepareMemory(itpd);
        wasMemoryPrepared = true;
    }
}

std::shared_ptr<MemoryDesc> RNN2::getSrcMemDesc(mkldnn::primitive_desc_iterator& primitive_desc_it, size_t idx) {
    return supportedPrimitiveDescriptors[0].getConfig().inConfs[idx].getMemDesc();
}

std::shared_ptr<MemoryDesc> RNN2::getDstMemDesc(mkldnn::primitive_desc_iterator& primitive_desc_it, size_t idx) {
    return supportedPrimitiveDescriptors[0].getConfig().outConfs[idx].getMemDesc();
}

void RNN2::execute(mkldnn::stream strm) {
    if (!prim)
        THROW_ERROR << "does not have initialized primitive to execute.";

    const auto src_data_mem = getParentEdgeAt(0)->getMemoryPtr();
    const auto dst_data_mem = getChildEdgeAt(0)->getMemoryPtr();

    const auto &wgh_data_mem = internalBlobMemory[0];
    const auto &wgh_stat_mem = internalBlobMemory[1];
    const auto &wgh_bias_mem = internalBlobMemory[2];

    std::unordered_map<int, memory> args {
        {DNNL_ARG_SRC_LAYER,     src_data_mem->GetPrimitive()},
        {DNNL_ARG_WEIGHTS_LAYER, wgh_data_mem->GetPrimitive()},
        {DNNL_ARG_WEIGHTS_ITER,  wgh_stat_mem->GetPrimitive()},
        {DNNL_ARG_BIAS,          wgh_bias_mem->GetPrimitive()},
        {DNNL_ARG_DST_LAYER,     dst_data_mem->GetPrimitive()},
    };

    int state_i_tags[] {DNNL_ARG_SRC_ITER, DNNL_ARG_SRC_ITER_C};
    int state_o_tags[] {DNNL_ARG_DST_ITER, DNNL_ARG_DST_ITER_C};
    for (size_t s = 0; s < S; s++) {
        args[state_i_tags[s]] = getParentEdgeAt(s+1)->getMemoryPtr()->GetPrimitive();
    }

    if (is_cell) {
        for (size_t s = 0; s < S; s++) {
            args[state_o_tags[s]] = getChildEdgesAtPort(s)[0]->getMemoryPtr()->GetPrimitive();
        }
    } else {
        size_t n_ports_with_init_states = outputShapes.size() - 1; // first is a sequence data
        for (size_t s = 0; s < std::min(S, n_ports_with_init_states); s++) {
            if (s < outputShapes.size()) {
                args[state_o_tags[s]] = getChildEdgesAtPort(s+1)[0]->getMemoryPtr()->GetPrimitive();
            }
        }
    }

    (*prim).execute(strm, args);
}

void RNN2::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

std::vector<VectorDims> RNN2::shapeInfer() const {
    std::vector<VectorDims> ret;
    auto Xdims = getParentEdgesAtPort(0)[0]->getMemory().GetShape().getStaticDims();

    const size_t N = m_op->input_batch_first ? Xdims[0] : Xdims[1];
    const size_t T = m_op->input_batch_first ? Xdims[1] : Xdims[0];
    const size_t D = m_op->m_num_directions;
    const size_t L = m_op->m_num_layers;
    const size_t H = m_op->m_hidden_size;

    if (m_op->output_batch_first) {
        // [N, D, T, H]
        ret.emplace_back(VectorDims{N, D, T, H});
    } else {
        ret.emplace_back(VectorDims{T, D, N, H});
    }
    // [N, L*D, H] 
    ret.emplace_back(VectorDims{N, L*D, H});
    ret.emplace_back(VectorDims{N, L*D, H});
    return ret;
}

void RNN2::cleanup() {
    if (!isDynamicNode()) {
        internalBlobs.clear();
    }

    for (auto it : fusedWith) {
        it->cleanup();
    }

    for (auto it : mergedWith) {
        it->cleanup();
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov