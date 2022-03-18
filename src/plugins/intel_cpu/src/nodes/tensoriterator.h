// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <graph.h>
#include <string>
#include <memory>
#include <vector>
#include <common/memory_desc_wrapper.hpp>

namespace ov {
namespace intel_cpu {
namespace node {

struct PortMap {
    // Data map rule
    int from; /**< Index of external data from ins/outs fields of node */
    int to;   /**< Index of internal data in iterator body */

    // Iteration rule
    int axis;      /**< Axis to iterate throught */
    int stride;    /**< Stride to iterate throught */
    int start;     /**< Start index of iteration range */
    int end;       /**< Last index of iteration range  */
    int part_size; /**< Part size which will be transfered to body subnetwork */
};

/**
 * Functor interface to perform some action with pointed tensors (captured in constructor)
 * Generally it's read, write or move data from specified tensors.
 * Action may depends on iteration index.
 */
class PortMapHelper {
public:
    virtual ~PortMapHelper() = default;
    virtual void execute(mkldnn::stream strm, int n_iter = -1) = 0;
protected:
    mkldnn::reorder reorder;
    mkldnn::memory mem_holder_src;
    mkldnn::memory mem_holder_dst;
};


/**
 * Functor interface to perform check of data tensor (captured in constructor)
 * Information extracted as int. Meaning of returned value is specific for
 * particular type of checker.
 */
class PortChecker {
public:
    virtual ~PortChecker() = default;
    virtual int getStatus() = 0;
protected:
    mkldnn::memory mem_holder;
};


/**
 * Class for storing intermediate output buffer state for dynamism when we don't know
 * final output shape but we should concatenate output after each iteration
 */
class DynamicBuffer {
public:
    DynamicBuffer(const MemoryPtr &from_, const std::vector<MemoryPtr> &to_, const PortMap &map_rule_);
    ~DynamicBuffer() = default;

    void execute(const mkldnn::engine& eng, const int iter);
    void transfer(const Node* node);

private:
    void init(const mkldnn::engine& eng);

    /* methods for resize and refill buffer */
    std::shared_ptr<mkldnn::memory> create_buffer(const mkldnn::engine& eng);
    void move_buffer(std::shared_ptr<mkldnn::memory> new_buffer);
    void move_data();

    static void copy(const uint8_t* src, uint8_t* dst, const size_t src_stride, const size_t dst_stride, const size_t count, const size_t len);
    static uint8_t* get_ptr(mkldnn::memory& prim);

    size_t len = 1lu;
    size_t count = 1lu;
    size_t elem_size = 0lu;
    ptrdiff_t chunk_offset_in_byte = 0;
    ptrdiff_t buffer_offset_in_byte = 0;

    MemoryPtr from;
    std::vector<MemoryPtr> to;
    PortMap map_rule;

    std::shared_ptr<mkldnn::memory> mem_holder_buffer;
};

class TensorIterator : public Node {
public:
    TensorIterator(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool isExecutable() const override { return true; }

    void setExtManager(const ExtensionManager::Ptr& extMgr) { ext_mng = extMgr; }

protected:
    //  needShapeInfer() should return false
    //  because we cannot resolve the output dimensions before the inference is completed
    bool needShapeInfer() const override { return false; };

    bool needPrepareParams() const override;
    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    void prepareInputPorts();
    void prepareOutputPorts();
    void prepareBackEdges();
    void prepareDynamicBackEdges();
    void prepareDynamicBuffers();
    void prepareLoopBodyCurrentIteration();
    void prepareContinueCond();
    void prepareInitialCond();
    void prepareTripCount();

    /* Dynamic support */
    void reshapeSubgraphInput();
    void reshapeAndFillOutput(mkldnn::stream strm);
    int getNumIteration(const std::vector<PortMap>& inputPortMap, const std::vector<PortMap>& outputPortMap) const;

    ExtensionManager::Ptr ext_mng;
    Graph sub_graph;
    std::vector<std::vector<MemoryPtr>> input_mems;
    std::vector<MemoryPtr> output_mem;

    std::vector<std::shared_ptr<PortMapHelper>>
        first_mappers,   /// < Applied once before loop
        last_mappers,    /// < Applied once after loop
        before_mappers,  /// < Applied before each iteration
        after_mappers,   /// < Applied after each iteration
        back_mappers;    /// < Applied before each iteration for dynamic shapes

    std::shared_ptr<PortChecker>
        trip_count_check,      /// < Perform check of trip count value. value >= -1
        initial_cond_check,    /// < Perform check of initial continue condition value. value [0, 1]
        continue_cond_check;   /// < Perform check of continue condition value of body. value [0, 1]

    std::vector<std::shared_ptr<DynamicBuffer>> buffers;

    std::vector<PortMap> inputPortMap;  //!< Input ports map
    std::vector<PortMap> outputPortMap;  //!< Output ports map
    std::vector<PortMap> backEdges;  //!< Back edges map

    std::vector<int> loopBodyCurrentIterationIdx;
    int loopBodyConditionOutputIdx = -1;
    int loopTripCountIdx = -1;
    int loopExecutionConditionIdx = -1;

    int lastUsedTripCount = -1;
    bool lastUsedCond = false;

    const std::shared_ptr<ov::Node> ngraphOp;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
