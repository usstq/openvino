// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/ric_fusion.hpp"

#include <memory>
#include <ngraph/log.hpp>
#include <ngraph/op/util/binary_elementwise_arithmetic.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/core/validation_util.hpp>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

namespace ngraph {
namespace pass {
namespace ric_attr {

// Attribute describes RIC type which we propagate.
// Also, it contains callback which can expand this attribute to the real RIC sub-graph.
// In addition, attribute has some functionality and properties for propagation.
class Attribute {
public:
    using callback_t = std::function<void(Input<Node>, const Attribute&)>;

    Attribute(std::vector<int64_t> order, int64_t axis, bool is_final = false, bool is_initial = false)
        : m_order(std::move(order)),
          m_axis(axis),
          m_is_final(is_final),
          m_is_initial(is_initial) {
        m_can_be_fused.emplace_back(std::make_shared<bool>(true));
    }

    // Method which is used to create a copy of attribute for further propagation.
    // TODO: can be removed and replaced with regular copy but we need to get rid of
    //       is_initial flag and use some other way to detect original RIC output.
    Attribute propagate() const {
        Attribute attr(m_order, m_axis);
        attr.m_can_be_fused = m_can_be_fused;
        return attr;
    }

    void set_is_final(bool is_final) {
        m_is_final = is_final;
    }

    void set_can_be_fused(bool can_be_fused) {
        std::for_each(m_can_be_fused.cbegin(),
                      m_can_be_fused.cend(),
                      [can_be_fused](const std::shared_ptr<bool>& state) {
                          *state = can_be_fused;
                      });
    }

    void set_callback(callback_t callback) {
        m_callback = std::move(callback);
    }

    // Apply callback to materialize RIC inside graph
    void operator()(Input<Node> input) const {
        m_callback(input, *this);
    }

    bool can_be_fused() const {
        return std::all_of(m_can_be_fused.cbegin(), m_can_be_fused.cend(), [](const std::shared_ptr<bool>& state) {
            return *state;
        });
    }

    // For cases when we propagate through operation with multiple inputs like Eltwise
    // we have to merge RIC attrs from all inputs. To check that given attr be merged with
    // current we check the order and axis which must be the same.
    bool can_be_merged_with(const Attribute& other) {
        return (m_order.empty() || other.m_order.empty() || m_order == other.m_order) && m_axis == other.m_axis;
    }

    // When merging two and more attrs for further propagation we have to keep can_be_fused references
    // for cases when fusion is not possible, so we can update all related attrs.
    void merge_with(const Attribute& other) {
        m_can_be_fused.insert(m_can_be_fused.end(), other.m_can_be_fused.begin(), other.m_can_be_fused.end());
    }

    const std::vector<int64_t>& get_order() const {
        return m_order;
    }

    void set_order(const std::vector<int64_t>& order) {
        m_order = order;
    }

    int64_t get_axis() const {
        return m_axis;
    }

    void set_axis(int64_t axis) {
        m_axis = axis;
    }

    bool is_final() const {
        return m_is_final;
    }

    bool is_initial() const {
        return m_is_initial;
    }

private:
    // empty order means that the order is default and must be n, n-1, ..., 0
    // according to the dimension values specified by m_axis
    std::vector<int64_t> m_order;
    int64_t m_axis;

    // Specifies whether RIC can be fused or not. vector is needed to keep references to other
    // attributes that were participated during merge.
    std::vector<std::shared_ptr<bool>> m_can_be_fused;

    // true - means that current RIC attribute is final and can be materialized
    // false - means that current RIC attribute is temporary and need only for propagation
    bool m_is_final;

    // true - means that current RIC attribute is an initial attribute and belongs to real RIC output
    // false - means that current RIC attribute is temporary and need only for propagation
    bool m_is_initial;

    // Callback specifies the action for RIC materialization for given input port.
    // In most cases it should insert Gather operation for the input.
    std::function<void(Input<Node>, const Attribute&)> m_callback = [](Input<Node>, const Attribute&) {};
};

namespace {

template <typename T>
using is_port = typename std::enable_if<!std::is_convertible<T, std::shared_ptr<Node>>::value>::type;

template <typename T, typename = is_port<T>>
void set(T port, const Attribute& ric_attr) {
    auto& attrs = port.get_rt_info();
    attrs["reverse_input_channel_index"] = ric_attr;
}

// Available only for output ports
void init(Output<Node> output, std::vector<int64_t> order, int64_t axis) {
    set(output, Attribute(std::move(order), axis, false, true));
}

template <typename T, typename = is_port<T>>
bool has(const T& port) {
    const auto& attrs = port.get_rt_info();
    return attrs.count("reverse_input_channel_index");
}

template <typename T, typename = is_port<T>>
Attribute get(const T& port) {
    const auto& attrs = port.get_rt_info();
    auto res = attrs.find("reverse_input_channel_index");
    if (res != attrs.end()) {
        return res->second.template as<Attribute>();
    }
    throw ngraph_error("reverse_input_channel_index is missing in given port");
}

template <typename T, typename = is_port<T>>
void erase(T port) {
    auto& rt_info = port.get_rt_info();
    rt_info.erase("reverse_input_channel_index");
}
}  // namespace
}  // namespace ric_attr

namespace init {
class SplitConcat : public ngraph::pass::MatcherPass {
public:
    SplitConcat() {
        MATCHER_SCOPE(SplitConcat);
        auto split_p = pattern::wrap_type<opset8::Split>();
        auto pattern_root = pattern::wrap_type<opset8::Concat>({split_p, split_p, split_p});

        auto callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto concat = ov::as_type_ptr<opset8::Concat>(pattern_map.at(pattern_root).get_node_shared_ptr());
            auto split = ov::as_type_ptr<opset8::Split>(pattern_map.at(split_p).get_node_shared_ptr());
            if (!concat || !split)
                return false;

            // Avoid cases with two consecutive Split->Concat
            if (ric_attr::has(split->input_value(0))) {
                return false;
            }

            std::vector<int64_t> order;
            order.reserve(split->get_num_splits());

            for (const auto& input : concat->inputs()) {
                auto split_output = input.get_source_output();
                if (split_output.get_node() != split.get())
                    return false;

                // Check that Concat is the only Split consumer and order of Split outputs
                // satisfies expected order for reverse input channel case.
                for (const auto& target_input : split_output.get_target_inputs()) {
                    if (target_input.get_node() != concat.get()) {
                        return false;
                    }
                    order.emplace_back(split_output.get_index());
                }
            }

            // Check that all order values are unique, otherwise it is not RIC
            std::set<int64_t> unique_values(order.cbegin(), order.cend());
            if (unique_values.size() != order.size()) {
                return false;
            }

            // Mark-up RIC output
            ric_attr::init(concat, order, concat->get_axis());
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Gather : public ngraph::pass::MatcherPass {
public:
    Gather() {
        MATCHER_SCOPE(Gather);
        auto input_p = pattern::any_input(pattern::has_static_rank());
        auto indices_p = pattern::any_input();
        auto axis_p = pattern::wrap_type<opset8::Constant>();
        auto pattern_root = pattern::wrap_type<opset8::Gather>({input_p, indices_p, axis_p});

        auto callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& output = pattern_map.at(pattern_root);

            auto axis = ov::get_constant_from_source(pattern_map.at(axis_p));
            if (!axis)
                return false;

            const auto axis_value = axis->cast_vector<int64_t>().at(0);

            if (ov::is_preprocesing_node(output.get_node_shared_ptr())) {
                ric_attr::init(output, {}, axis_value);
                return true;
            }

            auto order = ov::get_constant_from_source(pattern_map.at(indices_p));
            if (!order)
                return false;

            // Avoid cases with two consecutive Gathers
            if (ric_attr::has(pattern_map.at(input_p))) {
                return false;
            }

            // This constraint helps to avoid detection of other Gathers that do not perform RIC
            const auto& data_shape = m.get_match_root()->input(0).get_partial_shape();
            if (shape_size(order->get_shape()) == 1 || axis_value < 0 || axis_value >= data_shape.rank().get_length() ||
                data_shape[axis_value].is_dynamic() ||
                shape_size(order->get_shape()) != static_cast<size_t>(data_shape[axis_value].get_length())) {
                return false;
            }

            // Check that all order values are unique, otherwise it is not RIC
            const auto& order_values = order->cast_vector<int64_t>();
            std::set<int64_t> unique_values(order_values.cbegin(), order_values.cend());
            if (unique_values.size() != order_values.size()) {
                return false;
            }
            ric_attr::init(output, order_values, axis_value);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};
}  // namespace init

namespace prop {
namespace {
std::shared_ptr<opset8::Constant> create_const(const std::vector<int64_t>& values) {
    return opset8::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}
}  // namespace

class Binary : public ngraph::pass::MatcherPass {
public:
    Binary() {
        MATCHER_SCOPE(Binary);
        auto pattern_root = pattern::wrap_type<op::util::BinaryElementwiseArithmetic, opset8::FakeQuantize>();

        auto callback = [=](pattern::Matcher& m) {
            const auto& root = m.get_match_root();
            const auto& inputs = root->inputs();

            std::map<size_t, ric_attr::Attribute> attrs;
            for (const auto& input : inputs) {
                auto output = input.get_source_output();
                if (ric_attr::has(output)) {
                    attrs.insert({input.get_index(), ric_attr::get(output).propagate()});
                } else if (!ov::is_type<opset8::Constant>(output.get_node())) {
                    // If number of non-constant inputs and without RIC attr is greater than 0 we have to skip
                    // propagation because it is not efficient to have a lot of RIC copies on data path.
                    return false;
                }
            }

            if (attrs.empty())
                return false;

            // Check that all RIC attrs can be merged and then merge them
            auto ric = attrs.begin()->second;
            auto rank = root->get_input_partial_shape(attrs.begin()->first).rank();
            if (rank.is_dynamic())
                return false;
            auto data_rank = rank.get_length();

            for (const auto& item : attrs) {
                const auto& input_rank = root->get_input_partial_shape(item.first).rank();
                if (input_rank.is_static() && input_rank.get_length() == data_rank &&
                    ric.can_be_merged_with(item.second)) {
                    ric.merge_with(item.second);
                } else {
                    return false;
                }
            }

            for (const auto& input : inputs) {
                // Skip input that have RIC attribute
                if (attrs.count(input.get_index()))
                    continue;

                auto const_output = input.get_source_output();
                const auto& shape = const_output.get_shape();
                const int64_t& shape_rank = static_cast<int64_t>(shape.size());
                if (shape_rank > data_rank) {
                    // TODO: handle case when constant input broadcast another one
                    return false;
                }

                if (data_rank - shape_rank > ric.get_axis()) {
                    // we don't have to insert RIC for constant, so we keep propagating
                    ric_attr::set(m.get_match_value(), ric);
                    continue;
                }

                const int64_t& new_axis = ric.get_axis() - (data_rank - shape_rank);
                const auto& axis_dim = shape[new_axis];
                if (axis_dim == 1) {
                    // we don't have to insert RIC for constant, so we keep propagating
                    ric_attr::set(m.get_match_value(), ric);
                    continue;
                }

                // finally, insert RIC
                auto ric_const = ric;
                ric_const.set_axis(new_axis);
                ric_const.set_is_final(true);
                ric_const.set_callback([axis_dim](Input<Node> input, const ric_attr::Attribute& attr) {
                    auto output = input.get_source_output();
                    // Handle case when the RIC order is default
                    auto order = attr.get_order();
                    if (order.empty()) {
                        order.resize(axis_dim);
                        std::iota(order.rbegin(), order.rend(), 0);
                    }
                    auto gather =
                        std::make_shared<opset8::Gather>(output, create_const(order), create_const({attr.get_axis()}));
                    input.replace_source_output(gather);
                    // TODO: copy runtime info from RIC sub-graph
                });
                ric_attr::set(input, ric_const);
            }

            ric_attr::set(m.get_match_value(), ric);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Convolution : public ngraph::pass::MatcherPass {
public:
    Convolution() {
        MATCHER_SCOPE(Convolution);
        // Handle Convolution with Constant and FQ on weights. As Convolution is
        // a terminal node, so we do not propagate RIC attribute further and insert
        // final RIC attribute to the weights input.
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        auto pattern_root =
            pattern::wrap_type<opset8::Convolution>({input_p,
                                                     pattern::wrap_type<opset8::Constant, opset8::FakeQuantize>(
                                                         pattern::has_static_dim(1 /*output channel*/))});
        auto callback = [=](pattern::Matcher& m) {
            auto conv = m.get_match_root();
            auto ric = ric_attr::get(conv->input_value(0)).propagate();
            if (ric.get_axis() != 1)
                return false;

            ric.set_is_final(true);
            ric.set_callback([](Input<Node> input, const ric_attr::Attribute& attr) {
                const auto output_channel_index = 1;
                auto order = attr.get_order();
                // Handle case when the RIC order is default
                if (order.empty()) {
                    order.resize(input.get_partial_shape()[output_channel_index].get_length());
                    std::iota(order.rbegin(), order.rend(), 0);
                }
                auto weights = input.get_source_output();
                auto gather = std::make_shared<opset8::Gather>(weights,
                                                               create_const(order),
                                                               create_const({output_channel_index}));
                input.replace_source_output(gather);
                // TODO: copy runtime info from RIC sub-graph
            });

            if (auto fq = std::dynamic_pointer_cast<opset8::FakeQuantize>(conv->get_input_node_shared_ptr(1))) {
                // Set final RIC attr to the first FQ input
                ric_attr::set(fq->input(0), ric);

                // Apply Binary transformation for FQ to handle 1..5 inputs
                ric.set_is_final(false);
                ric_attr::set(fq->input_value(0), ric);  // set ric attr to simulate propagation flow
                Binary().apply(fq);
            } else {
                ric_attr::set(conv->input(1), ric);
            }
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class GroupConvolution : public ngraph::pass::MatcherPass {
public:
    GroupConvolution() {
        MATCHER_SCOPE(GroupConvolution);
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        auto pattern_root = pattern::wrap_type<opset8::GroupConvolution>(
            {input_p, pattern::wrap_type<opset8::Constant, opset8::FakeQuantize>(pattern::has_static_shape())});

        auto callback = [=](pattern::Matcher& m) {
            auto conv = m.get_match_root();
            const auto& weights_shape = conv->input_value(1).get_shape();
            const int64_t& group = static_cast<int64_t>(weights_shape.at(0));
            const int64_t& channels = static_cast<int64_t>(weights_shape.at(1));
            const int64_t& in_channels = static_cast<int64_t>(weights_shape.at(2));

            auto ric = ric_attr::get(conv->input_value(0)).propagate();
            auto order = ric.get_order();
            // Handle case when the RIC order is default
            if (order.empty()) {
                order.resize(group);
                std::iota(order.rbegin(), order.rend(), 0);
                ric.set_order(order);
            }

            if (in_channels != 1 || ric.get_order().size() != static_cast<size_t>(group) || ric.get_axis() != 1) {
                // TODO: insert RIC when group == 1
                return false;
            }

            // Update weights with RIC attribute
            auto ric_weights = ric;
            ric_weights.set_is_final(true);
            ric_weights.set_axis(0);
            ric_weights.set_callback([](Input<Node> input, const ric_attr::Attribute& attr) {
                auto weights = input.get_source_output();
                auto gather = std::make_shared<opset8::Gather>(weights,
                                                               create_const(attr.get_order()),
                                                               create_const({0} /* output channel */));
                input.replace_source_output(gather);
                // TODO: copy runtime info from RIC sub-graph
            });

            if (auto fq = std::dynamic_pointer_cast<opset8::FakeQuantize>(conv->get_input_node_shared_ptr(1))) {
                // Set final RIC attr to the first FQ input
                ric_attr::set(fq->input(0), ric_weights);

                // Apply Binary transformation for FQ to handle 1..5 inputs
                ric_weights.set_is_final(false);
                ric_attr::set(fq->input_value(0), ric_weights);  // set ric attr to simulate propagation flow
                Binary().apply(fq);
            } else {
                ric_attr::set(conv->input(1), ric_weights);
            }

            // Calculate new order for RIC propagation
            const int64_t output_channels = group * channels;
            std::vector<int64_t> new_order;
            new_order.reserve(output_channels);
            for (const auto& index : ric.get_order()) {
                for (int64_t pos = index * channels, i = 0; i < channels; ++i, ++pos) {
                    new_order.emplace_back(pos);
                }
            }
            assert(new_order.size() == static_cast<size_t>(output_channels));

            ric.set_order(new_order);
            ric_attr::set(conv->output(0), ric);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class ShapeOf : public ngraph::pass::MatcherPass {
public:
    ShapeOf() {
        MATCHER_SCOPE(ShapeOf);
        auto pattern_root = pattern::wrap_type<opset1::ShapeOf, opset8::ShapeOf>();

        auto callback = [=](pattern::Matcher& m) {
            // Skip propagation for ShapeOf path
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class PassThrough : public ngraph::pass::MatcherPass {
public:
    PassThrough() {
        MATCHER_SCOPE(PassThrough);
        auto pattern_root =
            pattern::wrap_type<op::util::UnaryElementwiseArithmetic, opset8::Convert, opset8::Pad, opset8::PRelu>();

        auto callback = [=](pattern::Matcher& m) {
            auto root = m.get_match_root();
            if (!ric_attr::has(root->input_value(0)))
                return false;
            ric_attr::set(root->output(0), ric_attr::get(root->input_value(0)).propagate());
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Transpose : public ngraph::pass::MatcherPass {
public:
    Transpose() {
        MATCHER_SCOPE(Transpose);
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        auto order_p = pattern::wrap_type<opset8::Constant>();
        auto pattern_root = pattern::wrap_type<opset8::Transpose>({input_p, order_p});

        auto callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto input = pattern_map.at(input_p);
            auto ric = ric_attr::get(input).propagate();

            auto order_node =
                std::dynamic_pointer_cast<opset8::Constant>(pattern_map.at(order_p).get_node_shared_ptr());
            auto order = order_node->cast_vector<int64_t>();

            int64_t new_axis = std::find(order.begin(), order.end(), ric.get_axis()) - order.begin();
            ric.set_axis(new_axis);

            ric_attr::set(m.get_match_value(), ric);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Unsupported : public ngraph::pass::MatcherPass {
public:
    Unsupported() {
        MATCHER_SCOPE(Unsupported);
        auto pattern_root = pattern::any_input();
        auto callback = [=](pattern::Matcher& m) {
            for (const auto& input : m.get_match_root()->input_values()) {
                if (ric_attr::has(input)) {
                    auto ric = ric_attr::get(input);
                    ric.set_can_be_fused(false);
                    NGRAPH_DEBUG << "Node is unsupported by RIC Fusion: " << *m.get_match_root() << std::endl;
                }
            }
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};
}  // namespace prop

namespace fuse {
namespace {
bool need_to_erase_ric(const Output<Node>& output) {
    if (!ric_attr::has(output))
        return false;
    const auto& ric = ric_attr::get(output);
    return ric.can_be_fused() && ric.is_initial();
}
}  // namespace

class InsertReverseInputChannel : public ngraph::pass::MatcherPass {
public:
    InsertReverseInputChannel() {
        MATCHER_SCOPE(InsertReverseInputChannel);
        auto pattern_root = pattern::any_input();
        auto callback = [](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            for (const auto& input : node->inputs()) {
                if (!ric_attr::has(input))
                    continue;
                const auto& ric = ric_attr::get(input);
                if (ric.can_be_fused() && ric.is_final()) {
                    ric(input);
                }
            }
            return false;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class EraseSplitConcat : public ngraph::pass::MatcherPass {
public:
    EraseSplitConcat() {
        MATCHER_SCOPE(EraseSplitConcat);
        auto input_p = pattern::any_input();
        auto split_p = pattern::wrap_type<opset8::Split>({input_p, pattern::any_input()});
        auto pattern_root = pattern::wrap_type<opset8::Concat>({split_p, split_p, split_p}, need_to_erase_ric);

        auto callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto output = pattern_map.at(pattern_root);
            auto input = pattern_map.at(input_p);
            output.replace(input);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class EraseGather : public ngraph::pass::MatcherPass {
public:
    EraseGather() {
        MATCHER_SCOPE(EraseGather);
        auto input_p = pattern::any_input();
        auto pattern_root = pattern::wrap_type<opset8::Gather>({input_p, pattern::any_input(), pattern::any_input()},
                                                               need_to_erase_ric);
        auto callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto output = pattern_map.at(pattern_root);
            auto input = pattern_map.at(input_p);
            output.replace(input);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};
}  // namespace fuse

bool ngraph::pass::ReverseInputChannelsFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    Manager m;
    m.set_per_pass_validation(false);

    // First we need to initialize and propagate RIC attributes through entire graph
    auto ric_prop = m.register_pass<GraphRewrite>();
    ric_prop->add_matcher<init::SplitConcat>();
    ric_prop->add_matcher<init::Gather>();
    ric_prop->add_matcher<prop::Convolution>();
    ric_prop->add_matcher<prop::GroupConvolution>();
    ric_prop->add_matcher<prop::Binary>();
    ric_prop->add_matcher<prop::ShapeOf>();
    ric_prop->add_matcher<prop::Transpose>();
    ric_prop->add_matcher<prop::PassThrough>();
    ric_prop->add_matcher<prop::Unsupported>();

    // TODO: validate attributes by request

    // Second we fuse available RIC into nodes and remove original nodes related to fused RIC
    auto ric_fuse = m.register_pass<GraphRewrite>();
    ric_fuse->add_matcher<fuse::InsertReverseInputChannel>();
    ric_fuse->add_matcher<fuse::EraseSplitConcat>();
    ric_fuse->add_matcher<fuse::EraseGather>();

    m.run_passes(model);
    return false;
}
}  // namespace pass
}  // namespace ngraph
