// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include "pattern_node.hpp"

namespace ov {
namespace intel_cpu {

const int _matcher_verbose = std::getenv("MATCHER_VERBOSE") ? (atoi(std::getenv("MATCHER_VERBOSE"))) : 0;

class AttributePredicate : public ngraph::AttributeVisitor {
    std::map<std::string, attr> attr_map;
    std::map<std::string, bool> attr_match;

public:
    AttributePredicate(const std::vector<attr>& attr) {
        for (auto& a : attr) {
            attr_map[a.name] = a;
            attr_match[a.name] = false;
        }
    }

    bool all_matched(bool verbose = false) {
        bool ret = true;
        for (auto& a : attr_match) {
            if (!a.second) {
                auto& attr = attr_map[a.first];
                verbose_log("     AttributePredicate: failed at ", attr.to_string());
            }
            ret = ret && a.second;
        }
        return ret;
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& value = a->get();
            attr_match[name] = it->second.predicate(value.to_string());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Shape>>(&adapter)) {
            ov::PartialShape value(a->get());
            attr_match[name] = it->second.predicate(value.to_string());
        } else {
            std::cout << "...." << name << ":" << it->second.to_string() << " vs ???" << std::endl;
            attr_match[name] = false;
        }
        /*
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& value = join(a->get());
            append_attribute(name.c_str(), value.c_str());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<ov::element::Type>>>(&adapter)) {
            const auto& value = join(a->get());
            append_attribute(name.c_str(), value.c_str());
        } else {
            append_attribute(name.c_str(), "?");
        }
        */
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }


    void on_adapter(const std::string& name, ngraph::ValueAccessor<int>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<float>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }
    /*
        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        template<class Container>
        inline std::string join(const Container& strs) {
            std::stringstream ss;
            ss << "[" << ov::intel_cpu::join(strs, ',') << "]";
            return ss.str();
        }
    */
};

bool attr_compatible(ov::Node& node, const std::vector<attr>& attr) {
    AttributePredicate vis(attr);
    node.visit_attributes(vis);
    return vis.all_matched(true);
}

}  // namespace intel_cpu
}  // namespace ov