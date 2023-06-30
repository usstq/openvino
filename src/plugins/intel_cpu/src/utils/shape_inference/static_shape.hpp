// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "cpu_types.h"
#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/shape.hpp"
#include "shape_infer_type_utils.hpp"
#include "static_dimension.hpp"

namespace ov {
namespace op {
    struct AutoBroadcastSpec;
}   // namespace op

namespace intel_cpu {
/// \brief Class representing a shape that must be totally static.
class StaticShape : public std::vector<StaticDimension> {
public:
    using ShapeContainer = StaticShape;

    StaticShape() = default;
    StaticShape(std::initializer_list<StaticDimension> init);
    StaticShape(const std::vector<StaticDimension::value_type>& dimensions);
    StaticShape(std::vector<StaticDimension> dimensions);

    StaticShape(const PartialShape&);

    static bool is_static() {
        return true;
    }
    static bool is_dynamic() {
        return false;
    }

    Rank rank() const {
        return Rank(size());
    }

    bool compatible(const StaticShape& s) const;
    bool same_scheme(const StaticShape& s) const;
    bool refines(const StaticShape& s) const;
    bool merge_rank(const Rank& r);

    ov::Shape to_shape() const;
    PartialShape to_partial_shape() const;

    friend std::ostream& operator<<(std::ostream& str, const StaticShape& shape);
    friend StaticShape operator+(const StaticShape& s1, const StaticShape& s2);
    bool operator==(const StaticShape& shape) const;
    bool operator!=(const StaticShape& shape) const;
    /// Get the max bounding shape
    Shape get_max_shape() const;
    /// Get the min bounding shape
    Shape get_min_shape() const;
    /// Get the unique shape
    Shape get_shape() const;
    static bool merge_into(StaticShape& dst, const StaticShape& src);
    static bool broadcast_merge_into(StaticShape& dst, const StaticShape& src, const ov::op::AutoBroadcastSpec& autob);
};

StaticShape operator+(const StaticShape& s1, const StaticShape& s2);
std::ostream& operator<<(std::ostream& str, const StaticShape& shape);

/**
 * @brief Main template for conditional static shape adapter which holds reference or container with CPU dimensions.
 */
template <class TDims>
class StaticShapeAdapter {};

using StaticShapeRef = StaticShapeAdapter<const VectorDims>;
using StaticShapeCon = StaticShapeAdapter<VectorDims>;  // Rename when when StaticShape class will be removed.

template <class T>
constexpr bool is_static_shape_adapter() {
    using U = typename std::decay<T>::type;
    return std::is_same<U, StaticShapeRef>::value || std::is_same<U, StaticShapeCon>::value;
}

/**
 * @brief The static shape adapter by copy value to VectorDims.
 *
 * This adapter is read/write stored VectorDims.
 */
template <>
class StaticShapeAdapter<VectorDims> {
    using TDims = VectorDims;
    using dim_type = typename TDims::value_type;

public:
    using ShapeContainer = StaticShapeCon;

    using value_type = StaticDimension;
    using iterator = typename TDims::iterator;
    using const_iterator = typename TDims::const_iterator;

    static_assert(std::is_same<dim_type, typename StaticDimension::value_type>::value,
                  "Static dimension must be of the same type as the CPU dimension.");
    static_assert(std::is_standard_layout<StaticDimension>::value,
                  "StaticShape must be standard layout to cast on CPU dimension type.");
    static_assert(sizeof(dim_type) == sizeof(StaticDimension),
                  "StaticDimension must have the same number of bytes as the CPU dimension type.");

    StaticShapeAdapter();
    StaticShapeAdapter(const TDims& dims);
    StaticShapeAdapter(TDims&& dims) noexcept;
    StaticShapeAdapter(std::initializer_list<dim_type> dims) noexcept : m_dims{dims} {}
    StaticShapeAdapter(const StaticShapeCon& other);
    StaticShapeAdapter(const ov::PartialShape&);

    const TDims& operator*() const& noexcept {
        return m_dims;
    }

    TDims& operator*() & noexcept {
        return m_dims;
    }

    const TDims&& operator*() const&& noexcept {
        return std::move(m_dims);
    }

    TDims&& operator*() && noexcept {
        return std::move(m_dims);
    }

    value_type& operator[](size_t i) {
        return reinterpret_cast<value_type&>(m_dims[i]);
    }

    const value_type& operator[](size_t i) const {
        return reinterpret_cast<const value_type&>(m_dims[i]);
    }

    //-- Shape functions
    static constexpr bool is_static() {
        return true;
    }

    static constexpr bool is_dynamic() {
        return !is_static();
    }

    template <class T>
    constexpr typename std::enable_if<is_static_shape_adapter<T>(), bool>::type compatible(
        const T& other) const {
        // for static shape compatible == both shape equals
        return *this == other;
    }

    template <class T>
    constexpr typename std::enable_if<is_static_shape_adapter<T>(), bool>::type same_scheme(
        const T& other) const {
        // for static shape same_scheme == compatible;
        return compatible(other);
    }

    ov::Rank rank() const;
    bool merge_rank(const ov::Rank& r);
    ov::Shape to_shape() const;
    ov::Shape get_max_shape() const;
    ov::Shape get_min_shape() const;
    ov::Shape get_shape() const;
    ov::PartialShape to_partial_shape() const;

    static bool merge_into(StaticShapeAdapter& dst, const StaticShapeAdapter& src);
    static bool broadcast_merge_into(StaticShapeAdapter& dst,
                                     const StaticShapeAdapter& src,
                                     const ov::op::AutoBroadcastSpec& autob);

    //-- Container functions
    const_iterator cbegin() const noexcept {
        return m_dims.cbegin();
    }

    const_iterator begin() const noexcept {
        return cbegin();
    }

    iterator begin() noexcept {
        return m_dims.begin();
    }

    const_iterator cend() const noexcept {
        return m_dims.cend();
    }

    const_iterator end() const noexcept {
        return cend();
    }

    iterator end() noexcept {
        return m_dims.end();
    }

    size_t size() const {
        return m_dims.size();
    }

    bool empty() const {
        return m_dims.empty();
    }

    void resize(size_t n) {
        m_dims.resize(n);
    }

    void reserve(size_t n) {
        m_dims.reserve(n);
    }

    iterator insert(iterator position, const value_type& value) {
        return m_dims.insert(position, value.get_length());
    }

    void insert(iterator position, size_t n, const value_type& val) {
        m_dims.insert(position, n, val.get_length());
    }

    template <class InputIterator>
    void insert(iterator position, InputIterator first, InputIterator last) {
        m_dims.insert(position, first, last);
    }

    void push_back(const dim_type value) {
        m_dims.push_back(value);
    }

    void push_back(const value_type& value) {
        m_dims.push_back(value.get_length());
    }

    template <class... Args>
    void emplace_back(Args&&... args) {
        m_dims.emplace_back(std::forward<Args>(args)...);
    }

private:
    TDims m_dims;
};

/**
 * @brief The static shape adapter by reference to VectorDims.
 *
 * This adapter is read-only access for VectorDims.
 */
template <>
class StaticShapeAdapter<const VectorDims> {
    using TDims = VectorDims;
    using dim_type = typename VectorDims::value_type;

public:
    using ShapeContainer = StaticShapeCon;

    using value_type = StaticDimension;
    using iterator = typename TDims::const_iterator;
    using const_iterator = typename TDims::const_iterator;

    static_assert(std::is_same<dim_type, typename StaticDimension::value_type>::value,
                  "Static dimension must be of the same type as the CPU dimension.");
    static_assert(std::is_standard_layout<StaticDimension>::value,
                  "StaticShape must be standard layout to cast on CPU dimension type.");
    static_assert(sizeof(dim_type) == sizeof(StaticDimension),
                  "StaticDimension must have the same number of bytes as the CPU dimension type.");

    constexpr StaticShapeAdapter() : m_dims{} {}
    constexpr StaticShapeAdapter(const TDims& dims) : m_dims{&dims} {}
    constexpr StaticShapeAdapter(const StaticShapeAdapter<const TDims>& other) : m_dims{other.m_dims} {}

    StaticShapeAdapter(const ov::PartialShape&);

    operator StaticShapeCon() const {
        return m_dims ? StaticShapeCon(*m_dims) : StaticShapeCon();
    }

    const TDims& operator*() const& noexcept {
        return *m_dims;
    }

    const value_type& operator[](size_t i) const {
        return reinterpret_cast<const value_type&>((*m_dims)[i]);
    }

    //-- Shape functions
    static constexpr bool is_static() {
        return true;
    }

    static constexpr bool is_dynamic() {
        return !is_static();
    }

    template <class T>
    constexpr typename std::enable_if<is_static_shape_adapter<T>(), bool>::type compatible(
        const T& other) const {
        // for static shape compatible == both shape equals
        return *this == other;
    }

    template <class T>
    constexpr typename std::enable_if<is_static_shape_adapter<T>(), bool>::type same_scheme(
        const T& other) const {
        // for static shape same_scheme == compatible;
        return compatible(other);
    }

    ov::Rank rank() const;
    bool merge_rank(const ov::Rank& r);
    ov::Shape to_shape() const;
    ov::Shape get_max_shape() const;
    ov::Shape get_min_shape() const;
    ov::Shape get_shape() const;
    ov::PartialShape to_partial_shape() const;

    //-- Container functions
    const_iterator cbegin() const noexcept {
        return m_dims ? m_dims->cbegin() : const_iterator{};
    }

    const_iterator begin() const noexcept {
        return cbegin();
    }

    const_iterator cend() const noexcept {
        return m_dims ? m_dims->cend() : const_iterator{};
    }

    const_iterator end() const noexcept {
        return cend();
    }

    size_t size() const noexcept {
        return m_dims ? m_dims->size() : 0;
    }

    bool empty() const {
        return m_dims ? m_dims->empty() : true;
    }

private:
    const TDims* m_dims = nullptr;
};

template <class T>
typename std::enable_if<is_static_shape_adapter<T>(), std::ostream&>::type operator<<(std::ostream& out, const T& shape) {
    out << '{';
    std::copy(shape.cbegin(), shape.cend() - 1, std::ostream_iterator<StaticDimension>(out, ","));
    if (!shape.empty()) {
        out << shape[shape.size() - 1];
    }
    out << '}';
    return out;
}

template <class T, class U>
constexpr typename std::enable_if<is_static_shape_adapter<T>() && is_static_shape_adapter<U>(), bool>::type operator==(
    const T& lhs,
    const U& rhs) {
    // The CPU dimension type and StaticDimension::value_type is same,
    // use CPU dimension type to compare in order to reduce number of conversions to StaticDimension.
    return (lhs.size() == rhs.size()) && (lhs.empty() || std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
}
}  // namespace intel_cpu
}  // namespace ov
