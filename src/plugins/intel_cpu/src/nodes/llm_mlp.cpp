// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_mlp.h"

#include <chrono>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#define ASSERT(cond) \
    if (!(cond)) { \
        OPENVINO_THROW(""); \
    }

template<int id = 0>
inline float get_delta_ms() {
    static auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    t0 = t1;
    return dt.count();
}

template <typename... Ts>
void easy_cout(const char* file, const char* func, int line, Ts... args) {
    std::string file_path(file);
    std::string file_name(file);

    auto last_sep = file_path.find_last_of('/');
    if (last_sep == std::string::npos)
        last_sep = file_path.find_last_of('\\');
    if (last_sep != std::string::npos)
        file_name = file_path.substr(last_sep + 1);

    std::string file_name_with_line = file_name + ":" + std::to_string(line);
    auto tag = file_name_with_line + " " + func + "()";

    std::stringstream ss;
    int dummy[sizeof...(Ts)] = {(ss << args, 0)...};
    std::cout << " \033[37;100m+" << std::fixed << std::setprecision(3) << get_delta_ms() << " ms\033[36;40m " << tag << " \033[0m " << ss.str() << "" << std::endl;
}

#define ECOUT(...) easy_cout(__FILE__, __func__, __LINE__, __VA_ARGS__)


namespace ov {
namespace intel_cpu {
namespace node {

namespace AMX_MLP {


using namespace dnnl::impl::cpu::x64;

struct TileConfig {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    void reset(int palette, int _startRow, const std::vector<std::pair<int, int>>& _rows_columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        unsigned long i;
        for (i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for (i = 0; i < _rows_columnsBytes.size(); i++) {
            rows[i] = _rows_columnsBytes[i].first;
            cols[i] = _rows_columnsBytes[i].second;
        }
        for (; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
    }
} __attribute__((__packed__));

class TileConfiger : public jit_generator {
public:
    TileConfiger() : jit_generator("TileConfiger") {
        create_kernel();
    }
    const char *name() const override { return "TileConfiger"; }
    const char *source_file() const override { return __FILE__; }

    void generate() override {
        Xbyak::Label release;
        test(abi_param1, abi_param1);
        jz(release);
        ldtilecfg(ptr[abi_param1]);
        ret();
        L(release);
        tilerelease();
        ret();
    }
};

// https://stackoverflow.com/questions/23690416/c-template-singleton-static-pointer-initialization-in-header-file
template <typename T>
class Singleton {
public:
    static T& get() {
        static T instance;
        return instance;
    }
};

class TileConfigScope {
public:
    std::vector<TileConfig> m_configs;

    TileConfigScope(TileConfig& cfg) : m_configs({cfg}) {
        (Singleton<TileConfiger>::get())(&cfg);
    };

    int cur_id = -1;
    TileConfigScope(const std::vector<TileConfig>& cfgs) : m_configs(cfgs) {
        cur_id = 0;
        (Singleton<TileConfiger>::get())(&m_configs[0]);
    };
    void load(int id) {
        if (cur_id == id)
            return;
        cur_id = id;
        (Singleton<TileConfiger>::get())(&m_configs[cur_id]);
    }
    ~TileConfigScope() {
        (Singleton<TileConfiger>::get())(nullptr);
    }
};

inline void transpose_m512i_16x16(__m512i& r0,
                                  __m512i& r1,
                                  __m512i& r2,
                                  __m512i& r3,
                                  __m512i& r4,
                                  __m512i& r5,
                                  __m512i& r6,
                                  __m512i& r7,
                                  __m512i& r8,
                                  __m512i& r9,
                                  __m512i& ra,
                                  __m512i& rb,
                                  __m512i& rc,
                                  __m512i& rd,
                                  __m512i& re,
                                  __m512i& rf) {
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

    t0 = _mm512_unpacklo_epi32(r0, r1);  //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
    t1 = _mm512_unpackhi_epi32(r0, r1);  //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    t2 = _mm512_unpacklo_epi32(r2, r3);  //  32  48  33  49 ...
    t3 = _mm512_unpackhi_epi32(r2, r3);  //  34  50  35  51 ...
    t4 = _mm512_unpacklo_epi32(r4, r5);  //  64  80  65  81 ...
    t5 = _mm512_unpackhi_epi32(r4, r5);  //  66  82  67  83 ...
    t6 = _mm512_unpacklo_epi32(r6, r7);  //  96 112  97 113 ...
    t7 = _mm512_unpackhi_epi32(r6, r7);  //  98 114  99 115 ...
    t8 = _mm512_unpacklo_epi32(r8, r9);  // 128 ...
    t9 = _mm512_unpackhi_epi32(r8, r9);  // 130 ...
    ta = _mm512_unpacklo_epi32(ra, rb);  // 160 ...
    tb = _mm512_unpackhi_epi32(ra, rb);  // 162 ...
    tc = _mm512_unpacklo_epi32(rc, rd);  // 196 ...
    td = _mm512_unpackhi_epi32(rc, rd);  // 198 ...
    te = _mm512_unpacklo_epi32(re, rf);  // 228 ...
    tf = _mm512_unpackhi_epi32(re, rf);  // 230 ...

    r0 = _mm512_unpacklo_epi64(t0, t2);  //   0  16  32  48 ...
    r1 = _mm512_unpackhi_epi64(t0, t2);  //   1  17  33  49 ...
    r2 = _mm512_unpacklo_epi64(t1, t3);  //   2  18  34  49 ...
    r3 = _mm512_unpackhi_epi64(t1, t3);  //   3  19  35  51 ...
    r4 = _mm512_unpacklo_epi64(t4, t6);  //  64  80  96 112 ...
    r5 = _mm512_unpackhi_epi64(t4, t6);  //  65  81  97 114 ...
    r6 = _mm512_unpacklo_epi64(t5, t7);  //  66  82  98 113 ...
    r7 = _mm512_unpackhi_epi64(t5, t7);  //  67  83  99 115 ...
    r8 = _mm512_unpacklo_epi64(t8, ta);  // 128 144 160 176 ...
    r9 = _mm512_unpackhi_epi64(t8, ta);  // 129 145 161 178 ...
    ra = _mm512_unpacklo_epi64(t9, tb);  // 130 146 162 177 ...
    rb = _mm512_unpackhi_epi64(t9, tb);  // 131 147 163 179 ...
    rc = _mm512_unpacklo_epi64(tc, te);  // 192 208 228 240 ...
    rd = _mm512_unpackhi_epi64(tc, te);  // 193 209 229 241 ...
    re = _mm512_unpacklo_epi64(td, tf);  // 194 210 230 242 ...
    rf = _mm512_unpackhi_epi64(td, tf);  // 195 211 231 243 ...

    t0 = _mm512_shuffle_i32x4(r0, r4, 0x88);  //   0  16  32  48   8  24  40  56  64  80  96  112 ...
    t1 = _mm512_shuffle_i32x4(r1, r5, 0x88);  //   1  17  33  49 ...
    t2 = _mm512_shuffle_i32x4(r2, r6, 0x88);  //   2  18  34  50 ...
    t3 = _mm512_shuffle_i32x4(r3, r7, 0x88);  //   3  19  35  51 ...
    t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd);  //   4  20  36  52 ...
    t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd);  //   5  21  37  53 ...
    t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd);  //   6  22  38  54 ...
    t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd);  //   7  23  39  55 ...
    t8 = _mm512_shuffle_i32x4(r8, rc, 0x88);  // 128 144 160 176 ...
    t9 = _mm512_shuffle_i32x4(r9, rd, 0x88);  // 129 145 161 177 ...
    ta = _mm512_shuffle_i32x4(ra, re, 0x88);  // 130 146 162 178 ...
    tb = _mm512_shuffle_i32x4(rb, rf, 0x88);  // 131 147 163 179 ...
    tc = _mm512_shuffle_i32x4(r8, rc, 0xdd);  // 132 148 164 180 ...
    td = _mm512_shuffle_i32x4(r9, rd, 0xdd);  // 133 149 165 181 ...
    te = _mm512_shuffle_i32x4(ra, re, 0xdd);  // 134 150 166 182 ...
    tf = _mm512_shuffle_i32x4(rb, rf, 0xdd);  // 135 151 167 183 ...

    r0 = _mm512_shuffle_i32x4(t0, t8, 0x88);  //   0  16  32  48  64  80  96 112 ... 240
    r1 = _mm512_shuffle_i32x4(t1, t9, 0x88);  //   1  17  33  49  66  81  97 113 ... 241
    r2 = _mm512_shuffle_i32x4(t2, ta, 0x88);  //   2  18  34  50  67  82  98 114 ... 242
    r3 = _mm512_shuffle_i32x4(t3, tb, 0x88);  //   3  19  35  51  68  83  99 115 ... 243
    r4 = _mm512_shuffle_i32x4(t4, tc, 0x88);  //   4 ...
    r5 = _mm512_shuffle_i32x4(t5, td, 0x88);  //   5 ...
    r6 = _mm512_shuffle_i32x4(t6, te, 0x88);  //   6 ...
    r7 = _mm512_shuffle_i32x4(t7, tf, 0x88);  //   7 ...
    r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd);  //   8 ...
    r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd);  //   9 ...
    ra = _mm512_shuffle_i32x4(t2, ta, 0xdd);  //  10 ...
    rb = _mm512_shuffle_i32x4(t3, tb, 0xdd);  //  11 ...
    rc = _mm512_shuffle_i32x4(t4, tc, 0xdd);  //  12 ...
    rd = _mm512_shuffle_i32x4(t5, td, 0xdd);  //  13 ...
    re = _mm512_shuffle_i32x4(t6, te, 0xdd);  //  14 ...
    rf = _mm512_shuffle_i32x4(t7, tf, 0xdd);  //  15  31  47  63  79  96 111 127 ... 255
}

#define _mm512_loadu_epi32(a) _mm512_load_epi32(a)
#define _mm512_storeu_epi32(a, b) _mm512_store_epi32(a, b)

inline void transpose_epi32_16x16(void* _dst, const void* src, int stride) {
    auto* dst = reinterpret_cast<uint32_t*>(_dst);
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    auto* pA = reinterpret_cast<const uint8_t*>(src);
    r0 = _mm512_loadu_epi32(pA);
    r1 = _mm512_loadu_epi32(pA + stride);
    r2 = _mm512_loadu_epi32(pA + 2 * stride);
    r3 = _mm512_loadu_epi32(pA + 3 * stride);
    r4 = _mm512_loadu_epi32(pA + 4 * stride);
    r5 = _mm512_loadu_epi32(pA + 5 * stride);
    r6 = _mm512_loadu_epi32(pA + 6 * stride);
    r7 = _mm512_loadu_epi32(pA + 7 * stride);
    r8 = _mm512_loadu_epi32(pA + 8 * stride);
    r9 = _mm512_loadu_epi32(pA + 9 * stride);
    ra = _mm512_loadu_epi32(pA + 10 * stride);
    rb = _mm512_loadu_epi32(pA + 11 * stride);
    rc = _mm512_loadu_epi32(pA + 12 * stride);
    rd = _mm512_loadu_epi32(pA + 13 * stride);
    re = _mm512_loadu_epi32(pA + 14 * stride);
    rf = _mm512_loadu_epi32(pA + 15 * stride);

    transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

    _mm512_storeu_epi32(dst, r0);
    _mm512_storeu_epi32(dst + 16, r1);
    _mm512_storeu_epi32(dst + 2 * 16, r2);
    _mm512_storeu_epi32(dst + 3 * 16, r3);
    _mm512_storeu_epi32(dst + 4 * 16, r4);
    _mm512_storeu_epi32(dst + 5 * 16, r5);
    _mm512_storeu_epi32(dst + 6 * 16, r6);
    _mm512_storeu_epi32(dst + 7 * 16, r7);
    _mm512_storeu_epi32(dst + 8 * 16, r8);
    _mm512_storeu_epi32(dst + 9 * 16, r9);
    _mm512_storeu_epi32(dst + 10 * 16, ra);
    _mm512_storeu_epi32(dst + 11 * 16, rb);
    _mm512_storeu_epi32(dst + 12 * 16, rc);
    _mm512_storeu_epi32(dst + 13 * 16, rd);
    _mm512_storeu_epi32(dst + 14 * 16, re);
    _mm512_storeu_epi32(dst + 15 * 16, rf);
}

// https://stackoverflow.com/questions/570669/checking-if-a-double-or-float-is-nan-in-c/57770634#57770634
static inline uint32_t load_ieee754_rep(float a) {
    uint32_t r;
    static_assert(sizeof r == sizeof a, "Unexpected sizes.");
    std::memcpy(&r, &a, sizeof a); // Generates movd instruction.
    return r;
}
constexpr uint32_t inf_float_shl1 = UINT32_C(0xff000000);
// The shift left removes the sign bit. The exponent moves into the topmost bits,
// so that plain unsigned comparison is enough.
static inline bool isnan2(float a)     { return load_ieee754_rep(a) << 1  > inf_float_shl1; }
static inline bool isinf2(float a)     { return load_ieee754_rep(a) << 1 == inf_float_shl1; }
static inline bool isfinite2(float a)  { return load_ieee754_rep(a) << 1  < inf_float_shl1; }

template<typename T>
struct tensor2D {
    int dims[2] = {0};
    std::shared_ptr<T> data;
    uint64_t capacity = 0;
    int stride = 0;
    bool force_compact = false;
    int padded_dim1 = 0;

    tensor2D() = default;

    operator bool() {
        return dims[0] * dims[1] > 0;
    }

    tensor2D(int d0, int d1, bool _force_compact = false) {
        capacity = 0;
        resize(d0, d1, _force_compact);
    }

    tensor2D(int d0, int d1, T * ext, int _stride) {
        capacity = 1;
        data = std::shared_ptr<T>(ext, [](void *) {});
        dims[0] = d0;
        dims[1] = d1;
        stride = _stride;
        padded_dim1 = stride / sizeof(T);
    }

    tensor2D<T> Tr() {
        tensor2D<T> ret(dims[1], dims[0]);
        for(int c0=0; c0 < dims[0]; ++c0) {
            for(int c1=0; c1 < dims[1]; ++c1) {
                ret(c1, c0) = (*this)(c0, c1);
            }
        }
        return ret;
    }
    tensor2D<T> clone() const {
        tensor2D<T> ret;
        ret.resize(dims[0], dims[1], force_compact);
        if (ret.stride == stride) {
            memcpy(ret.data.get(), data.get(), dims[0] * stride);
        }else{
            for(int i=0;i<dims[0];i++) {
                memcpy(&ret(i,0), &(*this)(i,0), ret.stride);
            }
        }
        return ret;
    }
    void resize(int d0, int d1, bool _force_compact = false) {
        force_compact = _force_compact;
        dims[0] = d0;
        dims[1] = d1;
        stride = d1 * sizeof(T);
        if ((stride % 64) && (!force_compact)) {
            ASSERT(false);
            //auto stride_fix = rndup(stride, 64);
            //logger() << "\tWarnning: stride " << stride << " is not aligned to cache line, will increase to " << stride_fix
            //          << " (" << stride_fix/64 << " cache lines)\n";
            //stride = stride_fix;
        }
        padded_dim1 = stride / sizeof(T);

        // resize method never shrink capacity, and extra T is added to put nan as test
        auto need_capacity = dims[0] * stride + sizeof(T);
        if (capacity < need_capacity) {
            capacity = need_capacity;
            // align begin address to cache line is vital, so tile load can
            // use all bandwidth (L1D/L2 only deliver data in unit of 64-byte aligned cache-line)

#ifdef ENABLE_NUMA
            if (USE_NUMA) {
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(numa_alloc_local(capacity)),
                            [need_capacity](void * p){ numa_free(p, need_capacity); });
            } else {
#else
            {
#endif
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(aligned_alloc(64, capacity)),
                            [](void * p) { ::free(p); });
            }
            if (reinterpret_cast<uintptr_t>(data.get()) % 64)
                std::cout << "WARNING: resize(), data is not cache-line aligned!" << std::endl;
        }
        // put a NaN at the end to test over-read
        // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
        #define INF 0xff80 
        #define NAN1 (INF + 1)
        if (sizeof(T) == 2) {
            *reinterpret_cast<uint16_t*>(data.get() + dims[0] * padded_dim1) = NAN1;
        }
        if (sizeof(T) == 4) {
            *reinterpret_cast<uint32_t*>(data.get() + dims[0] * padded_dim1) = (INF << 16) + 1;
        }
    }

    T & operator[](int i) {
        return data.get()[i];
    }

    const T & operator[](int i) const {
        return data.get()[i];
    }

    //https://stackoverflow.com/questions/1936399/c-array-operator-with-multiple-arguments
    T & operator()(int i0, int i1) {
        return (*this)[i0 * padded_dim1 + i1];
    }

    const T & operator()(int i0, int i1) const {
        return (*this)[i0 * padded_dim1 + i1];
    }

    void operator=(const T & v) {
        for(int k = 0; k<dims[0]*padded_dim1; k++)
            (*this)[k] = v;
    }

    tensor2D<T>& operator=(const tensor2D<T> & t2) {
        assert(dims[0]*dims[1] == t2.dims[0] * t2.dims[1]);
        for(int c0 = 0; c0 < dims[0]; c0++)
        for(int c1 = 0; c1 < dims[1]; c1++) {
            int k = c0*dims[1] + c1;
            auto c2 = k / t2.dims[1];
            auto c3 = k % t2.dims[1];
            (*this)(c0, c1) = t2(c2, c3);
        }
        return *this;
    }

    // move semantics
    tensor2D(tensor2D<T> && t2) {
        dims[0] = t2.dims[0];
        dims[1] = t2.dims[1];
        data = t2.data;
        capacity = t2.capacity;
        stride = t2.stride;
        padded_dim1 = t2.padded_dim1;
        force_compact = t2.force_compact;
        t2.capacity = 0;
        t2.data.reset();
    }

    tensor2D<T>&  operator=(tensor2D<T> && t2) {
        dims[0] = t2.dims[0];
        dims[1] = t2.dims[1];
        data = t2.data;
        capacity = t2.capacity;
        stride = t2.stride;
        padded_dim1 = t2.padded_dim1;
        force_compact = t2.force_compact;
        t2.capacity = 0;
        t2.data.reset();
        return *this;
    }

    bool operator==(const tensor2D<T> & rhs) const {
        return allclose(rhs);
    }

    bool allclose(const tensor2D<T> & rhs, float rtol=1e-05, float atol=1e-08) const {
        // 
        // absolute(a - b) <= (atol + rtol * absolute(b))
        //   here *this is a, rhs is b
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        float max_rtol = 0;
        float max_atol = 0;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            // with -ffast-math,  std::isnan, std::isinf,  x != x  always return false
            // so we need special logic to test nan here
            if (std::is_same<T, ov::bfloat16>::value ||
                std::is_same<T, float>::value) {
                float f0 = (*this)(i0,i1);
                float f1 = rhs(i0,i1);
                if (isnan2(f1) || isnan2(f0)) {
                    std::cout << " nan is found @(" << i0 << "," << i1 << ") : f0=" << f0 << ",  f1=" << f1 << std::endl;
                    return false;
                }
            }

            float a = (*this)(i0,i1);
            float b = rhs(i0,i1);

            auto absolute_diff = std::abs(a - b);
            auto absolute_b = std::abs(b);
            
            auto cur_rtol = (absolute_diff - atol)/absolute_b;
            auto cur_atol = absolute_diff - rtol * absolute_b;

            max_rtol = std::max(max_rtol, cur_rtol);
            max_atol = std::max(max_atol, cur_atol);
            if (absolute_diff <= (atol + rtol * absolute_b))
                continue;
            std::cout << " operator== failed at (" << i0 << ", " << i1 << ")  value "
                        << (*this)(i0,i1) << "!=" << rhs(i0,i1) << std::endl;
            std::cout << "to pass: adjust rtol to " << cur_rtol << " OR atol to " << cur_atol << std::endl;
            return false;
        }
        std::cout << "[all pass] with max rtol,atol=" << max_rtol << "," << max_atol << std::endl;
        return true;
    }
    bool compare(const tensor2D<T> & rhs, float tolerance) {
        float max_abs_diff = 0;
        float max_rel_diff = 0;
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            auto diff = std::fabs((*this)(i0,i1) - rhs(i0,i1));
            auto rel_diff = diff/std::fabs((*this)(i0,i1));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (std::fabs((*this)(i0,i1) > 0) && diff > 0)
                max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
        std::cout << "max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff;
        return tolerance > max_abs_diff;
    }
    friend std::ostream& operator<<(std::ostream& out, const tensor2D<T>& obj) {
        int i0;
        auto showline = [&](int i) {
            out << "[" << i << "," << 0 << "]: ";
            int i1;
            for(i1=0; i1<obj.dims[1] && i1 < 8; i1++) {
                out << +obj(i0,i1) << ",";
            }
            if (i1 < obj.dims[1]) out << "...";
            out << std::endl;
        };
        for(i0=0; i0 < obj.dims[0] && i0 < 32; i0++) {
            showline(i0);
        }
        if (i0 < obj.dims[0]) {
            out << "... ... ... ..." << std::endl;
            showline(obj.dims[0] - 1);
        }
        return out;
    }
};

class MKernel : public jit_generator {
public:
    TileConfig m_tile_cfg;

    int m_prefetch_Blines;

    // both A & B data will be prefetched from memory for next kernel invokation
    // and the prefetches are evenly distributed into each kernel.
    //
    // we first tackle the prefetching of B, because each time
    // we will call it with a new B, and run() will have to prefetch new B
    // for next round, so next B is also of size (KxN) elements
    //    distributes into (BM/32)*(BN/32) kernels:
    //    each kernel has (BK/32) iterations, thus each kernel iteration
    //    need to prefetch (BKxBN)/(BMxBNxBK/32768) = 32768/BM bfloat16-elements
    //    which is 1024/BM cache lines, this has to be determined at
    //    code-generation time. with BM=256, this is only 4.
    //
    // prefetch A can be done in unit of 32xBK elements, which must be evenly distributed
    // into (BN/32)*(BK/32) kernel iterations, each iteration prefetch/copy 32xBK/(BN*BK/1024) = 32768/BN
    // bfloat16-elements or 1024/BN cache lines. with BM=256, this is only 4 too.
    //
    // prefetch or copy?
    //   prefetch strided sub-matrix of A is tricky, consider each 32x32 AMX jit kernel has [BK/32] iterations
    //   and it's called (BN/32) times, each kernel must prefetch 32*BK/(BN/32) = (1024/BN)*BK elements
    //   since each kernel has [BK/32] loop iterations, each iteration fetch (1024/BN)*BK/(BK/32) = 1024*32/BN
    //   bytes.
    //
    //   when 1024 is not divisible by BN, it's fine, just prefetch more
    //
    // copy data from A to a ping-pong buffer has advantage:
    //    - read can be done in continous way most suitable for HW prefetcher
    //    - write to ping-pong buffer is within L2 cache, which should be fast
    //    - data transfer rate is small comparing to L2-bandwidth, shouldn't be a big issue for interleaved write to L2.
    //    - read from ping-pong buffer is much faster and free of odd-multiple-cache-line restriction.
    // so we prefer distribute the repacking of A sub-matrix into ping-pong buffer into kernel.
    // for BN=256, each kernel read 4*BK elements into ping-pong, each iteration read
    // 4*BK*sizeof(bfloat16)/(BK/32)=256bytes = 4-512bits zmm registers
    //
    //
    MKernel(int M_hint = 256) : jit_generator("MKernel") {
        setup(M_hint);
    }

    //  M_hint is only a hint for prefetching, set to 0 to avoid prefetch
    void setup(int M_hint = 0) {
        if (M_hint == 0) {
            m_prefetch_Blines = 0;
        } else {
            m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / M_hint;
        }

        create_kernel();
        tile_config_M(m_tile_cfg, M_hint);
    }

    // M can change w/o code-regeneration
    // with the help of :
    //  - m_BM_hint controls dynamic behaviour of the kernel
    //  - tile config controls behaviour of tileload & TMUL
    void tile_config_M(TileConfig& tile_cfg, int M) {
        auto rows0 = 16;
        auto rows1 = 16;
        if (M < 32) {
            // kernel is for processing Mtails
            if (M > 16) {
                rows0 = 16;
                rows1 = M - 16;
            } else {
                //  both A0 & A1 load from same memory, to avoid code-regeneration
                rows0 = rows1 = M;
            }
        }
        tile_cfg.reset(1,
                       0,
                       {
                           {rows0, 64},  // C00:0
                           {rows0, 64},  // C01:1
                           {rows1, 64},  // C10:2
                           {rows1, 64},  // C11:3
                           {rows0, 64},  // A0:4
                           {rows1, 64},  // A1:5
                           {16, 64},     // B0:6
                           {16, 64},     // B1:7
                       });
    }

    // row data is in layout [N, K], maybe smaller than [32, 16]
    template <typename T>
    void repackB(ov::bfloat16* dst, T* src, int N_stride, int N, int K) {
        if (N == 16 && K == 32 && std::is_same<T, ov::bfloat16>::value) {
            // SIMD optimized version
            // std::cout << "." << std::flush;
            transpose_epi32_16x16(dst, src, N_stride * sizeof(T));
            return;
        }

        assert(K <= 32);
        assert(N <= 16);
        int k = 0;
        ov::bfloat16 bf16zero(0.0f);
        for (; k < 32; k += 2) {
            int n = 0;
            bool is_k0_valid = (k) < K;
            bool is_k1_valid = (k + 1) < K;
            auto* psrc = src + k;
            for (; n < 16 && n < N; n++, psrc += N_stride) {
                *dst++ = is_k0_valid ? ov::bfloat16(psrc[0]) : bf16zero;
                *dst++ = is_k1_valid ? ov::bfloat16(psrc[1]) : bf16zero;
            }
            for (; n < 16; n++) {
                *dst++ = 0;
                *dst++ = 0;
            }
        }
    }

    // weight is supposed to be of shape[N, K], stride in unit of bytes
    // N should be m_BN
    // K should be m_BK
    template <typename T>
    tensor2D<ov::bfloat16> prepareB(T* p_weight, int stride, int N, int K) {
        tensor2D<ov::bfloat16> ret;
        ASSERT((N % 32) == 0);
        ASSERT((K % 32) == 0);
        // weight matrix is in unit of [N/32, Kx32]
        ret.resize(N / 32, K * 32, true);

        auto N_stride = stride / sizeof(T);
        for (int n = 0, blkn = 0; n < N; n += 32, blkn++) {
            for (int k = 0, blkk = 0; k < K; k += 32, blkk++) {
                // two adjacent 32x16 (512) block of weight: dst0 & dst1
                auto* dst0 = &ret(blkn, blkk * 1024);
                auto* dst1 = dst0 + 16 * 32;
                auto valid_k = (K - k) < 32 ? (K - k) : 32;

                auto* src0 = p_weight + n * N_stride + k;
                auto valid_n0 = (N - n) < 16 ? (N - n) : 16;
                repackB<T>(dst0, src0, N_stride, valid_n0, valid_k);

                auto* src1 = p_weight + (n + 16) * N_stride + k;
                auto valid_n1 = (N - (n + 16)) < 16 ? (N - (n + 16)) : 16;
                repackB<T>(dst1, src1, N_stride, valid_n1, valid_k);
            }
        }
        return ret;
    }

    // to save push/pop: do not use `abi_save_gpr_regs`
    uint8_t* prefetch_next_A_addr;

    struct call_args {
        const uint8_t* pA;  // bfloat16
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16
        const uint8_t* pC;  // float32
        int64_t strideC;    // in bytes
        const uint8_t* prefetch;
        int64_t k_tiles;  // K / 32
        int64_t do_accumulation;
        int64_t M;
    };
    const char *name() const override { return "MKernel"; }
    const char *source_file() const override { return __FILE__; }

    void generate() override {
        Xbyak::Reg64 reg_A_addr = abi_param2;
        Xbyak::Reg64 reg_A_stride = abi_param3;
        Xbyak::Reg64 reg_B_addr = abi_param4;
        Xbyak::Reg64 reg_C_addr = abi_param5;
        Xbyak::Reg64 reg_C_stride = abi_param6;

        Xbyak::Reg64 reg_ktiles = rax;
        Xbyak::Reg64 reg_B_stride = r10;
        Xbyak::Reg64 reg_A1_addr = r11;
        Xbyak::Reg64 reg_prefetch = r12;

        Xbyak::Tmm tmmC00 = tmm0;
        Xbyak::Tmm tmmC01 = tmm1;
        Xbyak::Tmm tmmC10 = tmm2;
        Xbyak::Tmm tmmC11 = tmm3;
        Xbyak::Tmm tmmA0 = tmm4;
        Xbyak::Tmm tmmA1 = tmm5;
        Xbyak::Tmm tmmB0 = tmm6;
        Xbyak::Tmm tmmB1 = tmm7;

        auto num_PFB = m_prefetch_Blines;
        int cur_PFB = 0;
        /*
                       B: 1x2 tiles
        A : 2x1 tiles  C: 2x2 tiles
        */
        Xbyak::Label loop_over_ktiles;
        Xbyak::Label skip_load;

        push(reg_prefetch);
        {
            auto reg_tmp = reg_B_stride;
            tilezero(tmmC00);
            tilezero(tmmC01);
            tilezero(tmmC10);
            tilezero(tmmC11);

            mov(reg_A_addr, ptr[abi_param1 + offsetof(call_args, pA)]);
            mov(reg_A_stride, ptr[abi_param1 + offsetof(call_args, strideA)]);
            mov(reg_B_addr, ptr[abi_param1 + offsetof(call_args, pB)]);
            mov(reg_C_addr, ptr[abi_param1 + offsetof(call_args, pC)]);
            mov(reg_C_stride, ptr[abi_param1 + offsetof(call_args, strideC)]);
            mov(reg_prefetch, ptr[abi_param1 + offsetof(call_args, prefetch)]);
            mov(reg_ktiles, ptr[abi_param1 + offsetof(call_args, k_tiles)]);

            lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
            lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);

            // reg_A1_addr = reg_A_addr if M <= 16 (to avoid tileloadd segmentfault)
            mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, M)]);
            cmp(reg_tmp, 16);
            cmovle(reg_A1_addr, reg_A_addr);

            mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, do_accumulation)]);
            and_(reg_tmp, 1);
            jz(skip_load);
            {
                auto reg_C1_addr = reg_tmp;
                tileloadd(tmmC00, ptr[reg_C_addr + reg_C_stride]);
                tileloadd(tmmC01, ptr[reg_C_addr + reg_C_stride + 64]);
                lea(reg_C1_addr, ptr[reg_C_addr + reg_C_stride * 8]);
                lea(reg_C1_addr, ptr[reg_C1_addr + reg_C_stride * 8]);
                tileloadd(tmmC10, ptr[reg_C1_addr + reg_C_stride]);
                tileloadd(tmmC11, ptr[reg_C1_addr + reg_C_stride + 64]);
            }
            L(skip_load);
        }

        mov(reg_B_stride, 64);

        auto const_A_steps = 64;

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tdpbf16ps(tmmC00, tmmA0, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

        // prefetch [K_tiles X 256] bytes

        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        // prefetch next sub-block B matrix
        if (cur_PFB < num_PFB) {
            for (int pi = cur_PFB; pi < num_PFB; pi++) {
                prefetcht2(ptr[reg_prefetch + pi * 64]);
            }
        }

        lea(reg_prefetch, ptr[reg_prefetch + 64 * num_PFB]);

        //}
        lea(reg_A_addr, ptr[reg_A_addr + const_A_steps]);
        lea(reg_A1_addr, ptr[reg_A1_addr + const_A_steps]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);
        dec(reg_ktiles);
        jnz(loop_over_ktiles, T_NEAR);

#if 0
        tilestored(ptr[reg_C_addr + reg_B_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024], tmmC01);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024 * 2], tmmC10);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024 * 3], tmmC11);
#else
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC01);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC10);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC11);
#endif
        pop(reg_prefetch);
        ret();
    }

    // run L2 cache blocking kernel with size:
    //    [BM, BK]*[BK, BN] => [BM, BN]
    //
    // prefetch of A can be done inside of this level of kernel
    // since it's done in unit of 32-rows
    // but prefetch of next B must be specified by caller.
    //
    void run(int M,  // actual M
             uint8_t* pA,
             int strideA,                         // A [M, K]
             tensor2D<ov::bfloat16>& repacked_B,  // B [N/32, K*32]
             uint8_t* pC,
             int strideC,          // C [M, N]
             uint8_t* prefetch_B,  // prefetch B
             bool do_accumulation) {
        call_args args;
        // number of blocks in N dimension (in unit of 32 columns)
        auto num_blkN = repacked_B.dims[0];
        auto K = repacked_B.dims[1] / 32;
        auto* pB = reinterpret_cast<uint8_t*>(&repacked_B[0]);
        auto strideB = repacked_B.stride;

        args.do_accumulation = do_accumulation;
        args.k_tiles = K / 32;
        args.strideA = strideA;
        args.strideC = strideC;
        args.prefetch = prefetch_B;
        assert((K % 32) == 0);

        auto prefetch_step = m_prefetch_Blines * 64 * args.k_tiles;

        // if (BM != m_BM_hint) it only effect prefetch of B which is not vital to function
        for (int m = 0; m < M; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            args.pB = pB;
            // prefetch_next_A_addr = pA + 32 * strideA;
            // if (m + 32 >= BM)
            //     prefetch_next_A_addr = pA;
            args.M = std::min(M - m, 32);
            args.pA = pA;
            for (int ni = 0; ni < num_blkN; ni++, args.pB += strideB, args.prefetch += prefetch_step) {
                args.pC = pC + ni * 32 * sizeof(float);
                (*this)(&args);
                //(*this)(pA, strideA, pB1, pC + ni * 32 * sizeof(float), strideC, prefetch_B);
                // prefetch_next_A_addr += 4 * strideA;
            }
        }
    }
};


class jit_base : public jit_generator {
public:
    const char * m_name;
    jit_base(const char * name) : jit_generator(name), m_name(name) {}
    const char *name() const override { return m_name; }
    const char *source_file() const override { return __FILE__; }

    #define Vx16(a) {a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a}
    struct _const_table {
        uint32_t exp_log2ef[16] = Vx16(0x3fb8aa3b);
        uint32_t exp_ln_flt_max_f[16] = Vx16(0x42b17218);
        uint32_t exp_ln_flt_min_f[16] = Vx16(0xc2aeac50);
        uint32_t ln2f[16] = Vx16(0x3f317218); // ln(2.0f)
        uint32_t exponent_bias[16] = Vx16(0x0000007f);
        float two[16] = Vx16(2.0f);
        float half[16] = Vx16(0.5f);
        float one[16] = Vx16(1.0f);   // p0=1.0f
        uint32_t exp_pol0[16] = Vx16(0x3f7ffffb);// p1 = 0.999999701f
        uint32_t exp_pol1[16] = Vx16(0x3efffee3);// p2 = 0.499991506f
        uint32_t exp_pol2[16] = Vx16(0x3e2aad40);// p3 = 0.166676521f
        uint32_t exp_pol3[16] = Vx16(0x3d2b9d0d);// p4 = 0.0418978221f
        uint32_t exp_pol4[16] = Vx16(0x3c07cfce);// p5 = 0.00828929059f
        uint32_t sign_bit[16] = Vx16(0x80000000);
    } const_table;
    static constexpr int n_mantissa_bits = 23;
    #define PTR_CONST(name) ptr[p_table + offsetof(_const_table, name)]

    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    void inject_exp(const Xbyak::Zmm &vmm_src, const Xbyak::Zmm &vmm_aux1, const Xbyak::Zmm &vmm_aux2, Xbyak::Reg64 p_table, Xbyak::Opmask k_mask) {
        // exp(x) =
        // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
        // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

        // get mask of values lower than log(FLT_MIN) to zero them in the output
        //compute_cmp_mask(vmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);
        vcmpps(k_mask, vmm_src, PTR_CONST(exp_ln_flt_min_f), _cmp_lt_os);

        vminps(vmm_src, vmm_src, PTR_CONST(exp_ln_flt_max_f));
        vmaxps(vmm_src, vmm_src, PTR_CONST(exp_ln_flt_min_f));
        vmovups(vmm_aux1, vmm_src);

        // calculate exp(x)
        // fx = x * log2ef + 0.5
        vmulps(vmm_src, vmm_src, PTR_CONST(exp_log2ef));
        vaddps(vmm_src, vmm_src, PTR_CONST(half));

        // tmp = floorf(fx)
        //vroundps(vmm_aux2, vmm_src, _op_floor);
        vrndscaleps(vmm_aux2, vmm_src, _op_floor & 0x3);

        // keep vmm_src = fx for further computations
        vmovups(vmm_src, vmm_aux2);

        // x = x - fx * ln2
        vfnmadd231ps(vmm_aux1, vmm_aux2, PTR_CONST(ln2f));

        // We do not count 2^n here, because n can reach 128 and 2^128 is not
        // representable by fp32, so to get around this problem, instead of computing
        // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
        // and 2 are numbers representable in fp32.

        // compute 2^(n-1)
        vsubps(vmm_src, vmm_src, PTR_CONST(one));
        vcvtps2dq(vmm_aux2, vmm_src);
        vpaddd(vmm_aux2, vmm_aux2, PTR_CONST(exponent_bias));

        vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits);
        // use vmm_src as tmp vmm_zero when applying mask
        vxorps(vmm_src, vmm_src, vmm_src);
        // set zeroes at those points which were < log(FLT_MIN)
        //blend_with_mask(vmm_aux2, vmm_src);
        vblendmps(vmm_aux2 | k_mask, vmm_aux2, vmm_src);
        //vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_mask);

        // compute polynomial
        vmovups(vmm_src, PTR_CONST(exp_pol4));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(exp_pol3));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(exp_pol2));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(exp_pol1));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(exp_pol0));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(one));
        // y = y * 2^n
        vmulps(vmm_src, vmm_src, vmm_aux2);
        vmulps(vmm_src, vmm_src, PTR_CONST(two));
    }

    void inject_sigmoid(const Xbyak::Zmm &vmm_sigmoid, const Xbyak::Zmm &vmm_src, const Xbyak::Zmm &vmm_aux1, const Xbyak::Zmm &vmm_aux2, Xbyak::Reg64 p_table, Xbyak::Opmask k_mask) {
        // sigmoid(x) = 1/(1+log(-x))
        vpxord(vmm_sigmoid, vmm_src, PTR_CONST(sign_bit)); // -x
        inject_exp(vmm_sigmoid, vmm_aux1, vmm_aux2, p_table, k_mask);  // log(-x)
        vaddps(vmm_sigmoid, vmm_sigmoid, PTR_CONST(one));  // 1.0f + log(-x)
        vrcp14ps(vmm_sigmoid, vmm_sigmoid);
    }

    void inject_silu(const Xbyak::Zmm &vmm_silu, const Xbyak::Zmm &vmm_src, const Xbyak::Zmm &vmm_aux1, const Xbyak::Zmm &vmm_aux2, Xbyak::Reg64 p_table, Xbyak::Opmask k_mask) {
        // silu(x) = x * sigmoid(x)
        inject_sigmoid(vmm_silu, vmm_src, vmm_aux1, vmm_aux2, p_table, k_mask);
        vmulps(vmm_silu, vmm_src, vmm_silu);
    }

    void inject_init(Xbyak::Reg64 p_table) {
        mov(p_table, reinterpret_cast<uintptr_t>(&const_table));        
    }
};

class GateUpCombine : public jit_base {
public:
    GateUpCombine() : jit_base("GateUpCombine") {
        create_kernel();
    }

    void generate() override {
        Xbyak::Label loop_begin;
        
        Xbyak::Reg64 src = abi_param1;
        Xbyak::Reg64 dst = abi_param2;
        Xbyak::Reg64 prefetch_dst = abi_param3;
        Xbyak::Reg64 BN = abi_param4;

        Xbyak::Reg64 loop_i = rax;
        Xbyak::Reg64 p_table = r10;
        const auto zmm_gate = zmm0;
        const auto zmm_up = zmm1;
        const auto zmm_aux1 = zmm2;
        const auto zmm_aux2 = zmm3;
        const auto zmm_silu = zmm4;
        const auto ymm_dst = ymm4;

        xor_(loop_i, loop_i);
        inject_init(p_table);

        shr(BN, 1); // BN = BN/2;
        align(64, false);
        L(loop_begin);
        {
            vmovups(zmm_gate, ptr[src + loop_i*8]);
            vmovups(zmm_up, ptr[src + loop_i*8 + 16*4]);
            inject_silu(zmm_silu, zmm_gate, zmm_aux1, zmm_aux2, p_table, k1);
            vmulps(zmm_up, zmm_up, zmm_silu);
            vcvtneps2bf16(ymm_dst, zmm_up);
            prefetchwt1(ptr[prefetch_dst + loop_i*2]);
            vmovdqu(ptr[dst + loop_i*2], ymm_dst);
        }
        add(loop_i, 16);
        cmp(loop_i, BN);
        jl(loop_begin, T_NEAR);

        ret();
    }

#if 0
    // m_do_reduce2 : false
    void operator()(float* src, ov::bfloat16* dst, ov::bfloat16* prefetch_dst, int BN) {
        for (int n = 0, i = 0; n < BN; n += 32, i += 16) {
            auto v_gate = _mm512_loadu_ps(src + n);
            auto v_up = _mm512_loadu_ps(src + n + 16);
            v_gate = silu_ps_avx512(v_gate);
            v_up = _mm512_mul_ps(v_gate, v_up);
            auto v_bh = _mm512_cvtneps_pbh(v_up);
            // Greate Optimization:
            //  following prefetchnta prevents L2 HW prefetcher prefetch interleaved
            //  channels belonging to other cores which will causes too much cross-core cache coherent cost.
            _mm_prefetch(prefetch_dst + i, _MM_HINT_ET1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), reinterpret_cast<__m256i&>(v_bh));
        }
    }
#endif
};

class ReduceAdd2bh : public jit_base {
public:
    bool m_do_reduce2;
    ReduceAdd2bh(bool do_reduce2) : jit_base("ReduceAdd2bh"), m_do_reduce2(do_reduce2) {
        create_kernel();
    }

#if 0
    // m_do_reduce2 : false
    void operator()(float* src0, ov::bfloat16* dst, ov::bfloat16* prefetch_dst, int BN) {
        for (int n = 0; n < work.BN; n += 32) {
            auto d0 = _mm512_loadu_ps(src + n);
            auto d1 = _mm512_loadu_ps(src + n + 16);
            auto v_bh = _mm512_cvtne2ps_pbh(d1, d0);
            _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
            _mm512_storeu_ps(dst + n, reinterpret_cast<__m512&>(v_bh));
        }
    }
    // m_do_reduce2 : true
    void operator()(float* src0, float* src1, ov::bfloat16* dst, ov::bfloat16* prefetch_dst, int BN) {
        for (int n = 0; n < BN; n += 32) {
            auto d0 = _mm512_loadu_ps(src0 + n);
            auto d0b = _mm512_loadu_ps(src1 + n);
            auto d1 = _mm512_loadu_ps(src0 + n + 16);
            auto d1b = _mm512_loadu_ps(src1 + n + 16);
            d0 = _mm512_add_ps(d0, d0b);
            d1 = _mm512_add_ps(d1, d1b);
            auto v_bh = _mm512_cvtne2ps_pbh(d1, d0);
            _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
            _mm512_storeu_ps(dst + n, reinterpret_cast<__m512&>(v_bh));
        }        
    }
#endif

    void generate() override {
        if (m_do_reduce2) {
            Xbyak::Reg64 src0 = abi_param1;
            Xbyak::Reg64 src1 = abi_param2;
            Xbyak::Reg64 dst = abi_param3;
            Xbyak::Reg64 prefetch_dst = abi_param4;
            Xbyak::Reg64 BN = abi_param5;
            Xbyak::Reg64 loop_i = rax;

            Xbyak::Label loop_begin;

            xor_(loop_i, loop_i);

            align(64, false);
            L(loop_begin);
            {
                vmovups(zmm0, ptr[src0 + loop_i*4]);
                vmovups(zmm1, ptr[src1 + loop_i*4]);
                vmovups(zmm2, ptr[src0 + loop_i*4 + 16*4]);
                vmovups(zmm3, ptr[src1 + loop_i*4 + 16*4]);
                vaddps(zmm0, zmm0, zmm1);
                vaddps(zmm2, zmm2, zmm3);
                vcvtne2ps2bf16(zmm4, zmm2, zmm0);
                prefetchwt1(ptr[prefetch_dst + loop_i*2]);
                vmovups(ptr[dst + loop_i*2], zmm4);
            }
            add(loop_i, 32);
            cmp(loop_i, BN);
            jl(loop_begin, T_NEAR);

            ret();
        } else {
            Xbyak::Reg64 src0 = abi_param1;
            Xbyak::Reg64 dst = abi_param2;
            Xbyak::Reg64 prefetch_dst = abi_param3;
            Xbyak::Reg64 BN = abi_param4;
            Xbyak::Reg64 loop_i = rax;

            Xbyak::Label loop_begin;

            xor_(loop_i, loop_i);

            align(64, false);
            L(loop_begin);
            {
                vmovups(zmm0, ptr[src0 + loop_i*4]);
                vmovups(zmm2, ptr[src0 + loop_i*4 + 16*4]);
                vcvtne2ps2bf16(zmm4, zmm2, zmm0);
                prefetchwt1(ptr[prefetch_dst + loop_i*2]);
                vmovups(ptr[dst + loop_i*2], zmm4);
            }
            add(loop_i, 32);
            cmp(loop_i, BN);
            jl(loop_begin, T_NEAR);

            ret();
        }
    }

};

class Linear {
public:
    struct Work {
        MKernel* p_jit_amx0 = nullptr;
        GateUpCombine* p_jit_gateup = nullptr;
        ReduceAdd2bh* p_jit_reduce2bh_1 = nullptr;
        ReduceAdd2bh* p_jit_reduce2bh_2 = nullptr;

        std::vector<tensor2D<ov::bfloat16>> weights;
        tensor2D<float> C;
        std::shared_ptr<std::atomic_int> sync_flag;
        int n0 = 0;
        int n1 = 0;
        int k0 = 0;
        int k1 = 0;
        int BN = 0;
        int blk_K_size = 0;
        operator bool() {
            return BN > 0;
        }

        // input : weight [N, K], setup repacks range of N [n_start, n_end)
        template <typename T>
        void setup(T* p_weight, int stride) {
            auto num_blk_K = (k1 - k0) / blk_K_size;
            auto* pw = p_weight + n0 * stride / sizeof(T) + k0;
            weights.resize(num_blk_K);
            for (int k = 0; k < num_blk_K; k++) {
                weights[k] = p_jit_amx0->prepareB(pw + k * blk_K_size, stride, BN, blk_K_size);
            }
        }

        void run(int M, uint8_t* pA, int strideA) {
            int num_blk_K = weights.size();

            TileConfig tile_cfg0;
            TileConfig tile_cfg1;

            auto Mtails = M % 32;
            auto Mbody = M - Mtails;
            p_jit_amx0->tile_config_M(tile_cfg0, 32);
            if (Mtails) {
                p_jit_amx0->tile_config_M(tile_cfg1, Mtails);
            }
            TileConfigScope tcfg({tile_cfg0, tile_cfg1});

            C.resize(Mbody + (Mtails ? 32 : 0), BN);

            pA += k0 * sizeof(ov::bfloat16);
            auto pC = reinterpret_cast<uint8_t*>(&C[0]);
            bool do_accumulation = false;
            for (int ki = 0; ki < num_blk_K; ki++) {
                tensor2D<ov::bfloat16>& blockB = weights[ki];
                tensor2D<ov::bfloat16>& blockB1 = weights[(ki + 1) < num_blk_K ? (ki + 1) : ki];

                if (Mbody) {
                    tcfg.load(0);
                    p_jit_amx0->run(Mbody,
                                    pA + ki * blk_K_size * sizeof(ov::bfloat16),
                                    strideA,
                                    blockB,
                                    pC,
                                    C.stride,
                                    reinterpret_cast<uint8_t*>(&blockB1[0]),
                                    do_accumulation);
                }

                if (Mtails) {
                    tcfg.load(1);
                    p_jit_amx0->run(Mtails,
                                    pA + ki * blk_K_size * sizeof(ov::bfloat16) + Mbody * strideA,
                                    strideA,
                                    blockB,
                                    pC + Mbody * C.stride,
                                    C.stride,
                                    reinterpret_cast<uint8_t*>(&blockB1[0]),
                                    do_accumulation);
                }
                do_accumulation = true;
            }
        }
    };
    std::vector<Work> works;

    int used_nthr = 0;
    bool do_splitK = false;

    Linear() {}

    // weight [N, K]
    // Gate & Up are interleaved in N dimension: 16-gate / 16-up
    // and post-ops will compute  silu(gate)*up in unit of 16 elements
    // and store out as bfloat16.
    template <typename T, int BM = 256>
    void setup(T* p_weight, int stride, int N, int K, bool _do_splitK = false) {
        static MKernel jit_amx0(BM);
        static GateUpCombine jit_gateup;
        static ReduceAdd2bh jit_reduce2bh_1(false);
        static ReduceAdd2bh jit_reduce2bh_2(true);
        const int blk_K_size = 256;
        // prepare weights, split N among threads
        // in unit of 32
        ASSERT((N % 32) == 0);
        ASSERT((K % blk_K_size) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_N = N / 32;
        auto num_blk_K = K / blk_K_size;
        works.resize(nthr);

        do_splitK = _do_splitK;
        auto K_splits = do_splitK ? 2 : 1;
        // every thread should do same amount of work, and some cores can be idle
        auto valid_nthr = nthr / K_splits;
        auto blkN_per_thread = (num_blk_N + valid_nthr - 1) / valid_nthr;
        auto start_blkN = 0;
        used_nthr = 0;
        auto blkK_per_thread = (num_blk_K + K_splits - 1) / K_splits;
        for (int ithr = 0; ithr < nthr; ithr += K_splits) {
            auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);

            if (blkN) {
                auto shared_atomic = std::make_shared<std::atomic_int>(0);
                auto start_blkK = 0;
                for (int ik = 0; ik < K_splits; ik++) {
                    auto blk_K = std::min(num_blk_K - start_blkK, blkK_per_thread);

                    auto& work = works[ithr + ik];

                    work.p_jit_amx0 = &jit_amx0;
                    work.p_jit_gateup = &jit_gateup;
                    work.p_jit_reduce2bh_1 = &jit_reduce2bh_1;
                    work.p_jit_reduce2bh_2 = &jit_reduce2bh_2;
                    work.sync_flag = shared_atomic;
                    work.blk_K_size = blk_K_size;

                    work.n0 = (start_blkN)*32;
                    work.n1 = (start_blkN + blkN) * 32;
                    work.BN = blkN * 32;
                    work.k0 = start_blkK * blk_K_size;
                    work.k1 = (start_blkK + blk_K) * blk_K_size;
                    start_blkK += blk_K;
                    used_nthr++;
                }
            }

            start_blkN += blkN;
        }

        ECOUT("Linear N,K=", N, ",", K, " used_nthr=", used_nthr, "  do_splitK=", do_splitK);

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(p_weight, stride);
            }
        });
    }

    /*
    // A bfloat16 [256,  num_blk_K * 256]
    void run(uint8_t* pA, int strideA, int M) {
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            auto& work = works[ithr];
            if (work) {
                work.run(M, pA, strideA);
            }
        }
    }

        void run(uint8_t* pA, int strideA, int M, float* dstC, int strideC) {
    #pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                auto& work = works[ithr];
                if (work) {
                    work.run(M, pA, strideA);
                    auto* src = &work.C[0];
                    auto* dst = dstC + work.n0;
                    auto strideS = work.C.stride / sizeof(*src);
                    auto strideD = strideC / sizeof(*dst);
                    for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                        for (int n = 0; n < work.BN; n += 32) {
                            auto d0 = _mm512_loadu_ps(src + n);
                            auto d1 = _mm512_loadu_ps(src + n + 16);
                            _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
                            _mm_prefetch(prefetch_dst + n + 16, _MM_HINT_ET1);
                            _mm512_storeu_ps(dst + n, d0);
                            _mm512_storeu_ps(dst + n + 16, d1);
                        }
                    }
                }
            }
        }
    */

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.run(M, pA, strideA);

                if (do_splitK) {
                    auto sync_id = work.sync_flag->fetch_add(1);
                    // (0,1) (2,3)
                    if (sync_id & 1) {
                        auto& peer = works[(ithr & 1) ? (ithr - 1) : (ithr + 1)];
                        // the other one has finished, we can do the reduce sum
                        auto* src0 = &work.C[0];
                        auto* src1 = &peer.C[0];
                        auto* dst = dstC + work.n0;
                        auto strideS = work.C.stride / sizeof(*src0);
                        auto strideD = strideC / sizeof(*dst);
                        for (int m = 0; m < M; m++, src0 += strideS, src1 += strideS, dst += strideD) {
                            // the prefetch distance is increased to ensure by the time store happens
                            // prefetch has done and no HW prefetcher is triggered
                            auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                            (*work.p_jit_reduce2bh_2)(src0, src1, dst, prefetch_dst, work.BN);
                        }
                    }
                } else {
                    auto* src = &work.C[0];
                    auto* dst = dstC + work.n0;
                    auto strideS = work.C.stride / sizeof(*src);
                    auto strideD = strideC / sizeof(*dst);
                    for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                        (*work.p_jit_reduce2bh_1)(src, dst, prefetch_dst, work.BN);
                    }
                }
            }
        });
    }

    // gate & up are interleaved: 16 gates + 16 up
    void runGateUp(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work.BN > 0) {
                work.run(M, pA, strideA);
                // K reduce is done, results of [M, BN] sub-block is ready in L2.
                // combine Gate & Up
                auto* src = &work.C[0];
                auto strideS = work.C.stride / sizeof(*src);
                auto* dst = dstC + (work.n0 / 2);  // important output is only half of the total N
                auto strideD = strideC / sizeof(*dst);
                for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                    auto* prefetch_dst = (m + 1 < M) ? (dst + strideD) : (dst);
                    (*work.p_jit_gateup)(src, dst, prefetch_dst, work.BN);
                }
            }
        });
    }
};

struct QKVProj : public LLMMLP::Executor  {
    QKVProj() {}
    struct Work {
        MKernel* p_jit_amx0 = nullptr;
        ReduceAdd2bh* p_jit2bh = nullptr;

        std::vector<tensor2D<ov::bfloat16>> weights;
        tensor2D<float> C;

        int output_id;
        int n0 = 0;
        int n1 = 0;
        int BN = 0;
        int BK = 0;
        operator bool() {
            return BN > 0;
        }
        int blk_K_size;

        ov::bfloat16 * p_raw_weights;

        // input : weight [N, K], setup repacks range of N [n_start, n_end)
        template <typename T>
        void setup(T* p_weight, int stride) {
            auto num_blk_K = BK / blk_K_size;
            auto* pw = p_weight + n0 * stride / sizeof(T);
            weights.resize(num_blk_K);
            for (int k = 0; k < num_blk_K; k++) {
                weights[k] = p_jit_amx0->prepareB(pw + k * blk_K_size, stride, BN, blk_K_size);
            }
        }

        void run(int M, uint8_t* pA, int strideA) {
            int num_blk_K = weights.size();

            TileConfig tile_cfg0;
            TileConfig tile_cfg1;

            auto Mtails = M % 32;
            auto Mbody = M - Mtails;
            p_jit_amx0->tile_config_M(tile_cfg0, 32);
            if (Mtails) {
                p_jit_amx0->tile_config_M(tile_cfg1, Mtails);
            }
            TileConfigScope tcfg({tile_cfg0, tile_cfg1});

            C.resize(Mbody + (Mtails ? 32 : 0), BN);

            auto pC = reinterpret_cast<uint8_t*>(&C[0]);
            bool do_accumulation = false;
            for (int ki = 0; ki < num_blk_K; ki++) {
                tensor2D<ov::bfloat16>& blockB = weights[ki];
                tensor2D<ov::bfloat16>& blockB1 = weights[(ki + 1) < num_blk_K ? (ki + 1) : ki];

                if (Mbody) {
                    tcfg.load(0);
                    p_jit_amx0->run(Mbody,
                                    pA + ki * blk_K_size * sizeof(ov::bfloat16),
                                    strideA,
                                    blockB,
                                    pC,
                                    C.stride,
                                    reinterpret_cast<uint8_t*>(&blockB1[0]),
                                    do_accumulation);
                }

                if (Mtails) {
                    tcfg.load(1);
                    p_jit_amx0->run(Mtails,
                                    pA + ki * blk_K_size * sizeof(ov::bfloat16) + Mbody * strideA,
                                    strideA,
                                    blockB,
                                    pC + Mbody * C.stride,
                                    C.stride,
                                    reinterpret_cast<uint8_t*>(&blockB1[0]),
                                    do_accumulation);
                }
                do_accumulation = true;
            }
        }
    };
    std::vector<Work> works;

    // q k v each have 1/3 or worker-thread
    void setup(ov::bfloat16* wq, ov::bfloat16* wk, ov::bfloat16* wv, int N, int K) {
        static MKernel jit_amx0(256);
        static ReduceAdd2bh jit_2bh(false);
        const int blk_K_size = 256;
        // prepare weights, split N among threads
        // in unit of 32
        ASSERT((N % 32) == 0);
        ASSERT((K % blk_K_size) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_N = N / 32;
        auto num_blk_K = K / blk_K_size;
        works.resize(nthr);

        int stride = K * sizeof(*wq);

        // every thread should do same amount of work, and some cores can be idle
        auto valid_nthr = nthr / 3;

        int cur_work_id = 0;
        auto create_works = [&](ov::bfloat16* pw, int output_id) {
            auto blkN_per_thread = (num_blk_N + valid_nthr - 1) / valid_nthr;
            auto start_blkN = 0;
            for (int ithr = 0; ithr < valid_nthr; ithr ++) {
                auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
                if (blkN) {
                    auto& work = works[cur_work_id++];
                    work.p_jit_amx0 = &jit_amx0;
                    work.p_jit2bh = &jit_2bh;
                    work.blk_K_size = blk_K_size;

                    work.n0 = (start_blkN)*32;
                    work.n1 = (start_blkN + blkN) * 32;
                    work.BN = blkN * 32;
                    work.BK = blk_K_size * num_blk_K;

                    work.output_id = output_id;
                    work.p_raw_weights = pw;
                }
                start_blkN += blkN;
            }
        };
        create_works(wq, 0);
        create_works(wk, 1);
        create_works(wv, 2);
        auto used_nthr = cur_work_id;

        ECOUT("QKVProj N,K=", N, ",", K, " used_nthr=", used_nthr);

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(work.p_raw_weights, stride);
            }
        });
    }

    void run(uint8_t* pA,
             int strideA,
             int M,
             ov::bfloat16* dst_q,
             int stride_q,
             ov::bfloat16* dst_k,
             int stride_k,
             ov::bfloat16* dst_v,
             int stride_v) {
        for (int m = 0; m < M;) {
            int BM = std::min(M - m, 256);

            ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
                auto& work = works[ithr];
                if (work.BN > 0) {
                    work.run(BM, pA, strideA);

                    // compress accumulation result into target
                    auto* src = &work.C[0];
                    auto stride_src = work.C.stride / sizeof(*src);
                    ov::bfloat16* dst = nullptr;
                    int stride_dst = 0;
                    if (work.output_id == 0) {
                        dst = dst_q + work.n0;
                        stride_dst = stride_q / sizeof(*dst);
                    }
                    if (work.output_id == 1) {
                        dst = dst_k + work.n0;
                        stride_dst = stride_k / sizeof(*dst);
                    }
                    if (work.output_id == 2) {
                        dst = dst_v + work.n0;
                        stride_dst = stride_v / sizeof(*dst);
                    }

                    for (int mi = 0; mi < BM; mi++, src += stride_src, dst += stride_dst) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (mi + 2 < BM) ? (dst + 2 * stride_dst) : (dst);
                        (*work.p_jit2bh)(src, dst, prefetch_dst, work.BN);
                    }
                }
            });
            m += BM;
            pA += BM * strideA;
            dst_q += BM * stride_q / sizeof(ov::bfloat16);
            dst_k += BM * stride_k / sizeof(ov::bfloat16);
            dst_v += BM * stride_v / sizeof(ov::bfloat16);
        }
    }
    /*
    void setup(ov::bfloat16* wq, ov::bfloat16* wk, ov::bfloat16* wv, int K, int N) {
        q_proj.setup(&wq[0], K*sizeof(ov::float16), N, K);
        k_proj.setup(&wk[0], K*sizeof(ov::float16), N, K);
        v_proj.setup(&wv[0], K*sizeof(ov::float16), N, K);
    }
    void run(uint8_t* pA, int strideA, int M,
             ov::bfloat16* dst_q, int stride_q,
             ov::bfloat16* dst_k, int stride_k,
             ov::bfloat16* dst_v, int stride_v) {
        for(int m = 0; m < M;) {
            int BM = std::min(M - m, 512);

            q_proj.run(pA, strideA, BM, dst_q, stride_q);
            k_proj.run(pA, strideA, BM, dst_k, stride_k);
            v_proj.run(pA, strideA, BM, dst_v, stride_v);

            m += BM;
            pA += BM*strideA;
            dst_q += BM*stride_q/sizeof(ov::bfloat16);
            dst_k += BM*stride_k/sizeof(ov::bfloat16);
            dst_v += BM*stride_v/sizeof(ov::bfloat16);
        }
    }
    */
};

struct MLP : LLMMLP::Executor {
    Linear gate_up;
    Linear down;
    MLP() {}

    tensor2D<ov::bfloat16> actUp;
    int m_N;
    int m_M = 0;

    // [M, K] x [N, K] => [M, N] x [K, N] => [M, K]
    // w_gate/w_up : [N, K]
    //     w_down  : [K, N]
    void setup(ov::bfloat16* pw_gate, ov::bfloat16* pw_up, ov::bfloat16* pw_down, int K, int N) {
        // [N, K] [N, K] interleave (16-16-...) into [2*N, K]
        tensor2D<ov::bfloat16> w_gate(N, K, pw_gate, K * sizeof(ov::bfloat16));
        tensor2D<ov::bfloat16> w_up(N, K, pw_up, K * sizeof(ov::bfloat16));
        tensor2D<ov::bfloat16> w_down(K, N, pw_down, N * sizeof(ov::bfloat16));

        tensor2D<ov::bfloat16> w_gate_up;
        w_gate_up.resize(2 * N, K, true);
        for (int n = 0; n < N; n += 16) {
            for (int i = 0; i < 16; i++)
                memcpy(&w_gate_up(2 * n + i, 0), &w_gate(n + i, 0), K * sizeof(ov::bfloat16));
            for (int i = 0; i < 16; i++)
                memcpy(&w_gate_up(2 * n + 16 + i, 0), &w_up(n + i, 0), K * sizeof(ov::bfloat16));
        }
        gate_up.setup(&w_gate_up[0], w_gate_up.stride, N * 2, K);
        down.setup(&w_down[0], w_down.stride, K, N, true);
        m_N = N;
    }

    void setM(int M) {
        if (m_M < M) {
            actUp.resize(M, m_N, false);
            m_M = M;
        }
    }

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
        for(int m = 0; m < M;) {
            int BM = std::min(M - m, 512);
            setM(BM);

            gate_up.runGateUp(pA, strideA, BM, &actUp[0], actUp.stride);
            down.run(reinterpret_cast<uint8_t*>(&actUp[0]), actUp.stride, BM, dstC, strideC);

            m += BM;
            pA += BM*strideA;
            dstC += BM*strideC/sizeof(ov::bfloat16);
        }
    }
};

};  // namespace AMX_MLP

LLMMLP::LLMMLP(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)), m_executor(nullptr) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const LLMMLPNode>(op);
    m_config = node->get_config();
}

void LLMMLP::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto srcPrecision = getOriginalInputPrecisionAtPort(0);

    auto rtPrecision = ov::element::bf16;
    auto weightPrecision = ov::element::bf16;

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // gate
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // up
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(3), false, -1);  // down

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;

    if (m_config.is_qkv_proj) {
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    } else {
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void LLMMLP::execute(dnnl::stream strm) {
    if (!m_executor) {
        if (m_config.is_qkv_proj) {
            auto exec = std::make_shared<AMX_MLP::QKVProj>();
            exec->setup(getSrcMemoryAtPort(1)->getDataAs<ov::bfloat16>(),
                    getSrcMemoryAtPort(2)->getDataAs<ov::bfloat16>(),
                    getSrcMemoryAtPort(3)->getDataAs<ov::bfloat16>(),
                    m_config.hidden_size,
                    m_config.hidden_size);
            m_executor = exec;
        } else {
            auto exec = std::make_shared<AMX_MLP::MLP>();
            exec->setup(getSrcMemoryAtPort(1)->getDataAs<ov::bfloat16>(),
                    getSrcMemoryAtPort(2)->getDataAs<ov::bfloat16>(),
                    getSrcMemoryAtPort(3)->getDataAs<ov::bfloat16>(),
                    m_config.hidden_size,
                    m_config.intermediate_size);
            m_executor = exec;
        }
    }

    auto input = getSrcMemoryAtPort(0);
    auto ishape = input->getStaticDims();
    uint8_t* pA = input->getDataAs<uint8_t>();
    int strideA = m_config.hidden_size * 2;
    int M = shape_size(ishape) / ishape[ishape.size()-1];

    if (m_config.is_qkv_proj) {
        auto exec = std::dynamic_pointer_cast<AMX_MLP::QKVProj>(m_executor);

        exec->run(pA, strideA, M,
                  getDstMemoryAtPort(0)->getDataAs<ov::bfloat16>(), m_config.hidden_size * 2,
                  getDstMemoryAtPort(1)->getDataAs<ov::bfloat16>(), m_config.hidden_size * 2,
                  getDstMemoryAtPort(2)->getDataAs<ov::bfloat16>(), m_config.hidden_size * 2);
    } else {
        auto exec = std::dynamic_pointer_cast<AMX_MLP::MLP>(m_executor);

        auto output = getDstMemoryAtPort(0);
        ov::bfloat16* dstC = output->getDataAs<ov::bfloat16>();
        int strideC = m_config.hidden_size * 2;

        //std::cout << "=== " << M << ", " << strideA << ", " << strideC << std::endl;
        exec->run(pA, strideA, M, dstC, strideC);
    }
}

bool LLMMLP::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const LLMMLPNode>(op);
        if (!node) {
            errorMessage = "Only LLMMLPNode operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
