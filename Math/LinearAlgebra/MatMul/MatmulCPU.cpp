/**
 * @file MatmulCPU.cpp
 * @author GreenBean
 * @brief CPU backend for matmul.
 * @version 0.1
 * @date 2026-10-18
 */
#include <algorithm>
#include <cstdint>
#include <future>
#include <type_traits>
#include <vector>

#include "Math/LinearAlgebra/MatMul/GemmKernels.h"
#include "Math/LinearAlgebra/MatMul/Matmul.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    namespace {
        /**
         * C rows [row_begin, row_end) of one batch, C(m x n) = A(m x k) * B(k x n), row-major.
         * C is fully overwritten. Packed dtypes compute per lane.
         * @brief Row-range GEMM kernel signature.
         * @tparam T Element type.
         */
        template<typename T>
        using GemmRowsFn = void(*)(const T*, const T*, T*, size_t, size_t, size_t, size_t);

        /**
         * Scalar fallback for FP32/FP64 in ikj order: the innermost j loop is unit-stride over both B and C,
         * so the compiler auto-vectorizes it even without the AVX TUs.
         * @brief Scalar row-range GEMM (fp32/fp64).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         */
        template<typename T>
        void gemm_rows_scalar(const T* a, const T* b, T* c, const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
            for (auto i = row_begin; i < row_end; ++i) {
                auto* c_row = c + i * n;
                for (auto j = size_t{0}; j < n; ++j) {
                    c_row[j] = T{0};
                }
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto a_ik = a[i * k + kk];
                    const auto* b_row = b + kk * n;
                    for (auto j = size_t{0}; j < n; ++j) {
                        c_row[j] += a_ik * b_row[j];
                    }
                }
            }
        }

        /**
         * Goes through the unaccelerated conversion pathway, untracked.
         * @brief Widen one fp16/bf16 lane to fp32.
         * @param v The fp16 or bf16 lane.
         * @return The lane as float32_t.
         */
        template<typename H>
        [[nodiscard]] float32_t widen_lane(const H& v) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return Intrinsics::detail::unaccel_fp16_to_fp32_untracked(v);
            } else {
                return Intrinsics::detail::unaccel_bf16_to_fp32(v);
            }
        }

        /**
         * Goes through the unaccelerated conversion pathway, untracked and PERMISSIVE.
         * @brief Narrow one fp32 accumulator to fp16/bf16.
         * @param v The fp32 accumulator.
         * @return The accumulator rounded to the half type.
         */
        template<typename H>
        [[nodiscard]] H narrow_lane(const float32_t v) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return Intrinsics::detail::unaccel_fp32_to_fp16_untracked(v);
            } else {
                return Intrinsics::detail::unaccel_fp32_to_bf16_untracked(v);
            }
        }

        /**
         * Maps float16x2_t to float16_t and bfloat16x2_t to bfloat16_t.
         * @brief Lane type of a packed dtype.
         * @tparam P Packed dtype.
         */
        template<typename P> struct packed_lane;
        template<> struct packed_lane<float16x2_t> { using type = float16_t; };
        template<> struct packed_lane<bfloat16x2_t> { using type = bfloat16_t; };
        template<typename P> using packed_lane_t = typename packed_lane<P>::type;

        /**
         * Reads the lane through its bit pattern, never through arithmetic.
         * @brief Low lane of a packed value.
         * @param bits The packed 32-bit unit.
         * @return The low 16 bits as the lane type.
         */
        template<typename H>
        [[nodiscard]] H lane_lo(const uint32_t bits) {
            return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits & 0xFFFFu));
        }

        /**
         * Reads the lane through its bit pattern, never through arithmetic.
         * @brief High lane of a packed value.
         * @param bits The packed 32-bit unit.
         * @return The high 16 bits as the lane type.
         */
        template<typename H>
        [[nodiscard]] H lane_hi(const uint32_t bits) {
            return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits >> 16));
        }

        /**
         * ijk order with an FP32 accumulator per output element,
         * matching the GPU backends' FP32-compute gemm_ex semantics.
         * @brief Scalar row-range GEMM (fp16/bf16, FP32 accumulate).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         */
        template<typename H>
        void gemm_rows_half(const H* a, const H* b, H* c, const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
            for (auto i = row_begin; i < row_end; ++i) {
                for (auto j = size_t{0}; j < n; ++j) {
                    auto acc = 0.0f;
                    for (auto kk = size_t{0}; kk < k; ++kk) {
                        acc += widen_lane(a[i * k + kk]) * widen_lane(b[kk * n + j]);
                    }
                    c[i * n + j] = narrow_lane<H>(acc);
                }
            }
        }

        /**
         * Each packed element holds two independent lanes, so every output element runs two FP32 accumulators (lo/hi)
         * and packs on store. ijk order, same FP32-accumulate semantics as gemm_rows_half.
         * @brief Scalar row-range GEMM (fp16x2/bf16x2, per-lane FP32 accumulate).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         */
        template<typename P>
        void gemm_rows_packed(const P* a, const P* b, P* c, const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
            using H = packed_lane_t<P>;
            for (auto i = row_begin; i < row_end; ++i) {
                for (auto j = size_t{0}; j < n; ++j) {
                    auto acc_lo = 0.0f;
                    auto acc_hi = 0.0f;
                    for (auto kk = size_t{0}; kk < k; ++kk) {
                        const auto a_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a[i * k + kk]);
                        const auto b_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(b[kk * n + j]);
                        acc_lo += widen_lane(lane_lo<H>(a_bits)) * widen_lane(lane_lo<H>(b_bits));
                        acc_hi += widen_lane(lane_hi<H>(a_bits)) * widen_lane(lane_hi<H>(b_bits));
                    }
                    const auto lo = static_cast<uint32_t>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(acc_lo)));
                    const auto hi = static_cast<uint32_t>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(acc_hi)));
                    c[i * n + j] = Intrinsics::detail::bit_cast_memcpy<P>(lo | (hi << 16));
                }
            }
        }

        /**
         * Modular (two's-complement wrap) arithmetic. Accumulating directly in the unsigned twin of T wraps every add,
         * and modular addition is associative, so this is bit-identical to the 32/64-bit-accumulator VNNI/BLAS pathways truncated on store.
         * ikj order keeps the innermost loop unit-stride for auto-vectorization.
         * @brief Scalar row-range GEMM (all integer dtypes, wrap semantics).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         */
        template<typename T>
        void gemm_rows_int(const T* a, const T* b, T* c, const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
            using UT = std::make_unsigned_t<T>;
            std::vector<UT> acc(n);
            for (auto i = row_begin; i < row_end; ++i) {
                std::fill(acc.begin(), acc.end(), UT{0});
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto a_ik = static_cast<UT>(a[i * k + kk]);
                    const auto* b_row = b + kk * n;
                    for (auto j = size_t{0}; j < n; ++j) {
                        acc[j] += static_cast<UT>(a_ik * static_cast<UT>(b_row[j]));
                    }
                }
                auto* c_row = c + i * n;
                for (auto j = size_t{0}; j < n; ++j) {
                    c_row[j] = static_cast<T>(acc[j]);
                }
            }
        }

        /**
         * @brief Sign-extended lo nibble (bits [3:0]) of a packed int4x2 byte, as int8_t.
         * @param v The packed byte.
         * @return The sign-extended lo nibble in [-8, 7].
         */
        [[nodiscard]] int8_t nib_lo(const int4x2_t v) {
            const auto nib = static_cast<int8_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
            return nib > 7 ? static_cast<int8_t>(nib - 16) : nib;
        }

        /**
         * @brief Sign-extended hi nibble (bits [7:4]) of a packed int4x2 byte, as int8_t.
         * @param v The packed byte.
         * @return The sign-extended hi nibble in [-8, 7].
         */
        [[nodiscard]] int8_t nib_hi(const int4x2_t v) {
            const auto nib = static_cast<int8_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) >> 4);
            return nib > 7 ? static_cast<int8_t>(nib - 16) : nib;
        }

        /**
         * Per-lane modular arithmetic (same contract as gemm_rows_int, two independent 4-bit lanes per element).
         * Each lane's products accumulate in uint32_t and wrap to the nibble on store.
         * @brief Scalar row-range GEMM (int4x2, per-lane wrap semantics).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         */
        void gemm_rows_int4x2(const int4x2_t* a, const int4x2_t* b, int4x2_t* c, const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
            for (auto i = row_begin; i < row_end; ++i) {
                for (auto j = size_t{0}; j < n; ++j) {
                    auto acc_lo = uint32_t{0};
                    auto acc_hi = uint32_t{0};
                    for (auto kk = size_t{0}; kk < k; ++kk) {
                        const auto av = a[i * k + kk];
                        const auto bv = b[kk * n + j];
                        acc_lo += static_cast<uint32_t>(static_cast<int32_t>(nib_lo(av)) * static_cast<int32_t>(nib_lo(bv)));
                        acc_hi += static_cast<uint32_t>(static_cast<int32_t>(nib_hi(av)) * static_cast<int32_t>(nib_hi(bv)));
                    }
                    const auto packed = static_cast<uint8_t>((acc_lo & 0x0Fu) | ((acc_hi & 0x0Fu) << 4));
                    c[i * n + j] = Intrinsics::detail::bit_cast_memcpy<int4x2_t>(packed);
                }
            }
        }

        /**
         * fp32/fp64 dispatch AVX512F -> AVX2+FMA -> scalar. fp16/bf16 dispatch to the VDPBF16PS / F16C-widened AVX512 kernels
         * (packed x2 included) where supported, otherwise to their scalar conversion-pathway kernels.
         * @brief Runtime SIMD dispatch for GEMM row kernels.
         * @param T Element type.
         * @return Function pointer to the fastest legal row kernel for this CPU.
         */
        template<typename T>
        [[nodiscard]] GemmRowsFn<T> pick_gemm_kernel() {
            if constexpr (std::is_same_v<T, float32_t>) {
                if (Math::LinearAlgebra::gemm_avx512_available()) {
                    return Math::LinearAlgebra::gemm_f32_rows_avx512;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_f32_rows_avx2;
                }
                return gemm_rows_scalar<T>;
            } else if constexpr (std::is_same_v<T, float64_t>) {
                if (Math::LinearAlgebra::gemm_avx512_available()) {
                    return Math::LinearAlgebra::gemm_f64_rows_avx512;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_f64_rows_avx2;
                }
                return gemm_rows_scalar<T>;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                if (Math::LinearAlgebra::gemm_avx512_f16c_available()) {
                    return Math::LinearAlgebra::gemm_f16_rows_avx512f16c;
                }
                return gemm_rows_half<T>;
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                if (Math::LinearAlgebra::gemm_avx512_bf16_available()) {
                    return Math::LinearAlgebra::gemm_bf16_rows_avx512bf16;
                }
                return gemm_rows_half<T>;
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                if (Math::LinearAlgebra::gemm_avx512_f16c_available()) {
                    return Math::LinearAlgebra::gemm_f16x2_rows_avx512f16c;
                }
                return gemm_rows_packed<T>;
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                if (Math::LinearAlgebra::gemm_avx512_bf16_available()) {
                    return Math::LinearAlgebra::gemm_bf16x2_rows_avx512bf16;
                }
                return gemm_rows_packed<T>;
            } else if constexpr (std::is_same_v<T, int8_t>) {
                if (Math::LinearAlgebra::gemm_avx512_vnni_available()) {
                    return Math::LinearAlgebra::gemm_i8_rows_avx512vnni;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_i8_rows_avx2;
                }
                return gemm_rows_int<T>;
            } else if constexpr (std::is_same_v<T, uint8_t>) {
                if (Math::LinearAlgebra::gemm_avx512_vnni_available()) {
                    return Math::LinearAlgebra::gemm_u8_rows_avx512vnni;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_u8_rows_avx2;
                }
                return gemm_rows_int<T>;
            } else if constexpr (std::is_same_v<T, int16_t>) {
                if (Math::LinearAlgebra::gemm_avx512_vnni_available()) {
                    return Math::LinearAlgebra::gemm_i16_rows_avx512vnni;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_i16_rows_avx2;
                }
                return gemm_rows_int<T>;
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                if (Math::LinearAlgebra::gemm_avx512_vnni_available()) {
                    return Math::LinearAlgebra::gemm_u16_rows_avx512vnni;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_u16_rows_avx2;
                }
                return gemm_rows_int<T>;
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if (Math::LinearAlgebra::gemm_avx512_available()) {
                    return Math::LinearAlgebra::gemm_i32_rows_avx512;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_i32_rows_avx2;
                }
                return gemm_rows_int<T>;
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                if (Math::LinearAlgebra::gemm_avx512_available()) {
                    return Math::LinearAlgebra::gemm_u32_rows_avx512;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_u32_rows_avx2;
                }
                return gemm_rows_int<T>;
            } else if constexpr (std::is_same_v<T, int64_t>) {
                if (Math::LinearAlgebra::gemm_avx512_dq_available()) {
                    return Math::LinearAlgebra::gemm_i64_rows_avx512dq;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_i64_rows_avx2;
                }
                return gemm_rows_int<T>;
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                if (Math::LinearAlgebra::gemm_avx512_dq_available()) {
                    return Math::LinearAlgebra::gemm_u64_rows_avx512dq;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_u64_rows_avx2;
                }
                return gemm_rows_int<T>;
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                if (Math::LinearAlgebra::gemm_avx512_vnni_available()) {
                    return Math::LinearAlgebra::gemm_i4x2_rows_avx512vnni;
                }
                if (Math::LinearAlgebra::gemm_avx2_available()) {
                    return Math::LinearAlgebra::gemm_i4x2_rows_avx2;
                }
                return gemm_rows_int4x2;
            } else {
                return gemm_rows_scalar<T>;
            }
        }

        /**
         * Splits the flattened (batch x row) unit space across the CPU thread pool; small products run inline.
         * Chunks are re-split at batch boundaries inside the worker. The shared futures are registered on all three tensors,
         * so freeing or syncing any of them drains the workers first, as the workers hold raw buffer pointers.
         * @brief Batched, row-blocked GEMM launcher.
         * @tparam T Element type.
         * @param kernel Row kernel chosen for the dtype.
         * @param a_data A buffer, batch x m x k.
         * @param b_data B buffer, batch x k x n.
         * @param c_data C buffer, batch x m x n.
         * @param batch Batch count.
         * @param m Rows of A and C.
         * @param k Inner dimension.
         * @param n Columns of B and C.
         * @param a Tensor owning a_data, receives the futures.
         * @param b Tensor owning b_data, receives the futures.
         * @param out Tensor owning c_data, receives the futures.
         */
        template<typename T>
        void launch_gemm_units(const GemmRowsFn<T> kernel, const T* a_data, const T* b_data, T* c_data,
                               const size_t batch, const size_t m, const size_t k, const size_t n,
                               const Tensor& a, const Tensor& b, const Tensor& out) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            const auto total = batch * m;
            /**
             * Walks the flattened (batch x row) units [unit_begin, unit_end), re-splitting at batch boundaries
             * so the kernel always sees one batch's base pointers.
             * @note Lambda subcall. Self-execution prohibited.
             * @brief Run the row kernel over one chunk of the flattened unit space.
             * @param unit_begin First flattened (batch x row) unit.
             * @param unit_end One past the last flattened unit.
             */
            const auto worker = [=](const size_t unit_begin, const size_t unit_end) {
                auto u = unit_begin;
                while (u < unit_end) {
                    const auto bi = u / m;
                    const auto row0 = u - bi * m;
                    const auto rows_here = std::min(m - row0, unit_end - u);
                    kernel(a_data + bi * m * k, b_data + bi * k * n, c_data + bi * m * n, k, n, row0, row0 + rows_here);
                    u += rows_here;
                }
            };
            if (t_count <= 1 || total * n * k <= 32768) {
                worker(0, total);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (total + t_count - 1) / t_count;
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto unit_begin = t * chunk;
                if (unit_begin >= total) {
                    break;
                }
                /**
                 * Binds one chunk of the unit space to the shared worker, for submission to the thread pool.
                 * @note Lambda subcall. Self-execution prohibited.
                 * @brief Thread pool task for one chunk of GEMM units.
                 */
                futures.push_back(threadMgr.submit([=] { worker(unit_begin, std::min(unit_begin + chunk, total)); }, Threading::TaskPriority::Normal).share());
            }
            a.give_futures(futures);
            b.give_futures(futures);
            out.give_futures(futures);
        }
    }

    /**
     * Validates that all three tensors live on the CPU, derives batch/m/k/n from the shapes, and dispatches on dtype.
     * Work is queued onto the thread pool; errors materialize at sync().
     * @brief CPU backend entry point for matmul.
     * @param a Left operand, m x k or batch x m x k.
     * @param b Right operand, k x n or batch x k x n.
     * @param out Preallocated output, m x n or batch x m x n.
     */
    void matmul_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
        if (a.device != Device::CPU || b.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(a.device));
        }
        const auto batch = a.ndim == 3 ? a.shape[0] : size_t{1};
        const auto m = a.shape[a.ndim == 3 ? 1 : 0];
        const auto k = a.shape[a.ndim == 3 ? 2 : 1];
        const auto n = b.shape[b.ndim == 3 ? 2 : 1];
        using DType = DataTypes::DType;
        switch (a.dtype) {
            case DType::Float32: {
                launch_gemm_units<float32_t>(pick_gemm_kernel<float32_t>(),
                                             static_cast<const float32_t*>(a.data), static_cast<const float32_t*>(b.data),
                                             static_cast<float32_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::Float64: {
                launch_gemm_units<float64_t>(pick_gemm_kernel<float64_t>(),
                                             static_cast<const float64_t*>(a.data), static_cast<const float64_t*>(b.data),
                                             static_cast<float64_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::Float16: {
                launch_gemm_units<float16_t>(pick_gemm_kernel<float16_t>(),
                                             static_cast<const float16_t*>(a.data), static_cast<const float16_t*>(b.data),
                                             static_cast<float16_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::BFloat16: {
                launch_gemm_units<bfloat16_t>(pick_gemm_kernel<bfloat16_t>(),
                                              static_cast<const bfloat16_t*>(a.data), static_cast<const bfloat16_t*>(b.data),
                                              static_cast<bfloat16_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::Float16x2: {
                launch_gemm_units<float16x2_t>(pick_gemm_kernel<float16x2_t>(),
                                               static_cast<const float16x2_t*>(a.data), static_cast<const float16x2_t*>(b.data),
                                               static_cast<float16x2_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::BFloat16x2: {
                launch_gemm_units<bfloat16x2_t>(pick_gemm_kernel<bfloat16x2_t>(),
                                                static_cast<const bfloat16x2_t*>(a.data), static_cast<const bfloat16x2_t*>(b.data),
                                                static_cast<bfloat16x2_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::Int8: {
                launch_gemm_units<int8_t>(pick_gemm_kernel<int8_t>(),
                                          static_cast<const int8_t*>(a.data), static_cast<const int8_t*>(b.data),
                                          static_cast<int8_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::UInt8: {
                launch_gemm_units<uint8_t>(pick_gemm_kernel<uint8_t>(),
                                           static_cast<const uint8_t*>(a.data), static_cast<const uint8_t*>(b.data),
                                           static_cast<uint8_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::Int16: {
                launch_gemm_units<int16_t>(pick_gemm_kernel<int16_t>(),
                                           static_cast<const int16_t*>(a.data), static_cast<const int16_t*>(b.data),
                                           static_cast<int16_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::UInt16: {
                launch_gemm_units<uint16_t>(pick_gemm_kernel<uint16_t>(),
                                            static_cast<const uint16_t*>(a.data), static_cast<const uint16_t*>(b.data),
                                            static_cast<uint16_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::Int32: {
                launch_gemm_units<int32_t>(pick_gemm_kernel<int32_t>(),
                                           static_cast<const int32_t*>(a.data), static_cast<const int32_t*>(b.data),
                                           static_cast<int32_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::UInt32: {
                launch_gemm_units<uint32_t>(pick_gemm_kernel<uint32_t>(),
                                            static_cast<const uint32_t*>(a.data), static_cast<const uint32_t*>(b.data),
                                            static_cast<uint32_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::Int64: {
                launch_gemm_units<int64_t>(pick_gemm_kernel<int64_t>(),
                                           static_cast<const int64_t*>(a.data), static_cast<const int64_t*>(b.data),
                                           static_cast<int64_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::UInt64: {
                launch_gemm_units<uint64_t>(pick_gemm_kernel<uint64_t>(),
                                            static_cast<const uint64_t*>(a.data), static_cast<const uint64_t*>(b.data),
                                            static_cast<uint64_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            case DType::Int4x2: {
                launch_gemm_units<int4x2_t>(pick_gemm_kernel<int4x2_t>(),
                                            static_cast<const int4x2_t*>(a.data), static_cast<const int4x2_t*>(b.data),
                                            static_cast<int4x2_t*>(out.data), batch, m, k, n, a, b, out);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }
}

namespace BeanTensor::Tensors {
    Tensor Tensor::matmul(const Tensor& other) const {
        if (this->ndim != other.ndim) {
            throw ErrorHandling::MismatchedShapes("rank differs (" + std::to_string(this->ndim) + " vs " + std::to_string(other.ndim) + ")");
        }
        if (this->ndim != 2 && this->ndim != 3) {
            throw ErrorHandling::MismatchedShapes("matmul supports rank-2 and rank-3 (batched) tensors");
        }
        if (this->dtype != other.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->device != other.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !other.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        const auto d = this->ndim == 3 ? size_t{1} : size_t{0}; // matrix dims start after the batch axis
        if (this->ndim == 3 && this->shape[0] != other.shape[0]) {
            throw ErrorHandling::MismatchedShapes("batch extents differ (" + std::to_string(this->shape[0]) + " vs " + std::to_string(other.shape[0]) + ")");
        }
        if (this->shape[d + 1] != other.shape[d]) {
            throw ErrorHandling::MismatchedShapes("inner dimensions differ (" + std::to_string(this->shape[d + 1]) + " vs " + std::to_string(other.shape[d]) + ")");
        }
        sync();
        other.sync();
        const auto out_shape = this->ndim == 3
            ? std::vector<size_t>{this->shape[0], this->shape[1], other.shape[2]}
            : std::vector<size_t>{this->shape[0], other.shape[1]};
        auto out = Tensor(out_shape, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::matmul_cpu(*this, other, out);
        } else if (this->device == Device::GPU) {
            detail::matmul_gpu(*this, other, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        out.wire_node(detail::gemm_g_node, *this, &other);
        return out;
    }
}
