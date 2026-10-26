/**
 * @file platform_topology_tests.cpp
 * @author GreenBean
 * @brief Platform topology / NT-store gating / CCD-routing tests - framework-free assertion suite.
 * Covers the CPU platform probes (LastLevelCacheBytes, NtStoreMinBytes, IncludedCcdTopology,
 * CcdGroupOfCurrentCpu, KeepWorkLocal), the affinity-restricted thread-pool submit
 * (submit_to_group pins the worker to the requested CCD group for the task's duration),
 * and the behavioral guards on the landed kernel changes: the fp32->fp16 overflow fixup
 * must be bit-identical to the scalar PERMISSIVE converter on inputs dense at the overflow
 * boundary, at odd sizes (vector loop + scalar tail), straddling the NT-store byte gate,
 * and with the STRICT/PERMISSIVE flag contract intact; plus end-to-end routing checks that
 * CCD-restricted scheduling leaves conversion and elementwise results bit-identical.
 * Also proves the newly CCD-wired op launchers execute and stay value-correct on both
 * sides of their gates: matmul (batched, including a C span above the LLC), bias_add,
 * relu, softmax_ce, equal, sum, mean, argmax, Adam, SGD, one autograd backward chain
 * (matmul -> bias_add -> relu -> sum), the tensor fills, and an elementwise add with
 * a span above the LLC - each sized past its launcher's pool-chunk gate, with the
 * gated byte span on both sides of LastLevelCacheBytes().
 * Also covers the work-stealing thread-pool rework: exactly-once execution
 * under 4-way submission, FIFO order on a single worker, the helping wait
 * (HelpWait) as the deadlock-free in-worker join under pool saturation,
 * nesting-safe CCD group pins, the three set_zero paths (routed chunk,
 * inline, and the in-worker inline fallback), and the worker telemetry
 * snapshot API (idle pool, snapshot under saturation, a deterministic HelpWaiting
 * observation via a non-pool gate future, exact completed counters, and the text
 * formatter).
 * @version 0.8
 * @date 2026-10-25
 */
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "Helpers/Alloc.h"

#if defined(__linux__)
#include <sched.h>
#endif

#include "Helpers/CPUThreading.h"
#include "Helpers/PoolMonitor.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/AVX2/DTypeConversions.h"
#include "Intrinsics/AVX512/DTypeConversions.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Tensor/Tensor.h"

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float16_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;
namespace bd = BeanTensor::Intrinsics::detail;
namespace bt = BeanTensor::Threading;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

// --- deterministic pseudo-random finite fp32 bit patterns (no <random>) ---

static uint32_t lcg_next(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

// Half the draws land in fp32-subnormal/zero territory (zero exponent is always
// finite); the other half get a random exponent in [1, 254] so they span the
// full finite range without ever producing Inf/NaN.
static uint32_t lcg_finite_bits(uint32_t& s) {
    const auto b = lcg_next(s);
    if ((b & 1u) == 0u) {
        return b & 0x807FFFFFu;
    }
    return (b & 0x807FFFFFu) | (((b >> 23) % 254u + 1u) << 23);
}

// --- 1: LLC / NT-store threshold sanity ---

static void test_llc_and_threshold() {
    constexpr size_t k256kb = size_t{256} * 1024u;
    constexpr size_t k1gb = size_t{1} * 1024u * 1024u * 1024u;
    constexpr size_t k64kb = size_t{64} * 1024u;

    const auto llc = LastLevelCacheBytes();
    const auto nt = NtStoreMinBytes();

    CHECK(llc >= k256kb);
    CHECK(llc <= k1gb);
    CHECK(nt == (llc / 2u > k64kb ? llc / 2u : k64kb));
    CHECK(nt >= k256kb / 2u);
    CHECK(nt <= k1gb / 2u);

    std::cout << "LastLevelCacheBytes() = " << llc << ", NtStoreMinBytes() = " << nt << "\n";
}

// --- 2: CCD topology invariants ---

static void test_ccd_topology() {
    const CpuCcdGroups& topology = IncludedCcdTopology();
    const auto num_cpus = static_cast<uint32_t>(std::thread::hardware_concurrency());

    CHECK(!topology.groups.empty());
    CHECK(topology.cpu_to_group.size() == num_cpus);

    // Pairwise disjoint, every member in range, union covers {0..num_cpus-1}.
    auto seen = std::vector<bool>(num_cpus, false);
    for (auto g = size_t{0}; g < topology.groups.size(); ++g) {
        CHECK(!topology.groups[g].empty());
        for (const auto cpu : topology.groups[g]) {
            CHECK(cpu < num_cpus);
            CHECK(topology.cpu_to_group[cpu] == g);
            CHECK(!seen[cpu]); // pairwise disjoint
            seen[cpu] = true;
        }
    }
    for (auto cpu = uint32_t{0}; cpu < num_cpus; ++cpu) {
        CHECK(seen[cpu]); // union == every logical CPU
    }

    CHECK(CcdGroupOfCurrentCpu() < topology.groups.size());
}

// --- 3: affinity-restricted submit ---

static void test_submit_to_group() {
#if defined(__linux__)
    const CpuCcdGroups& topology = IncludedCcdTopology();
    auto& pool = bt::GetCPUThreadPool();

    auto mu = std::mutex{};
    auto recorded = std::vector<std::pair<uint32_t, uint32_t>>{}; // (group, cpu)

    // One pinned task per group: the recorded CPU must be a member of that group.
    for (auto g = uint32_t{0}; g < topology.groups.size(); ++g) {
        const auto fut = pool.submit_to_group([g, &mu, &recorded] {
            const auto cpu = static_cast<uint32_t>(sched_getcpu());
            const auto lock = std::lock_guard{mu};
            recorded.emplace_back(g, cpu);
        }, g);
        fut.wait();
        const auto cpu_of = recorded.back().second;
        const auto& members = topology.groups[g];
        const auto found = std::find(members.begin(), members.end(), cpu_of) != members.end();
        CHECK(found); // pinned task observed on a CPU of its group
    }
    // One unrestricted task: it must simply run.
    auto ran = std::atomic<bool>{false};
    auto plain = pool.submit([&ran] { ran.store(true, std::memory_order_relaxed); });
    plain.wait();
    CHECK(ran.load());
#else
    std::cout << "submit_to_group test skipped (non-Linux)\n";
#endif
}

// --- 4: KeepWorkLocal boundary ---

static void test_keep_work_local() {
    const auto llc = LastLevelCacheBytes();
    CHECK(KeepWorkLocal(llc));      // exactly the LLC: local
    CHECK(!KeepWorkLocal(llc + 1)); // one byte past: not local
    CHECK(KeepWorkLocal(size_t{0}));
}

// --- 5: fp32 -> fp16 conversion boundary regression ---

using Fp16Kernel = std::vector<std::shared_future<void>>(*)(
    const float32_t*, float16_t*, const size_t&, std::atomic<bool>&, ConversionClampMethod);

// Probe through a tiny kernel launch (see tests/tensor_tests.cpp): this TU
// compiles with -mavx2 -mf16c only, so its own IncludedCPUFeatures() copy is
// blind for AVX512 - the kernel TU's runtime gate is the only authority. A
// thrown CPUIllegalInstruction means the CPU lacks the feature.
static bool probe_fp16_kernel(const Fp16Kernel kernel) {
    constexpr float32_t src[1] = {1.0f};
    auto dst = std::vector<float16_t>(1);
    const auto end = size_t{1};
    try {
        for (std::atomic flag{false}; auto& f : kernel(src, dst.data(), end, flag, ConversionClampMethod::PERMISSIVE)) {
            f.get();
        }
        return true;
    } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
        return false;
    }
}

// Input set dense at the fp16 overflow boundary, in fp32 bit patterns.
static std::vector<uint32_t> fp16_boundary_bits() {
    auto bits = std::vector<uint32_t>{
        0x00000000, 0x80000000,             // +-0.0
        0x7F800000, 0xFF800000,             // +-Inf
        0x7FC00000, 0xFFC00000,             // quiet NaN +/-
        0x7F800001, 0x7FA0BEEF, 0xFF800001, // signalling NaNs (several payloads)
        0x477FE000, 0xF77FE000,             // +-65504 (FP16_MAX, fixup trigger)
        0x7F7FFFFF, 0xFF7FFFFF,             // +-fp32 max (fixup trigger)
        0x38800000, 0xB8800000,             // +-2^-14 (fp16 min normal)
        0x387FFFFF, 0xB87FFFFF,             // just below 2^-14 (subnormal result)
        0x33800000, 0xB3800000,             // +-2^-24 (min fp16 subnormal)
        0x337FFFFF, 0xB37FFFFF,             // just below 2^-24 (flushes to zero)
    };
    // 65504 +- {0,1,2,3} ulp, both signs.
    for (const auto u : {uint32_t{0}, 1u, 2u, 3u}) {
        bits.push_back(0x477FE000u - u);
        bits.push_back(0x477FE000u + u);
        bits.push_back(0xF77FE000u - u);
        bits.push_back(0xF77FE000u + u);
    }
    // The RNE rounding boundary itself: 65519 / 65520 / 65521.
    for (const auto v : {65519.0f, 65520.0f, 65521.0f}) {
        bits.push_back(std::bit_cast<uint32_t>(v));
        bits.push_back(std::bit_cast<uint32_t>(-v));
    }
    // +-1e6: deep finite overflow (clamps to +-65504 via the fixup).
    bits.push_back(std::bit_cast<uint32_t>(1.0e6f));
    bits.push_back(std::bit_cast<uint32_t>(-1.0e6f));
    return bits;
}

// Deterministic per-index input: specials recur throughout the buffer (so they
// land in both the vector loop and the scalar tail), LCG finite values elsewhere.
static float32_t boundary_input(const size_t i, const std::vector<uint32_t>& specials, uint32_t& rng) {
    if (i < specials.size()) {
        return std::bit_cast<float32_t>(specials[i]);
    }
    if (i % 53u == 0u) {
        return std::bit_cast<float32_t>(specials[(i / 53u) % specials.size()]);
    }
    return std::bit_cast<float32_t>(lcg_finite_bits(rng));
}

// Runs one kernel over n elements and bit-compares every output lane against
// the scalar PERMISSIVE converter. Buffers are 64-byte aligned so the NT-store
// gate's alignment half is satisfied and the size half decides.
static void check_kernel_bit_exact(const char* const label, const Fp16Kernel kernel, const size_t n) {
    auto* const src = static_cast<float32_t*>(bt_aligned_alloc(n * sizeof(float32_t)));
    auto* const dst = static_cast<float16_t*>(bt_aligned_alloc(n * sizeof(float16_t)));

    const auto specials = fp16_boundary_bits();
    auto rng = uint32_t{0x2C179AC1u};
    for (auto i = size_t{0}; i < n; ++i) {
        src[i] = boundary_input(i, specials, rng);
    }

    const auto end = n;
    std::atomic flag{false};
    for (auto& f : kernel(src, dst, end, flag, ConversionClampMethod::PERMISSIVE)) {
        f.get();
    }

    auto ignored = false;
    for (auto i = size_t{0}; i < n; ++i) {
        const auto ref = bd::unaccel_fp32_to_fp16_tracked(src[i], ignored, ConversionClampMethod::PERMISSIVE);
        if (bd::bit_cast_memcpy<uint16_t>(dst[i]) != bd::bit_cast_memcpy<uint16_t>(ref)) {
            ++failures;
            std::cerr << "FAIL: " << label << " bit mismatch at [" << i << "] of " << n
                      << ": got=0x" << std::hex << static_cast<uint32_t>(bd::bit_cast_memcpy<uint16_t>(dst[i]))
                      << " ref=0x" << static_cast<uint32_t>(bd::bit_cast_memcpy<uint16_t>(ref)) << std::dec
                      << " src_bits=0x" << std::hex << std::bit_cast<uint32_t>(src[i]) << std::dec << "\n";
            break;
        }
    }

    std::free(src);
    std::free(dst);
}

// Flag contract on a small probe: the kernel's flag outcome must equal the
// scalar converter's on the same Inf/NaN/overflow inputs, for every method.
static void check_kernel_flags(const char* const label, const Fp16Kernel kernel) {
    const auto src = std::vector<float32_t>{5.0f, 1.0e6f, std::numeric_limits<float32_t>::infinity(),
                                            -std::numeric_limits<float32_t>::infinity(),
                                            std::numeric_limits<float32_t>::quiet_NaN()};
    constexpr ConversionClampMethod methods[3] = {
        ConversionClampMethod::STRICT, ConversionClampMethod::ALLOW_INF, ConversionClampMethod::PERMISSIVE};

    for (const auto method : methods) {
        auto scalar_flag = false;
        for (const auto& v : src) {
            static_cast<void>(bd::unaccel_fp32_to_fp16_tracked(v, scalar_flag, method));
        }
        auto dst = std::vector<float16_t>(src.size());
        const auto end = src.size();
        std::atomic flag{false};
        for (auto& f : kernel(src.data(), dst.data(), end, flag, method)) {
            f.get();
        }
        if (flag.load(std::memory_order_relaxed) != scalar_flag) {
            ++failures;
            std::cerr << "FAIL: " << label << " flag mismatch (method " << static_cast<int32_t>(method)
                      << "): kernel=" << flag.load() << " scalar=" << scalar_flag << "\n";
        }
    }

    // The headline contract, explicit: STRICT flags Inf/NaN, PERMISSIVE does not.
    auto strict_dst = std::vector<float16_t>(src.size());
    const auto strict_end = src.size();
    std::atomic strict_flag{false};
    for (auto& f : kernel(src.data(), strict_dst.data(), strict_end, strict_flag, ConversionClampMethod::STRICT)) {
        f.get();
    }
    CHECK(strict_flag.load());
    std::atomic perm_flag{false};
    for (auto& f : kernel(src.data(), strict_dst.data(), strict_end, perm_flag, ConversionClampMethod::PERMISSIVE)) {
        f.get();
    }
    CHECK(!perm_flag.load());
}

static void test_fp32_to_fp16_boundary() {
    // Sizes exercising the vector loop + scalar tail (37 = 2*16+5; 16389 = 16k+5),
    // then sizes straddling the NT-store gate: the fp16 kernel gates on
    // dst bytes > NtStoreMinBytes(), i.e. elements > nt/sizeof(float16_t).
    const auto nt_elements = NtStoreMinBytes() / sizeof(float16_t);
    const auto small_sizes = std::vector<size_t>{37, size_t{16} * 1024u + 5u};
    const auto gate_sizes = std::vector<size_t>{nt_elements - 8u, nt_elements + 8u};

    if (probe_fp16_kernel(&bd::avx512_fp32_to_fp16)) {
        for (const auto n : small_sizes) {
            check_kernel_bit_exact("avx512_fp32_to_fp16", &bd::avx512_fp32_to_fp16, n);
        }
        for (const auto n : gate_sizes) {
            check_kernel_bit_exact("avx512_fp32_to_fp16 (NT gate)", &bd::avx512_fp32_to_fp16, n);
        }
        check_kernel_flags("avx512_fp32_to_fp16", &bd::avx512_fp32_to_fp16);
        std::cout << "AVX512 fp32->fp16 boundary checks executed (incl. NT-gate straddle at "
                  << nt_elements - 8u << "/" << nt_elements + 8u << " elements)\n";
    } else {
        std::cout << "SKIP: AVX512 fp32->fp16 (CPU lacks the feature)\n";
    }

    // This TU carries -mavx2 -mf16c, so its IncludedCPUFeatures() copy is
    // authoritative for the AVX2/F16C gate (mirrors tests/tensor_tests.cpp).
    if (IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2) &&
        IncludedCPUFeatures().has_feature(CPUFeatureSetType_F16C)) {
        for (const auto n : small_sizes) {
            check_kernel_bit_exact("avx2_fp32_to_fp16", &bd::avx2_fp32_to_fp16, n);
        }
        for (const auto n : gate_sizes) {
            check_kernel_bit_exact("avx2_fp32_to_fp16 (NT gate)", &bd::avx2_fp32_to_fp16, n);
        }
        check_kernel_flags("avx2_fp32_to_fp16", &bd::avx2_fp32_to_fp16);
        std::cout << "AVX2 fp32->fp16 boundary checks executed (incl. NT-gate straddle)\n";
    } else {
        std::cout << "SKIP: AVX2 fp32->fp16 (CPU lacks the feature)\n";
    }
}

// --- 6: routing correctness under CCD restriction ---

// Values must be identical whether the pool chunks the work CCD-locally or not.
// n > threads*1024 forces pool chunking; the fp16 output (2n bytes) fits the
// LLC, so KeepWorkLocal holds and chunks are submitted via submit_to_group.
static void test_routing_correctness() {
    const auto n = size_t{40000};
    CHECK(n > static_cast<size_t>(IncludedCPUFeatures().threads) * 1024u);
    CHECK(KeepWorkLocal(n * sizeof(float16_t)));

    auto rng = uint32_t{0x5DEECE66u};
    auto values = std::vector<float32_t>(n);
    auto a = Tensor({n}, DType::Float32);
    auto b = Tensor({n}, DType::Float32);
    for (auto i = size_t{0}; i < n; ++i) {
        const auto va = std::bit_cast<float32_t>(lcg_finite_bits(rng));
        const auto vb = std::bit_cast<float32_t>(lcg_finite_bits(rng));
        values[i] = va;
        a.set<float32_t>({i}, va);
        b.set<float32_t>({i}, vb);
    }

    // Elementwise add: bit-identical to the scalar IEEE result.
    auto sum = a.add(b);
    for (auto i = size_t{0}; i < n; ++i) {
        const auto expect = values[i] + b.at<float32_t>({i});
        if (std::bit_cast<uint32_t>(sum.at<float32_t>({i})) != std::bit_cast<uint32_t>(expect)) {
            ++failures;
            std::cerr << "FAIL: routed add bit mismatch at [" << i << "] of " << n << "\n";
            break;
        }
    }

    // Tensor-level conversion: bit-identical to the scalar PERMISSIVE converter.
    auto c = Tensor({n}, DType::Float32);
    for (auto i = size_t{0}; i < n; ++i) {
        c.set<float32_t>({i}, values[i]);
    }
    c.convert_dtype(DType::Float16);
    c.sync();
    auto ignored = false;
    for (auto i = size_t{0}; i < n; ++i) {
        const auto ref = bd::unaccel_fp32_to_fp16_tracked(values[i], ignored, ConversionClampMethod::PERMISSIVE);
        if (bd::bit_cast_memcpy<uint16_t>(c.at<float16_t>({i})) != bd::bit_cast_memcpy<uint16_t>(ref)) {
            ++failures;
            std::cerr << "FAIL: routed convert_dtype bit mismatch at [" << i << "] of " << n << "\n";
            break;
        }
    }
}

// --- 7-19: newly CCD-wired op families under the restricted scheduler ---
//
// Every case below is sized past its launcher's pool-chunk gate (so the work is
// submitted to the pool at all) and lands its gated byte span on a chosen side of
// LastLevelCacheBytes() (so either submit_to_group or the unrestricted submit is
// provably taken - the gate predicates are CHECKed, not assumed). Values are
// verified against independent scalar/fp64 references, bit-exact where the
// compute contract allows it.

// Mixed-precision comparisons happen at float64 through explicit casts.
static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    return std::fabs(a - b) <= eps;
}

// One sampled element of a batched fp32 C against a k-loop fp64 reference.
static void check_matmul_element(const char* const label, Tensor& a, Tensor& b, Tensor& c,
                                 const size_t bi, const size_t i, const size_t j, const size_t k) {
    auto acc = float64_t{0};
    for (auto kk = size_t{0}; kk < k; ++kk) {
        acc += static_cast<float64_t>(a.at<float32_t>({bi, i, kk})) *
               static_cast<float64_t>(b.at<float32_t>({bi, kk, j}));
    }
    const auto got = static_cast<float64_t>(c.at<float32_t>({bi, i, j}));
    if (!near64(got, acc, 1e-3 * std::fabs(acc) + 1e-3)) {
        ++failures;
        std::cerr << "FAIL: " << label << " matmul mismatch at (" << bi << ", " << i << ", " << j
                  << "): got=" << got << " ref=" << acc << "\n";
    }
}

// Spreads `samples` flat C positions deterministically across all batches/rows/cols.
static void check_matmul_samples(const char* const label, Tensor& a, Tensor& b, Tensor& c,
                                 const size_t batch, const size_t m, const size_t n, const size_t k,
                                 const size_t samples) {
    const auto c_numel = batch * m * n;
    for (auto s = size_t{0}; s < samples; ++s) {
        const auto flat = (s * (c_numel / samples) + s) % c_numel;
        const auto bi = flat / (m * n);
        const auto i = (flat / n) % m;
        const auto j = flat % n;
        check_matmul_element(label, a, b, c, bi, i, j, k);
    }
}

// Sizes a "small" routed case for the machine running it: keep the classic
// element count while it fits half the detected LLC, clamp down on small-cache
// hardware (never below a 2x threaded-path floor), so the restricted-path
// assertion holds from the 256 KB fallback floor to gigabyte server caches.
// Valid while LLC > threads * 8 KB, which the 256 KB floor guarantees through
// 32 threads.
static size_t small_case_numel(const size_t want_elems, const size_t elem_bytes) {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    const auto cap = LastLevelCacheBytes() / 2u / elem_bytes;
    const auto floor = threads * 1024u * 2u;
    return std::max(std::min(want_elems, cap), floor);
}

static void test_routed_matmul() {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    const auto llc = LastLevelCacheBytes();

    // Restricted path: batched GEMM whose C span fits one CCD's LLC. m < the
    // per-thread chunk, so chunks straddle batch boundaries inside the worker.
    {
        constexpr size_t batch = 48;
        constexpr size_t m = 48;
        constexpr size_t k = 96;
        constexpr size_t n = 512;
        auto a = Tensor({batch, m, k}, DType::Float32);
        auto b = Tensor({batch, k, n}, DType::Float32);
        a.random(-1.0, 1.0, 1001);
        b.random(-1.0, 1.0, 1002);
        auto c = a.matmul(b);
        c.sync();

        const auto total = batch * m;
        const auto c_bytes = total * n * sizeof(float32_t);
        const auto chunk = (total + threads - 1) / threads;
        CHECK(total * n * k > size_t{32768}); // past the inline gate: pool-chunked
        CHECK(chunk > m);                     // chunks re-split at batch boundaries
        CHECK(KeepWorkLocal(c_bytes));        // restricted: submit_to_group

        check_matmul_samples("routed batched matmul", a, b, c, batch, m, n, k, 800);
        std::cout << "matmul batched 48x48x96x512 (C span " << c_bytes
                  << " B <= LLC " << llc << "): restricted path, sampled vs triple-loop reference\n";
    }

    // Unrestricted path: C span deliberately above the LLC.
    {
        constexpr size_t batch = 4;
        constexpr size_t n = 1024;
        constexpr size_t k = 40;
        const auto m = (llc + llc / 4u) / (batch * n * sizeof(float32_t));
        auto a = Tensor({batch, m, k}, DType::Float32);
        auto b = Tensor({batch, k, n}, DType::Float32);
        a.random(-1.0, 1.0, 1003);
        b.random(-1.0, 1.0, 1004);
        auto c = a.matmul(b);
        c.sync();

        const auto total = batch * m;
        const auto c_bytes = total * n * sizeof(float32_t);
        CHECK(total * n * k > size_t{32768});
        CHECK(!KeepWorkLocal(c_bytes));       // unrestricted: whole-pool submit

        check_matmul_samples("routed >LLC matmul", a, b, c, batch, m, n, k, 800);
        std::cout << "matmul batched 4x" << m << "x40x1024 (C span " << c_bytes
                  << " B > LLC " << llc << "): unrestricted path, sampled vs triple-loop reference\n";
    }
}

static void test_routed_bias_add() {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    const auto rows = small_case_numel(4000 * 128u, sizeof(float32_t)) / 128u;
    constexpr size_t cols = 128;
    auto t = Tensor({rows, cols}, DType::Float32);
    auto bias = Tensor({cols}, DType::Float32);
    t.random(-1.0, 1.0, 2001);
    bias.random(-1.0, 1.0, 2002);
    auto out = t.bias_add(bias);
    out.sync();

    const auto numel = rows * cols;
    CHECK(numel > threads * 1024u);          // past the inline gate: pool-chunked
    CHECK(KeepWorkLocal(numel * sizeof(float32_t))); // restricted

    auto bias_vals = std::vector<float32_t>(cols);
    for (auto c = size_t{0}; c < cols; ++c) {
        bias_vals[c] = bias.at<float32_t>({c});
    }
    auto bad = size_t{0};
    for (auto s = size_t{0}; s < 5000; ++s) {
        const auto i = (s * size_t{811}) % rows; // coprime strides: full row/col coverage
        const auto c = (s * size_t{131}) % cols;
        const auto expect = t.at<float32_t>({i, c}) + bias_vals[c]; // IEEE add, one rounding
        if (std::bit_cast<uint32_t>(out.at<float32_t>({i, c})) != std::bit_cast<uint32_t>(expect)) {
            if (++bad <= 5) {
                std::cerr << "FAIL: routed bias_add mismatch at (" << i << ", " << c << ")\n";
            }
        }
    }
    CHECK(bad == 0);
    std::cout << "bias_add 4000x128 (out span " << numel * sizeof(float32_t)
              << " B <= LLC): restricted path, sampled bit-exact vs broadcast sum\n";
}

static void test_routed_relu() {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    constexpr size_t n = 40000; // > threads*1024, 160 KB span: chunked + restricted
    auto x = Tensor({n}, DType::Float32);
    x.random(-1.0, 1.0, 3001);
    auto y = x.relu();
    y.sync();

    CHECK(n > threads * 1024u);
    CHECK(KeepWorkLocal(n * sizeof(float32_t)));

    auto bad = size_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        const auto v = x.at<float32_t>({i});
        const auto expect = v < 0.0f ? 0.0f : v;
        if (std::bit_cast<uint32_t>(y.at<float32_t>({i})) != std::bit_cast<uint32_t>(expect)) {
            if (++bad <= 5) {
                std::cerr << "FAIL: routed relu mismatch at [" << i << "]\n";
            }
        }
    }
    CHECK(bad == 0);
    std::cout << "relu 40000 elements (span " << n * sizeof(float32_t)
              << " B <= LLC): restricted path, full scan bit-exact vs scalar loop\n";
}

// Closed-form row loss for the planted pattern below: value p at column `col`,
// 0.0 in every other class, integer target class 1.
static float64_t planted_row_loss(const float64_t p, const size_t cols, const size_t col) {
    const auto s = static_cast<float64_t>(cols - 1u) * std::exp(-p) + 1.0;
    return p + std::log(s) - (col == size_t{1} ? p : 0.0);
}

static void run_routed_softmax_case(const size_t rows, const size_t cols, const bool expect_local, const char* const label) {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    auto logits = Tensor({rows, cols}, DType::Float32);
    auto targets = Tensor({rows}, DType::Int32);
    logits.set_zero(); // 0.0 in every class...
    targets.set_one(); // ...with target class 1 in every row
    for (auto r = size_t{0}; r < rows; ++r) {
        logits.set<float32_t>({r, r % cols}, 3.0f + static_cast<float32_t>(r % 5u));
    }
    auto out = logits.softmax_ce(targets);
    out.sync();

    auto want = float64_t{0};
    for (auto r = size_t{0}; r < rows; ++r) {
        want += planted_row_loss(3.0 + static_cast<float64_t>(r % 5u), cols, r % cols);
    }
    want /= static_cast<float64_t>(rows);

    const auto work_bytes = rows * cols * sizeof(float32_t);
    CHECK(rows > threads * 8u); // row-launcher chunk gate
    CHECK(KeepWorkLocal(work_bytes) == expect_local);
    CHECK(near64(static_cast<float64_t>(out.at<float32_t>({0})), want, 1e-3));
    std::cout << label << " softmax_ce " << rows << "x" << cols << " (work span " << work_bytes
              << " B, LLC " << LastLevelCacheBytes() << "): " << (expect_local ? "restricted" : "unrestricted")
              << " path, mean loss vs closed form\n";
}

static void test_routed_softmax_ce() {
    run_routed_softmax_case(1000, 64, true, "small");
    run_routed_softmax_case(LastLevelCacheBytes() / 4u / 64u + 4096u, 64, false, ">LLC");

    // Range error: a target outside [0, cols) is thrown by the worker and
    // rethrown at the caller-side drain, so it surfaces from softmax_ce itself.
    auto bad_logits = Tensor({300, 8}, DType::Float32); // 300 > threads*8: pool-chunked
    auto bad_targets = Tensor({300}, DType::Int32);
    bad_logits.set_zero();
    bad_targets.set_zero();
    bad_targets.set<int32_t>({0}, 99); // cols is 8
    auto threw = false;
    try {
        auto out = bad_logits.softmax_ce(bad_targets);
        static_cast<void>(out);
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_routed_equal() {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    constexpr size_t n = 40000; // mask span is n*1 byte: restricted either way
    auto a = Tensor({n}, DType::Float32);
    a.random(-1.0, 1.0, 4001);
    auto self = a.equal(a);
    self.sync();

    CHECK(n > threads * 1024u);
    CHECK(KeepWorkLocal(n)); // UInt8 output span

    auto bad = size_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        if (self.at<uint8_t>({i}) != 1) {
            if (++bad <= 5) {
                std::cerr << "FAIL: routed equal self-compare false at [" << i << "]\n";
            }
        }
    }
    CHECK(bad == 0);

    // One planted inequality: exactly one false lane.
    auto b = a.clone(true);
    constexpr size_t planted_at = 12345;
    b.set<float32_t>({planted_at}, 42.0f); // |a| <= 1, so this cannot collide
    auto mask = a.equal(b);
    mask.sync();
    CHECK(mask.at<uint8_t>({planted_at}) == 0);
    auto zeros = size_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        if (mask.at<uint8_t>({i}) == 0) {
            ++zeros;
        }
    }
    CHECK(zeros == 1);

    // Dtype mismatch keeps rejecting.
    auto other = Tensor({n}, DType::Int32);
    auto threw = false;
    try {
        auto out = a.equal(other);
        static_cast<void>(out);
    } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
        threw = true;
    }
    CHECK(threw);
    std::cout << "equal 40000 elements (mask span " << n
              << " B <= LLC): restricted path, self-compare all-true + one planted inequality\n";
}

// exact_limit pins the fp32 (2^24) / fp64 (2^40) range where integer-valued
// accumulation of ones is exact at ANY chunk order; the >LLC case uses fp64 so
// both "span above the LLC" and "exact count" hold at every LLC scale.
static void run_routed_sum_case(const size_t n, const bool expect_local, const char* const label,
                                const DType dtype, const size_t exact_limit) {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    auto ones = Tensor({n}, dtype);
    ones.set_one();
    auto total = ones.sum();
    total.sync();

    const auto in_bytes = n * (dtype == DType::Float64 ? sizeof(float64_t) : sizeof(float32_t));
    CHECK(n > threads * 1024u);
    CHECK(KeepWorkLocal(in_bytes) == expect_local);
    CHECK(n < exact_limit);
    if (dtype == DType::Float32) {
        CHECK(std::bit_cast<uint32_t>(total.at<float32_t>({0})) ==
              std::bit_cast<uint32_t>(static_cast<float32_t>(n)));
    } else {
        CHECK(std::bit_cast<uint64_t>(total.at<float64_t>({0})) ==
              std::bit_cast<uint64_t>(static_cast<float64_t>(n)));
    }
    std::cout << label << " sum of ones, numel " << n << " (input span " << in_bytes
              << " B, LLC " << LastLevelCacheBytes() << "): " << (expect_local ? "restricted" : "unrestricted")
              << " path, exact count\n";
}

static void test_routed_sum() {
    run_routed_sum_case(small_case_numel(2000000, sizeof(float32_t)), true, "small", DType::Float32, size_t{1} << 24);
    // Input span above the LLC: (LLC + LLC/2) bytes of fp64 — fp32 cannot be
    // both above a large LLC and below 2^24 elements, so the unrestricted case
    // verifies the exact count in the wider fp32->fp64... fp64 accumulator.
    run_routed_sum_case((LastLevelCacheBytes() + LastLevelCacheBytes() / 2u) / sizeof(float64_t), false, ">LLC",
                        DType::Float64, size_t{1} << 40);
}

static void run_routed_argmax_case(const size_t n, const size_t planted_at, const bool expect_local, const char* const label) {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    auto t = Tensor({n}, DType::Float32);
    t.random(-1.0, 1.0, 5001);
    t.set<float32_t>({planted_at}, 50.0f); // every other lane is in [-1, 1]
    const auto got = t.argmax();

    const auto in_bytes = n * sizeof(float32_t);
    CHECK(n > threads * 1024u);
    CHECK(KeepWorkLocal(in_bytes) == expect_local);
    CHECK(got == static_cast<uint64_t>(planted_at));
    std::cout << label << " argmax, numel " << n << " (input span " << in_bytes
              << " B, LLC " << LastLevelCacheBytes() << "): " << (expect_local ? "restricted" : "unrestricted")
              << " path, planted max at " << planted_at << "\n";
}

static void test_routed_argmax() {
    const auto small_n = small_case_numel(100000, sizeof(float32_t));
    run_routed_argmax_case(small_n, small_n / 2u + small_n / 4u, true, "small");
    // Input span above the LLC: (LLC + LLC/2) bytes of fp32.
    const auto n = (LastLevelCacheBytes() + LastLevelCacheBytes() / 2u) / sizeof(float32_t);
    run_routed_argmax_case(n, n - size_t{765433}, false, ">LLC");
}

// Mean routes through the same input-span gate as sum: chunks are CCD-restricted
// only when the INPUT fits one CCD's LLC (the output is a single scalar either way).
static void run_routed_mean_case(const size_t n, const bool expect_local, const char* const label) {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    auto ones = Tensor({n}, DType::Float32);
    ones.set_one();
    auto avg = ones.mean();
    avg.sync();

    const auto in_bytes = n * sizeof(float32_t);
    CHECK(n > threads * 1024u);
    CHECK(KeepWorkLocal(in_bytes) == expect_local);
    // Every value is exactly 1.0f and n < 2^24, so fp32 chunked accumulation
    // (any order) is exact: the mean must be exactly 1.0f.
    CHECK(n < (size_t{1} << 24));
    CHECK(std::bit_cast<uint32_t>(avg.at<float32_t>({0})) == std::bit_cast<uint32_t>(1.0f));
    std::cout << label << " mean of ones, numel " << n << " (input span " << in_bytes
              << " B, LLC " << LastLevelCacheBytes() << "): " << (expect_local ? "restricted" : "unrestricted")
              << " path, exact 1.0\n";
}

static void test_routed_mean() {
    run_routed_mean_case(small_case_numel(2000000, sizeof(float32_t)), true, "small");
    // Input span above the LLC: (LLC + LLC/2) bytes of fp32, with the classic
    // 48 MB floor so the exercised size on big-cache machines does not shrink.
    // Planted with a deterministic LCG pattern in [0.25, 0.75); the fp64
    // reference is accumulated while filling; the fp32 chunked mean differs
    // from a single-pass reference only by accumulation order.
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    const auto n = std::max(size_t{12582912},
                            (LastLevelCacheBytes() + LastLevelCacheBytes() / 2u) / sizeof(float32_t));
    auto t = Tensor({n}, DType::Float32);
    auto rng = uint32_t{0xC0FFEE1Du};
    auto ref = float64_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        const auto v = static_cast<float32_t>(0.25 + 0.5 * static_cast<float64_t>(lcg_next(rng) >> 8) / 16777216.0);
        t.set<float32_t>({i}, v);
        ref += static_cast<float64_t>(v);
    }
    auto avg = t.mean();
    avg.sync();

    const auto in_bytes = n * sizeof(float32_t);
    const auto want = ref / static_cast<float64_t>(n);
    CHECK(n > threads * 1024u);
    CHECK(!KeepWorkLocal(in_bytes));
    CHECK(near64(static_cast<float64_t>(avg.at<float32_t>({0})), want, 1e-4 * (1.0 + std::fabs(want))));
    std::cout << ">LLC mean, numel " << n << " (input span " << in_bytes << " B > LLC "
              << LastLevelCacheBytes() << "): unrestricted path, vs fp64 reference\n";
}

static void test_routed_adam_step() {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    constexpr size_t n = 40000; // > threads*1024; Adam's gate is 3 spans (w, m, v), so the
    // restricted case needs LLC >= ~1 MB; the sub-MB detection-failure floor would
    // need skip-logic (same class as the restricted matmul case below).
    auto w = Tensor({n}, DType::Float32);
    auto g = Tensor({n}, DType::Float32);
    w.random(-1.0, 1.0, 6001);
    g.random(-1.0, 1.0, 6002);
    w.sync();
    g.sync();

    auto w0 = std::vector<float32_t>(n);
    auto g0 = std::vector<float32_t>(n);
    for (auto i = size_t{0}; i < n; ++i) {
        w0[i] = w.at<float32_t>({i});
        g0[i] = g.at<float32_t>({i});
    }

    constexpr auto lr = 0.01;
    constexpr auto b1 = 0.9;
    constexpr auto b2 = 0.999;
    constexpr auto eps = 1e-8;
    w.adam_update(g, lr, b1, b2, eps); // first step on fresh moments: t = 1
    w.sync();

    CHECK(n > threads * 1024u);
    CHECK(KeepWorkLocal(3u * n * sizeof(float32_t)));

    auto bad = size_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        const auto gv = static_cast<float64_t>(g0[i]);
        const auto m = (1.0 - b1) * gv; // fp64 hand-computed mirror, t = 1
        const auto v = (1.0 - b2) * gv * gv;
        const auto step = (m / (1.0 - b1)) / (std::sqrt(v / (1.0 - b2)) + eps);
        const auto want = static_cast<float64_t>(w0[i]) - lr * step;
        if (!near64(static_cast<float64_t>(w.at<float32_t>({i})), want, 1e-4 * (1.0 + std::fabs(want)))) {
            if (++bad <= 5) {
                std::cerr << "FAIL: routed adam mismatch at [" << i << "]\n";
            }
        }
    }
    CHECK(bad == 0);
    std::cout << "adam_update one step, numel " << n << " (3 spans, "
              << 3u * n * sizeof(float32_t) << " B <= LLC): restricted path, vs hand-computed t=1 mirror\n";
}

static void test_routed_sgd_step() {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    constexpr size_t n = 40000;
    auto w = Tensor({n}, DType::Float32);
    auto g = Tensor({n}, DType::Float32);
    w.random(-1.0, 1.0, 6003);
    g.random(-1.0, 1.0, 6004);
    w.sync();
    g.sync();

    auto w0 = std::vector<float32_t>(n);
    auto g0 = std::vector<float32_t>(n);
    for (auto i = size_t{0}; i < n; ++i) {
        w0[i] = w.at<float32_t>({i});
        g0[i] = g.at<float32_t>({i});
    }

    w.sgd_update(g, 0.25); // power-of-two lr: fma and mul+add round identically
    w.sync();

    CHECK(n > threads * 1024u);
    CHECK(KeepWorkLocal(n * sizeof(float32_t)));

    auto bad = size_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        const auto expect = std::fma(-0.25f, g0[i], w0[i]);
        if (std::bit_cast<uint32_t>(w.at<float32_t>({i})) != std::bit_cast<uint32_t>(expect)) {
            if (++bad <= 5) {
                std::cerr << "FAIL: routed sgd mismatch at [" << i << "]\n";
            }
        }
    }
    CHECK(bad == 0);
    std::cout << "sgd_update one step, numel " << n << " (span " << n * sizeof(float32_t)
              << " B <= LLC): restricted path, bit-exact vs scalar fma\n";
}

// One backward through matmul -> bias_add -> relu -> mse with every intermediate
// pool-chunked (40000 elements, spans <= LLC), so the forward launchers AND the
// relu_g / bias_add_g backward launchers all take the restricted path. (sum()
// deliberately wires no node, so mse against a zero target terminates the graph.)
static void test_routed_autograd_chain() {
    constexpr size_t rows = 200;
    constexpr size_t cols = 200; // 40000 elements everywhere: > threads*1024 here
    constexpr size_t numel = rows * cols;
    auto x = Tensor({rows, cols}, DType::Float32, Device::CPU, true);
    auto w = Tensor({cols, cols}, DType::Float32, Device::CPU, true);
    auto b = Tensor({cols}, DType::Float32, Device::CPU, true);
    auto target = Tensor({rows, cols}, DType::Float32, Device::CPU, false);
    x.random(-0.5, 0.5, 7001);
    w.random(-0.5, 0.5, 7002);
    b.random(-0.5, 0.5, 7003);
    target.set_zero();

    auto z = x.matmul(w);
    auto zb = z.bias_add(b);
    auto a = zb.relu();
    auto loss = a.mse(target);
    loss.backward(DType::Float32);

    CHECK(KeepWorkLocal(numel * sizeof(float32_t)));

    // Read the leaves back and rebuild the whole forward in fp64 for the reference.
    auto xv = std::vector<float64_t>(rows * cols);
    auto wv = std::vector<float64_t>(cols * cols);
    auto bv = std::vector<float64_t>(cols);
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        xv[i] = static_cast<float64_t>(x.at<float32_t>({i / cols, i % cols}));
    }
    for (auto i = size_t{0}; i < cols * cols; ++i) {
        wv[i] = static_cast<float64_t>(w.at<float32_t>({i / cols, i % cols}));
    }
    for (auto c = size_t{0}; c < cols; ++c) {
        bv[c] = static_cast<float64_t>(b.at<float32_t>({c}));
    }

    auto zb_hat = std::vector<float64_t>(rows * cols);
    for (auto i = size_t{0}; i < rows; ++i) {
        for (auto k = size_t{0}; k < cols; ++k) {
            const auto x_ik = xv[i * cols + k];
            for (auto j = size_t{0}; j < cols; ++j) {
                zb_hat[i * cols + j] += x_ik * wv[k * cols + j];
            }
        }
    }

    // mse seed: d/da = 2*a/numel, then relu'(zb) selects on the sign of zb.
    // The borderline band marks where an fp32-vs-fp64 sign flip is possible.
    auto dz = std::vector<float64_t>(rows * cols);
    auto borderline = std::vector<bool>(rows * cols, false);
    auto borderline_rows = std::vector<bool>(rows, false);
    auto borderline_cols = std::vector<bool>(cols, false);
    for (auto i = size_t{0}; i < rows; ++i) {
        for (auto j = size_t{0}; j < cols; ++j) {
            const auto v = zb_hat[i * cols + j] + bv[j];
            if (std::fabs(v) < 1e-3) {
                borderline[i * cols + j] = true;
                borderline_rows[i] = true;
                borderline_cols[j] = true;
            }
            dz[i * cols + j] = v > 0.0 ? 2.0 * v / static_cast<float64_t>(numel) : 0.0;
        }
    }

    auto x_grad_bad = size_t{0};
    auto x_grad_checked = size_t{0};
    for (auto i = size_t{0}; i < rows; ++i) {
        if (borderline_rows[i]) {
            continue; // a relu edge in this row: fp32 mask may legitimately differ
        }
        for (auto k = size_t{0}; k < cols; ++k) {
            auto want = float64_t{0};
            for (auto j = size_t{0}; j < cols; ++j) {
                want += dz[i * cols + j] * wv[k * cols + j];
            }
            ++x_grad_checked;
            const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({i, k}));
            if (!near64(got, want, 1e-2 * std::fabs(want) + 1e-2)) {
                if (++x_grad_bad <= 5) {
                    std::cerr << "FAIL: routed autograd dx mismatch at (" << i << ", " << k
                              << "): got=" << got << " ref=" << want << "\n";
                }
            }
        }
    }
    CHECK(x_grad_checked > 0);
    CHECK(x_grad_bad == 0);

    auto w_grad_bad = size_t{0};
    auto w_grad_checked = size_t{0};
    for (auto k = size_t{0}; k < cols; ++k) {
        for (auto j = size_t{0}; j < cols; ++j) {
            if (borderline_cols[j]) {
                continue;
            }
            auto want = float64_t{0};
            for (auto i = size_t{0}; i < rows; ++i) {
                want += xv[i * cols + k] * dz[i * cols + j];
            }
            ++w_grad_checked;
            const auto got = static_cast<float64_t>(w.gradient().at<float32_t>({k, j}));
            if (!near64(got, want, 1e-2 * std::fabs(want) + 1e-2)) {
                if (++w_grad_bad <= 5) {
                    std::cerr << "FAIL: routed autograd dw mismatch at (" << k << ", " << j << ")\n";
                }
            }
        }
    }
    CHECK(w_grad_checked > 0);
    CHECK(w_grad_bad == 0);

    // dbias[c] = column sum of the deltas.
    auto b_grad_bad = size_t{0};
    for (auto j = size_t{0}; j < cols; ++j) {
        if (borderline_cols[j]) {
            continue;
        }
        auto want = float64_t{0};
        for (auto i = size_t{0}; i < rows; ++i) {
            want += dz[i * cols + j];
        }
        const auto got = static_cast<float64_t>(b.gradient().at<float32_t>({j}));
        if (!near64(got, want, 1e-5 * std::fabs(want) + 1e-7)) {
            ++b_grad_bad;
        }
    }
    CHECK(b_grad_bad == 0);
    std::cout << "autograd chain 200x200 matmul -> bias_add -> relu -> mse (all spans <= LLC):"
                 " restricted forward+backward, leaf grads vs analytical reference\n";
}

static void test_routed_fills() {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    constexpr size_t n = 40000; // > threads*1024, 160 KB span: chunked + restricted
    CHECK(n > threads * 1024u);
    CHECK(KeepWorkLocal(n * sizeof(float32_t)));

    // Fixed-value fill (set_one routes through the CCD-aware fill path).
    auto ones = Tensor({n}, DType::Float32);
    ones.set_one();
    ones.sync();
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        if (std::bit_cast<uint32_t>(ones.at<float32_t>({i})) != std::bit_cast<uint32_t>(1.0f)) {
            if (++bad <= 5) {
                std::cerr << "FAIL: routed set_one mismatch at [" << i << "]\n";
            }
        }
    }
    CHECK(bad == 0);

    // Random fill: same seed -> identical buffer (per-chunk seeding is deterministic).
    auto r1 = Tensor({n}, DType::Float32);
    auto r2 = Tensor({n}, DType::Float32);
    r1.random(0.0, 1.0, 42);
    r2.random(0.0, 1.0, 42);
    r1.sync();
    r2.sync();
    auto mismatches = size_t{0};
    auto distinct = size_t{0};
    auto prev = float32_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        const auto v1 = r1.at<float32_t>({i});
        if (std::bit_cast<uint32_t>(v1) != std::bit_cast<uint32_t>(r2.at<float32_t>({i}))) {
            ++mismatches;
        }
        if (i > 0 && std::bit_cast<uint32_t>(v1) != std::bit_cast<uint32_t>(prev)) {
            ++distinct;
        }
        if (v1 < 0.0f || v1 > 1.0f) {
            ++bad;
        }
        prev = v1;
    }
    CHECK(mismatches == 0);
    CHECK(distinct > 1000); // not a degenerate constant fill
    CHECK(bad == 0);

    // Range contract: out-of-representable-range bounds reject.
    auto threw = false;
    try {
        auto doomed = Tensor({n}, DType::Float16);
        doomed.random(0.0, 1e9, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // set_zero is a plain memset (not the routed fill path): spot-check zeros.
    auto zeros = Tensor({n}, DType::Float32);
    zeros.set_zero();
    zeros.sync();
    auto nonzero = size_t{0};
    for (auto i = size_t{0}; i < n; i += 977u) {
        if (std::bit_cast<uint32_t>(zeros.at<float32_t>({i})) != 0u) {
            ++nonzero;
        }
    }
    CHECK(nonzero == 0);
    std::cout << "fills: set_one + seeded random on 40000 elements (span " << n * sizeof(float32_t)
              << " B <= LLC): restricted path, determinism/range contracts intact\n";
}

static void test_routed_add_unrestricted() {
    const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
    // Elementwise binary op with the output span above the LLC.
    const auto n = (LastLevelCacheBytes() + LastLevelCacheBytes() / 2u) / sizeof(float32_t);
    auto a = Tensor({n}, DType::Float32);
    auto b = Tensor({n}, DType::Float32);
    a.random(-1.0, 1.0, 8001);
    b.random(-1.0, 1.0, 8002);
    auto c = a.add(b);
    c.sync();

    const auto out_bytes = n * sizeof(float32_t);
    CHECK(n > threads * 1024u);
    CHECK(!KeepWorkLocal(out_bytes));

    auto bad = size_t{0};
    constexpr size_t samples = 4096;
    for (auto s = size_t{0}; s < samples; ++s) {
        const auto i = (s * (n / samples) + s) % n;
        const auto expect = a.at<float32_t>({i}) + b.at<float32_t>({i}); // IEEE add, one rounding
        if (std::bit_cast<uint32_t>(c.at<float32_t>({i})) != std::bit_cast<uint32_t>(expect)) {
            if (++bad <= 5) {
                std::cerr << "FAIL: routed >LLC add mismatch at [" << i << "]\n";
            }
        }
    }
    CHECK(bad == 0);
    std::cout << "add " << n << " elements (out span " << out_bytes
              << " B > LLC " << LastLevelCacheBytes() << "): unrestricted path, sampled bit-exact\n";
}

// --- 20: work-stealing pool: exactly-once under 4-way submission ---

static void test_pool_exactly_once() {
    constexpr size_t task_count = 50000;
    constexpr size_t producers = 4;
    auto pool = bt::ThreadPool(producers);
    auto counts = std::vector<std::atomic<uint32_t>>(task_count);
    auto futures = std::vector<std::future<void>>(task_count); // pre-sized: producers write disjoint slots
    auto threads = std::vector<std::thread>{};
    for (auto p = size_t{0}; p < producers; p++) {
        threads.emplace_back([p, &counts, &pool, &futures] {
            for (auto i = p; i < task_count; i += producers) {
                futures[i] = pool.submit([i, &counts] {
                    counts[i].fetch_add(1, std::memory_order_relaxed);
                });
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    pool.wait_all();
    for (const auto& c : counts) {
        CHECK(c.load(std::memory_order_relaxed) == 1); // each ran exactly once
    }
    CHECK(pool.queue_size() == 0);
    std::cout << "pool exactly-once: 50000 tasks from 4 producers, all ran exactly once, queue drained\n";
}

// --- 21: single-worker FIFO by sequence ---

static void test_pool_fifo_single_worker() {
    constexpr size_t task_count = 500;
    auto pool = bt::ThreadPool(1); // one deque, no thieves: deterministic order
    auto mu = std::mutex{};
    auto order = std::vector<size_t>{};
    auto futures = std::vector<std::future<void>>{};
    futures.reserve(task_count);
    for (auto i = size_t{0}; i < task_count; i++) {
        futures.push_back(pool.submit([i, &mu, &order] {
            const auto lock = std::lock_guard{mu};
            order.push_back(i);
        }));
    }
    pool.wait_all();
    CHECK(order.size() == task_count);
    for (auto i = size_t{0}; i < order.size(); i++) {
        CHECK(order[i] == i); // FIFO within one priority level on a single worker
    }
    std::cout << "pool FIFO: 500 tasks on a 1-thread pool completed in exact submission order\n";
}

// --- 22: HelpWait joins children while the pool is saturated (no deadlock) ---

static void test_helping_wait_no_deadlock() {
    auto& pool = bt::GetCPUThreadPool();
    // Saturate first: 32 long fillers occupy every worker.
    auto fillers = std::vector<std::shared_future<void>>{};
    for (auto i = size_t{0}; i < 32; i++) {
        fillers.push_back(pool.submit([] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }).share());
    }
    constexpr size_t child_count = 64;
    auto child_runs = std::vector<std::atomic<uint32_t>>(child_count);
    const auto parent = pool.submit([&pool, &child_runs] {
        auto children = std::vector<std::shared_future<void>>{};
        for (auto i = size_t{0}; i < child_count; i++) {
            children.push_back(pool.submit([i, &child_runs] {
                child_runs[i].fetch_add(1, std::memory_order_relaxed);
            }).share());
        }
        pool.HelpWait(children); // must run children itself, not block on them
    }).share();
    parent.get(); // a plain blocking f.get() in the parent would deadlock here
    for (const auto& f : fillers) {
        f.wait();
    }
    auto total = uint32_t{0};
    for (const auto& c : child_runs) {
        CHECK(c.load(std::memory_order_relaxed) == 1);
        total += c.load(std::memory_order_relaxed);
    }
    CHECK(total == child_count);
    std::cout << "helping wait: parent joined 64 children under 32 saturated fillers, no deadlock\n";
}

// --- 23: nested group pins keep the outer pin and the user thread intact ---

static void test_nested_group_pin() {
#if defined(__linux__)
    const CpuCcdGroups& topology = IncludedCcdTopology();
    auto& pool = bt::GetCPUThreadPool();
    auto before = cpu_set_t{};
    CPU_ZERO(&before);
    CHECK(sched_getaffinity(0, sizeof(before), &before) == 0);

    auto child_cpu = std::atomic<uint32_t>{std::numeric_limits<uint32_t>::max()};
    auto child_ran = std::atomic<bool>{false};
    // Group-0 parent; group-1 child (an out-of-range index on single-group
    // topologies degrades to plain execution by contract — completion is
    // asserted either way).
    const auto parent = pool.submit_to_group([&pool, &child_cpu, &child_ran] {
        const auto child = pool.submit_to_group([&child_cpu, &child_ran] {
            child_cpu.store(static_cast<uint32_t>(sched_getcpu()), std::memory_order_relaxed);
            child_ran.store(true, std::memory_order_relaxed);
        }, uint32_t{1}).share();
        pool.HelpWait(child);
    }, uint32_t{0}).share();
    parent.get();
    CHECK(child_ran.load());
    if (topology.groups.size() > 1) {
        const auto cpu = child_cpu.load();
        const auto& members = topology.groups[1];
        CHECK(std::find(members.begin(), members.end(), cpu) != members.end());
    }

    auto after = cpu_set_t{};
    CPU_ZERO(&after);
    CHECK(sched_getaffinity(0, sizeof(after), &after) == 0);
    CHECK(CPU_EQUAL(&before, &after)); // submitting thread's affinity untouched
    std::cout << "nested group pin: group-0 parent helped a group-1 child, pins and caller affinity intact\n";
#else
    std::cout << "nested group pin test skipped (non-Linux)\n";
#endif
}

// --- 24: set_zero paths: routed chunk, inline, and in-worker fallback ---

static void test_set_zero_paths() {
    // 4 MB of fp32: past the 256 KB inline threshold, so the chunk path runs.
    const auto n_big = (size_t{4} * 1024u * 1024u) / sizeof(float32_t);
    CHECK(n_big * sizeof(float32_t) > size_t{256} * 1024u);

    // (a) user thread, pool-drained CCD-routed chunk path.
    auto big = Tensor({n_big}, DType::Float32);
    big.random(-1.0, 1.0, 9101);
    big.sync();
    big.set_zero();
    auto nonzero = size_t{0};
    for (auto i = size_t{0}; i < n_big; i += 4099u) {
        if (std::bit_cast<uint32_t>(big.at<float32_t>({i})) != 0u) {
            ++nonzero;
        }
    }
    CHECK(nonzero == 0);

    // (b) small tensor: inline path.
    auto small = Tensor({1024}, DType::Float32);
    small.set_one();
    small.sync();
    small.set_zero();
    for (auto i = size_t{0}; i < 1024; i++) {
        CHECK(std::bit_cast<uint32_t>(small.at<float32_t>({i})) == 0u);
    }

    // (c) inside a pool worker: inline fallback (the worker must not submit
    // chunk work back into the pool). The tensor carries no pending futures —
    // sync() inside a worker remains a documented deadlock.
    auto worker_t = Tensor({n_big}, DType::Float32);
    for (auto i = size_t{0}; i < n_big; i += 8191u) {
        worker_t.set<float32_t>({i}, 7.0f);
    }
    auto& pool = bt::GetCPUThreadPool();
    auto done = std::atomic<bool>{false};
    const auto fut = pool.submit([&worker_t, &done] {
        worker_t.set_zero();
        done.store(true, std::memory_order_relaxed);
    }).share();
    fut.get();
    CHECK(done.load());
    for (auto i = size_t{0}; i < n_big; i += 4099u) {
        CHECK(std::bit_cast<uint32_t>(worker_t.at<float32_t>({i})) == 0u);
    }
    std::cout << "set_zero: 4 MB routed chunk path, 4 KB inline path, in-worker inline fallback — all zero\n";
}

// --- 25: snapshot of an idle local pool ---

static void test_pool_snapshot_idle() {
    auto pool = bt::ThreadPool(4);
    pool.wait_all(); // empty: returns immediately
    const auto snap = pool.snapshot();
    CHECK(snap.worker_count == 4);
    CHECK(snap.workers.size() == 4);
    CHECK(snap.pending_tasks_total == 0);
    CHECK(!snap.stopping);
    CHECK(snap.snapshot_ms > 0);
    for (const auto& w : snap.workers) {
        CHECK(w.status == bt::WorkerStatus::Idle);
        CHECK(w.pinned_group == bt::no_pinned_group);
        CHECK(w.task_sequence == 0);
        CHECK(w.task_running_ms == 0);
        CHECK(w.futures_awaited == 0);
        CHECK(w.tasks_completed == 0);
        CHECK(w.tasks_stolen == 0);
        CHECK(w.tasks_helped == 0);
        CHECK(w.local_queue_depth == 0);
    }
    const auto text = bt::format_snapshot(snap);
    CHECK(!text.empty());
    std::cout << "snapshot idle: 4 workers all Idle with zero counters and zero depths\n";
}

// --- 26: snapshot under saturation observes HelpWaiting ---

static void test_pool_snapshot_under_load() {
    auto& pool = bt::GetCPUThreadPool();
    const auto worker_count = static_cast<size_t>(IncludedCPUFeatures().threads);

    // Fillers saturate the pool (more than workers, so some stay pending); each
    // records its worker via the TLS index.
    auto filler_per_worker = std::vector<std::atomic<uint32_t>>(worker_count);
    auto fillers = std::vector<std::shared_future<void>>{};
    for (auto i = size_t{0}; i < 40; i++) {
        fillers.push_back(pool.submit([&filler_per_worker] {
            filler_per_worker[bt::detail::worker_index].fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }).share());
    }

    // First sample while the fillers saturate the pool. Settle first:
    // task_running_ms is whole milliseconds, and a worker claimed its filler
    // <1ms ago still reports 0.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto first_snap = pool.snapshot();

    for (const auto& f : fillers) {
        f.wait();
    }

    // Filler identity via the TLS worker index: every recorded index in range,
    // every filler ran exactly once.
    auto filler_total = uint32_t{0};
    for (const auto& c : filler_per_worker) {
        filler_total += c.load(std::memory_order_relaxed);
    }
    CHECK(filler_total == 40);

    CHECK(!first_snap.stopping);
    CHECK(first_snap.pending_tasks_total > 0); // 40 fillers on fewer workers, some still queued
    auto busy_first = size_t{0};
    for (const auto& w : first_snap.workers) {
        const bool running = w.status == bt::WorkerStatus::RunningOwn ||
                             w.status == bt::WorkerStatus::RunningStolen ||
                             w.status == bt::WorkerStatus::RunningHelped;
        if (running) {
            ++busy_first;
            CHECK(w.task_running_ms > 0); // a busy worker reports live run time
            CHECK(w.task_sequence > 0);
        }
    }
    CHECK(busy_first > 0); // the fillers keep the workers busy

    // HelpWaiting observation, made deterministic: the parent waits on a
    // non-pool gate future, so its deque is empty and nothing is stealable and
    // it parks in the HelpWaiting 100us sleep loop. The pre-fix version had the
    // parent help its own children, which flickers through microsecond-scale
    // HelpWaiting windows between 30ms helped tasks — a 5ms sampler can
    // legitimately miss every one (soaked ~1/3000 runs), which is a test race,
    // not a pool bug.
    auto gate_prom = std::make_shared<std::promise<void>>();
    auto gate = gate_prom->get_future().share();
    const auto parent = pool.submit([&pool, &gate] {
        pool.HelpWait(gate);
    }).share();

    auto observed_help_waiting = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (
        parent.wait_for(std::chrono::milliseconds(5)) != std::future_status::ready &&
        std::chrono::steady_clock::now() < deadline
    ) {
        const auto snap = pool.snapshot();
        for (const auto& w : snap.workers) {
            if (w.status == bt::WorkerStatus::HelpWaiting && w.futures_awaited > 0) {
                observed_help_waiting = true;
            }
        }
        if (observed_help_waiting) {
            break;
        }
    }
    gate_prom->set_value();
    parent.get();
    CHECK(observed_help_waiting); // sampled the parent's HelpWaiting state

    const auto text = bt::format_snapshot(first_snap);
    CHECK(!text.empty());
    CHECK(text.find("Running") != std::string::npos || text.find("HelpWaiting") != std::string::npos);
    const auto newlines = static_cast<size_t>(std::count(text.begin(), text.end(), '\n'));
    CHECK(newlines == first_snap.workers.size() + 1); // header + one line per worker
    std::cout << "snapshot under load (first sample):\n" << text;
    std::cout << "snapshot under load: pending>0, workers Running, HelpWaiting awaited>0 observed\n";
}

// --- 27: exact completed counters on a single-worker pool ---

static void test_pool_snapshot_counters() {
    constexpr size_t task_count = 1000;
    auto pool = bt::ThreadPool(1); // single worker: no stealing, deterministic counters
    auto futures = std::vector<std::future<void>>{};
    futures.reserve(task_count);
    for (auto i = size_t{0}; i < task_count; i++) {
        futures.push_back(pool.submit([] {}));
    }
    pool.wait_all();
    std::this_thread::sleep_for(std::chrono::seconds(1)); // settle
    const auto snap = pool.snapshot();
    CHECK(snap.pending_tasks_total == 0);
    for (const auto& w : snap.workers) {
        CHECK(w.status == bt::WorkerStatus::Idle);
        CHECK(w.tasks_completed == task_count);
        CHECK(w.tasks_stolen == 0);
        CHECK(w.tasks_helped == 0);
        CHECK(w.local_queue_depth == 0);
    }
    std::cout << "snapshot counters: 1000 tasks on a 1-worker pool -> tasks_completed == 1000 exactly\n";
}

void test_pool_monitor_render() {
    bt::PoolSnapshot snap{};
    snap.snapshot_ms = 12345;
    snap.worker_count = 4;
    snap.pending_tasks_total = 7;
    snap.stopping = false;
    for (uint32_t i = 0; i < 4; ++i) {
        bt::WorkerSnapshot w{};
        w.worker_id = i;
        w.status = i == 0 ? bt::WorkerStatus::HelpWaiting : (i == 1 ? bt::WorkerStatus::RunningStolen : bt::WorkerStatus::Idle);
        w.pinned_group = i == 1 ? 0 : bt::no_pinned_group;
        w.task_running_ms = i == 0 ? 42 : (i == 1 ? 20000 : 0);
        w.futures_awaited = i == 0 ? 3 : 0;
        w.tasks_completed = 100 + i;
        w.tasks_stolen = i;
        w.tasks_helped = i * 2;
        w.local_queue_depth = i == 1 ? 2 : 0;
        snap.workers.push_back(w);
    }
    const auto frame = bt::render_dashboard(snap, 10'000);
    CHECK(frame.find("BeanTensor pool") != std::string::npos);
    CHECK(frame.find("pending=7") != std::string::npos);
    CHECK(frame.find("WAIT") != std::string::npos);
    CHECK(frame.find("STEAL") != std::string::npos);
    CHECK(frame.find("a3") != std::string::npos);      // awaited futures shown on the WAIT cell
    CHECK(frame.find("G0") != std::string::npos);      // pin marker
    CHECK(frame.find("\x1b[41m") != std::string::npos); // stuck worker over threshold gets red background
    std::cout << "monitor render: fabricated snapshot -> header, statuses, awaited/pin markers, stuck flag all present\n";
}

int main() {
    std::cout << "PlatformTopologyTests" << std::endl;
    test_llc_and_threshold();
    test_ccd_topology();
    test_submit_to_group();
    test_keep_work_local();
    test_fp32_to_fp16_boundary();
    test_routing_correctness();
    test_routed_matmul();
    test_routed_bias_add();
    test_routed_relu();
    test_routed_softmax_ce();
    test_routed_equal();
    test_routed_sum();
    test_routed_argmax();
    test_routed_mean();
    test_routed_adam_step();
    test_routed_sgd_step();
    test_routed_autograd_chain();
    test_routed_fills();
    test_routed_add_unrestricted();
    test_pool_exactly_once();
    test_pool_fifo_single_worker();
    test_helping_wait_no_deadlock();
    test_nested_group_pin();
    test_set_zero_paths();
    test_pool_snapshot_idle();
    test_pool_snapshot_under_load();
    test_pool_snapshot_counters();
    test_pool_monitor_render();
    if (failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << failures << " TEST(S) FAILED" << std::endl;
    return 1;
}
