// NPU/NPUBackend.cpp
// BeanTensor UCA — XDNA2 NPU backend implementation.

#include "NPUBackend.hpp"

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace BeanTensor {

// N is baked into the xclbin at AOT compile time (fma_graph.py -n 1024).
static constexpr int FMA_N = 1024;

struct NPUBackend::Impl {
    xrt::device      device;
    xrt::xclbin      xclbin_obj;
    xrt::hw_context  context;
    xrt::kernel      kernel;
    std::vector<uint32_t> insts;

    // Cached BOs — allocated once on first call, reused thereafter.
    xrt::bo bo_instr;
    xrt::bo bo_in;
    xrt::bo bo_out;
    uint16_t* mapped_in  = nullptr;
    uint16_t* mapped_out = nullptr;

    bool loaded     = false;
    bool bos_ready  = false;
};

static std::vector<uint32_t> load_insts(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open instruction file: " + path);
    f.seekg(0, std::ios::end);
    const auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz % 4 != 0)
        throw std::runtime_error("Instruction file size not a multiple of 4");
    std::vector<uint32_t> v(sz / 4);
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

NPUBackend::NPUBackend()
    : impl_(std::make_unique<Impl>())
{
    impl_->device = xrt::device(0);
}

NPUBackend::~NPUBackend() = default;

void NPUBackend::ensure_loaded()
{
    if (impl_->loaded) return;

    const std::string xclbin_path = FMA_XCLBIN_PATH;
    const std::string insts_path  = FMA_INSTS_PATH;

    if (!std::filesystem::exists(xclbin_path))
        throw std::runtime_error("fma.xclbin not found: " + xclbin_path);
    if (!std::filesystem::exists(insts_path))
        throw std::runtime_error("fma_insts.bin not found: " + insts_path);

    impl_->xclbin_obj = xrt::xclbin(xclbin_path);
    impl_->device.register_xclbin(impl_->xclbin_obj);
    impl_->context = xrt::hw_context(impl_->device, impl_->xclbin_obj.get_uuid());
    impl_->kernel  = xrt::kernel(impl_->context, "MLIR_AIE");
    impl_->insts   = load_insts(insts_path);
    impl_->loaded  = true;
}

void NPUBackend::fma_bf16(
    const uint16_t* a,
    const uint16_t* b,
    const uint16_t* c,
    uint16_t*       out,
    int             N)
{
    ensure_loaded();

    if (N != FMA_N)
        throw std::invalid_argument(
            "fma_bf16: N must be 1024 (xclbin AOT-compiled for N=1024); "
            "recompile fma_graph.py with -n <desired_N> to change this");

    constexpr size_t elem_bytes = FMA_N * sizeof(uint16_t);
    constexpr size_t in_bytes   = 3 * elem_bytes;

    auto &k = impl_->kernel;

    // ── One-time BO allocation ────────────────────────────────────────────────
    if (!impl_->bos_ready) {
        impl_->bo_instr = xrt::bo(impl_->device,
                                  impl_->insts.size() * sizeof(uint32_t),
                                  XCL_BO_FLAGS_CACHEABLE,
                                  k.group_id(1));
        impl_->bo_in    = xrt::bo(impl_->device, in_bytes,
                                  XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
        impl_->bo_out   = xrt::bo(impl_->device, elem_bytes,
                                  XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));

        std::memcpy(impl_->bo_instr.map<void*>(),
                    impl_->insts.data(),
                    impl_->insts.size() * sizeof(uint32_t));
        impl_->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        impl_->mapped_in  = impl_->bo_in.map<uint16_t*>();
        impl_->mapped_out = impl_->bo_out.map<uint16_t*>();
        impl_->bos_ready  = true;
    }

    // ── Pack [a | b | c] into the single input BO ─────────────────────────────
    std::memcpy(impl_->mapped_in,             a, elem_bytes);
    std::memcpy(impl_->mapped_in + FMA_N,     b, elem_bytes);
    std::memcpy(impl_->mapped_in + 2 * FMA_N, c, elem_bytes);
    impl_->bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ── Dispatch ──────────────────────────────────────────────────────────────
    auto run = k(3u,
                 impl_->bo_instr,
                 static_cast<uint32_t>(impl_->insts.size()),
                 impl_->bo_in,
                 impl_->bo_out);
    const auto state = run.wait(30000);  // ms
    if (state != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("NPU FMA kernel timed out or errored");

    impl_->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(out, impl_->mapped_out, elem_bytes);
}

bool NPUBackend::is_available() noexcept
{
    try   { xrt::device dev(0); return true; }
    catch (...) { return false; }
}

} // namespace BeanTensor