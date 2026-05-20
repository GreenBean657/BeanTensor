#pragma once
// NPU/NPUBackend.hpp
// BeanTensor UCA — XDNA2 NPU backend interface.
// Include this from your UCA dispatch code.

#include <cstdint>
#include <memory>
#include <vector>

namespace BeanTensor {

    class NPUBackend {
    public:
        NPUBackend();
        ~NPUBackend();

        NPUBackend(const NPUBackend&)            = delete;
        NPUBackend& operator=(const NPUBackend&) = delete;
        NPUBackend(NPUBackend&&)                 = default;

        // BF16 FMA: out[i] = a[i] * b[i] + c[i]
        // All buffers: N bfloat16 elements as raw uint16_t.
        // N must be a multiple of 16.
        void fma_bf16(
            const uint16_t* a,
            const uint16_t* b,
            const uint16_t* c,
            uint16_t*       out,
            int             N);

        // Returns false on systems without XDNA2 NPU.
        static bool is_available() noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        void ensure_loaded();
    };

} // namespace BeanTensor