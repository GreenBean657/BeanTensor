//===- fma.cc - BF16 FMA kernel for XDNA2 (AIE2P / Strix Halo) -----------===//
//
// BeanTensor UCA — NPU FMA kernel
// Computes: out[i] = abc[i] * abc[N+i] + abc[2*N+i]  (element-wise BF16)
//
// Input layout:  packed buffer [a0..aN-1 | b0..bN-1 | c0..cN-1]
// Output:        [out0..outN-1]
// N:             runtime parameter. Must be a multiple of VEC (16)
//                Must be a multiple of VEC (16). Currently 1024.
//
// Packed-input layout is required because CoreTile has only 2 input DMA
// channels; we use 1 (packed input) + 1 (output). The host is responsible
// for packing before dispatch.
//
// NOTE on pipelining: AIE_PREPARE_FOR_PIPELINING is intentionally NOT used
// here. With 3 live bf16 vector loads per iteration the AIE2P scheduler
// produces incorrect register liveness and the loop emits inf/nan. If you
// later split the packed input into separate ObjectFifos (via a MemTile
// stage) each load has its own stream and the pragma can be reintroduced.
//
// Compiled by Peano: clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf
//                    -DNDEBUG -I${MLIR_AIE_INSTALL}/include
//                    -I${MLIR_AIE_SRC}/aie_kernels
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"

// ── Public C entry point ──────────────────────────────────────────────────────
extern "C" {

void fma_bf16_kernel(bfloat16 *__restrict abc_in,
                     bfloat16 *__restrict out,
                     int32_t N)
{
    constexpr int     VEC = 16;

    bfloat16 *__restrict pA   = abc_in;
    bfloat16 *__restrict pB   = abc_in + N;
    bfloat16 *__restrict pC   = abc_in + 2 * N;
    bfloat16 *__restrict pOut = out;

    for (int i = 0; i < N / VEC; i++) {
        aie::vector<bfloat16, VEC> va = aie::load_v<VEC>(pA);  pA += VEC;
        aie::vector<bfloat16, VEC> vb = aie::load_v<VEC>(pB);  pB += VEC;
        aie::vector<bfloat16, VEC> vc = aie::load_v<VEC>(pC);  pC += VEC;

        // a*b in accumulator, narrow to bf16, then add c.
        aie::accum<accfloat, VEC>  acc   = aie::mul(va, vb);
        aie::vector<bfloat16, VEC> vprod = acc.template to_vector<bfloat16>();
        aie::vector<bfloat16, VEC> vout  = aie::add(vprod, vc);

        aie::store_v(pOut, vout);  pOut += VEC;
    }
}

} // extern "C"