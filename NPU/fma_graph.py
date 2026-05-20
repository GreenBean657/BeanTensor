# NPU/fma_graph.py
# Outputs MLIR to stdout for aiecc.
# Usage: python3 fma_graph.py -d npu2 -n 1024 > fma.mlir
#
# Input layout: packed buffer [a | b | c] (3*N bfloat16 elements)
# Output:       [out] (N bfloat16 elements)
# This uses only 2 DMA channels (1 in, 1 out) to fit CoreTile constraints.

import numpy as np
import argparse

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_


def fma_bf16_design(dev, n: int):
    """BF16 FMA: out = a * b + c, packed input [a|b|c], N elements each."""

    dtype     = np.uint16                                    # bfloat16 as uint16
    in_ty     = np.ndarray[(3 * n,), np.dtype[dtype]]        # packed [a|b|c]
    out_ty    = np.ndarray[(n,),     np.dtype[dtype]]        # output

    # ── 1 input FIFO + 1 output FIFO = 2 DMA channels, fits CoreTile ─────────
    of_in  = ObjectFifo(in_ty,  name="in_abc")
    of_out = ObjectFifo(out_ty, name="out")

    # ── Kernel: maps to fma_bf16_kernel(abc_in, out, N) in fma.cc ─────────────
    fma_kernel = Kernel(
        "fma_bf16_kernel",
        "fma.o",
        [in_ty, out_ty, np.int32],          # no more N
    )

    # ── Worker ────────────────────────────────────────────────────────────────
    def core_fn(of_in, of_out, fma_kernel, N):
        elem_in  = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        fma_kernel(elem_in, elem_out, N)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod(), fma_kernel, n])

    # ── Runtime sequence ──────────────────────────────────────────────────────
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (abc, o):
        rt.start(worker)
        rt.fill(of_in.prod(), abc)
        rt.drain(of_out.cons(), o, wait=True)

    print(Program(dev, rt).resolve_program())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--device",  default="npu2",
                   choices=["npu", "npu2"])
    p.add_argument("-n", "--n-elems", type=int, default=1024,
                   help="Elements per section (N); total input = 3*N uint16")
    args = p.parse_args()

    dev = NPU2() if args.device == "npu2" else NPU1Col1()
    fma_bf16_design(dev, args.n_elems)