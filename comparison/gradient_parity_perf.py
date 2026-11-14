#!/usr/bin/env python3
"""PyTorch comparison column for the gradient parity performance harness.

Reads the shape plan emitted by `BeanTensorV5Benchmarks --emit-shapes` (default
benchmark/shapes.txt) and times one forward + one backward pass for the same
families and sizes, emitting CSV rows that match `benchmark/gradient_parity_perf.cpp`.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def timed(fn, reps, device):
    fn()  # warmup
    samples = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def run_elementwise(op, dims, device, reps):
    (n,) = dims
    a = torch.randn(n, device=device, requires_grad=True)
    b = torch.randn(n, device=device, requires_grad=True)
    if op == "add":
        return timed(lambda: (a + b).sum().backward(), reps, device)
    return timed(lambda: (a * b).sum().backward(), reps, device)


def run_gemm(_op, dims, device, reps):
    m, k, n = dims
    a = torch.randn(m, k, device=device, requires_grad=True)
    b = torch.randn(k, n, device=device, requires_grad=True)
    return timed(lambda: (a @ b).sum().backward(), reps, device)


def run_reduction(op, dims, device, reps):
    if op != "sum":
        return None
    rows, cols = dims
    a = torch.randn(rows, cols, device=device, requires_grad=True)
    return timed(lambda: a.sum().backward(), reps, device)


def run_loss(_op, dims, device, reps):
    rows, classes = dims
    logits = torch.randn(rows, classes, device=device, requires_grad=True)
    labels = torch.randint(0, classes, (rows,), device=device)
    return timed(lambda: F.cross_entropy(logits, labels).backward(), reps, device)


def run_convolution(_op, dims, device, reps):
    n, cin, side, _unused, cout, k1, k2 = dims
    x = torch.randn(n, cin, side, side, device=device, requires_grad=True)
    w = torch.randn(cout, cin, k1, k2, device=device, requires_grad=True)
    bias = torch.randn(cout, device=device, requires_grad=True)
    return timed(
        lambda: F.conv2d(x, w, bias, stride=1, padding=0).sum().backward(),
        reps,
        device,
    )


def run_spatial(_op, dims, device, reps):
    n, c, side, _ = dims
    x = torch.randn(n, c, side, side, device=device, requires_grad=True)
    return timed(
        lambda: F.max_pool2d(x, 2).sum().backward(),
        reps,
        device,
    )


def run_normalization(_op, dims, device, reps):
    n, c, side, _ = dims
    x = torch.randn(n, c, side, side, device=device, requires_grad=True)
    gamma = torch.randn(c, device=device, requires_grad=True)
    beta = torch.randn(c, device=device, requires_grad=True)
    return timed(
        lambda: F.batch_norm(
            x, running_mean=None, running_var=None, weight=gamma, bias=beta,
            training=True, momentum=0.1, eps=1e-5,
        ).sum().backward(),
        reps,
        device,
    )


def run_shape(family, op, dims, device, reps):
    if family == "elementwise":
        return run_elementwise(op, dims, device, reps)
    if family == "gemm":
        return run_gemm(op, dims, device, reps)
    if family == "reduction":
        return run_reduction(op, dims, device, reps)
    if family == "loss":
        return run_loss(op, dims, device, reps)
    if family == "convolution":
        return run_convolution(op, dims, device, reps)
    if family == "spatial":
        return run_spatial(op, dims, device, reps)
    if family == "normalization":
        return run_normalization(op, dims, device, reps)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="benchmark/shapes.txt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "gpu":
        device_name = "cuda"
    device = torch.device(device_name)
    reps = 1 if args.smoke else args.reps

    shapes_path = Path(args.shapes)
    if not shapes_path.exists():
        print(f"shapes file not found: {shapes_path}", file=sys.stderr)
        return 1

    torch.manual_seed(7)
    with open(shapes_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0] != "shape":
                continue
            _, family, op, *dim_strs = parts
            dims = [int(d) for d in dim_strs]
            median_ms = run_shape(family, op, dims, device, reps)
            if median_ms is None:
                continue
            print(f"{family},{op},{'x'.join(dim_strs)},{median_ms:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
