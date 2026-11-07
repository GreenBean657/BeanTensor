#!/usr/bin/env python3
"""PyTorch comparison column for the op-level benchmark harness.

Reads the shape plan emitted by `benchmark --emit-shapes` (default
benchmark/shapes.txt) and times the identical ops at the identical sizes, so
docs/benchmarks.md can put BeanTensor and PyTorch numbers side by side.
Intentionally dependency-free beyond torch: parsing the shape file replaces
sharing size logic across languages.

Run: comparison/.venv/bin/python comparison/bench_pytorch.py [shapes-file]
"""
import sys
import time

import torch

REPS = 5
WARMUP = 1


def timed(fn):
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPS):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def run_shape(family, op, dims):
    if family == "elementwise":
        (n,) = dims
        a = torch.randn(n)
        b = torch.randn(n)
        if op == "add":
            return timed(lambda: a + b)
        return timed(lambda: a * b)
    if family == "gemm":
        m, k, n = dims
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        return timed(lambda: a @ b)
    if family == "reduction":
        rows, cols = dims
        a = torch.randn(rows, cols)
        if op == "sum":
            return timed(lambda: a.sum())
        if op == "max":
            return timed(lambda: a.max())
        return timed(lambda: a.argmax(dim=1))
    if family == "loss":
        rows, classes = dims
        logits = torch.randn(rows, classes)
        labels = torch.randint(0, classes, (rows,))
        fn = torch.nn.CrossEntropyLoss()
        return timed(lambda: fn(logits, labels))
    if family == "convolution":
        n, cin, side, _unused, cout, _k1, _k2 = dims
        x = torch.randn(n, cin, side, side)
        w = torch.randn(cout, cin, 3, 3)
        bias = torch.randn(cout)
        return timed(lambda: torch.nn.functional.conv2d(x, w, bias, stride=1, padding=0))
    if family == "spatial":
        n, c, side, _ = dims
        x = torch.randn(n, c, side, side)
        return timed(lambda: torch.nn.functional.max_pool2d(x, 2))
    if family == "normalization":
        n, c, side, _ = dims
        x = torch.randn(n, c, side, side)
        gamma = torch.randn(c)
        beta = torch.randn(c)
        return timed(lambda: torch.nn.functional.batch_norm(
            x, running_mean=None, running_var=None, weight=gamma, bias=beta,
            training=True, momentum=0.1, eps=1e-5))
    raise ValueError(f"unknown shape line: {family} {op} {dims}")


def main():
    shapes_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark/shapes.txt"
    torch.manual_seed(7)
    with open(shapes_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0] != "shape":
                continue
            _, family, op, *dim_strs = parts
            dims = [int(d) for d in dim_strs]
            median_ms = run_shape(family, op, dims)
            print(f"{family},{op},{'x'.join(dim_strs)},{median_ms:.4f}")


if __name__ == "__main__":
    main()
