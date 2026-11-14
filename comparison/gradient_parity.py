#!/usr/bin/env python3
"""PyTorch side of the gradient parity harness.

Reads the manifest and raw little-endian tensor bytes emitted by
comparison/gradient_parity.cpp, rebuilds the identical autograd graphs in
PyTorch, and compares every gradient against the C++ reference bytes. Nonzero
exit on the first failed case after printing max abs/rel error and the worst
element.
"""
import argparse
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DTYPE_MAP = {
    "Float32": (np.float32, torch.float32),
    "Float64": (np.float64, torch.float64),
    "Int32": (np.int32, torch.int32),
    "Int64": (np.int64, torch.int64),
}


def load_tensor(path: str, dtype_name: str, shape: Tuple[int, ...],
                device: torch.device) -> torch.Tensor:
    np_type, torch_dtype = DTYPE_MAP[dtype_name]
    np_dtype = np.dtype(np_type).newbyteorder("<")
    raw = np.fromfile(path, dtype=np_dtype)
    expected = int(np.prod(shape, dtype=np.int64))
    if raw.size != expected:
        raise ValueError(f"{path}: expected {expected} elements, found {raw.size}")
    return torch.from_numpy(raw.reshape(shape).copy()).to(torch_dtype).to(device)


def parse_manifest(manifest_path: Path) -> List[dict]:
    cases: List[dict] = []
    current: dict = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "case":
                if current:
                    cases.append(current)
                current = {
                    "name": parts[1],
                    "grad_dtype": parts[2],
                    "abs_tol": float(parts[3]),
                    "rel_tol": float(parts[4]),
                    "tensors": {},
                }
            elif parts[0] == "tensor":
                if not current:
                    raise ValueError("tensor line before first case")
                name = parts[1]
                dtype_name = parts[2]
                shape_end = len(parts) - 1
                shape = tuple(int(d) for d in parts[3:shape_end])
                filename = parts[-1]
                current["tensors"][name] = (dtype_name, shape, filename)
            else:
                raise ValueError(f"unknown manifest line: {line}")
    if current:
        cases.append(current)
    return cases


def _is_both_nan(a: float, b: float) -> bool:
    return math.isnan(a) and math.isnan(b)


def _is_both_inf(a: float, b: float) -> bool:
    return math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0)


def compare_gradients(ref: np.ndarray, actual: np.ndarray, abs_tol: float,
                      rel_tol: float) -> Tuple[bool, str]:
    if ref.shape != actual.shape:
        return False, f"shape mismatch: {ref.shape} vs {actual.shape}"
    if ref.dtype != actual.dtype:
        return False, f"dtype mismatch: {ref.dtype} vs {actual.dtype}"

    flat_ref = ref.ravel()
    flat_actual = actual.ravel()
    max_abs = 0.0
    max_rel = 0.0
    worst_idx = 0
    for i in range(flat_ref.size):
        r = float(flat_ref[i])
        a = float(flat_actual[i])
        if _is_both_nan(r, a):
            continue
        if _is_both_inf(r, a):
            continue
        if math.isnan(r) != math.isnan(a) or math.isinf(r) != math.isinf(a):
            return False, (
                f"NaN/Inf mismatch at index {i}: ref={r} actual={a}"
            )
        abs_err = abs(r - a)
        rel_err = abs_err / max(abs(r), 1e-12)
        if abs_err > max_abs or rel_err > max_rel:
            max_abs = abs_err
            max_rel = rel_err
            worst_idx = i

    if max_abs > abs_tol and max_rel > rel_tol:
        return False, (
            f"worst element at flat index {worst_idx}: ref={flat_ref[worst_idx]} "
            f"actual={flat_actual[worst_idx]} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} (tol abs={abs_tol:.6e} rel={rel_tol:.6e})"
        )
    return True, f"max_abs={max_abs:.6e} max_rel={max_rel:.6e}"


def _tensors_for_case(case: dict, data_dir: Path,
                      device: torch.device) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for name, (dtype_name, shape, rel_path) in case["tensors"].items():
        tensors[name] = load_tensor(str(data_dir / rel_path), dtype_name, shape, device)
    return tensors


def run_case(case: dict, data_dir: Path, device: torch.device,
             perturb: str) -> Tuple[bool, str]:
    name = case["name"]
    tensors = _tensors_for_case(case, data_dir, device)
    for t in tensors.values():
        if t.dtype.is_floating_point:
            t.requires_grad_(True)

    if name == "broadcast_add":
        c = tensors["a"] + tensors["b"]
        loss = c.sum()
    elif name == "broadcast_mul":
        c = tensors["a"] * tensors["b"]
        loss = c.sum()
    elif name == "broadcast_div":
        c = tensors["a"] / tensors["b"]
        loss = c.sum()
    elif name == "matmul_bias_relu_softmax_ce":
        h = tensors["x"] @ tensors["weight"] + tensors["bias"]
        y = F.relu(h)
        loss = F.cross_entropy(y, tensors["labels"].to(torch.int64))
    elif name == "mse_linear":
        pred = tensors["x"] @ tensors["weight"] + tensors["bias"]
        loss = F.mse_loss(pred, tensors["target"])
    elif name == "conv_bn_pool_flatten":
        # C++ conv weight is (in_c, k, k, out_c); PyTorch wants (out_c, in_c, k, k).
        w_cpp = tensors["conv_weight"]
        in_c, k1, k2, out_c = w_cpp.shape
        w_pt = w_cpp.permute(3, 0, 1, 2).reshape(out_c, in_c, k1, k2)
        t = F.conv2d(tensors["x"], w_pt, tensors["conv_bias"], stride=1, padding=1)
        t = F.batch_norm(
            t, None, None, tensors["bn_gamma"], tensors["bn_beta"],
            training=True, momentum=0.1, eps=1e-5
        )
        t = F.relu(t)
        t = F.max_pool2d(t, 2, stride=2)
        t = torch.flatten(t, start_dim=1)
        logits = t @ tensors["linear_weight"] + tensors["linear_bias"]
        loss = F.cross_entropy(logits, tensors["labels"].to(torch.int64))
    elif name == "dropout_fixed_mask":
        y = tensors["x"] * tensors["mask"]
        loss = y.sum()
    elif name == "chained_fanout":
        y = tensors["x"] * 2.0
        z1 = y + 1.0
        z2 = y * y
        prod = z1 * z2
        loss = prod.mean()
    else:
        return False, f"unknown case: {name}"

    loss.backward()

    grad_order = {
        "broadcast_add": ["grad_a", "grad_b"],
        "broadcast_mul": ["grad_a", "grad_b"],
        "broadcast_div": ["grad_a", "grad_b"],
        "matmul_bias_relu_softmax_ce": ["grad_x", "grad_weight", "grad_bias"],
        "mse_linear": ["grad_x", "grad_weight", "grad_bias"],
        "conv_bn_pool_flatten": [
            "grad_x", "grad_conv_weight", "grad_conv_bias",
            "grad_bn_gamma", "grad_bn_beta", "grad_linear_weight", "grad_linear_bias"
        ],
        "dropout_fixed_mask": ["grad_x"],
        "chained_fanout": ["grad_x"],
    }

    for grad_name in grad_order[name]:
        param_name = grad_name.replace("grad_", "", 1)
        param = tensors[param_name]
        if param.grad is None:
            return False, f"{grad_name}: PyTorch produced no gradient"
        ref = load_tensor(
            str(data_dir / case["tensors"][grad_name][2]),
            case["tensors"][grad_name][0],
            case["tensors"][grad_name][1],
            torch.device("cpu")
        )
        actual = param.grad.detach().cpu().numpy()
        if perturb == grad_name:
            actual = actual * 1.1
        ok, msg = compare_gradients(
            ref.detach().cpu().numpy(), actual, case["abs_tol"], case["rel_tol"]
        )
        if not ok:
            return False, f"{grad_name}: {msg}"

    return True, "all gradients match"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="comparison/generated-data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--perturb", default="",
                        help="name of a gradient to scale by 1.1 (negative control)")
    parser.add_argument("--corrupt-manifest", action="store_true",
                        help="prepend garbage to the manifest before parsing (negative control)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.txt"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    if args.corrupt_manifest:
        original = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text("garbage line\n" + original, encoding="utf-8")

    try:
        cases = parse_manifest(manifest_path)
    except ValueError as e:
        print(f"manifest parse error: {e}", file=sys.stderr)
        return 1
    finally:
        if args.corrupt_manifest:
            manifest_path.write_text(original, encoding="utf-8")

    device_name = args.device
    if device_name == "gpu":
        device_name = "cuda"
    device = torch.device(device_name)
    failed = False
    for case in cases:
        try:
            ok, msg = run_case(case, data_dir, device, args.perturb)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"exception: {e}"
        status = "ok" if ok else "FAIL"
        print(f"{status}: {case['name']}: {msg}")
        if not ok:
            failed = True
            break

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
