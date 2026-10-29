#!/usr/bin/env python3
"""PyTorch equivalent of the BeanTensorV5 MNIST demo in main.cpp.

Matches the C++ version as closely as practical:
- Model: Linear(784, 128) -> ReLU -> Linear(128, 10)
- Loss: softmax + cross-entropy (nn.CrossEntropyLoss)
- Optimizer: Adam, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
- Batch size: 100, epochs: 3, shuffle seed: 1234
- Input pixels scaled to [0, 1]
- Test accuracy computed by argmax over logits

Data pipeline matches main.cpp too: the whole split is decoded once into a
single tensor before timing starts (main.cpp keeps the raw IDX bytes in memory),
each epoch shuffles an index permutation, and batches are gathered by index.
The per-sample torchvision transform + DataLoader collate path is NOT used: it
costs ~2.2 s/epoch of Python overhead and would dominate the comparison.
The running loss is accumulated on the device, so there is no host sync per step.

--flush-denormal enables torch.set_flush_denormal(True) (FTZ/DAZ on CPU).
Off by default (PyTorch's default); without it, Adam state drifting into
denormals slows later CPU epochs.

Runs the same training twice: once on CPU, then once on the best available
accelerator (CUDA, then HIP, then MPS, then CPU again if nothing else).
"""

import argparse
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import datasets


INPUTS = 784
HIDDEN = 128
CLASSES = 10
BATCH = 100
EPOCHS = 3
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
SHUFFLE_SEED = 1234
MNIST_DIR = "datasets/mnist"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_model(model: nn.Module) -> str:
    lines = ["Sequential {"]
    for name, module in model.named_children():
        lines.append(f"  [{name}] {module}")
    lines.append("}")
    return "\n".join(lines)


def format_size(bytes_value: int) -> str:
    if bytes_value < 1024:
        return f"{bytes_value} B"
    if bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.1f} KB"
    return f"{bytes_value / (1024 * 1024):.1f} MB"


def describe_state(model: nn.Module) -> str:
    lines = ["Sequential {"]
    for name, module in model.named_children():
        lines.append(f"  [{name}] {module}")
        if isinstance(module, nn.Linear):
            for pname, param in module.named_parameters():
                size_bytes = param.numel() * param.element_size()
                lines.append(
                    f"    {pname}: {param.dtype} {list(param.shape)}  "
                    f"{format_size(size_bytes)}"
                )
    lines.append("}")
    return "\n".join(lines)


def make_model() -> nn.Module:
    return nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(INPUTS, HIDDEN)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(HIDDEN, CLASSES)),
            ]
        )
    )


def load_split(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one MNIST split into [N, 784] float32 pixels in [0, 1] and int64 labels."""
    split = datasets.MNIST(MNIST_DIR, train=train, download=True)
    images = split.data.to(torch.float32).div_(255.0).view(-1, INPUTS)
    return images, split.targets.to(torch.int64)


def train_and_evaluate(
    device: torch.device,
    train_split: tuple[torch.Tensor, torch.Tensor],
    test_split: tuple[torch.Tensor, torch.Tensor],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    # One-time upload, outside the timed region (main.cpp loads before timing too).
    train_images, train_labels = (t.to(device) for t in train_split)
    test_images, test_labels = (t.to(device) for t in test_split)
    train_count = train_images.shape[0] // BATCH * BATCH
    test_count = test_images.shape[0] // BATCH * BATCH
    generator = torch.Generator().manual_seed(SHUFFLE_SEED)

    model = make_model().to(device)
    print(describe_model(model))
    print(f"total: {count_parameters(model):,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPSILON
    )

    run_start = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        steps = 0

        order = torch.randperm(train_images.shape[0], generator=generator).to(device)
        for start in range(0, train_count, BATCH):
            indices = order[start : start + BATCH]
            images = train_images[indices]
            labels = train_labels[indices]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach()
            steps += 1

        model.eval()
        correct = torch.zeros((), dtype=torch.int64, device=device)
        with torch.no_grad():
            for start in range(0, test_count, BATCH):
                outputs = model(test_images[start : start + BATCH])
                predictions = outputs.argmax(dim=1)
                correct += (predictions == test_labels[start : start + BATCH]).sum()

        # .item() syncs the device, so the epoch timer below covers all queued work.
        accuracy = 100.0 * correct.item() / test_count
        mean_loss = loss_sum.item() / steps
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch + 1}/{EPOCHS}  "
            f"mean loss {mean_loss:.6f}  "
            f"test accuracy {accuracy:.2f}%  "
            f"time {epoch_seconds:.5f} s"
        )

    total_seconds = time.perf_counter() - run_start
    print(f"total time {total_seconds:.3f} s")
    print("\nafter training:")
    print(describe_state(model))


def pick_accelerator() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--flush-denormal",
        action="store_true",
        help="enable FTZ/DAZ for CPU float math (torch.set_flush_denormal)",
    )
    args = parser.parse_args()
    if args.flush_denormal:
        if not torch.set_flush_denormal(True):
            raise SystemExit("--flush-denormal: not supported on this CPU")
        print("flush-denormal: on")

    train_split = load_split(train=True)
    test_split = load_split(train=False)

    print(f"Loaded {train_split[0].shape[0]} training and {test_split[0].shape[0]} test images")

    # CPU run
    train_and_evaluate(torch.device("cpu"), train_split, test_split)

    # GPU run (if available)
    accel = pick_accelerator()
    if accel.type == "cpu":
        print("\nNo accelerator available; skipping GPU run.")
    else:
        train_and_evaluate(accel, train_split, test_split)


if __name__ == "__main__":
    main()
