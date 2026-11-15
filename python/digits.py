#!/usr/bin/env python3
"""digits.py — BeanTensor Python-bindings MNIST MLP demo.

Replays main.cpp (the C++ BeanTensorV5 demo) call-for-call through the
pybind11 surface: same seed, same layer construction order, same loader
settings, same training loop, same epoch-line format. On a CPU build the
epoch accuracy/loss fields match the C++ binary bit-for-bit (wall-time
fields excluded by construction of the comparison).

Run from the repository root:
    PYTHONPATH=scratch/build-bindings/Bindings python3 python/digits.py
"""

import argparse
import sys
import time

import beantensor as bt

K_INPUTS = 784
K_HIDDEN = 128
K_CLASSES = 10
K_BATCH = 100
K_EPOCHS = 3
K_LEARNING_RATE = 0.001
K_BETA1 = 0.9
K_BETA2 = 0.999
K_EPSILON = 1e-8
K_SHUFFLE_SEED = 1234
K_DATA_DIR = "datasets/mnist"


def run_training(train_images, train_labels, test_images, test_labels, device):
    print()
    print("=" * 60)
    print(f"Device: {'CPU' if device == bt.Device.CPU else 'GPU'}")
    print("=" * 60)

    net = bt.Sequential([
        bt.Linear(K_INPUTS, K_HIDDEN, bt.DType.Float32, device),
        bt.Relu(),
        bt.Linear(K_HIDDEN, K_CLASSES, bt.DType.Float32, device),
    ])
    print(net)

    train_loader = bt.DataLoader(train_images, train_labels, K_BATCH, True, K_SHUFFLE_SEED, True)
    test_loader = bt.DataLoader(test_images, test_labels, K_BATCH, False, None, True)

    run_start = time.perf_counter()
    for epoch in range(K_EPOCHS):
        epoch_start = time.perf_counter()
        train_loader.reset()
        loss_acc = None
        steps = 0
        while True:
            batch_opt = train_loader.next_batch()
            if batch_opt is None:
                break
            batch, labels = batch_opt

            net.zero_grad()
            out = net.forward(batch)
            loss = bt.softmax_cross_entropy(out, labels)
            loss.backward(bt.DType.Float32)
            net.adam_step(K_LEARNING_RATE, K_BETA1, K_BETA2, K_EPSILON)

            # main.cpp accumulates into a zeroed Float32 scalar; 0.0f + x is
            # exactly x in IEEE, so seeding the chain from the first loss is
            # bit-identical (the binding surface exposes no shape constructor
            # for a standalone zero tensor).
            loss_acc = loss if loss_acc is None else loss_acc.add(loss)
            steps += 1
        for _, param in net.named_parameters().items():
            param.sync()
        net.eval()
        accuracy, mean_loss = bt.evaluate(net, test_loader)
        net.train()
        epoch_seconds = time.perf_counter() - epoch_start
        print(f"epoch {epoch + 1}/{K_EPOCHS}"
              f"  train mean loss {loss_acc.item() / steps:.6g}"
              f"  test accuracy {accuracy * 100:.6g}%"
              f"  test mean loss {mean_loss:.6g}"
              f"  time {epoch_seconds:.6g} s")
    total_seconds = time.perf_counter() - run_start
    print(f"total time {total_seconds:.6g} s")
    print("\nafter training:")
    print(net)
    return net, test_loader


def main():
    parser = argparse.ArgumentParser(description="BeanTensor MNIST MLP demo (Python bindings)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="training device (gpu requires a CUDA/HIP build)")
    parser.add_argument("--data-dir", default=K_DATA_DIR,
                        help="directory holding the uncompressed MNIST IDX files")
    parser.add_argument("--save", default="digits_mlp.bt",
                        help="path of the .bt checkpoint written after training")
    args = parser.parse_args()

    device = bt.Device.CPU if args.device == "cpu" else bt.Device.GPU

    bt.set_seed(K_SHUFFLE_SEED)

    try:
        train_images, train_labels = bt.load_mnist_flat(args.data_dir, True, bt.DType.Float32, device)
        test_images, test_labels = bt.load_mnist_flat(args.data_dir, False, bt.DType.Float32, device)
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    print(f"Loaded {train_images.shape[0]} training and {test_images.shape[0]} test images")

    try:
        net, test_loader = run_training(train_images, train_labels, test_images, test_labels, device)
    except RuntimeError as exc:
        if device == bt.Device.GPU:
            print(f"GPU run failed (CPU-only build?): {exc}", file=sys.stderr)
            return 1
        raise

    bt.save(net, args.save, True, bt.CheckpointLevel.Full)
    print(f"\nsaved checkpoint to {args.save}")
    reloaded = bt.load(args.save, device)
    reloaded.eval()
    accuracy, mean_loss = bt.evaluate(reloaded, test_loader)
    print(f"reloaded: test accuracy {accuracy * 100:.6g}%  test mean loss {mean_loss:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
