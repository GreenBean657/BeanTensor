#!/usr/bin/env python3
"""PyTorch counterpart of the BeanTensor CNN MNIST demo (cnn.cpp).

Same architecture, optimizer, epochs, and batch size so docs/cnn-results.md
can apply the project pass bar (BeanTensor >= PyTorch - 1pp) on the same
machine. Downloads MNIST through torchvision into ./datasets/mnist-torch
(kept separate from the demo's uncompressed IDX files).

Run: comparison/.venv/bin/python comparison/cnn_pytorch.py
"""
import argparse
import time
import torch
import torch.nn as nn
import torchvision

EPOCHS = 2
BATCH = 128
LR = 1e-3


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.25),
            nn.Linear(16 * 5 * 5, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=60000)
    parser.add_argument("--test-rows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    torch.manual_seed(1234)
    device = "cpu"
    train_set = torchvision.datasets.MNIST(
        "./datasets/mnist-torch", train=True, download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_set = torchvision.datasets.MNIST(
        "./datasets/mnist-torch", train=False, download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_subset = torch.utils.data.Subset(train_set, range(min(args.train_rows, 60000)))
    test_subset = torch.utils.data.Subset(test_set, range(min(args.test_rows, 10000)))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=BATCH, shuffle=False)

    model = Cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        steps = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                preds = model(images.to(device)).argmax(dim=1)
                correct += (preds == labels.to(device)).sum().item()
        accuracy = correct / len(test_subset)
        print(f"epoch {epoch + 1}/{args.epochs}  train mean loss {total / steps:.6f}"
              f"  test accuracy {accuracy * 100:.2f}%")
    print(f"total time {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
