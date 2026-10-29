#!/usr/bin/env bash
# fetch_mnist.sh — download the MNIST IDX files that main.cpp reads.
# Contract: fetch_mnist.sh [destination-dir]
#   Default destination is data/mnist (relative to the working directory,
#   matching where main.cpp looks). Files already present are kept.
#   Exit non-zero on a missing tool or a failed download.

set -euo pipefail
DEST="${1:-datasets/mnist}"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"
FILES=(
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
)

for tool in curl gunzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "fetch_mnist.sh: required tool '$tool' not found" >&2
        exit 1
    fi
done

mkdir -p "$DEST"
for name in "${FILES[@]}"; do
    target="$DEST/$name"
    if [[ -f "$target" ]]; then
        echo "keep    $target"
        continue
    fi
    echo "fetch   $target"
    curl -fsSL -o "$target.gz.part" "$BASE_URL/$name.gz"
    mv "$target.gz.part" "$target.gz"
    gunzip -f "$target.gz"
done
echo "done: MNIST IDX files are in $DEST"
