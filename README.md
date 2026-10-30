[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/GreenBean657/badge.svg)](https://hub.xerj.org/contributors/GreenBean657)
[![YES: AI](https://raw.githubusercontent.com/GreenBean657/pro-ai-badge/master/badge.png)](https://github.com/GreenBean657/pro-ai-badge)

# BeanTensor

BeanTensor is a from-scratch C++23 tensor and machine-learning framework with CPU (scalar, AVX2, AVX-512), CUDA, and HIP/ROCm backends. It provides tensors, autograd, neural-network modules, and first-order optimizers built directly on the CUDA/ROCm runtimes and compiler intrinsics, with no external ML libraries.

## Build

BeanTensor targets exactly one GPU backend, or none at all for a CPU-only build. The CMake defaults are `USE_CUDA=ON` and `USE_HIP=OFF`; always pass both flags explicitly so the default never silently chooses a backend for you:

```bash
cmake -B build -S . -DUSE_CUDA=ON -DUSE_HIP=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6
```

Other valid configurations:

```bash
# HIP/ROCm
cmake -B build -S . -DUSE_HIP=ON -DUSE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release

# CPU only
cmake -B build -S . -DUSE_CUDA=OFF -DUSE_HIP=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
```

**CUDA backend toolchain:** the build is pinned to CUDA 12.9 (the last toolkit with sm_70/Volta support). On distros with glibc ≥ 2.41 (Ubuntu 25.10+, Fedora 42+, current Arch) you must apply a one-time 6-line patch to a CUDA toolkit header; the build verification fails and prints the exact command if it is missing. See [CUDA_GLIBC_PATCH.md](CUDA_GLIBC_PATCH.md) for what the patch does and why it is safe.

**HIP/ROCm backend toolchain:** requires ROCm with `hip` and `rocblas`.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — user-visible semantics: backends, tensors, autograd, modules, and training utilities.
- [`docs/quickstart.md`](docs/quickstart.md) — copy-paste MNIST training example and CPU-only build command.
- [`docs/mnist-results.md`](docs/mnist-results.md) — end-to-end MNIST acceptance results and repro commands.
- API reference: `cmake --build build --target docs` (requires Doxygen).
- GitHub Pages: enable Pages in the repo settings (`Settings → Pages → Source: GitHub Actions`) to publish the Doxygen site from the workflow in `.github/workflows/docs.yml`.

## License

MIT License

Copyright (c) 2026 BeanTensor

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
