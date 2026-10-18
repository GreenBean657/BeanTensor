# BeanTensor
BeanTensor is a C++ library built for HIP, CUDA, and CPU (AVX512 accelerated!) neural network inferencing.
Other libraries have patches for support like this, but it's often buggy or falls apart in high dimensions.
BeanTensor is built on the principle of GROUND UP support for all hardware - obscure still deserves support. 


# Usage
The library is under construction. At this current time, documentation is not constructed. 
Usage is at your own risk, and the API is unlikely to function fully or at all.

# Building
Backends: `-DUSE_HIP=ON` (default), `-DUSE_CUDA=ON`, or CPU-only (both off). Mutually exclusive.

**CUDA backend requirement:** the build is pinned to CUDA 12.9 (the last toolkit with
sm_70/Volta support), and on distros with glibc ≥ 2.41 (Ubuntu 25.10+, Fedora 42+, current
Arch) you must apply a one-time 6-line patch to a CUDA toolkit header or nothing compiles.
The build verifies this and prints the exact command if it's missing. Full explanation of
what the patch does and why it's safe: [CUDA_GLIBC_PATCH.md](CUDA_GLIBC_PATCH.md).

# Abilities

**BeanTensor is intended for:**
- NVIDIA GPUs of Volta Architecture (Compute SM_70) or newer.
- AMD GPUs of RDNA2 or newer, supported by ROCM 10. CDNA is not supported.
- 64 bit CPUs. AVX2, AVX512, AVX512BF, and other accelerating instructions are not required, but will be used when possible.

**BeanTensor is not intended for:**
- Intel & AMD NPUs attached to x86 CPUs (though this is planned in the future!)
- TPUs, including google coral.
- 32 bit processors.
- Legacy Nvidia & AMD cards, including Pascal & RDNA 1 + Vega. Support for Pascal and basic RDNA is planned, but not a current priority and that plan may change as AI evolves. 

**BeanTensor will target a few major floating point and integer precision's, including:**
- FP64 (double)
- FP32 (float)
- FP16 (half)
- BF16 (bhalf)
- INT64 (long)
- INT32 (int)
- INT16 (short)
- INT8 (byte)
- UINT64 (unsigned long)
- UINT32 (unsigned int)
- UINT16 (unsigned short)
- UINT8 (unsigned byte)
When tensor core acceleration exists and the operation permits it, I intend for BeanTensor to use the accelerated tensor pathway.
Additional precision to maximize Tensor implementations are planned in the future, but may not land within the capstone deadline. 


# Contributing
Pull Requests are welcome! 
This is a capstone project for Neumont University. At the moment, pull requests will be considered, but possibly recoded. 

I welcome suggestions, feature recommendations, and any other feedback - even if negative (though constructive is better!)


# Contact
You can contact me at my email: admin@beantensor.org

Or on discord: @greenbeanofficial

Website: https://beantensor.org

# License
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
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
