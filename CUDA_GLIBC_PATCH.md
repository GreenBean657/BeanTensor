# CUDA 12.9 + glibc >= 2.41: the math_functions.h patch

Building BeanTensor's CUDA backend (`-DUSE_CUDA=ON`) on a current Linux distro
requires patching one CUDA toolkit header. This doc covers what the patch does,
why it's needed, and why it's safe.

The repo enforces it. `Verifications/cuda_glibc_noexcept.sh` runs on every
CUDA-configured build and fails the build, printing the exact fix command, if
the toolkit header is unpatched. You can't silently build a broken configuration.

## The patch

Six function declarations in
`/usr/local/cuda-12.9/targets/x86_64-linux/include/crt/math_functions.h`
need `noexcept(true)` appended before the semicolon:

- `cospi`, `cospif`
- `sinpi`, `sinpif`
- `rsqrt`, `rsqrtf`

```bash
sudo sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ +(double|float) +(cospi|cospif|sinpi|sinpif|rsqrt|rsqrtf)\((double|float) x\));/\1 noexcept(true);/' /usr/local/cuda-12.9/targets/x86_64-linux/include/crt/math_functions.h
```

## Why the conflict exists

Three facts collide:

1. BeanTensor targets sm_70 (Volta). CUDA 13 dropped Volta, so the CUDA backend
   is pinned to CUDA 12.9, the last toolkit that can emit sm_70 binaries.
2. glibc 2.41 added the C23 math functions. From glibc 2.41 onward (Ubuntu
   25.10+, Fedora 42+, current Arch), `<math.h>` declares `cospi`, `sinpi`,
   `rsqrt` and their float variants. In C++ mode glibc declares them
   `noexcept(true)` through its `__THROW` macro.
3. CUDA 12.9's header declares the same six functions without `noexcept`.
   `crt/math_functions.h` predates the glibc change. CUDA 13 fixed the mismatch
   upstream, but CUDA 13 can't target sm_70 (point 1).

In C++, two declarations of the same function with different exception
specifications are a hard error:

```error
error: exception specification is incompatible with that of previous
function "cospi"
```

Every `.cu` file pulls in both headers, so no CUDA file compiles on an affected
distro until the declarations agree.

The glibc declarations sit behind `_GNU_SOURCE`, which the `g++` driver defines
by default, so this fires on every nvcc invocation no matter what your code does.

## Why a header patch instead of a compile flag

The flag-level alternatives were tried:

- `-U_GNU_SOURCE` hides the glibc prototypes, but glibc's feature macros are
  evaluated once per translation unit, and libstdc++'s own headers (`<mutex>`,
  `<cwchar>`, etc.) need GNU extensions. The build dies with around 100 errors
  inside the standard library instead.
- Shadowing the header with `-I` doesn't work. CUDA's headers include
  `math_functions.h` with quoted includes, which resolve relative to the
  toolkit directory first. The toolkit copy always wins.
- Suppressing the diagnostic isn't possible. nvcc's frontend emits this error
  without a diagnostic number.

The declarations have to agree, and glibc is the system C library, so the CUDA
header is the one that gets patched.

## Why the patch is safe

- `noexcept` is a compile-time contract. It changes no calling convention, no
  symbol names, no generated code. A patched toolkit produces bit-identical
  binaries.
- It's a header, not a binary. Nothing gets recompiled; the text file just
  matches what CUDA 13 ships upstream.
- It's reversible. Reinstalling or upgrading the toolkit package restores the
  original header, and the build verification will immediately tell you the
  patch is gone.

## Caveats

- The patch is per machine, per toolkit install. A fresh CUDA 12.9 install
  needs it re-applied. If you forget, the build fails and prints the command.
  That's intentional.
- HIP/ROCm and CPU-only builds are unaffected. This applies only to
  `-DUSE_CUDA=ON` on distros with glibc >= 2.41.