This folder contains test cases for all modules, for the purpose of BeanTensor coding standard. 
All executable targets should end in .sh, or this document & CMakeLists should be appended if an additional path is needed. 
Any file in here lacking .sh is ignored. 

no_float80.sh - Prevent float80_t (long double) from being used in any module.
- Per my standard, usage of float80_t encourages divergence - this x87 type is not supported on all platforms, and especially not on GPU code.
- x87 support not being present on GPUs means CPU vs GPU pathways diverage (CPU code touches float80_t, but GPU doesn't? That's encouraging rounding errors.)
- float80_t occupies uneven registers, and lacks any hardware support. FPU's software emulate it - it's a waste of cycles.
**float80_t should never be used in this project.**
**If I ever find a valid reason to use this datatype, I will remove this unit test - right now, it's a no.**