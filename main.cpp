#include <hip/hip_runtime.h>
#include <iostream>

#include "Tensor/tensor.h"
#include <Core/Casting.h>
#include <Helper/hipcheck.h>
#include <Tensor/tensor.h>

#include <Ops/Algebra/MatMul.h>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <Tensor/tensor.h>
using namespace BeanTensor;
int main()
{
    xrt::device device(0);

    auto xclbin = device.load_xclbin("matmul.xclbin");
    return 0;
}

