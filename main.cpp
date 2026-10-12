#include <iostream>

#include "Helpers/Hardware.h"
#include "Tensor/Tensor.h"
#include "Tensor/DTypes.h"


int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << sizeof(bool) << std::endl;
    CPUFeatureSets cpu_features = {};
    std::cout << sizeof(uint8_t) << std::endl;
    return 0;
}
