/**
 * @file BeanTensor.h
 * @author GreenBean
 * @brief The umbrella header: one include pulls in the whole public surface
 * — tensors, every module, the DataLoader, metrics, the RNG, CSV loading,
 * and .bt serialization — and the short namespace aliases (nn, data, io,
 * metrics, rnd) turn the deeply nested project namespaces into something a
 * training script actually wants to type. C++ is a first-class user of this
 * library; this header is the front door.
 *
 * Typical use:
 *     #include "BeanTensor.h"
 *     using namespace BeanTensor;
 *     auto net = nn::make_sequential(std::make_unique<nn::Linear>(784, 128),
 *                                    std::make_unique<nn::Relu>(),
 *                                    std::make_unique<nn::Linear>(128, 10));
 * @version 0.2
 * @date 2026-11-10
 */
#pragma once

#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"

#include "Math/Loss/Loss.h"

#include "Module/BatchNorm.h"
#include "Module/Conv2d.h"
#include "Module/Dropout.h"
#include "Module/Flatten.h"
#include "Module/Gelu.h"
#include "Module/Linear.h"
#include "Module/MaxPool2d.h"
#include "Module/AvgPool2d.h"
#include "Module/Module.h"
#include "Module/ModuleStream.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Module/Sigmoid.h"
#include "Module/Softmax.h"
#include "Module/Tanh.h"

#include "Data/Csv.h"
#include "Data/DataLoader.h"

#include "Metrics/Metrics.h"
#include "Random/Generator.h"
#include "Serialization/Serialization.h"

namespace BeanTensor {
    /**
     * The two names every training script touches, lifted to the root so
     * `using namespace BeanTensor` is enough.
     */
    using Tensors::Device;
    using Tensors::Tensor;

    /**
     * Short alias for the module layer namespace: layers, Sequential, and
     * the training conveniences live here.
     */
    namespace nn = Modules;
    /**
     * Short alias for the data namespace: DataLoader and CSV loading.
     */
    namespace data = Data;
    /**
     * Short alias for the metrics namespace: accuracy and evaluate.
     */
    namespace metrics = Metrics;
    /**
     * Short alias for the random namespace: the Generator and set_seed.
     */
    namespace rnd = Random;
    /**
     * Short alias for the serialization namespace: .bt save/load and the
     * checkpoint levels.
     */
    namespace io = Serialization;
}
