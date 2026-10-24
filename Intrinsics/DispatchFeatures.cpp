/**
 * @file DispatchFeatures.cpp
 * @author GreenBean
 * @brief Feature gate for the CPU conversion dispatcher.
 * Compiled with the full ISA flag union so IncludedCPUFeatures() sees every compile-time macro. It holds no conversion code,
 * so the union never reaches the scalar fallback in Dispatcher.cpp.
 * @version 0.1
 * @date 2026-10-23
 */
#include "Intrinsics/Dispatcher.h"

namespace BeanTensor::Intrinsics::detail {

    /**
     * Forwards to IncludedCPUFeatures(), which combines the compile-time macro for each feature with __builtin_cpu_supports.
     * @brief True when the feature is compiled in and reported by the CPU.
     * @param feature The feature to test.
     * @return Whether the dispatcher may take the fast path gated on this feature.
     */
    bool dispatch_has_feature(const CPUFeatureSetTypes feature) {
        return IncludedCPUFeatures().has_feature(feature);
    }
}
