/**
 * @file Serialization.h
 * @author GreenBean
 * @brief The .bt model-export entry points: save(Module -> file) and
 * load(file -> Module). A .bt file is a versioned container holding a
 * ModuleSpec architecture tree (layer types, constructor hyperparameters,
 * named children) plus a flat stream of named tensor records written in
 * registry traversal order — parameters first, then named state tensors
 * (BatchNorm running statistics). Values are bit-exact: the save->load
 * roundtrip on one machine is byte-identical per tensor (CRC32-verified),
 * and CPU-saved -> GPU-loaded holds within the existing CPU<->GPU conversion
 * contract because load re-reads dtypes verbatim and only the device move is
 * backend-defined. Tensor payloads are optionally LZ77-compressed per record
 * (stored only when smaller). Errors raise ErrorHandling::* synchronously;
 * file I/O is not part of the async tensor-op model.
 * @version 0.1
 * @date 2026-11-05
 */
#pragma once

#include <memory>
#include <string>

#include "Module/Module.h"
#include "Module/ModuleSpec.h"
#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Serialization {

    /**
     * How much optimizer state a .bt save carries. Weights writes version-1
     * files (architecture + parameters + named state tensors — the original
     * v1 layout). WeightsAndGrads appends per-parameter gradient records.
     * Full additionally appends per-parameter Adam moments and the step
     * counter, so a mid-train checkpoint resumes training bit-reproducibly.
     * @brief Optimizer-state depth of a .bt save.
     */
    enum class CheckpointLevel : uint8_t {
        Weights,
        WeightsAndGrads,
        Full,
    };

    /**
     * Checkpoint a model that is being trained concurrently is not
     * whole-file coherent (parameters are written record by record); sync
     * and quiesce the model first for a reproducible checkpoint.
     * @brief Export a module tree to a .bt file.
     * @throws ErrorHandling::SerializationError the file cannot be opened or written.
     * @throws ErrorHandling::NotSerializable a module in the tree does not export a spec.
     */
    void save(const Modules::Module&, const std::string& path, bool compress = true,
              CheckpointLevel level = CheckpointLevel::Weights);

    /**
     * @brief Import a module tree from a .bt file.
     * @throws ErrorHandling::SerializationError the file cannot be opened or read.
     * @throws ErrorHandling::CorruptFile bad magic, truncation, CRC mismatch, or a record that does not match the rebuilt architecture.
     * @throws ErrorHandling::UnsupportedVersion the file version is newer than this build reads.
     * @throws ErrorHandling::UnknownModuleType the spec names a type no factory constructs.
     */
    [[nodiscard]] std::unique_ptr<Modules::Module> load(const std::string& path, Tensors::Device device = Tensors::Device::CPU);
}
