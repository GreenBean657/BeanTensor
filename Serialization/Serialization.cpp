/**
 * @file Serialization.cpp
 * @author GreenBean
 * @brief Implementation of the .bt writer/reader declared in Serialization.h:
 * little-endian field I/O, the recursive spec record, the CRC32 checksum, the
 * dtype tag table, the registry-ordered tensor stream, the layer factory
 * table used at load time, and the version-3 trailing extra-record section
 * (the optional metadata blob) with its skip-unknown-kind reader.
 * @version 0.3
 * @date 2026-11-10
 */
#include "Serialization/Serialization.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Module/AvgPool2d.h"
#include "Module/BatchNorm.h"
#include "Module/Conv2d.h"
#include "Module/Dropout.h"
#include "Module/Flatten.h"
#include "Module/Gelu.h"
#include "Module/Linear.h"
#include "Module/MaxPool2d.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Module/Sigmoid.h"
#include "Module/Softmax.h"
#include "Module/Tanh.h"
#include "Serialization/Lz77.h"

namespace BeanTensor::Serialization {
    namespace {
        // ---- Format constants (v1, little-endian everywhere) ----
        // 'B', 'T', '0', '1' as fixed-width constants: the char type is
        // banned outside Tensor/DTypes.h, so the magic is byte literals.
        constexpr std::array<uint8_t, 4> k_magic = {0x42, 0x54, 0x30, 0x31};
        constexpr uint32_t k_version = 1;
        constexpr uint32_t k_version_checkpoint = 2;
        constexpr uint32_t k_version_metadata = 3;
        // Section presence travels in the flags field: bit 0 = gradient
        // section, bit 1 = adam section, bit 2 = trailing extra-record
        // section (today: the optional metadata record). Weights-only files
        // keep version 1 and flags 0 (the original v1 layout); checkpoint
        // files are version 2; any file carrying metadata is version 3.
        constexpr uint32_t k_flag_grads = 1u;
        constexpr uint32_t k_flag_adam = 2u;
        constexpr uint32_t k_flag_metadata = 4u;
        constexpr uint32_t k_flag_mask = k_flag_grads | k_flag_adam | k_flag_metadata;
        constexpr uint8_t k_compression_none = 0;
        constexpr uint8_t k_compression_lz = 1;
        // Extra-record kinds (the trailing section, version 3). Kinds are
        // append-only; a v3 reader skips kinds it does not know.
        constexpr uint8_t k_kind_metadata = 1;
        constexpr std::string_view k_metadata_name = "btui.config";
        // Format sanity limits: every file-controlled length/count/rank is
        // checked against these BEFORE anything is allocated or reserved, so
        // a malformed file raises CorruptFile instead of bad_alloc/length_error.
        constexpr uint32_t k_max_string = 1u << 20;      // 1 MiB names/values
        constexpr uint32_t k_max_children = 1u << 16;
        constexpr uint32_t k_max_records = 1u << 20;
        constexpr uint64_t k_max_tensor_bytes = uint64_t{1} << 34; // 16 GiB
        constexpr uint32_t k_max_spec_depth = 64;

        // ---- DType <-> tag table. Explicit and append-only: existing tags
        // must never be reassigned, only extended, so old files keep parsing. ----
        constexpr uint8_t k_tag_int8 = 1;
        constexpr uint8_t k_tag_int16 = 2;
        constexpr uint8_t k_tag_int32 = 3;
        constexpr uint8_t k_tag_int64 = 4;
        constexpr uint8_t k_tag_uint16 = 5;
        constexpr uint8_t k_tag_uint32 = 6;
        constexpr uint8_t k_tag_uint64 = 7;
        constexpr uint8_t k_tag_float16 = 8;
        constexpr uint8_t k_tag_bfloat16 = 9;
        constexpr uint8_t k_tag_float32 = 10;
        constexpr uint8_t k_tag_float64 = 11;
        constexpr uint8_t k_tag_float16x2 = 12;
        constexpr uint8_t k_tag_bfloat16x2 = 13;
        constexpr uint8_t k_tag_uint8 = 15;
        constexpr uint8_t k_tag_int4x2 = 14;

        [[nodiscard]] uint8_t dtype_tag(const DataTypes::DType dt) {
            switch (dt) {
                case DataTypes::DType::Int8: return k_tag_int8;
                case DataTypes::DType::Int16: return k_tag_int16;
                case DataTypes::DType::Int32: return k_tag_int32;
                case DataTypes::DType::Int64: return k_tag_int64;
                case DataTypes::DType::UInt8: return k_tag_uint8;
                case DataTypes::DType::UInt16: return k_tag_uint16;
                case DataTypes::DType::UInt32: return k_tag_uint32;
                case DataTypes::DType::UInt64: return k_tag_uint64;
                case DataTypes::DType::Float16: return k_tag_float16;
                case DataTypes::DType::BFloat16: return k_tag_bfloat16;
                case DataTypes::DType::Float32: return k_tag_float32;
                case DataTypes::DType::Float64: return k_tag_float64;
                case DataTypes::DType::Float16x2: return k_tag_float16x2;
                case DataTypes::DType::BFloat16x2: return k_tag_bfloat16x2;
                case DataTypes::DType::Int4x2: return k_tag_int4x2;
            }
            throw ErrorHandling::CorruptFile("unmapped dtype at write time");
        }

        [[nodiscard]] DataTypes::DType tag_dtype(const uint8_t tag) {
            switch (tag) {
                case k_tag_int8: return DataTypes::DType::Int8;
                case k_tag_int16: return DataTypes::DType::Int16;
                case k_tag_int32: return DataTypes::DType::Int32;
                case k_tag_int64: return DataTypes::DType::Int64;
                case k_tag_uint8: return DataTypes::DType::UInt8;
                case k_tag_uint16: return DataTypes::DType::UInt16;
                case k_tag_uint32: return DataTypes::DType::UInt32;
                case k_tag_uint64: return DataTypes::DType::UInt64;
                case k_tag_float16: return DataTypes::DType::Float16;
                case k_tag_bfloat16: return DataTypes::DType::BFloat16;
                case k_tag_float32: return DataTypes::DType::Float32;
                case k_tag_float64: return DataTypes::DType::Float64;
                case k_tag_float16x2: return DataTypes::DType::Float16x2;
                case k_tag_bfloat16x2: return DataTypes::DType::BFloat16x2;
                case k_tag_int4x2: return DataTypes::DType::Int4x2;
                [[unlikely]] default: throw ErrorHandling::CorruptFile("unknown dtype tag " + std::to_string(tag));;
            }
        }

        // ---- CRC32 (IEEE, reflected, polynomial 0xEDB88320) ----
        [[nodiscard]] uint32_t crc32(const uint8_t* data, const size_t n) {
            static constexpr auto table = [] {
                auto t = std::array<uint32_t, 256>{};
                for (uint32_t i = 0; i < 256; ++i) {
                    auto c = i;
                    for (int32_t k = 0; k < 8; ++k) {
                        c = (c & 1U) != 0U ? (0xEDB88320U ^ (c >> 1U)) : (c >> 1U);
                    }
                    t[i] = c;
                }
                return t;
            }();
            auto crc = uint32_t{0xFFFFFFFFU};
            for (size_t i = 0; i < n; ++i) {
                crc = table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8U);
            }
            return crc ^ 0xFFFFFFFFU;
        }

        // ---- Little-endian field I/O on streams ----
        template <typename T>
        void write_scalar(std::ostream& os, const T v) {
            for (size_t i = 0; i < sizeof(T); ++i) {
                os.put(static_cast<char>((static_cast<uint64_t>(v) >> (i * 8U)) & 0xFFU));
            }
        }

        template <typename T>
        [[nodiscard]] T read_scalar(std::istream& is, const char* what) {
            auto v = uint64_t{0};
            for (size_t i = 0; i < sizeof(T); ++i) {
                const auto c = is.get();
                if (c == std::char_traits<char>::eof()) {
                    throw ErrorHandling::CorruptFile(std::string("truncated reading ") + what);
                }
                v |= static_cast<uint64_t>(static_cast<uint8_t>(c)) << (i * 8U);
            }
            return static_cast<T>(v);
        }

        void write_string(std::ostream& os, const std::string& s) {
            write_scalar<uint32_t>(os, static_cast<uint32_t>(s.size()));
            os.write(s.data(), static_cast<std::streamsize>(s.size()));
        }

        [[nodiscard]] std::string read_string(std::istream& is, const char* what) {
            const auto len = read_scalar<uint32_t>(is, what);
            if (len > k_max_string) {
                throw ErrorHandling::CorruptFile(std::string("unreasonable ") + what + " length " + std::to_string(len));
            }
            auto s = std::string(len, '\0');
            is.read(s.data(), static_cast<std::streamsize>(len));
            if (static_cast<uint32_t>(is.gcount()) != len) {
                throw ErrorHandling::CorruptFile(std::string("truncated reading ") + what);
            }
            return s;
        }

        // Bytes from the current position to end-of-file; bounds every
        // file-controlled allocation against what is actually on disk.
        [[nodiscard]] uint64_t remaining_bytes(std::istream& is) {
            const auto pos = is.tellg();
            if (pos < 0) {
                return 0;
            }
            is.seekg(0, std::ios::end);
            const auto end = is.tellg();
            is.seekg(pos);
            if (end < pos) {
                return 0;
            }
            return static_cast<uint64_t>(end - pos);
        }

        // ---- Spec tree records ----
        void write_spec(std::ostream& os, const Modules::ModuleSpec& spec) {
            write_string(os, spec.type);
            write_scalar<uint32_t>(os, static_cast<uint32_t>(spec.hyperparameters.size()));
            for (const auto&[k, v] : spec.hyperparameters) {
                write_string(os, k);
                write_string(os, v);
            }
            write_scalar<uint32_t>(os, static_cast<uint32_t>(spec.children.size()));
            for (const auto&[child_name, child] : spec.children) {
                write_string(os, child_name);
                write_spec(os, child);
            }
        }

        [[nodiscard]] Modules::ModuleSpec read_spec(std::istream& is, const uint32_t depth) {
            if (depth > k_max_spec_depth) {
                throw ErrorHandling::CorruptFile("module spec nesting exceeds the depth limit");
            }
            auto spec = Modules::ModuleSpec{};
            spec.type = read_string(is, "module type");
            const auto n_hyper = read_scalar<uint32_t>(is, "hyperparameter count");
            if (n_hyper > k_max_children) {
                throw ErrorHandling::CorruptFile("unreasonable hyperparameter count " + std::to_string(n_hyper));
            }
            for (uint32_t i = 0; i < n_hyper; ++i) {
                auto k = read_string(is, "hyperparameter name");
                auto v = read_string(is, "hyperparameter value");
                spec.hyperparameters.emplace_back(std::move(k), std::move(v));
            }
            const auto n_children = read_scalar<uint32_t>(is, "children count");
            if (n_children > k_max_children) {
                throw ErrorHandling::CorruptFile("unreasonable children count " + std::to_string(n_children));
            }
            for (uint32_t i = 0; i < n_children; ++i) {
                auto child_name = read_string(is, "child name");
                spec.children.emplace_back(std::move(child_name), read_spec(is, depth + 1u));
            }
            return spec;
        }

        // ---- Tensor-record helpers ----
        struct TensorRecord {
            std::string name;
            uint8_t compression = k_compression_none;
            DataTypes::DType dtype = DataTypes::DType::Float32;
            std::vector<size_t> shape;
            std::vector<uint8_t> raw;
        };

        void write_record(std::ostream& os, const TensorRecord& rec, const bool compress) {
            write_string(os, rec.name);
            os.put(static_cast<char>(dtype_tag(rec.dtype)));
            write_scalar<uint32_t>(os, static_cast<uint32_t>(rec.shape.size()));
            for (const auto d : rec.shape) {
                write_scalar<uint64_t>(os, static_cast<uint64_t>(d));
            }
            write_scalar<uint32_t>(os, crc32(rec.raw.data(), rec.raw.size()));
            std::vector<uint8_t> stored;
            uint8_t flag = k_compression_none;
            if (compress) {
                auto candidate = lz_compress(rec.raw.data(), rec.raw.size());
                if (candidate.size() < rec.raw.size()) {
                    stored = std::move(candidate);
                    flag = k_compression_lz;
                }
            }
            if (flag == k_compression_none) {
                stored = rec.raw;
            }
            os.put(static_cast<char>(flag));
            write_scalar<uint64_t>(os, static_cast<uint64_t>(rec.raw.size()));
            write_scalar<uint64_t>(os, static_cast<uint64_t>(stored.size()));
            os.write(reinterpret_cast<const char*>(stored.data()), static_cast<std::streamsize>(stored.size()));
        }

        [[nodiscard]] TensorRecord read_record(std::istream& is) {
            auto rec = TensorRecord{};
            rec.name = read_string(is, "tensor name");
            rec.dtype = tag_dtype(static_cast<uint8_t>(is.get()));
            if (is.eof()) {
                throw ErrorHandling::CorruptFile("truncated reading tensor dtype tag");
            }
            const auto rank = read_scalar<uint32_t>(is, "tensor rank");
            if (rank > BT_TENSOR_MAX_DIMS) {
                throw ErrorHandling::CorruptFile("tensor rank " + std::to_string(rank) + " exceeds the maximum " + std::to_string(BT_TENSOR_MAX_DIMS));
            }
            rec.shape.reserve(rank);
            for (uint32_t i = 0; i < rank; ++i) {
                rec.shape.push_back(static_cast<size_t>(read_scalar<uint64_t>(is, "tensor shape")));
            }
            const auto crc_expected = read_scalar<uint32_t>(is, "tensor crc");
            rec.compression = static_cast<uint8_t>(is.get());
            if (is.eof()) {
                throw ErrorHandling::CorruptFile("truncated reading tensor compression flag");
            }
            if (rec.compression != k_compression_none && rec.compression != k_compression_lz) {
                throw ErrorHandling::CorruptFile("unknown compression flag " + std::to_string(rec.compression));
            }
            const auto raw_size = read_scalar<uint64_t>(is, "tensor raw size");
            const auto stored_size = read_scalar<uint64_t>(is, "tensor stored size");
            if (raw_size > k_max_tensor_bytes || stored_size > k_max_tensor_bytes) {
                throw ErrorHandling::CorruptFile("tensor byte size exceeds the format limit");
            }
            if (stored_size > remaining_bytes(is)) {
                throw ErrorHandling::CorruptFile("tensor payload overruns the file");
            }
            auto stored = std::vector<uint8_t>(stored_size);
            is.read(reinterpret_cast<char*>(stored.data()), static_cast<std::streamsize>(stored_size));
            if (static_cast<uint64_t>(is.gcount()) != stored_size) {
                throw ErrorHandling::CorruptFile("truncated reading tensor payload");
            }
            if (rec.compression == k_compression_lz) {
                rec.raw = lz_decompress(stored.data(), stored.size(), static_cast<size_t>(raw_size));
            } else {
                if (stored_size != raw_size) {
                    throw ErrorHandling::CorruptFile("stored size disagrees with raw size on an uncompressed record");
                }
                rec.raw = std::move(stored);
            }
            if (crc32(rec.raw.data(), rec.raw.size()) != crc_expected) {
                throw ErrorHandling::CorruptFile("CRC32 mismatch on tensor '" + rec.name + "'");
            }
            return rec;
        }

        // True when a parsed record's rank and extents match a parameter,
        // before any tensor is constructed from file-controlled metadata.
        [[nodiscard]] bool shape_matches_param(const TensorRecord& rec, const Tensors::Tensor& param) {
            if (rec.shape.size() != param.rank()) {
                return false;
            }
            for (size_t d = 0; d < rec.shape.size(); ++d) {
                if (rec.shape[d] != param.dim(d)) {
                    return false;
                }
            }
            return true;
        }

        // ---- Tensor slots of a module tree, in the exact order save writes
        // them: the subtree-flat named_parameters() list, then named state
        // tensors collected per module in a named_children() recursion. Both
        // sides of the format derive the order from the same tree shape, so
        // load can verify record-by-record. ----
        void collect_state_slots(Modules::Module& m, const std::string& prefix,
                                 std::vector<std::pair<std::string, Tensors::Tensor*>>& out) {
            for (auto&[name, tensor] : m.state_tensors()) {
                out.emplace_back(prefix + name, tensor);
            }
            for (const auto&[child_name, child] : m.named_children()) {
                collect_state_slots(*const_cast<Modules::Module*>(child), prefix + child_name + ".", out);
            }
        }

        [[nodiscard]] std::vector<std::pair<std::string, Tensors::Tensor*>> tensor_slots(Modules::Module& root) {
            auto out = root.named_parameters();
            collect_state_slots(root, "", out);
            return out;
        }

        void collect_state_records(const Modules::Module& m, const std::string& prefix, std::vector<TensorRecord>& out) {
            for (const auto&[name, tensor] : m.state_tensors()) {
                auto& t = *tensor;
                t.sync();
                auto rec = TensorRecord{};
                rec.name = prefix + name;
                rec.dtype = t.get_dtype();
                for (size_t d = 0; d < t.rank(); ++d) {
                    rec.shape.push_back(t.dim(d));
                }
                rec.raw.resize(t.byte_count());
                t.read_element(0, rec.raw.data(), rec.raw.size());
                out.push_back(std::move(rec));
            }
            for (const auto&[child_name, child] : m.named_children()) {
                collect_state_records(*child, prefix + child_name + ".", out);
            }
        }

        // Skip one tensor record WITHOUT decompressing: read the framing
        // fields, bounds-check, then seek past the stored payload. Used by
        // read_metadata, which must reach the trailing section without
        // paying LZ77 on the way through.
        void skip_record(std::istream& is) {
            static_cast<void>(read_string(is, "tensor name"));
            const auto tag = is.get();
            if (tag == std::char_traits<char>::eof()) {
                throw ErrorHandling::CorruptFile("truncated reading tensor dtype tag");
            }
            const auto rank = read_scalar<uint32_t>(is, "tensor rank");
            if (rank > BT_TENSOR_MAX_DIMS) {
                throw ErrorHandling::CorruptFile("tensor rank exceeds the maximum");
            }
            for (uint32_t i = 0; i < rank; ++i) {
                static_cast<void>(read_scalar<uint64_t>(is, "tensor shape"));
            }
            static_cast<void>(read_scalar<uint32_t>(is, "tensor crc"));
            const auto compression = is.get();
            if (compression == std::char_traits<char>::eof()) {
                throw ErrorHandling::CorruptFile("truncated reading tensor compression flag");
            }
            static_cast<void>(read_scalar<uint64_t>(is, "tensor raw size"));
            const auto stored_size = read_scalar<uint64_t>(is, "tensor stored size");
            if (stored_size > k_max_tensor_bytes) {
                throw ErrorHandling::CorruptFile("tensor byte size exceeds the format limit");
            }
            if (stored_size > remaining_bytes(is)) {
                throw ErrorHandling::CorruptFile("tensor payload overruns the file");
            }
            is.seekg(static_cast<std::streamoff>(stored_size), std::ios::cur);
            if (!is) {
                throw ErrorHandling::CorruptFile("truncated reading tensor payload");
            }
        }

        // ---- Extra records (the trailing section, version 3) ----
        void write_extra_record(std::ostream& os, const uint8_t kind, const std::string& name, const std::string& payload) {
            os.put(static_cast<char>(kind));
            write_string(os, name);
            write_scalar<uint32_t>(os, crc32(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
            write_scalar<uint64_t>(os, static_cast<uint64_t>(payload.size()));
            os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        }

        // One extra record, skipped when the kind is unknown (v3 forward
        // compat); the known metadata kind is CRC-verified and returned.
        // Returns true only when a metadata payload was captured.
        [[nodiscard]] bool read_extra_record(std::istream& is, std::string& metadata_out) {
            const auto kind_byte = is.get();
            if (kind_byte == std::char_traits<char>::eof()) {
                throw ErrorHandling::CorruptFile("truncated reading extra record kind");
            }
            const auto kind = static_cast<uint8_t>(kind_byte);
            auto name = read_string(is, "extra record name");
            const auto crc_expected = read_scalar<uint32_t>(is, "extra record crc");
            const auto size = read_scalar<uint64_t>(is, "extra record size");
            if (size > k_max_tensor_bytes) {
                throw ErrorHandling::CorruptFile("extra record size exceeds the format limit");
            }
            if (size > remaining_bytes(is)) {
                throw ErrorHandling::CorruptFile("extra record payload overruns the file");
            }
            if (kind != k_kind_metadata) {
                is.seekg(static_cast<std::streamoff>(size), std::ios::cur);
                if (!is) {
                    throw ErrorHandling::CorruptFile("truncated reading extra record payload");
                }
                return false;
            }
            auto payload = std::string(static_cast<size_t>(size), '\0');
            is.read(payload.data(), static_cast<std::streamsize>(size));
            if (static_cast<uint64_t>(is.gcount()) != size) {
                throw ErrorHandling::CorruptFile("truncated reading extra record payload");
            }
            if (crc32(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()) != crc_expected) {
                throw ErrorHandling::CorruptFile("CRC32 mismatch on extra record '" + name + "'");
            }
            if (name == k_metadata_name) {
                metadata_out = std::move(payload);
                return true;
            }
            return false;
        }

        // ---- Layer factory table ----
        using Factory = std::function<std::unique_ptr<Modules::Module>(const Modules::ModuleSpec&, DataTypes::DType, Tensors::Device)>;

        [[nodiscard]] const std::unordered_map<std::string, Factory>& factories() {
            static const auto table = [] {
                auto t = std::unordered_map<std::string, Factory>();
                t.emplace("Linear", [](const Modules::ModuleSpec& s, const DataTypes::DType dt, const Tensors::Device dv) {
                    return Modules::Linear::construct(s, dt, dv);
                });
                t.emplace("Conv2d", [](const Modules::ModuleSpec& s, const DataTypes::DType dt, const Tensors::Device dv) {
                    return Modules::Conv2d::construct(s, dt, dv);
                });
                t.emplace("MaxPool2d", [](const Modules::ModuleSpec& s, const DataTypes::DType, const Tensors::Device) {
                    return Modules::MaxPool2d::construct(s, DataTypes::DType::Float32, Tensors::Device::CPU);
                });
                t.emplace("AvgPool2d", [](const Modules::ModuleSpec& s, const DataTypes::DType, const Tensors::Device) {
                    return Modules::AvgPool2d::construct(s, DataTypes::DType::Float32, Tensors::Device::CPU);
                });
                t.emplace("Dropout", [](const Modules::ModuleSpec& s, const DataTypes::DType, const Tensors::Device) {
                    return Modules::Dropout::construct(s, DataTypes::DType::Float32, Tensors::Device::CPU);
                });
                t.emplace("Flatten", [](const Modules::ModuleSpec&, const DataTypes::DType, const Tensors::Device) {
                    return std::make_unique<Modules::Flatten>();
                });
                t.emplace("Sigmoid", [](const Modules::ModuleSpec&, const DataTypes::DType, const Tensors::Device) {
                    return std::make_unique<Modules::Sigmoid>();
                });
                t.emplace("Tanh", [](const Modules::ModuleSpec&, const DataTypes::DType, const Tensors::Device) {
                    return std::make_unique<Modules::Tanh>();
                });
                t.emplace("Gelu", [](const Modules::ModuleSpec&, const DataTypes::DType, const Tensors::Device) {
                    return std::make_unique<Modules::Gelu>();
                });
                t.emplace("Softmax", [](const Modules::ModuleSpec&, const DataTypes::DType, const Tensors::Device) {
                    return std::make_unique<Modules::Softmax>();
                });
                t.emplace("Relu", [](const Modules::ModuleSpec&, const DataTypes::DType, const Tensors::Device) {
                    return std::make_unique<Modules::Relu>();
                });
                t.emplace("BatchNorm", [](const Modules::ModuleSpec& s, const DataTypes::DType dt, const Tensors::Device dv) {
                    return Modules::BatchNorm::construct(s, dt, dv);
                });
                t.emplace("Sequential", [](const Modules::ModuleSpec&, const DataTypes::DType, const Tensors::Device) {
                    return std::make_unique<Modules::Sequential>();
                });
                return t;
            }();
            return table;
        }

        [[nodiscard]] std::unique_ptr<Modules::Module> build_module(const Modules::ModuleSpec& spec, const DataTypes::DType dtype, const Tensors::Device device, const uint32_t depth = 0) {
            if (depth > k_max_spec_depth) {
                throw ErrorHandling::CorruptFile("module spec nesting exceeds the depth limit");
            }
            const auto& table = factories();
            const auto it = table.find(spec.type);
            if (it == table.end()) {
                throw ErrorHandling::UnknownModuleType(spec.type);
            }
            // Constructor/adoption failures (range violations, duplicate
            // names, non-container adoption) are format errors here: the file
            // is malformed, so the caller gets CorruptFile, never an
            // implementation error leaking out of load().
            std::unique_ptr<Modules::Module> module;
            try {
                module = it->second(spec, dtype, device);
                for (auto&[child_name, child_spec] : spec.children) {
                    module->adopt(child_name, build_module(child_spec, dtype, device, depth + 1u));
                }
            } catch (const ErrorHandling::SerializationError&) {
                throw;
            } catch (const std::exception& e) {
                throw ErrorHandling::CorruptFile(std::string("invalid module spec for type '") + spec.type + "': " + e.what());
            }
            return module;
        }
    }

    /**
     * Writes a module tree to a .bt file. Traverses the registry via the
     * virtual Module::spec()/Module::named_children() hooks — no switch on
     * class names — and writes every parameter plus every named state tensor
     * with a CRC32 checksum over the raw bytes. When @p compress is true, a
     * record whose LZ77 form is smaller than its raw form is stored
     * compressed; otherwise it is stored raw.
     * @brief Export a module tree to a .bt file.
     * @param module The module tree to export.
     * @param path Destination file path, replaced when present.
     * @param compress Whether to store LZ77-compressed records when smaller.
     * @param level How much optimizer state to include; Weights writes the
     *        original version-1 file, the other levels write version 2 with
     *        gradient and/or Adam sections appended.
     * @throws ErrorHandling::SerializationError the file cannot be opened or written.
     * @throws ErrorHandling::NotSerializable a module in the tree does not export a spec.
     */
    void save(const Modules::Module& module, const std::string& path, const bool compress, const CheckpointLevel level, const std::string metadata) {
        auto out = std::ofstream(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw ErrorHandling::SerializationError("cannot open '" + path + "' for writing");
        }
        out.write(reinterpret_cast<const char*>(k_magic.data()), static_cast<std::streamsize>(k_magic.size()));
        const auto with_grads = level == CheckpointLevel::WeightsAndGrads || level == CheckpointLevel::Full;
        // Any metadata payload makes the file version 3 (old builds then
        // fail cleanly with UnsupportedVersion instead of mis-parsing);
        // absent metadata keeps the byte layout of the existing versions.
        const auto has_metadata = !metadata.empty();
        const auto version = has_metadata ? k_version_metadata
            : with_grads || level == CheckpointLevel::Full ? k_version_checkpoint : k_version;
        write_scalar<uint32_t>(out, version);
        const auto flags = (with_grads ? k_flag_grads : 0u) | (level == CheckpointLevel::Full ? k_flag_adam : 0u)
            | (has_metadata ? k_flag_metadata : 0u);
        write_scalar<uint32_t>(out, flags);
        write_spec(out, module.spec());

        // Parameters: the subtree-flat qualified list, in registry order.
        auto records = std::vector<TensorRecord>();
        for (const auto&[name, tensor] : module.named_parameters()) {
            auto& t = *tensor;
            t.sync();
            auto rec = TensorRecord{};
            rec.name = name;
            rec.dtype = t.get_dtype();
            for (size_t d = 0; d < t.rank(); ++d) {
                rec.shape.push_back(t.dim(d));
            }
            rec.raw.resize(t.byte_count());
            t.read_element(0, rec.raw.data(), rec.raw.size());
            records.push_back(std::move(rec));
        }
        // Named state tensors (BatchNorm running statistics), per module.
        collect_state_records(module, "", records);

        write_scalar<uint32_t>(out, static_cast<uint32_t>(records.size()));
        for (const auto& rec : records) {
            write_record(out, rec, compress);
        }

        // Checkpoint sections (version 2): parameter slots in the same order
        // as the parameter records above. Gradients first, then Adam moments
        // with the step counter. State tensors (BatchNorm running stats) have
        // no optimizer state and are not repeated here.
        const auto params = module.named_parameters();
        if (level == CheckpointLevel::WeightsAndGrads || level == CheckpointLevel::Full) {
            write_scalar<uint32_t>(out, static_cast<uint32_t>(params.size()));
            for (const auto&[name, tensor] : params) {
                auto& t = *const_cast<Tensors::Tensor*>(tensor);
                const bool has_grad = t.has_grad();
                out.put(static_cast<char>(has_grad ? 1 : 0));
                if (has_grad) {
                    auto& g = t.gradient();
                    g.sync();
                    auto rec = TensorRecord{};
                    rec.name = name + "/grad";
                    rec.dtype = g.get_dtype();
                    for (size_t d = 0; d < g.rank(); ++d) {
                        rec.shape.push_back(g.dim(d));
                    }
                    rec.raw.resize(g.byte_count());
                    g.read_element(0, rec.raw.data(), rec.raw.size());
                    write_record(out, rec, compress);
                }
            }
        }
        if (level == CheckpointLevel::Full) {
            write_scalar<uint32_t>(out, static_cast<uint32_t>(params.size()));
            for (const auto&[name, tensor] : params) {
                auto& t = *const_cast<Tensors::Tensor*>(tensor);
                // The locked snapshot keeps the step count and moments one
                // generation even under a concurrent adam_update.
                const auto snap = t.adam_snapshot();
                write_scalar<uint8_t>(out, snap.has_value() ? 1u : 0u);
                if (snap.has_value()) {
                    write_scalar<uint64_t>(out, snap->step_count);
                    const Tensors::Tensor* m_v[] = {&snap->m, &snap->v};
                    const char* suffixes[] = {"/adam_m", "/adam_v"};
                    for (size_t k = 0; k < 2; ++k) {
                        auto& m = *const_cast<Tensors::Tensor*>(m_v[k]);
                        m.sync();
                        auto rec = TensorRecord{};
                        rec.name = name + suffixes[k];
                        rec.dtype = m.get_dtype();
                        for (size_t d = 0; d < m.rank(); ++d) {
                            rec.shape.push_back(m.dim(d));
                        }
                        rec.raw.resize(m.byte_count());
                        m.read_element(0, rec.raw.data(), rec.raw.size());
                        write_record(out, rec, compress);
                    }
                }
            }
        }
        // Trailing extra-record section (version 3): today exactly one
        // record — the opaque metadata blob the engine carries without
        // interpreting. Absent metadata means no section at all, so a
        // metadata-free file keeps the old layout byte-for-byte.
        if (has_metadata) {
            write_scalar<uint32_t>(out, 1u);
            write_extra_record(out, k_kind_metadata, std::string{k_metadata_name}, metadata);
        }
        out.flush();
        if (!out) {
            throw ErrorHandling::SerializationError("failed writing '" + path + "'");
        }
    }

    /**
     * Reads a .bt file back into a module tree. The architecture is rebuilt
     * from the spec tree by the factory table (one entry per shipped layer
     * type), children are adopted in file order, and every tensor record is
     * verified — name, dtype, shape — against the rebuilt tree before its
     * bytes are written in. Gradient and Adam sections, when the file
     * carries them, are restored onto the rebuilt parameters after the
     * device move, so a reloaded checkpoint continues training
     * bit-reproducibly.
     * @brief Import a module tree from a .bt file.
     * @param path Source file path.
     * @param device Device to hold the rebuilt tensors on.
     * @return The reconstructed module tree.
     * @throws ErrorHandling::SerializationError the file cannot be opened or read.
     * @throws ErrorHandling::CorruptFile bad magic, truncation, CRC mismatch, or a record that does not match the rebuilt architecture.
     * @throws ErrorHandling::UnsupportedVersion the file version is newer than this build reads.
     * @throws ErrorHandling::UnknownModuleType the spec names a type no factory constructs.
     */
    std::unique_ptr<Modules::Module> load(const std::string& path, const Tensors::Device device) {
        auto in = std::ifstream(path, std::ios::binary);
        if (!in) {
            throw ErrorHandling::SerializationError("cannot open '" + path + "' for reading");
        }
        auto magic = std::array<uint8_t, 4>{};
        in.read(reinterpret_cast<char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
        if (in.gcount() != 4 || magic != k_magic) {
            throw ErrorHandling::CorruptFile("bad magic — not a .bt file (or wrong format version)");
        }
        const auto version = read_scalar<uint32_t>(in, "format version");
        if (version > k_version_metadata) {
            throw ErrorHandling::UnsupportedVersion("file version " + std::to_string(version) + ", this build reads up to " + std::to_string(k_version_metadata));
        }
        if (version == 0) {
            throw ErrorHandling::CorruptFile("undefined .bt version 0");
        }
        const auto flags = read_scalar<uint32_t>(in, "flags");
        if ((flags & ~k_flag_mask) != 0u) {
            throw ErrorHandling::CorruptFile("unsupported .bt flags " + std::to_string(flags));
        }
        if (version == k_version && flags != 0u) {
            throw ErrorHandling::CorruptFile("version 1 .bt file carries checkpoint flags " + std::to_string(flags));
        }
        if (version == k_version_checkpoint && (flags & k_flag_metadata) != 0u) {
            throw ErrorHandling::CorruptFile("version 2 .bt file carries the metadata flag — metadata files are version 3");
        }
        auto spec = read_spec(in, 0);
        auto module = build_module(spec, DataTypes::DType::Float32, Tensors::Device::CPU);

        const auto n_records = read_scalar<uint32_t>(in, "tensor record count");
        if (n_records > k_max_records) {
            throw ErrorHandling::CorruptFile("unreasonable tensor record count " + std::to_string(n_records));
        }
        auto slots = tensor_slots(*module);
        if (slots.size() != n_records) {
            throw ErrorHandling::CorruptFile("file holds " + std::to_string(n_records) + " tensor records but the rebuilt architecture has " + std::to_string(slots.size()) + " slots");
        }
        for (size_t i = 0; i < slots.size(); ++i) {
            auto rec = read_record(in);
            auto&[expected_name, tensor] = slots[i];
            if (rec.name != expected_name) {
                throw ErrorHandling::CorruptFile("tensor record '" + rec.name + "' does not match the expected slot '" + expected_name + "'");
            }
            if (rec.dtype != tensor->get_dtype()) {
                // The record's dtype is authoritative: restore it before the
                // bytes land. Conversion is value-exact per the existing
                // conversion contract, and raw bytes are then written into
                // the converted buffer, so the roundtrip stays bit-exact.
                tensor->convert_dtype(rec.dtype);
                tensor->sync();
            }
            if (rec.shape.size() != tensor->rank()) {
                throw ErrorHandling::CorruptFile("rank mismatch on '" + rec.name + "'");
            }
            for (size_t d = 0; d < rec.shape.size(); ++d) {
                if (rec.shape[d] != tensor->dim(d)) {
                    throw ErrorHandling::CorruptFile("shape mismatch on '" + rec.name + "'");
                }
            }
            if (rec.raw.size() != tensor->byte_count()) {
                throw ErrorHandling::CorruptFile("byte size mismatch on '" + rec.name + "'");
            }
            tensor->write_element(0, rec.raw.data(), rec.raw.size());
        }
        if (device != Tensors::Device::CPU) {
            for (auto&[_, tensor] : slots) {
                tensor->to(device);
            }
        }

        // Checkpoint sections (version 2): gradients, then Adam moments. The
        // device move happens first so restored buffers are built directly on
        // the target device (write_element drives the H2D copy), keeping
        // optimizer state valid where the parameters live.
        const auto param_slots = module->named_parameters();
        if ((flags & k_flag_grads) != 0u) {
            const auto n_grads = read_scalar<uint32_t>(in, "gradient record count");
            if (n_grads != param_slots.size()) {
                throw ErrorHandling::CorruptFile("gradient record count does not match the parameter count");
            }
            for (size_t i = 0; i < param_slots.size(); ++i) {
                const auto present = read_scalar<uint8_t>(in, "gradient presence flag");
                if (present > 1) {
                    throw ErrorHandling::CorruptFile("malformed gradient presence flag");
                }
                if (present == 0) {
                    continue;
                }
                auto rec = read_record(in);
                auto& param = *param_slots[i].second;
                if (rec.name != param_slots[i].first + "/grad") {
                    throw ErrorHandling::CorruptFile("gradient record name mismatch on '" + rec.name + "'");
                }
                // Validate against the parameter BEFORE constructing anything
                // from file-controlled extents: a malformed file must raise
                // CorruptFile, never a construction-time allocation error or
                // a post-hoc shape complaint. (Gradient dtypes are pinned per
                // leaf by backward and may legitimately differ from the
                // parameter's, so only the shape is checked here.)
                if (!shape_matches_param(rec, param)) {
                    throw ErrorHandling::CorruptFile("gradient record shape does not match the parameter on '" + rec.name + "'");
                }
                auto g = Tensors::Tensor(rec.shape, rec.dtype, param.get_device(), false);
                if (rec.raw.size() != g.byte_count()) {
                    throw ErrorHandling::CorruptFile("gradient byte size mismatch on '" + rec.name + "'");
                }
                g.write_element(0, rec.raw.data(), rec.raw.size());
                param.restore_grad(std::move(g));
            }

        }
        if ((flags & k_flag_adam) != 0u) {
            const auto n_adam = read_scalar<uint32_t>(in, "adam record count");
            if (n_adam != param_slots.size()) {
                throw ErrorHandling::CorruptFile("adam record count does not match the parameter count");
            }
            for (size_t i = 0; i < param_slots.size(); ++i) {
                const auto present = read_scalar<uint8_t>(in, "adam presence flag");
                if (present > 1) {
                    throw ErrorHandling::CorruptFile("malformed adam presence flag");
                }
                if (present == 0) {
                    continue;
                }
                const auto step_count = read_scalar<uint64_t>(in, "adam step count");
                auto rec_m = read_record(in);
                auto rec_v = read_record(in);
                auto& param = *param_slots[i].second;
                if (rec_m.name != param_slots[i].first + "/adam_m" || rec_v.name != param_slots[i].first + "/adam_v") {
                    throw ErrorHandling::CorruptFile("adam record name mismatch on '" + param_slots[i].first + "'");
                }
                // Moment buffers must match the parameter's shape AND dtype
                // (adam_update resets entries whose dtype drifts), validated
                // before any allocation so malformed files stay CorruptFile.
                if (!shape_matches_param(rec_m, param) || !shape_matches_param(rec_v, param) ||
                    rec_m.dtype != param.get_dtype() || rec_v.dtype != param.get_dtype()) {
                    throw ErrorHandling::CorruptFile("adam record shape or dtype does not match the parameter on '" + param_slots[i].first + "'");
                }
                auto m = Tensors::Tensor(rec_m.shape, rec_m.dtype, param.get_device(), false);
                auto v = Tensors::Tensor(rec_v.shape, rec_v.dtype, param.get_device(), false);
                if (rec_m.raw.size() != m.byte_count() || rec_v.raw.size() != v.byte_count()) {
                    throw ErrorHandling::CorruptFile("adam byte size mismatch on '" + param_slots[i].first + "'");
                }
                m.write_element(0, rec_m.raw.data(), rec_m.raw.size());
                v.write_element(0, rec_v.raw.data(), rec_v.raw.size());
                param.adam_restore_moments(m, v, step_count);
            }
        }

        // Trailing extra-record section (version 3 only): the metadata
        // record is CRC-verified and ignored (load does not consume it —
        // read_metadata is the accessor), and unknown kinds are SKIPPED so
        // a future v3 file with new record kinds still loads here.
        if (version == k_version_metadata) {
            const auto n_extra = read_scalar<uint32_t>(in, "extra record count");
            if (n_extra > k_max_records) {
                throw ErrorHandling::CorruptFile("unreasonable extra record count " + std::to_string(n_extra));
            }
            auto ignored_metadata = std::string{};
            for (uint32_t i = 0; i < n_extra; ++i) {
                static_cast<void>(read_extra_record(in, ignored_metadata));
            }
        }

        // Trailing bytes after everything the version defines mean the file
        // was appended to or mis-parsed; refuse rather than silently ignore.
        if (in.peek() != std::char_traits<char>::eof()) {
            throw ErrorHandling::CorruptFile("trailing bytes after the last record");
        }
        return module;
    }

    /**
     * Reads only as much of a .bt file as needed to reach the trailing
     * extra-record section: header, spec tree, then each tensor/gradient/adam
     * record is skipped via its framing (seekg past the stored payload — no
     * decompression). The metadata record (kind 1, name "btui.config") is
     * CRC-verified before its payload is returned; unknown kinds are skipped
     * (the version-3 forward-compat rule). The engine never parses the blob.
     * @brief Read a .bt file's metadata payload.
     * @param path Source file path.
     * @return The metadata record's payload, or "" when the file carries none.
     * @throws ErrorHandling::SerializationError the file cannot be opened or read.
     * @throws ErrorHandling::CorruptFile bad magic, truncation, or a CRC mismatch on the metadata record.
     * @throws ErrorHandling::UnsupportedVersion the file version is newer than this build reads.
     */
    std::string read_metadata(const std::string& path) {
        auto in = std::ifstream(path, std::ios::binary);
        if (!in) {
            throw ErrorHandling::SerializationError("cannot open '" + path + "' for reading");
        }
        auto magic = std::array<uint8_t, 4>{};
        in.read(reinterpret_cast<char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
        if (in.gcount() != 4 || magic != k_magic) {
            throw ErrorHandling::CorruptFile("bad magic — not a .bt file (or wrong format version)");
        }
        const auto version = read_scalar<uint32_t>(in, "format version");
        if (version > k_version_metadata) {
            throw ErrorHandling::UnsupportedVersion("file version " + std::to_string(version) + ", this build reads up to " + std::to_string(k_version_metadata));
        }
        if (version == 0) {
            throw ErrorHandling::CorruptFile("undefined .bt version 0");
        }
        const auto flags = read_scalar<uint32_t>(in, "flags");
        if ((flags & ~k_flag_mask) != 0u) {
            throw ErrorHandling::CorruptFile("unsupported .bt flags " + std::to_string(flags));
        }
        static_cast<void>(read_spec(in, 0));
        const auto n_records = read_scalar<uint32_t>(in, "tensor record count");
        if (n_records > k_max_records) {
            throw ErrorHandling::CorruptFile("unreasonable tensor record count " + std::to_string(n_records));
        }
        for (uint32_t i = 0; i < n_records; ++i) {
            skip_record(in);
        }
        if ((flags & k_flag_grads) != 0u) {
            const auto n_grads = read_scalar<uint32_t>(in, "gradient record count");
            for (uint32_t i = 0; i < n_grads; ++i) {
                const auto present = read_scalar<uint8_t>(in, "gradient presence flag");
                if (present > 1) {
                    throw ErrorHandling::CorruptFile("malformed gradient presence flag");
                }
                if (present != 0u) {
                    skip_record(in);
                }
            }
        }
        if ((flags & k_flag_adam) != 0u) {
            const auto n_adam = read_scalar<uint32_t>(in, "adam record count");
            for (uint32_t i = 0; i < n_adam; ++i) {
                const auto present = read_scalar<uint8_t>(in, "adam presence flag");
                if (present > 1) {
                    throw ErrorHandling::CorruptFile("malformed adam presence flag");
                }
                if (present != 0u) {
                    static_cast<void>(read_scalar<uint64_t>(in, "adam step count"));
                    skip_record(in);
                    skip_record(in);
                }
            }
        }
        if (version != k_version_metadata) {
            return "";
        }
        const auto n_extra = read_scalar<uint32_t>(in, "extra record count");
        if (n_extra > k_max_records) {
            throw ErrorHandling::CorruptFile("unreasonable extra record count " + std::to_string(n_extra));
        }
        auto metadata = std::string{};
        for (uint32_t i = 0; i < n_extra; ++i) {
            static_cast<void>(read_extra_record(in, metadata));
        }
        return metadata;
    }
}
