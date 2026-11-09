/**
 * @file Csv.cpp
 * @author GreenBean
 * @brief Implementation of the CSV dataset loader declared in Csv.h. The
 * parser is single-pass and locale-independent (std::from_chars for all
 * numeric parsing): it reads the file line by line, splits on the inferred
 * or chosen delimiter, classifies each column once from its first non-empty
 * cell, validates every row, and bulk-writes the tensors with a single
 * write_element per buffer.
 * @version 0.1
 * @date 2026-11-09
 */
#include "Data/Csv.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"

namespace BeanTensor::Data {
    namespace {
        using Tensors::Device;
        using Tensors::Tensor;
        using DType = DataTypes::DType;

        [[nodiscard]] char delimiter_char(const CsvDelimiter d) {
            switch (d) {
                case CsvDelimiter::Comma: return ',';
                case CsvDelimiter::Semicolon: return ';';
                case CsvDelimiter::Tab: return '\t';
                case CsvDelimiter::Auto: break;
            }
            throw ErrorHandling::UnReachableCode();
        }

        [[nodiscard]] char infer_delimiter(const std::string& first_line) {
            const auto count = [&first_line](const char d) {
                return static_cast<size_t>(std::count(first_line.begin(), first_line.end(), d));
            };
            const auto comma = count(',');
            const auto semi = count(';');
            const auto tab = count('\t');
            if (semi > comma && semi > tab) {
                return ';';
            }
            if (tab > comma && tab > semi) {
                return '\t';
            }
            return ',';
        }

        [[nodiscard]] std::vector<std::string> split_line(const std::string& line, const char delim) {
            auto cells = std::vector<std::string>{};
            auto start = size_t{0};
            while (true) {
                const auto pos = line.find(delim, start);
                if (pos == std::string::npos) {
                    cells.push_back(line.substr(start));
                    break;
                }
                cells.push_back(line.substr(start, pos - start));
                start = pos + 1;
            }
            // Tolerate CRLF line endings on the final cell.
            auto& last = cells.back();
            if (!last.empty() && last.back() == '\r') {
                last.pop_back();
            }
            return cells;
        }

        [[nodiscard]] bool blank(const std::string& line) {
            return line.find_first_not_of(" \t\r") == std::string::npos;
        }

        /**
         * Locale-independent full-string parse as float64_t. Accepts the
         * same syntax std::from_chars does (plain decimal, exponent, inf/nan
         * spellings).
         */
        [[nodiscard]] bool parse_number(const std::string& cell, float64_t& out) {
            if (cell.empty()) {
                return false;
            }
            const auto* begin = cell.data();
            const auto* end = cell.data() + cell.size();
            const auto res = std::from_chars(begin, end, out);
            return res.ec == std::errc() && res.ptr == end;
        }

        [[nodiscard]] bool is_integral(const float64_t v) {
            // (comparison without ==/!= keeps -Werror=float-equal quiet)
            return std::isfinite(v) && !(std::fabs(v - std::trunc(v)) > 0.0) &&
                   v >= -9007199254740992.0 && v <= 9007199254740992.0;
        }

        enum class ColumnKind : uint8_t {
            Numeric,
            Categorical,
        };
    }

    CsvDataset load_csv(const std::string& path, CsvOptions options) {
        auto in = std::ifstream(path);
        if (!in) {
            throw ErrorHandling::SerializationError("cannot open '" + path + "' for reading");
        }

        // Collect the physical lines, skipping blanks, and pick the delimiter
        // from the first kept line when inference is requested.
        auto lines = std::vector<std::string>{};
        auto raw = std::string{};
        while (std::getline(in, raw)) {
            if (!blank(raw)) {
                lines.push_back(raw);
            }
        }
        if (lines.empty()) {
            throw ErrorHandling::EmptyFile("'" + path + "' has no data rows");
        }
        const auto delim = options.delimiter == CsvDelimiter::Auto
            ? infer_delimiter(lines.front())
            : delimiter_char(options.delimiter);

        auto header = std::vector<std::string>{};
        auto first_data = size_t{0};
        if (options.has_header.has_value()) {
            if (*options.has_header) {
                header = split_line(lines.front(), delim);
                first_data = 1;
            }
        } else {
            // Automatic: the first line is a header when any cell fails to
            // parse as a number.
            const auto cells = split_line(lines.front(), delim);
            auto all_numeric = true;
            for (const auto& cell : cells) {
                auto v = float64_t{0};
                if (!parse_number(cell, v)) {
                    all_numeric = false;
                    break;
                }
            }
            if (!all_numeric) {
                header = cells;
                first_data = 1;
            }
        }
        if (first_data >= lines.size()) {
            throw ErrorHandling::EmptyFile("'" + path + "' has no data rows");
        }

        // Column count comes from the first data row; every other row must match.
        const auto first_cells = split_line(lines[first_data], delim);
        const auto n_cols = first_cells.size();
        if (n_cols < 2) {
            throw ErrorHandling::RaggedCsv("need at least a feature and a label column, found " + std::to_string(n_cols));
        }

        // Label column index.
        auto label_idx = n_cols - 1;
        if (options.label_column.has_value()) {
            label_idx = n_cols;
            for (size_t i = 0; i < n_cols; ++i) {
                const auto name = i < header.size() && !header[i].empty()
                    ? header[i]
                    : "col" + std::to_string(i);
                if (name == *options.label_column) {
                    label_idx = i;
                    break;
                }
            }
            if (label_idx >= n_cols) {
                throw ErrorHandling::NameError("no column named '" + *options.label_column + "'");
            }
        }

        // Classify each column from its first non-empty cell.
        const auto n_rows = lines.size() - first_data;
        auto kinds = std::vector<ColumnKind>(n_cols, ColumnKind::Numeric);
        auto seen = std::vector<bool>(n_cols, false);
        auto any_unseen = [&seen] {
            for (const auto s : seen) {
                if (!s) {
                    return true;
                }
            }
            return false;
        };
        for (auto r = size_t{0}; r < n_rows && any_unseen(); ++r) {
            const auto cells = split_line(lines[first_data + r], delim);
            if (cells.size() != n_cols) {
                throw ErrorHandling::RaggedCsv("row " + std::to_string(r) + " has " + std::to_string(cells.size()) +
                                               " cells, expected " + std::to_string(n_cols));
            }
            for (size_t c = 0; c < n_cols; ++c) {
                if (seen[c] || cells[c].empty()) {
                    continue;
                }
                auto v = float64_t{0};
                kinds[c] = parse_number(cells[c], v) ? ColumnKind::Numeric : ColumnKind::Categorical;
                seen[c] = true;
            }
        }

        // Feature column indices and names.
        auto feature_idx = std::vector<size_t>{};
        auto ds = CsvDataset{};
        for (size_t c = 0; c < n_cols; ++c) {
            if (c == label_idx) {
                continue;
            }
            feature_idx.push_back(c);
            ds.feature_names.push_back(c < header.size() && !header[c].empty() ? header[c] : "col" + std::to_string(c));
        }

        // Numeric staging: everything parses into a float64 buffer; stats,
        // null filling, and normalization happen here, and the dtype
        // conversion to options.feature_dtype is the last step.
        const auto fdt = options.feature_dtype;
        const auto total = n_rows * feature_idx.size();
        auto f64 = std::vector<float64_t>(total, std::numeric_limits<float64_t>::quiet_NaN());
        auto labels = std::vector<int32_t>(n_rows);
        auto label_ids = std::unordered_map<std::string, int32_t>{};

        for (auto r = size_t{0}; r < n_rows; ++r) {
            const auto cells = split_line(lines[first_data + r], delim);
            if (cells.size() != n_cols) {
                throw ErrorHandling::RaggedCsv("row " + std::to_string(r) + " has " + std::to_string(cells.size()) +
                                               " cells, expected " + std::to_string(n_cols));
            }
            for (size_t f = 0; f < feature_idx.size(); ++f) {
                const auto& cell = cells[feature_idx[f]];
                auto v = std::numeric_limits<float64_t>::quiet_NaN();
                if (!cell.empty()) {
                    if (kinds[feature_idx[f]] != ColumnKind::Numeric ||
                        !parse_number(cell, v)) {
                        throw ErrorHandling::NonNumericValue("'" + cell + "' at row " + std::to_string(r) +
                                                             ", column '" + ds.feature_names[f] + "'");
                    }
                } else if (kinds[feature_idx[f]] != ColumnKind::Numeric) {
                    throw ErrorHandling::NonNumericValue("empty categorical cell at row " + std::to_string(r) +
                                                         ", column '" + ds.feature_names[f] + "'");
                }
                const auto slot = r * feature_idx.size() + f;
                // Integer dtypes validate here, before fill/normalize, on the
                // parsed values (filled values are derived from the same
                // column, and normalization is only sensible for float kinds).
                if (fdt == DType::Int64 || fdt == DType::Int32) {
                    if (!cell.empty() && !is_integral(v)) {
                        throw ErrorHandling::NonNumericValue("'" + cell + "' is not integral at row " + std::to_string(r));
                    }
                    if (fdt == DType::Int32 && !cell.empty() &&
                        (v < static_cast<float64_t>(INT32_MIN) || v > static_cast<float64_t>(INT32_MAX))) {
                        throw ErrorHandling::NonNumericValue("'" + cell + "' does not fit Int32 at row " + std::to_string(r));
                    }
                }
                f64[slot] = v;
            }
            // Labels: ids follow first-seen order; empty labels are errors.
            const auto& cell = cells[label_idx];
            if (cell.empty()) {
                throw ErrorHandling::NonNumericValue("empty label at row " + std::to_string(r));
            }
            if (const auto it = label_ids.find(cell); it != label_ids.end()) {
                labels[r] = it->second;
            } else {
                const auto id = static_cast<int32_t>(label_ids.size());
                label_ids.emplace(cell, id);
                ds.label_names.push_back(cell);
                labels[r] = id;
            }
        }

        // NaN-skipped per-column stats over the parsed values.
        const auto n_feat = feature_idx.size();
        ds.null_count = 0;
        for (size_t f = 0; f < n_feat; ++f) {
            auto min_v = std::numeric_limits<float64_t>::max();
            auto max_v = std::numeric_limits<float64_t>::lowest();
            auto sum = float64_t{0};
            auto seen_count = size_t{0};
            for (auto r = size_t{0}; r < n_rows; ++r) {
                const auto v = f64[r * n_feat + f];
                if (std::isnan(v)) {
                    ++ds.null_count;
                    continue;
                }
                min_v = v < min_v ? v : min_v;
                max_v = v > max_v ? v : max_v;
                sum += v;
                ++seen_count;
            }
            if (seen_count == 0) {
                min_v = std::numeric_limits<float64_t>::quiet_NaN();
                max_v = std::numeric_limits<float64_t>::quiet_NaN();
            }
            ds.column_min.push_back(min_v);
            ds.column_max.push_back(max_v);
            ds.column_mean.push_back(seen_count == 0 ? std::numeric_limits<float64_t>::quiet_NaN() : sum / static_cast<float64_t>(seen_count));
        }

        // Null policy.
        switch (options.nulls) {
            case CsvNullPolicy::KeepNaN:
                break;
            case CsvNullPolicy::Error:
                if (ds.null_count != 0) {
                    throw ErrorHandling::NullValue(std::to_string(ds.null_count) + " empty cell(s) in '" + path + "'");
                }
                break;
            case CsvNullPolicy::FillMean:
            case CsvNullPolicy::FillMedian:
            case CsvNullPolicy::FillConstant: {
                auto medians = std::vector<float64_t>{};
                if (options.nulls == CsvNullPolicy::FillMedian) {
                    for (size_t f = 0; f < n_feat; ++f) {
                        auto vals = std::vector<float64_t>{};
                        for (auto r = size_t{0}; r < n_rows; ++r) {
                            const auto v = f64[r * n_feat + f];
                            if (!std::isnan(v)) {
                                vals.push_back(v);
                            }
                        }
                        std::sort(vals.begin(), vals.end());
                        medians.push_back(vals.empty() ? float64_t{0} : vals[vals.size() / 2]);
                    }
                }
                for (auto r = size_t{0}; r < n_rows; ++r) {
                    for (size_t f = 0; f < n_feat; ++f) {
                        const auto slot = r * n_feat + f;
                        if (!std::isnan(f64[slot])) {
                            continue;
                        }
                        switch (options.nulls) {
                            case CsvNullPolicy::FillMean:
                                f64[slot] = ds.column_mean[f];
                                break;
                            case CsvNullPolicy::FillMedian:
                                f64[slot] = medians[f];
                                break;
                            case CsvNullPolicy::FillConstant:
                                f64[slot] = options.fill_value;
                                break;
                            default: throw ErrorHandling::UnReachableCode();
                        }
                    }
                }
                break;
            }
        }

        // Normalization (labels untouched).
        switch (options.normalize) {
            case CsvNormalize::None:
                break;
            case CsvNormalize::MinMax: {
                for (size_t f = 0; f < n_feat; ++f) {
                    const auto span = ds.column_max[f] - ds.column_min[f];
                    for (auto r = size_t{0}; r < n_rows; ++r) {
                        const auto slot = r * n_feat + f;
                        if (std::isnan(f64[slot])) {
                            continue;
                        }
                        f64[slot] = !(std::fabs(span) > 0.0)
                            ? float64_t{0}
                            : (f64[slot] - ds.column_min[f]) / span;
                    }
                }
                break;
            }
            case CsvNormalize::Standardize: {
                for (size_t f = 0; f < n_feat; ++f) {
                    auto sum_sq = float64_t{0};
                    auto count = size_t{0};
                    for (auto r = size_t{0}; r < n_rows; ++r) {
                        const auto v = f64[r * n_feat + f];
                        if (std::isnan(v)) {
                            continue;
                        }
                        const auto d = v - ds.column_mean[f];
                        sum_sq += d * d;
                        ++count;
                    }
                    const auto std_dev = count == 0 ? float64_t{0} : std::sqrt(sum_sq / static_cast<float64_t>(count));
                    for (auto r = size_t{0}; r < n_rows; ++r) {
                        const auto slot = r * n_feat + f;
                        if (std::isnan(f64[slot])) {
                            continue;
                        }
                        f64[slot] = !(std::fabs(std_dev) > 0.0)
                            ? float64_t{0}
                            : (f64[slot] - ds.column_mean[f]) / std_dev;
                    }
                }
                break;
            }
        }

        // Dtype conversion + bulk-write (single host->tensor copy each).
        const auto shape = std::vector<size_t>{n_rows, n_feat};
        ds.features = Tensor{shape, fdt, Device::CPU, false};
        switch (fdt) {
            case DType::Float64:
                ds.features.write_element(0, f64.data(), f64.size() * sizeof(float64_t));
                break;
            case DType::Float32: {
                auto f32 = std::vector<float32_t>(f64.begin(), f64.end());
                ds.features.write_element(0, f32.data(), f32.size() * sizeof(float32_t));
                break;
            }
            case DType::Int64: {
                auto i64 = std::vector<int64_t>(f64.size());
                for (size_t i = 0; i < f64.size(); ++i) {
                    i64[i] = std::isnan(f64[i]) ? int64_t{0} : static_cast<int64_t>(f64[i]);
                }
                ds.features.write_element(0, i64.data(), i64.size() * sizeof(int64_t));
                break;
            }
            case DType::Int32: {
                auto i32 = std::vector<int32_t>(f64.size());
                for (size_t i = 0; i < f64.size(); ++i) {
                    i32[i] = std::isnan(f64[i]) ? int32_t{0} : static_cast<int32_t>(f64[i]);
                }
                ds.features.write_element(0, i32.data(), i32.size() * sizeof(int32_t));
                break;
            }
            default: throw ErrorHandling::UnReachableCode();
        }
        ds.labels = Tensor{std::vector<size_t>{n_rows}, DType::Int32, Device::CPU, false};
        ds.labels.write_element(0, labels.data(), labels.size() * sizeof(int32_t));
        return ds;
    }
}
