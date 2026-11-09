/**
 * @file Csv.h
 * @author GreenBean
 * @brief CSV dataset loading: a file on disk becomes a tensor pair the
 * existing DataLoader consumes directly — features {N, F} in a caller-chosen
 * numeric dtype and integer class labels {N} with the string mapping kept
 * and queryable through CsvDataset::label_names. Scope ruling (docs/roadmap.md):
 * CSV is the supported standard data format for this task; the MNIST IDX
 * reader stays demo-private and formats like Parquet or Arrow are out of
 * scope. Parsing contract (docs/architecture.md): header detection is
 * automatic unless overridden; the delimiter is inferred from comma,
 * semicolon, or tab unless overridden; a column is numeric when its first
 * non-empty cell parses as a number and categorical otherwise, with label
 * ids in first-seen order; empty cells in numeric columns become NaN while
 * empty cells in categorical columns are an error; blank lines are skipped.
 * @version 0.1
 * @date 2026-11-09
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Data {

    /**
     * The supported CSV field separators. Auto lets the loader infer from
     * the first line (comma, semicolon, or tab).
     * @brief CSV delimiter selection.
     */
    enum class CsvDelimiter : uint8_t {
        Auto,
        Comma,
        Semicolon,
        Tab,
    };

    /**
     * How empty numeric cells are handled.
     * @brief CSV null policy.
     */
    enum class CsvNullPolicy : uint8_t {
        /** Keep NaN (default). */
        KeepNaN,
        /** Raise ErrorHandling::NullValue when any cell is empty. */
        Error,
        /** Fill each column's nulls with its NaN-skipped mean. */
        FillMean,
        /** Fill each column's nulls with its NaN-skipped median. */
        FillMedian,
        /** Fill nulls with CsvOptions::fill_value. */
        FillConstant,
    };

    /**
     * Per-column feature scaling, applied after null filling and before
     * the dtype conversion.
     * @brief CSV feature normalization.
     */
    enum class CsvNormalize : uint8_t {
        /** No scaling (default). */
        None,
        /** (x - min) / (max - min); constant columns become all zeros. */
        MinMax,
        /** (x - mean) / population standard deviation; constant columns become all zeros. */
        Standardize,
    };

    /**
     * Parsing knobs; every field defaults to the documented automatic
     * behavior.
     * @brief CSV loading options.
     */
    struct CsvOptions {
        /**
         * Header presence. Unset = automatic: the first line is a header
         * when any of its cells fails to parse as a number.
         */
        std::optional<bool> has_header;
        /**
         * Field separator. Auto infers; an explicit value wins.
         */
        CsvDelimiter delimiter = CsvDelimiter::Auto;
        /**
         * Name of the label column (default: the last column).
         */
        std::optional<std::string> label_column;
        /**
         * Storage dtype of the numeric feature columns (Float64 by default;
         * Int32/Int64 require every value in a column to be integral).
         * Conversion happens after null filling and normalization.
         */
        DataTypes::DType feature_dtype = DataTypes::DType::Float64;
        /**
         * Null handling for empty numeric cells. Default KeepNaN.
         */
        CsvNullPolicy nulls = CsvNullPolicy::KeepNaN;
        /**
         * The value used when CsvNullPolicy::FillConstant is chosen.
         */
        float64_t fill_value = 0.0;
        /**
         * Per-column feature scaling. Default None. The label column is
         * never scaled.
         */
        CsvNormalize normalize = CsvNormalize::None;
    };

    /**
     * A parsed CSV dataset, ready for DataLoader.
     * @brief Dataset tensor pair plus column metadata.
     */
    struct CsvDataset {
        /**
         * Default construct: empty rank-1 tensors; load_csv fills both.
         * @brief Construct an empty dataset.
         */
        CsvDataset()
            : features(std::vector<size_t>{0}, DataTypes::DType::Float64, Tensors::Device::CPU, false),
              labels(std::vector<size_t>{0}, DataTypes::DType::Int32, Tensors::Device::CPU, false) {
        }

        /**
         * Numeric feature columns, {row_count, feature_count}, in
         * CsvOptions::feature_dtype.
         */
        Tensors::Tensor features;
        /**
         * Class ids of the label column, {row_count} Int32, in first-seen
         * order; index CsvDataset::label_names by these ids.
         */
        Tensors::Tensor labels;
        /**
         * Label id -> original string, in first-seen order.
         */
        std::vector<std::string> label_names;
        /**
         * Feature column names: from the header row, or "col<index>" when
         * the file has no header.
         */
        std::vector<std::string> feature_names;
        /**
         * Total empty numeric cells found while parsing (before any fill).
         */
        size_t null_count = 0;
        /**
         * Per feature column: NaN-skipped minimum of the parsed values
         * (before filling or normalization).
         */
        std::vector<float64_t> column_min;
        /**
         * Per feature column: NaN-skipped maximum of the parsed values
         * (before filling or normalization).
         */
        std::vector<float64_t> column_max;
        /**
         * Per feature column: NaN-skipped mean of the parsed values
         * (before filling or normalization).
         */
        std::vector<float64_t> column_mean;
    };

    /**
     * Parses a CSV file into a dataset tensor pair. The label column
     * becomes Int32 class ids (first-seen order); every other column must
     * be uniformly numeric or categorical. Value-in/value-out; errors
     * raise ErrorHandling::*.
     * @brief Load a CSV file as a DataLoader-ready dataset.
     * @param path Filesystem path to the CSV file.
     * @param options Parsing options; defaults are fully automatic.
     * @return The dataset pair, label mapping, and column names.
     * @throws ErrorHandling::SerializationError the file cannot be opened.
     * @throws ErrorHandling::EmptyFile the file has no data rows.
     * @throws ErrorHandling::RaggedCsv a row has a different cell count than the rest.
     * @throws ErrorHandling::NullValue CsvNullPolicy::Error is chosen and any cell is empty.
     * @throws ErrorHandling::NonNumericValue a numeric-column cell fails to parse (including empty categorical cells and fractional values under an integer feature dtype).
     */
    [[nodiscard]] CsvDataset load_csv(const std::string& path, CsvOptions options = {});
}
