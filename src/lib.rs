//! Random data generator UDFs for Apache DataFusion.
//!
//! The crate exports DataFusion UDF constructors. It does not register
//! functions or own a `SessionContext`; your DataFusion session code chooses
//! which generators to install.
//!
//! ```no_run
//! use datafusion::prelude::SessionContext;
//!
//! let ctx = SessionContext::new();
//! for udf in datafusion_randgen::all_udfs() {
//!     ctx.register_udf(udf);
//! }
//! for udaf in datafusion_randgen::all_udafs() {
//!     ctx.register_udaf(udaf);
//! }
//! ```
//!
//! Exported UDFs are volatile. For row-wise generators, null required inputs
//! produce null output for that row. Invalid ranges and distribution parameters
//! return DataFusion errors.
//!
//! Generator UDFs except `randgen_nullable` accept an optional trailing
//! `null_probability Float64` argument. When supplied, each generated row is
//! null with that probability. The probability may be scalar or row-wise, and
//! must be finite and inside `0.0..=1.0`. A null probability value produces
//! null output for that row. Native generator nullability does not write
//! generated values under nulls. Use `randgen_nullable(value, probability)` when
//! nullability must wrap an arbitrary expression.
//!
//! With `column-choice-parquet` or `column-choice-arrow-ipc` enabled,
//! `randgen_roaring_agg(source_column)` builds a serialized roaring set and
//! `randgen_column_choice(values[, null_probability])` samples from it.

#![deny(missing_docs)]

use datafusion_expr::{AggregateUDF, ScalarUDF};

pub use crate::randgen::bool::Bool;
pub use crate::randgen::choice::Choice;
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
pub use crate::randgen::column_choice::{ColumnChoice, RoaringAgg};
pub use crate::randgen::date32::Date32;
pub use crate::randgen::float64_normal::Float64Normal;
pub use crate::randgen::float64_uniform::Float64Uniform;
pub use crate::randgen::int64_normal::Int64Normal;
pub use crate::randgen::int64_uniform::Int64Uniform;
pub use crate::randgen::nullable::Nullable;
pub use crate::randgen::timestamp_millisecond::TimestampMillisecond;
pub use crate::randgen::uint64_normal::UInt64Normal;
pub use crate::randgen::uint64_uniform::UInt64Uniform;
pub use crate::randgen::utf8::Utf8;

/// Concrete implementation modules for the exported random generator UDFs.
///
/// Prefer the top-level `*_udf` constructors or [`all_udfs`]. The module stays
/// public for callers that need direct access to the `ScalarUDFImpl` types.
pub mod randgen;

/// Builds `randgen_int64_uniform(min, max[, null_probability])`.
///
/// The UDF returns an `Int64` sampled from the inclusive range `min..=max`.
/// Null bounds produce null output for that row. Non-null bounds must satisfy
/// `min <= max`.
pub fn int64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(Int64Uniform::new())
}

/// Builds `randgen_uint64_uniform(min, max[, null_probability])`.
///
/// The UDF returns a `UInt64` sampled from the inclusive range `min..=max`.
/// The generator covers the full `UInt64` domain, including values that cannot
/// be represented by `Int64`. Arguments may be `UInt64` values or nonnegative
/// signed integer values. Null bounds produce null output for that row.
/// Non-null bounds must satisfy `min <= max`.
pub fn uint64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(UInt64Uniform::new())
}

/// Builds `randgen_float64_uniform(min, max[, null_probability])`.
///
/// The UDF returns a `Float64` sampled from the inclusive range `min..=max`.
/// Bounds and their span must be finite, and `min <= max`.
pub fn float64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(Float64Uniform::new())
}

/// Builds `randgen_float64_normal(mean, stddev[, null_probability])`.
///
/// The UDF returns a `Float64` sampled from a normal distribution. Arguments
/// must be finite, and `stddev` must be greater than zero.
pub fn float64_normal_udf() -> ScalarUDF {
    ScalarUDF::from(Float64Normal::new())
}

/// Builds `randgen_int64_normal(min, max, mean, stddev[, null_probability])`.
///
/// The UDF returns an `Int64` in the inclusive range `min..=max`. The caller
/// supplies `stddev`; the range is a truncation bound, not an input used to
/// calculate standard deviation. `mean` may be outside the range. Sampling uses
/// a rounded f64 normal and retries values outside the truncation range. Very
/// low-probability tail ranges can error after a bounded number of retries.
/// `min <= max`, and `stddev` must be finite and greater than zero.
pub fn int64_normal_udf() -> ScalarUDF {
    ScalarUDF::from(Int64Normal::new())
}

/// Builds `randgen_uint64_normal(min, max, mean, stddev[, null_probability])`.
///
/// The UDF returns a `UInt64` in the inclusive range `min..=max`. The caller
/// supplies `stddev`; the range is a truncation bound, not an input used to
/// calculate standard deviation. `mean` may be outside the range.
/// `min`, `max`, and `mean` may be `UInt64` values or nonnegative signed integer
/// values. Sampling uses a rounded f64 normal and retries values outside the
/// truncation range. Very low-probability tail ranges can error after a bounded
/// number of retries. `min <= max`, and `stddev` must be finite and greater than
/// zero.
pub fn uint64_normal_udf() -> ScalarUDF {
    ScalarUDF::from(UInt64Normal::new())
}

/// Builds `randgen_bool(probability[, null_probability])`.
///
/// The UDF returns `true` with the given probability. The first probability
/// controls true/false output; the optional second probability controls null
/// output. Probabilities must be finite and inside `0.0..=1.0`.
pub fn bool_udf() -> ScalarUDF {
    ScalarUDF::from(Bool::new())
}

/// Builds `randgen_utf8(characters, min_length, max_length[, null_probability])`.
///
/// The UDF returns a UTF-8 string made from the distinct characters in
/// `characters`. Length bounds are measured in characters and must satisfy
/// `0 <= min_length <= max_length`.
pub fn utf8_udf() -> ScalarUDF {
    ScalarUDF::from(Utf8::new())
}

/// Builds `randgen_choice(choices[, null_probability])`.
///
/// The UDF accepts a `List<T>` and returns a randomly selected item with type
/// `T`. Each non-null list must contain at least one element.
pub fn choice_udf() -> ScalarUDF {
    ScalarUDF::from(Choice::new())
}

/// Builds `randgen_nullable(value, probability)`.
///
/// The UDF returns `value` with the same data type and replaces rows with null
/// at the supplied probability. Existing nulls stay null. The probability must
/// be finite and inside `0.0..=1.0`. Rows that become null are rebuilt as null
/// values instead of carrying hidden input values under a validity bitmap.
pub fn nullable_udf() -> ScalarUDF {
    ScalarUDF::from(Nullable::new())
}

/// Builds `randgen_column_choice(values[, null_probability])`.
///
/// Samples with replacement from a scalar serialized roaring set produced by
/// `randgen_roaring_agg`.
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
pub fn column_choice_udf() -> ScalarUDF {
    ScalarUDF::from(ColumnChoice::new())
}

/// Builds column-choice scalar UDFs.
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
pub fn column_choice_udfs() -> Vec<ScalarUDF> {
    crate::randgen::column_choice::column_choice_udfs()
}

/// Builds `randgen_roaring_agg(source_column)`.
///
/// The aggregate accepts `UInt32` or `UInt64` and returns a serialized roaring
/// value for `randgen_column_choice`.
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
pub fn roaring_agg_udaf() -> AggregateUDF {
    AggregateUDF::from(RoaringAgg::new())
}

/// Builds column-choice aggregate UDFs.
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
pub fn column_choice_udafs() -> Vec<AggregateUDF> {
    vec![roaring_agg_udaf()]
}

/// Builds `randgen_date32(min, max[, null_probability])`.
///
/// The UDF returns a `Date32` sampled from the inclusive day range `min..=max`.
/// Non-null bounds must satisfy `min <= max`.
pub fn date32_udf() -> ScalarUDF {
    ScalarUDF::from(Date32::new())
}

/// Builds `randgen_timestamp_millisecond(min, max[, null_probability])`.
///
/// The UDF returns a millisecond timestamp sampled from the inclusive range
/// `min..=max`. Both arguments must use millisecond precision and matching
/// timezones.
pub fn timestamp_millisecond_udf() -> ScalarUDF {
    ScalarUDF::from(TimestampMillisecond::new())
}

/// Builds all scalar UDFs exported by this crate.
///
/// Register these when a session should expose the whole generator set. With
/// a column-choice source feature enabled, the list includes
/// `randgen_column_choice`. Use [`all_udafs`] to register aggregate UDFs.
pub fn all_udfs() -> Vec<ScalarUDF> {
    let udfs = vec![
        int64_uniform_udf(),
        uint64_uniform_udf(),
        float64_uniform_udf(),
        float64_normal_udf(),
        int64_normal_udf(),
        uint64_normal_udf(),
        bool_udf(),
        utf8_udf(),
        choice_udf(),
        nullable_udf(),
        date32_udf(),
        timestamp_millisecond_udf(),
    ];

    #[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
    let udfs = {
        let mut udfs = udfs;
        udfs.extend(column_choice_udfs());
        udfs
    };

    udfs
}

/// Builds all aggregate UDFs exported by this crate.
pub fn all_udafs() -> Vec<AggregateUDF> {
    let udafs = vec![];

    #[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
    let udafs = {
        let mut udafs = udafs;
        udafs.extend(column_choice_udafs());
        udafs
    };

    udafs
}
