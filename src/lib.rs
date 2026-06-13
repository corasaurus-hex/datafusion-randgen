//! Random data generator UDFs for Apache DataFusion.
//!
//! This crate exports `ScalarUDF` constructors. It does not register functions
//! or own a `SessionContext`; the application that owns the DataFusion session
//! chooses which generators to install.
//!
//! ```no_run
//! use datafusion::prelude::SessionContext;
//!
//! let ctx = SessionContext::new();
//! for udf in datafusion_randgen::all_udfs() {
//!     ctx.register_udf(udf);
//! }
//! ```
//!
//! Every exported UDF is volatile. Null required inputs produce null output for
//! that row. Invalid ranges and distribution parameters return DataFusion
//! errors.

#![deny(missing_docs)]

use datafusion_expr::ScalarUDF;

pub use crate::randgen::bool::Bool;
pub use crate::randgen::choice::Choice;
pub use crate::randgen::date32::Date32;
pub use crate::randgen::float64_normal::Float64Normal;
pub use crate::randgen::float64_uniform::Float64Uniform;
pub use crate::randgen::int64_normal::Int64Normal;
pub use crate::randgen::int64_uniform::Int64Uniform;
pub use crate::randgen::timestamp_millisecond::TimestampMillisecond;
pub use crate::randgen::uint64_normal::UInt64Normal;
pub use crate::randgen::uint64_uniform::UInt64Uniform;
pub use crate::randgen::utf8::Utf8;

/// Concrete implementation modules for the exported random generator UDFs.
///
/// Most callers should use the top-level `*_udf` constructors or [`all_udfs`].
/// This module is public for callers that need direct access to the
/// `ScalarUDFImpl` types.
pub mod randgen;

/// Builds `randgen_int64_uniform(min, max)`.
///
/// The UDF returns an `Int64` sampled from the inclusive range `min..=max`.
/// Null bounds produce null output for that row. Non-null bounds must satisfy
/// `min <= max`.
pub fn int64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(Int64Uniform::new())
}

/// Builds `randgen_uint64_uniform(min, max)`.
///
/// The UDF returns a `UInt64` sampled from the inclusive range `min..=max`.
/// This covers the full `UInt64` domain, including values that cannot be
/// represented by `Int64`. Arguments may be `UInt64` values or nonnegative
/// signed integer values. Null bounds produce null output for that row. Non-null
/// bounds must satisfy `min <= max`.
pub fn uint64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(UInt64Uniform::new())
}

/// Builds `randgen_float64_uniform(min, max)`.
///
/// The UDF returns a `Float64` sampled from the inclusive range `min..=max`.
/// Bounds and their span must be finite, and `min <= max`.
pub fn float64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(Float64Uniform::new())
}

/// Builds `randgen_float64_normal(mean, stddev)`.
///
/// The UDF returns a `Float64` sampled from a normal distribution. Arguments
/// must be finite, and `stddev` must be greater than zero.
pub fn float64_normal_udf() -> ScalarUDF {
    ScalarUDF::from(Float64Normal::new())
}

/// Builds `randgen_int64_normal(min, max, mean, stddev)`.
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

/// Builds `randgen_uint64_normal(min, max, mean, stddev)`.
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

/// Builds `randgen_bool(probability)`.
///
/// The UDF returns `true` with the given probability. The probability must be
/// finite and inside `0.0..=1.0`.
pub fn bool_udf() -> ScalarUDF {
    ScalarUDF::from(Bool::new())
}

/// Builds `randgen_utf8(characters, min_length, max_length)`.
///
/// The UDF returns a UTF-8 string made from the distinct characters in
/// `characters`. Length bounds are measured in characters and must satisfy
/// `0 <= min_length <= max_length`.
pub fn utf8_udf() -> ScalarUDF {
    ScalarUDF::from(Utf8::new())
}

/// Builds `randgen_choice(choices)`.
///
/// The UDF accepts a `List<T>` and returns a randomly selected item with type
/// `T`. Each non-null list must contain at least one element.
pub fn choice_udf() -> ScalarUDF {
    ScalarUDF::from(Choice::new())
}

/// Builds `randgen_date32(min, max)`.
///
/// The UDF returns a `Date32` sampled from the inclusive day range `min..=max`.
/// Non-null bounds must satisfy `min <= max`.
pub fn date32_udf() -> ScalarUDF {
    ScalarUDF::from(Date32::new())
}

/// Builds `randgen_timestamp_millisecond(min, max)`.
///
/// The UDF returns a millisecond timestamp sampled from the inclusive range
/// `min..=max`. Both arguments must use millisecond precision and matching
/// timezones.
pub fn timestamp_millisecond_udf() -> ScalarUDF {
    ScalarUDF::from(TimestampMillisecond::new())
}

/// Builds all UDFs exported by this crate.
///
/// Use this when a session should expose the whole generator set.
pub fn all_udfs() -> Vec<ScalarUDF> {
    vec![
        int64_uniform_udf(),
        uint64_uniform_udf(),
        float64_uniform_udf(),
        float64_normal_udf(),
        int64_normal_udf(),
        uint64_normal_udf(),
        bool_udf(),
        utf8_udf(),
        choice_udf(),
        date32_udf(),
        timestamp_millisecond_udf(),
    ]
}
