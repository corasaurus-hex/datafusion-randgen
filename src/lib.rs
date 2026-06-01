//! Random data generator UDFs for Apache DataFusion.
//!
//! This crate exports `ScalarUDF` constructors. It doesn't register functions
//! for you or own a `SessionContext`; the application that owns the DataFusion
//! session decides which generators to install.
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
//! All exported functions are volatile. Required null inputs produce null
//! outputs for that row. Invalid ranges and distribution parameters return
//! DataFusion errors.

#![deny(missing_docs)]

use datafusion_expr::ScalarUDF;

pub use crate::randgen::bool::Bool;
pub use crate::randgen::choice::Choice;
pub use crate::randgen::date32::Date32;
pub use crate::randgen::float64_normal::Float64Normal;
pub use crate::randgen::float64_uniform::Float64Uniform;
pub use crate::randgen::int64_uniform::Int64Uniform;
pub use crate::randgen::timestamp_millisecond::TimestampMillisecond;
pub use crate::randgen::utf8::Utf8;

/// Implementation modules for the exported random generator UDFs.
///
/// Most callers should use the top-level `*_udf` constructors or [`all_udfs`].
/// The module remains public so advanced users can construct or inspect the
/// concrete `ScalarUDFImpl` types directly.
pub mod randgen;

/// Builds `randgen_int64_uniform(min, max)`.
///
/// The UDF returns an `Int64` sampled from the inclusive range `min..=max`.
/// Null bounds produce null output for that row. Non-null bounds must satisfy
/// `min <= max`.
pub fn int64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(Int64Uniform::new())
}

/// Builds `randgen_float64_uniform(min, max)`.
///
/// The UDF returns a `Float64` sampled from the inclusive range `min..=max`.
/// Bounds must be finite, the distance between them must be finite, and
/// `min <= max`.
pub fn float64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(Float64Uniform::new())
}

/// Builds `randgen_float64_normal(mean, stddev)`.
///
/// The UDF returns a `Float64` sampled from a normal distribution. Arguments
/// must be finite and `stddev` must be greater than zero.
pub fn float64_normal_udf() -> ScalarUDF {
    ScalarUDF::from(Float64Normal::new())
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

/// Builds every UDF exported by this crate.
///
/// Use this when a session should expose the whole generator set.
pub fn all_udfs() -> Vec<ScalarUDF> {
    vec![
        int64_uniform_udf(),
        float64_uniform_udf(),
        float64_normal_udf(),
        bool_udf(),
        utf8_udf(),
        choice_udf(),
        date32_udf(),
        timestamp_millisecond_udf(),
    ]
}
