//! Random data generator UDFs for Apache DataFusion.
//!
//! The crate exposes UDF values and leaves registration to the DataFusion
//! application that owns the `SessionContext`.
//!
//! ```no_run
//! use datafusion::prelude::SessionContext;
//!
//! let ctx = SessionContext::new();
//! for udf in datafusion_randgen::all_udfs() {
//!     ctx.register_udf(udf);
//! }
//! ```

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

/// Random data generator UDF implementations.
pub mod randgen;

/// Creates the `randgen_int64_uniform(min, max)` UDF.
pub fn int64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(Int64Uniform::new())
}

/// Creates the `randgen_float64_uniform(min, max)` UDF.
pub fn float64_uniform_udf() -> ScalarUDF {
    ScalarUDF::from(Float64Uniform::new())
}

/// Creates the `randgen_float64_normal(mean, stddev)` UDF.
pub fn float64_normal_udf() -> ScalarUDF {
    ScalarUDF::from(Float64Normal::new())
}

/// Creates the `randgen_bool(probability)` UDF.
pub fn bool_udf() -> ScalarUDF {
    ScalarUDF::from(Bool::new())
}

/// Creates the `randgen_utf8(characters, min_length, max_length)` UDF.
pub fn utf8_udf() -> ScalarUDF {
    ScalarUDF::from(Utf8::new())
}

/// Creates the `randgen_choice(choices)` UDF.
pub fn choice_udf() -> ScalarUDF {
    ScalarUDF::from(Choice::new())
}

/// Creates the `randgen_date32(min, max)` UDF.
pub fn date32_udf() -> ScalarUDF {
    ScalarUDF::from(Date32::new())
}

/// Creates the `randgen_timestamp_millisecond(min, max)` UDF.
pub fn timestamp_millisecond_udf() -> ScalarUDF {
    ScalarUDF::from(TimestampMillisecond::new())
}

/// Creates every UDF exported by this crate.
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
