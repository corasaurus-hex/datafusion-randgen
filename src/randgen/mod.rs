//! UDF implementation modules.
//!
//! The top-level constructors wrap these modules. They stay public for callers
//! that need direct access to a UDF implementation type.

/// Boolean generator implementation.
pub mod bool;
/// List-choice generator implementation.
pub mod choice;
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
/// Aggregate-backed column-choice implementation.
pub mod column_choice;
/// Date32 generator implementation.
pub mod date32;
/// Float64 normal generator implementation.
pub mod float64_normal;
/// Float64 uniform generator implementation.
pub mod float64_uniform;
/// Int64 normal generator implementation.
pub mod int64_normal;
/// Int64 uniform generator implementation.
pub mod int64_uniform;
pub(crate) mod integer_normal;
/// Nullability wrapper implementation.
pub mod nullable;
/// Millisecond timestamp generator implementation.
pub mod timestamp_millisecond;
/// UInt64 normal generator implementation.
pub mod uint64_normal;
/// UInt64 uniform generator implementation.
pub mod uint64_uniform;
/// UTF-8 string generator implementation.
pub mod utf8;
pub(crate) mod utils;

#[cfg(test)]
pub(crate) mod test_helpers;
