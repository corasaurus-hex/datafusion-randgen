//! Concrete `ScalarUDFImpl` types for the random generators.
//!
//! The top-level constructors wrap these modules. They stay public for callers
//! that need direct access to a UDF implementation type. Registration code can
//! use `datafusion_randgen::all_udfs()` or one of the `*_udf` helpers.

/// Implementation for `randgen_bool`.
pub mod bool;
/// Implementation for `randgen_choice`.
pub mod choice;
#[cfg(feature = "column-choice-parquet")]
/// Implementation for `randgen_column_choice`.
pub mod column_choice;
/// Implementation for `randgen_date32`.
pub mod date32;
/// Implementation for `randgen_float64_normal`.
pub mod float64_normal;
/// Implementation for `randgen_float64_uniform`.
pub mod float64_uniform;
/// Implementation for `randgen_int64_normal`.
pub mod int64_normal;
/// Implementation for `randgen_int64_uniform`.
pub mod int64_uniform;
pub(crate) mod integer_normal;
/// Implementation for `randgen_nullable`.
pub mod nullable;
/// Implementation for `randgen_timestamp_millisecond`.
pub mod timestamp_millisecond;
/// Implementation for `randgen_uint64_normal`.
pub mod uint64_normal;
/// Implementation for `randgen_uint64_uniform`.
pub mod uint64_uniform;
/// Implementation for `randgen_utf8`.
pub mod utf8;
pub(crate) mod utils;

#[cfg(test)]
pub(crate) mod test_helpers;
