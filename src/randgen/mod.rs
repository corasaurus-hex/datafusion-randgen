//! Concrete `ScalarUDFImpl` types for the random generators.
//!
//! These modules back the top-level constructors. They are public for callers
//! that need direct access to a UDF implementation type. Registration code
//! usually only needs `datafusion_randgen::all_udfs()` or one of the `*_udf`
//! helpers.

/// Implementation for `randgen_bool`.
pub mod bool;
/// Implementation for `randgen_choice`.
pub mod choice;
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
