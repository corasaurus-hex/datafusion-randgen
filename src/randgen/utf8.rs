//! UTF-8 string generator UDF.
//!
//! `randgen_utf8(characters, min_length, max_length[, null_probability])`
//! builds strings from the distinct characters in `characters`. Lengths are
//! measured in characters, not bytes. Bounds must satisfy
//! `0 <= min_length <= max_length`, and the generated array must fit Arrow's
//! `Utf8` offset limit. The optional null probability must be finite and within
//! `0.0..=1.0`.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, builder::StringBuilder};
use arrow_schema::DataType;
use datafusion_common::exec_err;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use rand::RngExt;

use crate::randgen::utils::{NullProbability, optional_args, three_array_args};

const MAX_UTF8_ARRAY_BYTES: i64 = i32::MAX as i64;

#[derive(Debug)]
struct Alphabet {
    characters: Vec<char>,
    max_character_bytes: usize,
}

#[derive(Debug)]
struct RowSpec {
    alphabet: Arc<Alphabet>,
    min_length: i64,
    max_length: i64,
}

fn parse_alphabet(characters: &str, name: &str) -> Result<Arc<Alphabet>> {
    let mut seen = HashSet::new();
    let mut alphabet = Vec::new();
    for character in characters.chars() {
        if seen.insert(character) {
            alphabet.push(character);
        }
    }

    if alphabet.is_empty() {
        return exec_err!("{name} requires at least one allowed character");
    }

    let max_character_bytes = alphabet
        .iter()
        .map(|character| character.len_utf8())
        .max()
        .unwrap();

    Ok(Arc::new(Alphabet {
        characters: alphabet,
        max_character_bytes,
    }))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_utf8(characters, min_length, max_length[, null_probability])`.
pub struct Utf8 {
    signature: &'static Signature,
}

static UTF8_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::one_of(
        vec![
            TypeSignature::Exact(vec![DataType::Utf8, DataType::Int64, DataType::Int64]),
            TypeSignature::Exact(vec![
                DataType::Utf8,
                DataType::Int64,
                DataType::Int64,
                DataType::Float64,
            ]),
        ],
        Volatility::Volatile,
    )
});

impl Utf8 {
    /// Creates the `randgen_utf8` implementation.
    pub fn new() -> Self {
        Self {
            signature: &UTF8_SIGNATURE,
        }
    }
}

impl Default for Utf8 {
    fn default() -> Self {
        Self::new()
    }
}

impl Utf8 {
    fn invoke_scalar_args(
        &self,
        characters: Option<&str>,
        min_length: Option<i64>,
        max_length: Option<i64>,
        number_rows: usize,
        null_probability: &NullProbability,
    ) -> Result<ColumnarValue> {
        let (Some(characters), Some(min_length), Some(max_length)) =
            (characters, min_length, max_length)
        else {
            let mut builder = StringBuilder::with_capacity(number_rows, 0);
            for _ in 0..number_rows {
                builder.append_null();
            }
            return Ok(ColumnarValue::Array(Arc::new(builder.finish())));
        };

        let alphabet = parse_alphabet(characters, self.name())?;
        let mut total_max_bytes = 0_i64;
        for _ in 0..number_rows {
            if min_length < 0 || max_length < 0 || min_length > max_length {
                return exec_err!("{} requires 0 <= min_length <= max_length", self.name());
            }

            let Some(row_max_bytes) = max_length.checked_mul(alphabet.max_character_bytes as i64)
            else {
                return exec_err!(
                    "{} generated Utf8 output exceeds the Arrow Utf8 byte limit of {MAX_UTF8_ARRAY_BYTES}",
                    self.name()
                );
            };
            let Some(new_total_max_bytes) = total_max_bytes.checked_add(row_max_bytes) else {
                return exec_err!(
                    "{} generated Utf8 output exceeds the Arrow Utf8 byte limit of {MAX_UTF8_ARRAY_BYTES}",
                    self.name()
                );
            };
            if new_total_max_bytes > MAX_UTF8_ARRAY_BYTES {
                return exec_err!(
                    "{} generated Utf8 output exceeds the Arrow Utf8 byte limit of {MAX_UTF8_ARRAY_BYTES}",
                    self.name()
                );
            }
            total_max_bytes = new_total_max_bytes;
        }

        let mut rng = rand::rng();
        let mut builder = StringBuilder::with_capacity(number_rows, 0);
        for row in 0..number_rows {
            if null_probability.is_null(row, &mut rng, self.name())? {
                builder.append_null();
                continue;
            }
            let length = rng.random_range(min_length..=max_length) as usize;
            for _ in 0..length {
                let index = rng.random_range(0..alphabet.characters.len());
                builder
                    .write_char(alphabet.characters[index])
                    .map_err(|_| {
                        DataFusionError::Execution(format!(
                            "{} failed to write generated output",
                            self.name()
                        ))
                    })?;
            }
            builder.append_value("");
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

impl ScalarUDFImpl for Utf8 {
    fn name(&self) -> &str {
        "randgen_utf8"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let ([characters, min_length, max_length], null_probability) =
            optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::Utf8(characters)),
            ColumnarValue::Scalar(ScalarValue::Int64(min_length)),
            ColumnarValue::Scalar(ScalarValue::Int64(max_length)),
        ) = (&characters, &min_length, &max_length)
        {
            return self.invoke_scalar_args(
                characters.as_deref(),
                *min_length,
                *max_length,
                number_rows,
                &null_probability,
            );
        }

        let (characters_array, min_length_array, max_length_array) = three_array_args(
            vec![characters, min_length, max_length],
            (DataType::Utf8, "Utf8, Int64, Int64 arguments"),
            (DataType::Int64, "Utf8, Int64, Int64 arguments"),
            (DataType::Int64, "Utf8, Int64, Int64 arguments"),
            number_rows,
            self.name(),
        )?;
        let characters_values = characters_array.as_string::<i32>();
        let min_length_values = min_length_array.as_primitive::<Int64Type>();
        let max_length_values = max_length_array.as_primitive::<Int64Type>();

        let mut total_max_bytes = 0_i64;
        let mut local_alphabets = HashMap::new();
        let mut row_specs = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if characters_values.is_null(row)
                || min_length_values.is_null(row)
                || max_length_values.is_null(row)
            {
                row_specs.push(None);
                continue;
            }

            let characters = characters_values.value(row);
            let alphabet = match local_alphabets.get(characters) {
                Some(alphabet) => Arc::clone(alphabet),
                None => {
                    let alphabet = parse_alphabet(characters, self.name())?;
                    local_alphabets.insert(characters, Arc::clone(&alphabet));
                    alphabet
                }
            };

            let min_length = min_length_values.value(row);
            let max_length = max_length_values.value(row);
            if min_length < 0 || max_length < 0 || min_length > max_length {
                return exec_err!("{} requires 0 <= min_length <= max_length", self.name());
            }

            let Some(row_max_bytes) = max_length.checked_mul(alphabet.max_character_bytes as i64)
            else {
                return exec_err!(
                    "{} generated Utf8 output exceeds the Arrow Utf8 byte limit of {MAX_UTF8_ARRAY_BYTES}",
                    self.name()
                );
            };
            let Some(new_total_max_bytes) = total_max_bytes.checked_add(row_max_bytes) else {
                return exec_err!(
                    "{} generated Utf8 output exceeds the Arrow Utf8 byte limit of {MAX_UTF8_ARRAY_BYTES}",
                    self.name()
                );
            };
            if new_total_max_bytes > MAX_UTF8_ARRAY_BYTES {
                return exec_err!(
                    "{} generated Utf8 output exceeds the Arrow Utf8 byte limit of {MAX_UTF8_ARRAY_BYTES}",
                    self.name()
                );
            }
            total_max_bytes = new_total_max_bytes;
            row_specs.push(Some(RowSpec {
                alphabet,
                min_length,
                max_length,
            }));
        }

        let mut rng = rand::rng();
        let mut builder = StringBuilder::with_capacity(number_rows, 0);
        for (row, row_spec) in row_specs.into_iter().enumerate() {
            let Some(row_spec) = row_spec else {
                builder.append_null();
                continue;
            };
            if null_probability.is_null(row, &mut rng, self.name())? {
                builder.append_null();
                continue;
            }

            let length = rng.random_range(row_spec.min_length..=row_spec.max_length) as usize;
            for _ in 0..length {
                let index = rng.random_range(0..row_spec.alphabet.characters.len());
                builder
                    .write_char(row_spec.alphabet.characters[index])
                    .map_err(|_| {
                        DataFusionError::Execution(format!(
                            "{} failed to write generated output",
                            self.name()
                        ))
                    })?;
            }
            builder.append_value("");
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use crate::randgen::test_helpers::querying::{query_result, query_to_string_values};

    use super::*;

    #[tokio::test]
    async fn utf8_values_use_allowed_characters_and_lengths() {
        let values = query_to_string_values(
            ScalarUDF::from(Utf8::new()),
            "SELECT randgen_utf8('ABC', 2, 8) FROM generate_series(1, 1000)",
        )
        .await;
        for value in values {
            let value = value.unwrap();
            assert!((2..=8).contains(&value.chars().count()));
            assert!(value.chars().all(|character| "ABC".contains(character)));
        }
    }

    #[tokio::test]
    async fn utf8_single_character_fixed_length_is_deterministic() {
        let values = query_to_string_values(
            ScalarUDF::from(Utf8::new()),
            "SELECT randgen_utf8('A', 5, 5) FROM generate_series(1, 100)",
        )
        .await;
        assert!(values.iter().all(|value| value.as_deref() == Some("AAAAA")));
    }

    #[tokio::test]
    async fn utf8_invalid_lengths_error() {
        let result = query_result(
            ScalarUDF::from(Utf8::new()),
            "SELECT randgen_utf8('ABC', 8, 2) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn utf8_empty_character_set_errors() {
        let result = query_result(
            ScalarUDF::from(Utf8::new()),
            "SELECT randgen_utf8('', 1, 2) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[test]
    fn utf8_character_set_is_deduplicated() {
        let alphabet = parse_alphabet("AAB😀😀C", "randgen_utf8").unwrap();
        assert_eq!(alphabet.characters, vec!['A', 'B', '😀', 'C']);
    }

    #[tokio::test]
    async fn utf8_rejects_output_larger_than_arrow_utf8_offsets() {
        let result = query_result(
            ScalarUDF::from(Utf8::new()),
            "SELECT randgen_utf8('A', 2147483648, 2147483648) FROM generate_series(1, 1)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn utf8_rejects_multibyte_output_larger_than_arrow_utf8_offsets() {
        let result = query_result(
            ScalarUDF::from(Utf8::new()),
            "SELECT randgen_utf8('😀', 536870912, 536870912) FROM generate_series(1, 1)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn utf8_array_args_propagate_nulls() {
        let values = query_to_string_values(
            ScalarUDF::from(Utf8::new()),
            "SELECT randgen_utf8(characters, min_length, max_length) FROM (VALUES ('A', 3, 3), (CAST(NULL AS STRING), 1, 2), ('B', CAST(NULL AS BIGINT), 2), ('C', 1, CAST(NULL AS BIGINT))) AS t(characters, min_length, max_length)",
        )
        .await;

        assert_eq!(values, vec![Some("AAA".to_owned()), None, None, None]);
    }

    #[tokio::test]
    async fn utf8_array_args_reuse_repeated_alphabets_per_invocation() {
        let values = query_to_string_values(
            ScalarUDF::from(Utf8::new()),
            "SELECT randgen_utf8(characters, min_length, max_length) FROM (VALUES ('A', 1, 1), ('A', 2, 2)) AS t(characters, min_length, max_length)",
        )
        .await;

        assert_eq!(values, vec![Some("A".to_owned()), Some("AA".to_owned())]);
    }
}
