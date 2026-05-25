use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::{Array, AsArray, StringArray};
use datafusion::arrow::datatypes::{DataType, Int64Type};
use datafusion::common::{exec_err, internal_err};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use rand::Rng;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Utf8 {
    signature: &'static Signature,
}

static UTF8_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(
        vec![DataType::Utf8, DataType::Int64, DataType::Int64],
        Volatility::Volatile,
    )
});

impl Utf8 {
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

impl ScalarUDFImpl for Utf8 {
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        let [characters, min_length, max_length]: [ColumnarValue; 3] = match args.try_into() {
            Ok(args) => args,
            Err(_) => return internal_err!("{} expects exactly three arguments", self.name()),
        };

        if characters.data_type() != DataType::Utf8
            || min_length.data_type() != DataType::Int64
            || max_length.data_type() != DataType::Int64
        {
            return internal_err!("{} expects Utf8, Int64, Int64 arguments", self.name());
        }

        let characters_array = characters.into_array_of_size(number_rows)?;
        let min_length_array = min_length.into_array_of_size(number_rows)?;
        let max_length_array = max_length.into_array_of_size(number_rows)?;
        let characters_values = characters_array.as_string::<i32>();
        let min_length_values = min_length_array.as_primitive::<Int64Type>();
        let max_length_values = max_length_array.as_primitive::<Int64Type>();

        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if characters_values.is_null(row)
                || min_length_values.is_null(row)
                || max_length_values.is_null(row)
            {
                values.push(None);
                continue;
            }

            let alphabet = characters_values.value(row);
            let alphabet = alphabet.chars().collect::<Vec<_>>();
            if alphabet.is_empty() {
                return exec_err!("{} requires at least one allowed character", self.name());
            }

            let min_length = min_length_values.value(row);
            let max_length = max_length_values.value(row);
            if min_length < 0 || max_length < 0 || min_length > max_length {
                return exec_err!("{} requires 0 <= min_length <= max_length", self.name());
            }

            let length = rng.random_range(min_length..=max_length) as usize;
            let mut value = String::new();
            for _ in 0..length {
                let index = rng.random_range(0..alphabet.len());
                value.push(alphabet[index]);
            }
            values.push(Some(value));
        }

        Ok(ColumnarValue::Array(Arc::new(StringArray::from(values))))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::ScalarUDF;

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
}
