//! Nullability wrapper UDF.
//!
//! `randgen_nullable(value, probability)` returns `value` with its original
//! type and replaces rows with null at the supplied probability. Existing nulls
//! stay null. Rows that become null are rebuilt as null values instead of only
//! overlaying a validity bitmap on the input array.

use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::{Array, ArrayRef, new_empty_array};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{Result, ScalarValue, exec_err, internal_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};

use crate::randgen::utils::{NullProbability, coerce_float64_argument, exact_args};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_nullable(value, probability)`.
///
/// The implementation clears values under rows that become null, which avoids
/// leaking hidden physical values through writers that preserve Arrow buffers.
pub struct Nullable {
    signature: &'static Signature,
}

static NULLABLE_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::user_defined(Volatility::Volatile));

fn apply_nullability_safely(
    value_array: ArrayRef,
    null_probability: &NullProbability,
    name: &str,
) -> Result<ArrayRef> {
    let mut rng = rand::rng();
    let mut values = Vec::with_capacity(value_array.len());
    let data_type = value_array.data_type().clone();
    let mut changed = false;

    for row in 0..value_array.len() {
        if value_array.is_null(row) || null_probability.is_null(row, &mut rng, name)? {
            changed = true;
            values.push(ScalarValue::try_from(&data_type)?);
        } else {
            values.push(ScalarValue::try_from_array(value_array.as_ref(), row)?);
        }
    }

    if !changed {
        return Ok(value_array);
    }

    ScalarValue::iter_to_array(values)
}

impl Nullable {
    /// Creates the `randgen_nullable` implementation.
    pub fn new() -> Self {
        Self {
            signature: &NULLABLE_SIGNATURE,
        }
    }
}

impl Default for Nullable {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for Nullable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_nullable"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [data_type, _] => Ok(data_type.clone()),
            _ => internal_err!("{} expects exactly two arguments", self.name()),
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        match args.arg_fields {
            [value, _] => Ok(Arc::new(Field::new(
                self.name(),
                value.data_type().clone(),
                true,
            ))),
            _ => internal_err!("{} expects exactly two arguments", self.name()),
        }
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 2 {
            let argument = if arg_types.len() == 1 {
                "argument"
            } else {
                "arguments"
            };
            return exec_err!(
                "{} expects exactly 2 arguments, got {} {argument}",
                self.name(),
                arg_types.len()
            );
        }

        Ok(vec![
            arg_types[0].clone(),
            coerce_float64_argument(&arg_types[1], self.name())?,
        ])
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args,
            number_rows,
            return_field,
            ..
        } = args;
        let [value, probability] = exact_args(args, self.name())?;

        if number_rows == 0 {
            return Ok(ColumnarValue::Array(new_empty_array(
                return_field.data_type(),
            )));
        }

        let value_array = value.into_array_of_size(number_rows)?;
        let null_probability =
            NullProbability::from_optional_arg(Some(probability), number_rows, self.name())?;

        Ok(ColumnarValue::Array(apply_nullability_safely(
            value_array,
            &null_probability,
            self.name(),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{ArrayRef, Float64Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field};
    use datafusion_common::{ScalarValue, config::ConfigOptions};
    use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl};

    use crate::randgen::test_helpers::querying::{
        query_result, query_to_string_values, query_to_values,
    };

    use super::*;

    fn direct_args(
        args: Vec<ColumnarValue>,
        return_type: DataType,
        number_rows: usize,
    ) -> ScalarFunctionArgs {
        let arg_fields = args
            .iter()
            .enumerate()
            .map(|(index, arg)| Arc::new(Field::new(format!("arg{index}"), arg.data_type(), true)))
            .collect();

        ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field: Arc::new(Field::new("randgen_nullable", return_type, true)),
            config_options: Arc::new(ConfigOptions::default()),
        }
    }

    fn direct_array(
        args: Vec<ColumnarValue>,
        return_type: DataType,
        number_rows: usize,
    ) -> ArrayRef {
        let result = Nullable::new()
            .invoke_with_args(direct_args(args, return_type, number_rows))
            .unwrap();
        let ColumnarValue::Array(array) = result else {
            panic!("expected array result");
        };
        array
    }

    #[test]
    fn nullable_zero_scalar_probability_preserves_input_without_nulls() {
        let input: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
        let output = direct_array(
            vec![
                ColumnarValue::Array(input.clone()),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))),
            ],
            DataType::Int64,
            input.len(),
        );

        let values = output.as_primitive::<Int64Type>();
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            vec![Some(1), None, Some(3)]
        );
    }

    #[test]
    fn nullable_array_probability_filters_value_buffers() {
        let input: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")]));
        let probabilities: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.0_f64),
            Some(1.0_f64),
            Some(0.0_f64),
        ]));
        let output = direct_array(
            vec![
                ColumnarValue::Array(input.clone()),
                ColumnarValue::Array(probabilities),
            ],
            DataType::Utf8,
            input.len(),
        );

        let strings = output.as_string::<i32>();
        assert_eq!(
            strings.iter().collect::<Vec<_>>(),
            vec![Some("a"), None, Some("c")]
        );

        let output_data = output.to_data();
        assert_eq!(output_data.buffers()[1].as_slice(), b"ac");
    }

    #[tokio::test]
    async fn nullable_zero_probability_preserves_values() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable(value, 0.0) FROM (VALUES (1), (2), (3)) AS t(value)",
            DataType::Int64,
        )
        .await;

        assert_eq!(values, vec![Some(1), Some(2), Some(3)]);
    }

    #[tokio::test]
    async fn nullable_one_probability_returns_nulls() {
        let values = query_to_string_values(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable('UTC', 1.0) FROM generate_series(1, 16)",
        )
        .await;

        assert!(values.iter().all(Option::is_none));
    }

    #[tokio::test]
    async fn nullable_integer_probability_literals_are_coerced() {
        let values = query_to_string_values(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable('UTC', 1) FROM generate_series(1, 16)",
        )
        .await;

        assert!(values.iter().all(Option::is_none));
    }

    #[tokio::test]
    async fn nullable_null_probability_returns_nulls() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable(7, CAST(NULL AS DOUBLE)) FROM generate_series(1, 16)",
            DataType::Int64,
        )
        .await;

        assert!(values.iter().all(Option::is_none));
    }

    #[tokio::test]
    async fn nullable_half_probability_smoke() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable(7, 0.5) FROM generate_series(1, 1000)",
            DataType::Int64,
        )
        .await;

        assert!(values.contains(&Some(7)));
        assert!(values.contains(&None));
    }

    #[tokio::test]
    async fn nullable_preserves_existing_nulls() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable(value, 0.0) FROM (VALUES (1), (CAST(NULL AS BIGINT)), (3)) AS t(value)",
            DataType::Int64,
        )
        .await;

        assert_eq!(values, vec![Some(1), None, Some(3)]);
    }

    #[tokio::test]
    async fn nullable_row_varying_probability() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable(value, p) FROM (VALUES (1, 0.0), (2, 1.0), (3, 0.0)) AS t(value, p)",
            DataType::Int64,
        )
        .await;

        assert_eq!(values, vec![Some(1), None, Some(3)]);
    }

    #[tokio::test]
    async fn nullable_invalid_probability_errors() {
        let result = query_result(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable(7, 1.1) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn nullable_negative_probability_errors() {
        let result = query_result(
            ScalarUDF::from(Nullable::new()),
            "SELECT randgen_nullable(7, -0.1) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }
}
