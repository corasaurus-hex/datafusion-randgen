//! List choice random generator.
//!
//! `randgen_choice(choices[, null_probability])` samples one item from a
//! `List<T>` and returns type `T`. Non-null lists must contain at least one
//! element. Null lists produce null output for that row. The optional null
//! probability must be finite and within `0.0..=1.0`.

use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::builder::StringBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, ListArray, PrimitiveArray, new_empty_array, new_null_array};
use arrow_schema::{DataType, Field};
use datafusion_common::Result;
use datafusion_common::{ScalarValue, exec_err, internal_err, plan_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use rand::Rng;

use crate::randgen::utils::{NullProbability, coerce_float64_argument, optional_args};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_choice(choices[, null_probability])`.
pub struct Choice {
    signature: &'static Signature,
}

static CHOICE_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::user_defined(Volatility::Volatile));

fn scalar_list_choices(choices: &ListArray, name: &str) -> Result<Option<ArrayRef>> {
    if choices.len() != 1 {
        return internal_err!("{name} scalar List value must contain exactly one row");
    }
    if choices.is_null(0) {
        return Ok(None);
    }

    let row_choices = choices.value(0);
    if row_choices.is_empty() {
        return exec_err!("{name} requires at least one choice");
    }

    Ok(Some(row_choices))
}

fn choose_from_scalar_utf8_list(
    choices: &ListArray,
    number_rows: usize,
    name: &str,
    null_probability: &NullProbability,
) -> Result<ColumnarValue> {
    let Some(row_choices) = scalar_list_choices(choices, name)? else {
        return Ok(ColumnarValue::Array(new_null_array(
            &DataType::Utf8,
            number_rows,
        )));
    };

    let row_choices = row_choices.as_string::<i32>();
    let mut rng = rand::rng();
    let mut builder = StringBuilder::with_capacity(number_rows, 0);
    for row in 0..number_rows {
        if null_probability.is_null(row, &mut rng, name)? {
            builder.append_null();
            continue;
        }
        let choice_index = rng.random_range(0..row_choices.len());
        if row_choices.is_null(choice_index) {
            builder.append_null();
        } else {
            builder.append_value(row_choices.value(choice_index));
        }
    }

    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
}

fn choose_from_scalar_primitive_list<T>(
    choices: &ListArray,
    data_type: &DataType,
    number_rows: usize,
    name: &str,
    null_probability: &NullProbability,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
{
    let Some(row_choices) = scalar_list_choices(choices, name)? else {
        return Ok(ColumnarValue::Array(new_null_array(data_type, number_rows)));
    };

    let row_choices = row_choices.as_primitive::<T>();
    let mut rng = rand::rng();
    if row_choices.null_count() == 0 && matches!(null_probability, NullProbability::None) {
        let mut values = Vec::with_capacity(number_rows);
        for _ in 0..number_rows {
            let choice_index = rng.random_range(0..row_choices.len());
            values.push(row_choices.value(choice_index));
        }

        return Ok(ColumnarValue::Array(Arc::new(
            PrimitiveArray::<T>::from_iter_values(values),
        )));
    }

    let mut values = Vec::with_capacity(number_rows);
    for row in 0..number_rows {
        if null_probability.is_null(row, &mut rng, name)? {
            values.push(None);
            continue;
        }
        let choice_index = rng.random_range(0..row_choices.len());
        if row_choices.is_null(choice_index) {
            values.push(None);
        } else {
            values.push(Some(row_choices.value(choice_index)));
        }
    }

    Ok(ColumnarValue::Array(Arc::new(
        values.into_iter().collect::<PrimitiveArray<T>>(),
    )))
}

impl Choice {
    /// Creates the `randgen_choice` implementation.
    pub fn new() -> Self {
        Self {
            signature: &CHOICE_SIGNATURE,
        }
    }
}

impl Default for Choice {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for Choice {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_choice"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [DataType::List(field)] => Ok(field.data_type().clone()),
            [DataType::List(field), DataType::Float64] => Ok(field.data_type().clone()),
            [data_type] => plan_err!("{} expects a List argument, got {data_type}", self.name()),
            [_, data_type] => plan_err!(
                "{} expects a Float64 null probability, got {data_type}",
                self.name()
            ),
            _ => plan_err!("{} expects one or two arguments", self.name()),
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        match args.arg_fields {
            [field] | [field, _] => match field.data_type() {
                DataType::List(item_field) => Ok(Arc::new(Field::new(
                    self.name(),
                    item_field.data_type().clone(),
                    true,
                ))),
                data_type => plan_err!("{} expects a List argument, got {data_type}", self.name()),
            },
            _ => plan_err!("{} expects one or two arguments", self.name()),
        }
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 1 && arg_types.len() != 2 {
            return exec_err!("{} expects one or two arguments", self.name());
        }

        let mut coerced = vec![arg_types[0].clone()];
        if let Some(null_probability_type) = arg_types.get(1) {
            coerced.push(coerce_float64_argument(null_probability_type, self.name())?);
        }
        Ok(coerced)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args,
            number_rows,
            return_field,
            ..
        } = args;
        let ([choices], null_probability) = optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;

        if number_rows == 0 {
            return Ok(ColumnarValue::Array(new_empty_array(
                return_field.data_type(),
            )));
        }
        if let ColumnarValue::Scalar(ScalarValue::List(list)) = &choices
            && matches!(
                return_field.data_type(),
                DataType::Utf8 | DataType::Int64 | DataType::Float64
            )
        {
            let DataType::List(item_field) = list.data_type() else {
                return internal_err!("{} expects a List argument", self.name());
            };
            if item_field.data_type() != return_field.data_type() {
                return internal_err!("{} return field does not match list item type", self.name());
            }
            return match return_field.data_type() {
                DataType::Utf8 => {
                    choose_from_scalar_utf8_list(list, number_rows, self.name(), &null_probability)
                }
                DataType::Int64 => choose_from_scalar_primitive_list::<Int64Type>(
                    list,
                    return_field.data_type(),
                    number_rows,
                    self.name(),
                    &null_probability,
                ),
                DataType::Float64 => choose_from_scalar_primitive_list::<Float64Type>(
                    list,
                    return_field.data_type(),
                    number_rows,
                    self.name(),
                    &null_probability,
                ),
                _ => unreachable!("matches! limits choice scalar specializations"),
            };
        }

        let choices = choices.into_array_of_size(number_rows)?;
        let DataType::List(item_field) = choices.data_type() else {
            return internal_err!("{} expects a List argument", self.name());
        };
        if item_field.data_type() != return_field.data_type() {
            return internal_err!("{} return field does not match list item type", self.name());
        }

        let choices = choices.as_list::<i32>();
        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if choices.is_null(row) {
                values.push(ScalarValue::try_from(return_field.data_type())?);
                continue;
            }

            let row_choices = choices.value(row);
            if row_choices.is_empty() {
                return exec_err!("{} requires at least one choice", self.name());
            }
            if null_probability.is_null(row, &mut rng, self.name())? {
                values.push(ScalarValue::try_from(return_field.data_type())?);
                continue;
            }

            let choice_index = rng.random_range(0..row_choices.len());
            values.push(ScalarValue::try_from_array(
                row_choices.as_ref(),
                choice_index,
            )?);
        }

        Ok(ColumnarValue::Array(ScalarValue::iter_to_array(values)?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::{Float64Type, Int64Type};
    use arrow_array::{Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{datasource::MemTable, prelude::SessionContext};
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl};

    use crate::randgen::test_helpers::querying::{
        query_result, query_to_string_values, query_to_values,
    };

    use super::*;

    #[tokio::test]
    async fn choice_returns_values_from_list() {
        let values = query_to_string_values(
            ScalarUDF::from(Choice::new()),
            "SELECT randgen_choice(['UTC', 'America/New_York', 'Europe/London']) FROM generate_series(1, 1000)",
        )
        .await;
        for value in values {
            assert!(matches!(
                value.as_deref(),
                Some("UTC" | "America/New_York" | "Europe/London")
            ));
        }
    }

    #[tokio::test]
    async fn choice_single_value_is_deterministic() {
        let values = query_to_string_values(
            ScalarUDF::from(Choice::new()),
            "SELECT randgen_choice(['UTC']) FROM generate_series(1, 100)",
        )
        .await;
        assert!(values.iter().all(|value| value.as_deref() == Some("UTC")));
    }

    #[tokio::test]
    async fn choice_repeated_values_act_as_weights() {
        let values = query_to_string_values(
            ScalarUDF::from(Choice::new()),
            "SELECT randgen_choice(['UTC', 'UTC', 'America/New_York']) FROM generate_series(1, 1000)",
        )
        .await;
        assert!(values.iter().any(|value| value.as_deref() == Some("UTC")));
        assert!(
            values
                .iter()
                .any(|value| value.as_deref() == Some("America/New_York"))
        );
    }

    #[tokio::test]
    async fn choice_int64_scalar_list_returns_int64_values() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Choice::new()),
            "SELECT randgen_choice([7]) FROM generate_series(1, 10)",
            DataType::Int64,
        )
        .await;

        assert!(values.iter().all(|value| *value == Some(7)));
    }

    #[tokio::test]
    async fn choice_float64_scalar_list_returns_float64_values() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Choice::new()),
            "SELECT randgen_choice([3.5]) FROM generate_series(1, 10)",
            DataType::Float64,
        )
        .await;

        assert!(values.iter().all(|value| *value == Some(3.5)));
    }

    #[tokio::test]
    async fn choice_empty_list_errors() {
        let result = query_result(
            ScalarUDF::from(Choice::new()),
            "SELECT randgen_choice([]) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn choice_non_list_argument_errors() {
        let result = query_result(
            ScalarUDF::from(Choice::new()),
            "SELECT randgen_choice('UTC') FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn choice_empty_batch_returns_empty_array() {
        let ctx = SessionContext::new();
        ctx.register_udf(ScalarUDF::from(Choice::new()));

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let table = MemTable::try_new(
            Arc::clone(&schema),
            vec![vec![RecordBatch::new_empty(Arc::clone(&schema))]],
        )
        .unwrap();
        ctx.register_table("empty_table", Arc::new(table)).unwrap();

        let batches = ctx
            .sql("SELECT randgen_choice(['UTC']) FROM empty_table")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert!(!batches.is_empty());
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        assert!(
            batches
                .iter()
                .all(|batch| batch.column(0).data_type() == &DataType::Utf8)
        );
    }

    #[test]
    fn choice_list_column_preserves_null_rows() {
        let choices = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(7)]),
            None,
            Some(vec![Some(9)]),
        ]);
        let result = Choice::new()
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![ColumnarValue::Array(Arc::new(choices))],
                arg_fields: vec![Arc::new(Field::new(
                    "choices",
                    DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                    true,
                ))],
                number_rows: 3,
                return_field: Arc::new(Field::new("randgen_choice", DataType::Int64, true)),
                config_options: Arc::new(ConfigOptions::default()),
            })
            .unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };
        let values = array.as_primitive::<Int64Type>();
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            vec![Some(7), None, Some(9)]
        );
    }
}
