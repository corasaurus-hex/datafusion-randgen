use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::builder::StringBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ListArray, new_empty_array, new_null_array};
use arrow_schema::{DataType, Field};
use datafusion_common::Result;
use datafusion_common::{ScalarValue, exec_err, internal_err, plan_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use rand::Rng;

use crate::randgen::utils::exact_args;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Choice {
    signature: &'static Signature,
}

static CHOICE_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::any(1, Volatility::Volatile));

fn choose_from_scalar_utf8_list(
    choices: &ListArray,
    number_rows: usize,
    name: &str,
) -> Result<ColumnarValue> {
    if choices.len() != 1 {
        return internal_err!("{name} scalar List value must contain exactly one row");
    }
    if choices.is_null(0) {
        return Ok(ColumnarValue::Array(new_null_array(
            &DataType::Utf8,
            number_rows,
        )));
    }

    let row_choices = choices.value(0);
    if row_choices.is_empty() {
        return exec_err!("{name} requires at least one choice");
    }

    let row_choices = row_choices.as_string::<i32>();
    let mut rng = rand::rng();
    let mut builder = StringBuilder::with_capacity(number_rows, 0);
    for _ in 0..number_rows {
        let choice_index = rng.random_range(0..row_choices.len());
        if row_choices.is_null(choice_index) {
            builder.append_null();
        } else {
            builder.append_value(row_choices.value(choice_index));
        }
    }

    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
}

impl Choice {
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
            [data_type] => plan_err!("{} expects a List argument, got {data_type}", self.name()),
            _ => plan_err!("{} expects exactly one argument", self.name()),
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<Arc<Field>> {
        match args.arg_fields {
            [field] => match field.data_type() {
                DataType::List(item_field) => Ok(Arc::new(Field::new(
                    self.name(),
                    item_field.data_type().clone(),
                    true,
                ))),
                data_type => plan_err!("{} expects a List argument, got {data_type}", self.name()),
            },
            _ => plan_err!("{} expects exactly one argument", self.name()),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args,
            number_rows,
            return_field,
            ..
        } = args;
        let [choices] = exact_args(args, self.name())?;

        if number_rows == 0 {
            return Ok(ColumnarValue::Array(new_empty_array(
                return_field.data_type(),
            )));
        }
        if let ColumnarValue::Scalar(ScalarValue::List(list)) = &choices {
            if matches!(return_field.data_type(), DataType::Utf8) {
                let DataType::List(item_field) = list.data_type() else {
                    return internal_err!("{} expects a List argument", self.name());
                };
                if item_field.data_type() != return_field.data_type() {
                    return internal_err!(
                        "{} return field does not match list item type",
                        self.name()
                    );
                }
                return choose_from_scalar_utf8_list(list, number_rows, self.name());
            }
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

    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{datasource::MemTable, prelude::SessionContext};
    use datafusion_expr::ScalarUDF;

    use crate::randgen::test_helpers::querying::{query_result, query_to_string_values};

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
    async fn choice_empty_list_errors() {
        let result = query_result(
            ScalarUDF::from(Choice::new()),
            "SELECT randgen_choice(arrow_cast([], 'List(Utf8)')) FROM generate_series(1, 10)",
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
}
