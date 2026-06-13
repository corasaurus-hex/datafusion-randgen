use std::sync::Arc;

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, BooleanArray, PrimitiveArray, StringArray};
use arrow_schema::{DataType, Field};
use datafusion_common::{ScalarValue, config::ConfigOptions};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};

pub(crate) fn call_args(
    args: Vec<ColumnarValue>,
    return_type: DataType,
    return_name: &str,
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
        return_field: Arc::new(Field::new(return_name.to_owned(), return_type, true)),
        config_options: Arc::new(ConfigOptions::default()),
    }
}

pub(crate) fn scalar(value: ScalarValue) -> ColumnarValue {
    ColumnarValue::Scalar(value)
}

pub(crate) fn invoke_array(
    udf: &impl ScalarUDFImpl,
    args: Vec<ColumnarValue>,
    return_type: DataType,
    number_rows: usize,
) -> ArrayRef {
    let result = udf
        .invoke_with_args(call_args(args, return_type, udf.name(), number_rows))
        .unwrap();
    let ColumnarValue::Array(array) = result else {
        panic!("expected an array result");
    };
    assert_eq!(array.len(), number_rows);
    array
}

pub(crate) fn primitive_values<T>(array: &ArrayRef) -> Vec<Option<T::Native>>
where
    T: ArrowPrimitiveType,
{
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap()
        .iter()
        .collect()
}

pub(crate) fn bool_values(array: &ArrayRef) -> Vec<Option<bool>> {
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
        .iter()
        .collect()
}

pub(crate) fn string_values(array: &ArrayRef) -> Vec<Option<String>> {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect()
}

pub(crate) fn scalar_choice_int64(values: &[i64]) -> ColumnarValue {
    let values = values
        .iter()
        .map(|value| ScalarValue::Int64(Some(*value)))
        .collect::<Vec<_>>();

    scalar(ScalarValue::List(ScalarValue::new_list_nullable(
        &values,
        &DataType::Int64,
    )))
}

pub(crate) fn scalar_choice_utf8(values: &[&str]) -> ColumnarValue {
    let values = values
        .iter()
        .map(|value| ScalarValue::Utf8(Some((*value).to_owned())))
        .collect::<Vec<_>>();

    scalar(ScalarValue::List(ScalarValue::new_list_nullable(
        &values,
        &DataType::Utf8,
    )))
}
