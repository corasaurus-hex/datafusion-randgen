use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::{Result, internal_err};
use datafusion_expr::ColumnarValue;

pub(crate) fn exact_args<const N: usize>(
    args: Vec<ColumnarValue>,
    name: &str,
) -> Result<[ColumnarValue; N]> {
    match args.try_into() {
        Ok(args) => Ok(args),
        Err(_) => {
            let argument = if N == 1 { "argument" } else { "arguments" };
            internal_err!("{name} expects exactly {N} {argument}")
        }
    }
}

fn array_arg(
    value: ColumnarValue,
    expected_type: &DataType,
    number_rows: usize,
    name: &str,
    expected_name: &str,
) -> Result<ArrayRef> {
    if value.data_type() != *expected_type {
        return internal_err!("{name} expects {expected_name}");
    }

    value.into_array_of_size(number_rows)
}

type ExpectedArg = (DataType, &'static str);

pub(crate) fn one_array_arg(
    args: Vec<ColumnarValue>,
    expected: ExpectedArg,
    number_rows: usize,
    name: &str,
) -> Result<ArrayRef> {
    let [value] = exact_args(args, name)?;
    let (expected_type, expected_name) = expected;
    array_arg(value, &expected_type, number_rows, name, expected_name)
}

pub(crate) fn two_array_args(
    args: Vec<ColumnarValue>,
    first: ExpectedArg,
    second: ExpectedArg,
    number_rows: usize,
    name: &str,
) -> Result<(ArrayRef, ArrayRef)> {
    let [first_value, second_value] = exact_args(args, name)?;
    let (first_type, first_name) = first;
    let (second_type, second_name) = second;
    Ok((
        array_arg(first_value, &first_type, number_rows, name, first_name)?,
        array_arg(second_value, &second_type, number_rows, name, second_name)?,
    ))
}

pub(crate) fn three_array_args(
    args: Vec<ColumnarValue>,
    first: ExpectedArg,
    second: ExpectedArg,
    third: ExpectedArg,
    number_rows: usize,
    name: &str,
) -> Result<(ArrayRef, ArrayRef, ArrayRef)> {
    let [first_value, second_value, third_value] = exact_args(args, name)?;
    let (first_type, first_name) = first;
    let (second_type, second_name) = second;
    let (third_type, third_name) = third;
    Ok((
        array_arg(first_value, &first_type, number_rows, name, first_name)?,
        array_arg(second_value, &second_type, number_rows, name, second_name)?,
        array_arg(third_value, &third_type, number_rows, name, third_name)?,
    ))
}

pub(crate) fn four_array_args(
    args: Vec<ColumnarValue>,
    first: ExpectedArg,
    second: ExpectedArg,
    third: ExpectedArg,
    fourth: ExpectedArg,
    number_rows: usize,
    name: &str,
) -> Result<(ArrayRef, ArrayRef, ArrayRef, ArrayRef)> {
    let [first_value, second_value, third_value, fourth_value] = exact_args(args, name)?;
    let (first_type, first_name) = first;
    let (second_type, second_name) = second;
    let (third_type, third_name) = third;
    let (fourth_type, fourth_name) = fourth;
    Ok((
        array_arg(first_value, &first_type, number_rows, name, first_name)?,
        array_arg(second_value, &second_type, number_rows, name, second_name)?,
        array_arg(third_value, &third_type, number_rows, name, third_name)?,
        array_arg(fourth_value, &fourth_type, number_rows, name, fourth_name)?,
    ))
}
