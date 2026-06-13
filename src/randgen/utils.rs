use std::fmt::Display;

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, ArrayRef, PrimitiveArray};
use arrow_schema::DataType;
use datafusion_common::{Result, exec_err, internal_err};
use datafusion_expr::ColumnarValue;
use rand::Rng;
use rand::distr::uniform::SampleUniform;

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

pub(crate) fn coerce_uint64_argument(data_type: &DataType, name: &str) -> Result<DataType> {
    match data_type {
        DataType::Null
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Ok(DataType::UInt64),
        DataType::Dictionary(_, value_type) => coerce_uint64_argument(value_type, name),
        _ => exec_err!("{name} expects UInt64 or nonnegative signed integer arguments"),
    }
}

pub(crate) fn coerce_float64_argument(data_type: &DataType, name: &str) -> Result<DataType> {
    match data_type {
        DataType::Null
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => Ok(DataType::Float64),
        DataType::Dictionary(_, value_type) => coerce_float64_argument(value_type, name),
        _ => exec_err!("{name} expects a Float64-compatible stddev argument"),
    }
}

fn validate_inclusive_range<T>(min: T, max: T, name: &str) -> Result<()>
where
    T: PartialOrd + Display,
{
    if min > max {
        return exec_err!("{name} requires min <= max, got min {min} and max {max}");
    }

    Ok(())
}

pub(crate) fn primitive_range_scalar_array<T>(
    min: Option<T::Native>,
    max: Option<T::Native>,
    number_rows: usize,
    name: &str,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy + SampleUniform + PartialOrd + Display,
{
    let mut rng = rand::rng();
    if let (Some(min), Some(max)) = (min, max) {
        if number_rows == 0 {
            return Ok(PrimitiveArray::<T>::from_iter_values(
                Vec::<T::Native>::new(),
            ));
        }

        validate_inclusive_range(min, max, name)?;

        let mut values = Vec::with_capacity(number_rows);
        for _ in 0..number_rows {
            values.push(rng.random_range(min..=max));
        }

        return Ok(PrimitiveArray::<T>::from_iter_values(values));
    }

    let mut values = Vec::with_capacity(number_rows);
    for _ in 0..number_rows {
        values.push(None);
    }

    Ok(values.into_iter().collect::<PrimitiveArray<T>>())
}

pub(crate) fn primitive_range_array<T>(
    min_values: &PrimitiveArray<T>,
    max_values: &PrimitiveArray<T>,
    number_rows: usize,
    name: &str,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy + SampleUniform + PartialOrd + Display,
{
    let mut rng = rand::rng();
    if min_values.null_count() == 0 && max_values.null_count() == 0 {
        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            let min = min_values.value(row);
            let max = max_values.value(row);
            validate_inclusive_range(min, max, name)?;
            values.push(rng.random_range(min..=max));
        }

        return Ok(PrimitiveArray::<T>::from_iter_values(values));
    }

    let mut values = Vec::with_capacity(number_rows);
    for row in 0..number_rows {
        if min_values.is_null(row) || max_values.is_null(row) {
            values.push(None);
            continue;
        }

        let min = min_values.value(row);
        let max = max_values.value(row);
        validate_inclusive_range(min, max, name)?;
        values.push(Some(rng.random_range(min..=max)));
    }

    Ok(values.into_iter().collect::<PrimitiveArray<T>>())
}
