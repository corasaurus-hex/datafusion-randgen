#![no_main]

use std::collections::HashSet;
use std::sync::Arc;

use arbitrary::{Arbitrary, Unstructured};
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Float64Type, Int64Type, TimestampMillisecondType, UInt32Type,
    UInt64Type,
};
use arrow_array::{ArrayRef, BooleanArray, PrimitiveArray, StringArray};
use arrow_schema::{DataType, Field, TimeUnit};
use datafusion_common::{Result, ScalarValue, config::ConfigOptions};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_randgen::{
    Bool, Choice, ColumnChoice, Date32, Float64Normal, Float64Uniform, Int64Normal, Int64Uniform,
    Nullable, TimestampMillisecond, UInt64Normal, UInt64Uniform, Utf8,
};
use libfuzzer_sys::fuzz_target;
use roaring::{RoaringBitmap, RoaringTreemap};

#[derive(Arbitrary, Debug, Clone, Copy)]
enum ProbabilityArg {
    Omitted,
    Null,
    Zero,
    One,
    Half,
    Fraction(u8),
    Negative,
    TooLarge,
    Nan,
    Infinity,
}

#[derive(Arbitrary, Debug)]
enum DirectCase {
    Int64Uniform {
        min: i64,
        span: u16,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    UInt64Uniform {
        min: u64,
        span: u16,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    Float64Uniform {
        min: i32,
        width: u16,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    Float64Normal {
        mean: i32,
        stddev: u16,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    Int64Normal {
        min: i16,
        span: u8,
        mean_offset: u8,
        stddev: u16,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    UInt64Normal {
        min: u16,
        span: u8,
        mean_offset: u8,
        stddev: u16,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    Bool {
        probability: u8,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    Utf8 {
        alphabet: u8,
        min_length: u8,
        extra_length: u8,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    ChoiceInt64 {
        choices: [i16; 4],
        len: u8,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    NullableInt64 {
        value: i64,
        probability: ProbabilityArg,
        rows: u8,
    },
    Date32 {
        min: i32,
        span: u16,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    TimestampMillisecond {
        min: i64,
        span: u16,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    ColumnChoiceUInt32 {
        values: [u32; 4],
        len: u8,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    ColumnChoiceUInt64 {
        values: [u64; 4],
        len: u8,
        rows: u8,
        null_probability: ProbabilityArg,
    },
    ColumnChoiceCorruptBinary {
        bytes: [u8; 16],
        rows: u8,
        null_probability: ProbabilityArg,
    },
}

fuzz_target!(|bytes: &[u8]| {
    let mut unstructured = Unstructured::new(bytes);
    let Ok(case) = DirectCase::arbitrary(&mut unstructured) else {
        return;
    };
    case.run();
});

impl DirectCase {
    fn run(self) {
        match self {
            Self::Int64Uniform {
                min,
                span,
                rows,
                null_probability,
            } => {
                let max = i64_max_for_span(min, span);
                let mut args = vec![
                    scalar(ScalarValue::Int64(Some(min))),
                    scalar(ScalarValue::Int64(Some(max))),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&Int64Uniform::new(), args, DataType::Int64, rows) {
                    assert_range::<Int64Type>(&array, min, max);
                }
            }
            Self::UInt64Uniform {
                min,
                span,
                rows,
                null_probability,
            } => {
                let max = min.saturating_add(u64::from(span));
                let mut args = vec![
                    scalar(ScalarValue::UInt64(Some(min))),
                    scalar(ScalarValue::UInt64(Some(max))),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&UInt64Uniform::new(), args, DataType::UInt64, rows)
                {
                    assert_range::<UInt64Type>(&array, min, max);
                }
            }
            Self::Float64Uniform {
                min,
                width,
                rows,
                null_probability,
            } => {
                let min = f64::from(min) / 10.0;
                let max = min + f64::from(width) / 10.0;
                let mut args = vec![
                    scalar(ScalarValue::Float64(Some(min))),
                    scalar(ScalarValue::Float64(Some(max))),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) =
                    invoke_array(&Float64Uniform::new(), args, DataType::Float64, rows)
                {
                    assert!(primitive_values::<Float64Type>(&array).iter().all(|value| {
                        value.is_none_or(|value| value.is_finite() && (min..=max).contains(&value))
                    }));
                }
            }
            Self::Float64Normal {
                mean,
                stddev,
                rows,
                null_probability,
            } => {
                let mean = f64::from(mean) / 10.0;
                let stddev = f64::from(stddev.max(1)) / 10.0;
                let mut args = vec![
                    scalar(ScalarValue::Float64(Some(mean))),
                    scalar(ScalarValue::Float64(Some(stddev))),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) =
                    invoke_array(&Float64Normal::new(), args, DataType::Float64, rows)
                {
                    assert!(
                        primitive_values::<Float64Type>(&array)
                            .iter()
                            .all(|value| value.is_none_or(f64::is_finite))
                    );
                }
            }
            Self::Int64Normal {
                min,
                span,
                mean_offset,
                stddev,
                rows,
                null_probability,
            } => {
                let min = i64::from(min);
                let max = min + i64::from(span);
                let mean = min + i64::from(mean_offset % (span.saturating_add(1)));
                let stddev = f64::from(stddev.max(1)) / 10.0;
                let mut args = vec![
                    scalar(ScalarValue::Int64(Some(min))),
                    scalar(ScalarValue::Int64(Some(max))),
                    scalar(ScalarValue::Int64(Some(mean))),
                    scalar(ScalarValue::Float64(Some(stddev))),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&Int64Normal::new(), args, DataType::Int64, rows) {
                    assert_range::<Int64Type>(&array, min, max);
                }
            }
            Self::UInt64Normal {
                min,
                span,
                mean_offset,
                stddev,
                rows,
                null_probability,
            } => {
                let min = u64::from(min);
                let max = min + u64::from(span);
                let mean = min + u64::from(mean_offset % (span.saturating_add(1)));
                let stddev = f64::from(stddev.max(1)) / 10.0;
                let mut args = vec![
                    scalar(ScalarValue::UInt64(Some(min))),
                    scalar(ScalarValue::UInt64(Some(max))),
                    scalar(ScalarValue::UInt64(Some(mean))),
                    scalar(ScalarValue::Float64(Some(stddev))),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&UInt64Normal::new(), args, DataType::UInt64, rows)
                {
                    assert_range::<UInt64Type>(&array, min, max);
                }
            }
            Self::Bool {
                probability,
                rows,
                null_probability,
            } => {
                let probability = f64::from(probability) / 255.0;
                let mut args = vec![scalar(ScalarValue::Float64(Some(probability)))];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&Bool::new(), args, DataType::Boolean, rows) {
                    if probability == 0.0 {
                        assert!(
                            bool_values(&array)
                                .iter()
                                .all(|value| value.is_none_or(|value| !value))
                        );
                    }
                    if probability == 1.0 {
                        assert!(
                            bool_values(&array)
                                .iter()
                                .all(|value| value.is_none_or(|value| value))
                        );
                    }
                }
            }
            Self::Utf8 {
                alphabet,
                min_length,
                extra_length,
                rows,
                null_probability,
            } => {
                let alphabet = alphabet_for(alphabet);
                let min_length = i64::from(min_length % 8);
                let max_length = min_length + i64::from(extra_length % 8);
                let mut args = vec![
                    scalar(ScalarValue::Utf8(Some(alphabet.to_owned()))),
                    scalar(ScalarValue::Int64(Some(min_length))),
                    scalar(ScalarValue::Int64(Some(max_length))),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&Utf8::new(), args, DataType::Utf8, rows) {
                    let allowed = alphabet.chars().collect::<HashSet<_>>();
                    assert!(string_values(&array).iter().all(|value| {
                        value.as_ref().is_none_or(|value| {
                            (min_length as usize..=max_length as usize)
                                .contains(&value.chars().count())
                                && value.chars().all(|character| allowed.contains(&character))
                        })
                    }));
                }
            }
            Self::ChoiceInt64 {
                choices,
                len,
                rows,
                null_probability,
            } => {
                let len = usize::from(len % 4) + 1;
                let choices = choices[..len]
                    .iter()
                    .map(|value| i64::from(*value))
                    .collect::<Vec<_>>();
                let mut args = vec![scalar_choice_int64(&choices)];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&Choice::new(), args, DataType::Int64, rows) {
                    let allowed = choices.iter().copied().collect::<HashSet<_>>();
                    assert!(
                        primitive_values::<Int64Type>(&array)
                            .iter()
                            .all(|value| value.is_none_or(|value| allowed.contains(&value)))
                    );
                }
            }
            Self::NullableInt64 {
                value,
                probability,
                rows,
            } => {
                let args = vec![
                    scalar(ScalarValue::Int64(Some(value))),
                    probability_scalar(probability),
                ];
                if let Ok(array) = invoke_array(&Nullable::new(), args, DataType::Int64, rows) {
                    assert!(
                        primitive_values::<Int64Type>(&array)
                            .iter()
                            .all(|output| output.is_none() || *output == Some(value))
                    );
                }
            }
            Self::Date32 {
                min,
                span,
                rows,
                null_probability,
            } => {
                let max = i32_max_for_span(min, span);
                let mut args = vec![
                    scalar(ScalarValue::Date32(Some(min))),
                    scalar(ScalarValue::Date32(Some(max))),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&Date32::new(), args, DataType::Date32, rows) {
                    assert_range::<Date32Type>(&array, min, max);
                }
            }
            Self::TimestampMillisecond {
                min,
                span,
                rows,
                null_probability,
            } => {
                let max = i64_max_for_span(min, span);
                let data_type = DataType::Timestamp(TimeUnit::Millisecond, None);
                let mut args = vec![
                    scalar(ScalarValue::TimestampMillisecond(Some(min), None)),
                    scalar(ScalarValue::TimestampMillisecond(Some(max), None)),
                ];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&TimestampMillisecond::new(), args, data_type, rows)
                {
                    assert_range::<TimestampMillisecondType>(&array, min, max);
                }
            }
            Self::ColumnChoiceUInt32 {
                values,
                len,
                rows,
                null_probability,
            } => {
                let len = usize::from(len % 5);
                let values = values[..len].to_vec();
                let mut args = vec![scalar(ScalarValue::Binary(Some(bitmap_bytes(&values))))];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&ColumnChoice::new(), args, DataType::UInt32, rows)
                {
                    let allowed = values.iter().copied().collect::<HashSet<_>>();
                    assert!(
                        primitive_values::<UInt32Type>(&array)
                            .iter()
                            .all(|value| value.is_none_or(|value| allowed.contains(&value)))
                    );
                }
            }
            Self::ColumnChoiceUInt64 {
                values,
                len,
                rows,
                null_probability,
            } => {
                let len = usize::from(len % 5);
                let values = values[..len].to_vec();
                let mut args = vec![scalar(ScalarValue::LargeBinary(Some(treemap_bytes(
                    &values,
                ))))];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&ColumnChoice::new(), args, DataType::UInt64, rows)
                {
                    let allowed = values.iter().copied().collect::<HashSet<_>>();
                    assert!(
                        primitive_values::<UInt64Type>(&array)
                            .iter()
                            .all(|value| value.is_none_or(|value| allowed.contains(&value)))
                    );
                }
            }
            Self::ColumnChoiceCorruptBinary {
                bytes,
                rows,
                null_probability,
            } => {
                let mut args = vec![scalar(ScalarValue::Binary(Some(bytes.to_vec())))];
                append_probability(&mut args, null_probability);
                if let Ok(array) = invoke_array(&ColumnChoice::new(), args, DataType::UInt32, rows)
                {
                    assert_eq!(array.len(), bounded_rows(rows));
                    assert_eq!(array.data_type(), &DataType::UInt32);
                }
            }
        }
    }
}

fn bounded_rows(rows: u8) -> usize {
    usize::from(rows % 33)
}

fn i64_max_for_span(min: i64, span: u16) -> i64 {
    (min as i128 + i128::from(span)).min(i64::MAX as i128) as i64
}

fn i32_max_for_span(min: i32, span: u16) -> i32 {
    (i64::from(min) + i64::from(span)).min(i64::from(i32::MAX)) as i32
}

fn alphabet_for(index: u8) -> &'static str {
    match index % 5 {
        0 => "A",
        1 => "ABC",
        2 => "AABC",
        3 => "01xyz",
        _ => "A\u{1f600}",
    }
}

fn scalar(value: ScalarValue) -> ColumnarValue {
    ColumnarValue::Scalar(value)
}

fn probability_value(probability: ProbabilityArg) -> Option<f64> {
    match probability {
        ProbabilityArg::Omitted => Some(0.0),
        ProbabilityArg::Null => None,
        ProbabilityArg::Zero => Some(0.0),
        ProbabilityArg::One => Some(1.0),
        ProbabilityArg::Half => Some(0.5),
        ProbabilityArg::Fraction(value) => Some(f64::from(value) / 255.0),
        ProbabilityArg::Negative => Some(-0.1),
        ProbabilityArg::TooLarge => Some(1.1),
        ProbabilityArg::Nan => Some(f64::NAN),
        ProbabilityArg::Infinity => Some(f64::INFINITY),
    }
}

fn probability_scalar(probability: ProbabilityArg) -> ColumnarValue {
    scalar(ScalarValue::Float64(probability_value(probability)))
}

fn append_probability(args: &mut Vec<ColumnarValue>, probability: ProbabilityArg) {
    if matches!(probability, ProbabilityArg::Omitted) {
        return;
    }
    args.push(probability_scalar(probability));
}

fn invoke_array(
    udf: &impl ScalarUDFImpl,
    args: Vec<ColumnarValue>,
    return_type: DataType,
    rows: u8,
) -> Result<ArrayRef> {
    let number_rows = bounded_rows(rows);
    let arg_fields = args
        .iter()
        .enumerate()
        .map(|(index, arg)| Arc::new(Field::new(format!("arg{index}"), arg.data_type(), true)))
        .collect();
    udf.invoke_with_args(ScalarFunctionArgs {
        args,
        arg_fields,
        number_rows,
        return_field: Arc::new(Field::new(udf.name(), return_type, true)),
        config_options: Arc::new(ConfigOptions::default()),
    })?
    .into_array_of_size(number_rows)
}

fn primitive_values<T>(array: &ArrayRef) -> Vec<Option<T::Native>>
where
    T: ArrowPrimitiveType,
{
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("primitive array")
        .iter()
        .collect()
}

fn string_values(array: &ArrayRef) -> Vec<Option<String>> {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("string array")
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect()
}

fn bool_values(array: &ArrayRef) -> Vec<Option<bool>> {
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("boolean array")
        .iter()
        .collect()
}

fn assert_range<T>(array: &ArrayRef, min: T::Native, max: T::Native)
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    assert!(
        primitive_values::<T>(array)
            .iter()
            .all(|value| value.is_none_or(|value| min <= value && value <= max))
    );
}

fn scalar_choice_int64(values: &[i64]) -> ColumnarValue {
    let values = values
        .iter()
        .map(|value| ScalarValue::Int64(Some(*value)))
        .collect::<Vec<_>>();

    scalar(ScalarValue::List(ScalarValue::new_list_nullable(
        &values,
        &DataType::Int64,
    )))
}

fn bitmap_bytes(values: &[u32]) -> Vec<u8> {
    let mut bitmap = RoaringBitmap::new();
    for value in values {
        bitmap.insert(*value);
    }
    let mut bytes = Vec::with_capacity(bitmap.serialized_size());
    bitmap.serialize_into(&mut bytes).unwrap();
    bytes
}

fn treemap_bytes(values: &[u64]) -> Vec<u8> {
    let mut treemap = RoaringTreemap::new();
    for value in values {
        treemap.insert(*value);
    }
    let mut bytes = Vec::with_capacity(treemap.serialized_size());
    treemap.serialize_into(&mut bytes).unwrap();
    bytes
}
