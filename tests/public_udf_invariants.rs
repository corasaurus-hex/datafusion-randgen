use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Float64Type, Int64Type, TimestampMillisecondType, UInt64Type,
};
use arrow_array::{ArrayRef, BooleanArray, PrimitiveArray, StringArray};
use arrow_schema::{DataType, Field, TimeUnit};
use datafusion_common::{ScalarValue, config::ConfigOptions};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_randgen::{
    Bool, Choice, Date32, Float64Normal, Float64Uniform, Int64Normal, Int64Uniform,
    TimestampMillisecond, UInt64Normal, UInt64Uniform, Utf8,
};
use proptest::prelude::*;

const PROP_ROWS: usize = 64;
const STRESS_ROWS: usize = 16_384;

fn call_args(
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

fn scalar(value: ScalarValue) -> ColumnarValue {
    ColumnarValue::Scalar(value)
}

fn invoke_array(
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

fn primitive_values<T>(array: &ArrayRef) -> Vec<Option<T::Native>>
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

fn bool_values(array: &ArrayRef) -> Vec<Option<bool>> {
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
        .iter()
        .collect()
}

fn string_values(array: &ArrayRef) -> Vec<Option<String>> {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect()
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

fn scalar_choice_utf8(values: &[&str]) -> ColumnarValue {
    let values = values
        .iter()
        .map(|value| ScalarValue::Utf8(Some((*value).to_owned())))
        .collect::<Vec<_>>();

    scalar(ScalarValue::List(ScalarValue::new_list_nullable(
        &values,
        &DataType::Utf8,
    )))
}

fn i64_range_strategy() -> impl Strategy<Value = (i64, i64)> {
    (any::<i64>(), 0_u16..=4096).prop_map(|(min, span)| {
        let max = (min as i128 + span as i128).min(i64::MAX as i128) as i64;
        (min, max)
    })
}

fn u64_range_strategy() -> impl Strategy<Value = (u64, u64)> {
    (any::<u64>(), 0_u16..=4096).prop_map(|(min, span)| (min, min.saturating_add(span as u64)))
}

fn i32_range_strategy() -> impl Strategy<Value = (i32, i32)> {
    (any::<i32>(), 0_u16..=4096).prop_map(|(min, span)| {
        let max = (min as i64 + span as i64).min(i32::MAX as i64) as i32;
        (min, max)
    })
}

fn timestamp_range_strategy() -> impl Strategy<Value = (i64, i64)> {
    (any::<i64>(), 0_u16..=4096).prop_map(|(min, span)| {
        let max = (min as i128 + span as i128).min(i64::MAX as i128) as i64;
        (min, max)
    })
}

fn i64_normal_strategy() -> impl Strategy<Value = (i64, i64, i64, f64)> {
    (any::<i64>(), 0_u16..=512, 0_u16..=512, 1_u16..=1000).prop_map(
        |(min, span, mean_index, stddev)| {
            let max = (min as i128 + span as i128).min(i64::MAX as i128) as i64;
            let range_len = (max as i128 - min as i128 + 1) as u16;
            let mean = (min as i128 + (mean_index % range_len) as i128) as i64;
            (min, max, mean, f64::from(stddev) / 10.0)
        },
    )
}

fn u64_normal_strategy() -> impl Strategy<Value = (u64, u64, u64, f64)> {
    (any::<u64>(), 0_u16..=512, 0_u16..=512, 1_u16..=1000).prop_map(
        |(min, span, mean_index, stddev)| {
            let max = min.saturating_add(span as u64);
            let range_len = (max - min + 1) as u16;
            let mean = min + u64::from(mean_index % range_len);
            (min, max, mean, f64::from(stddev) / 10.0)
        },
    )
}

fn alphabet_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            Just('A'),
            Just('B'),
            Just('C'),
            Just('X'),
            Just('Y'),
            Just('Z'),
            Just('0'),
            Just('1'),
        ],
        1..12,
    )
    .prop_map(|characters| characters.into_iter().collect())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn int64_uniform_outputs_stay_in_the_requested_range((min, max) in i64_range_strategy()) {
        let values = primitive_values::<Int64Type>(&invoke_array(
            &Int64Uniform::new(),
            vec![scalar(ScalarValue::Int64(Some(min))), scalar(ScalarValue::Int64(Some(max)))],
            DataType::Int64,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(|value| (min..=max).contains(&value))));
    }

    #[test]
    fn uint64_uniform_outputs_stay_in_the_requested_range((min, max) in u64_range_strategy()) {
        let values = primitive_values::<UInt64Type>(&invoke_array(
            &UInt64Uniform::new(),
            vec![scalar(ScalarValue::UInt64(Some(min))), scalar(ScalarValue::UInt64(Some(max)))],
            DataType::UInt64,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(|value| (min..=max).contains(&value))));
    }

    #[test]
    fn float64_uniform_outputs_are_finite_and_in_range(min in -1_000_000_i64..=1_000_000, width in 0_u32..=1_000_000) {
        let min = min as f64 / 10.0;
        let max = min + width as f64 / 10.0;
        let values = primitive_values::<Float64Type>(&invoke_array(
            &Float64Uniform::new(),
            vec![scalar(ScalarValue::Float64(Some(min))), scalar(ScalarValue::Float64(Some(max)))],
            DataType::Float64,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(|value| value.is_finite() && (min..=max).contains(&value))));
    }

    #[test]
    fn float64_normal_outputs_are_finite(mean in -1_000_000_i64..=1_000_000, stddev in 1_u32..=1_000_000) {
        let mean = mean as f64 / 10.0;
        let stddev = stddev as f64 / 100.0;
        let values = primitive_values::<Float64Type>(&invoke_array(
            &Float64Normal::new(),
            vec![scalar(ScalarValue::Float64(Some(mean))), scalar(ScalarValue::Float64(Some(stddev)))],
            DataType::Float64,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(f64::is_finite)));
    }

    #[test]
    fn bool_outputs_are_non_null_for_valid_probabilities(probability in 0_u32..=1_000_000) {
        let probability = probability as f64 / 1_000_000.0;
        let values = bool_values(&invoke_array(
            &Bool::new(),
            vec![scalar(ScalarValue::Float64(Some(probability)))],
            DataType::Boolean,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(Option::is_some));
        if probability == 0.0 {
            prop_assert!(values.iter().all(|value| *value == Some(false)));
        }
        if probability == 1.0 {
            prop_assert!(values.iter().all(|value| *value == Some(true)));
        }
    }

    #[test]
    fn utf8_outputs_use_only_allowed_characters(alphabet in alphabet_strategy(), min_length in 0_u8..=8, extra_length in 0_u8..=8) {
        let max_length = min_length + extra_length;
        let allowed = alphabet.chars().collect::<HashSet<_>>();
        let values = string_values(&invoke_array(
            &Utf8::new(),
            vec![
                scalar(ScalarValue::Utf8(Some(alphabet))),
                scalar(ScalarValue::Int64(Some(i64::from(min_length)))),
                scalar(ScalarValue::Int64(Some(i64::from(max_length)))),
            ],
            DataType::Utf8,
            PROP_ROWS,
        ));

        for value in values {
            let value = value.unwrap();
            let length = value.chars().count();
            prop_assert!((usize::from(min_length)..=usize::from(max_length)).contains(&length));
            prop_assert!(value.chars().all(|character| allowed.contains(&character)));
        }
    }

    #[test]
    fn choice_outputs_come_from_the_input_list(choices in prop::collection::vec(-10_000_i64..=10_000, 1..16)) {
        let allowed = choices.iter().copied().collect::<HashSet<_>>();
        let values = primitive_values::<Int64Type>(&invoke_array(
            &Choice::new(),
            vec![scalar_choice_int64(&choices)],
            DataType::Int64,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(|value| allowed.contains(&value))));
    }

    #[test]
    fn date32_outputs_stay_in_the_requested_range((min, max) in i32_range_strategy()) {
        let values = primitive_values::<Date32Type>(&invoke_array(
            &Date32::new(),
            vec![scalar(ScalarValue::Date32(Some(min))), scalar(ScalarValue::Date32(Some(max)))],
            DataType::Date32,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(|value| (min..=max).contains(&value))));
    }

    #[test]
    fn timestamp_millisecond_outputs_stay_in_the_requested_range((min, max) in timestamp_range_strategy()) {
        let data_type = DataType::Timestamp(TimeUnit::Millisecond, None);
        let values = primitive_values::<TimestampMillisecondType>(&invoke_array(
            &TimestampMillisecond::new(),
            vec![
                scalar(ScalarValue::TimestampMillisecond(Some(min), None)),
                scalar(ScalarValue::TimestampMillisecond(Some(max), None)),
            ],
            data_type,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(|value| (min..=max).contains(&value))));
    }

    #[test]
    fn int64_normal_outputs_stay_in_the_truncation_range((min, max, mean, stddev) in i64_normal_strategy()) {
        let values = primitive_values::<Int64Type>(&invoke_array(
            &Int64Normal::new(),
            vec![
                scalar(ScalarValue::Int64(Some(min))),
                scalar(ScalarValue::Int64(Some(max))),
                scalar(ScalarValue::Int64(Some(mean))),
                scalar(ScalarValue::Float64(Some(stddev))),
            ],
            DataType::Int64,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(|value| (min..=max).contains(&value))));
    }

    #[test]
    fn uint64_normal_outputs_stay_in_the_truncation_range((min, max, mean, stddev) in u64_normal_strategy()) {
        let values = primitive_values::<UInt64Type>(&invoke_array(
            &UInt64Normal::new(),
            vec![
                scalar(ScalarValue::UInt64(Some(min))),
                scalar(ScalarValue::UInt64(Some(max))),
                scalar(ScalarValue::UInt64(Some(mean))),
                scalar(ScalarValue::Float64(Some(stddev))),
            ],
            DataType::UInt64,
            PROP_ROWS,
        ));

        prop_assert!(values.iter().all(|value| value.is_some_and(|value| (min..=max).contains(&value))));
    }
}

#[test]
fn uniform_integer_udfs_handle_full_type_ranges() {
    let int64_values = primitive_values::<Int64Type>(&invoke_array(
        &Int64Uniform::new(),
        vec![
            scalar(ScalarValue::Int64(Some(i64::MIN))),
            scalar(ScalarValue::Int64(Some(i64::MAX))),
        ],
        DataType::Int64,
        512,
    ));
    assert!(int64_values.iter().all(Option::is_some));

    let uint64_values = primitive_values::<UInt64Type>(&invoke_array(
        &UInt64Uniform::new(),
        vec![
            scalar(ScalarValue::UInt64(Some(0))),
            scalar(ScalarValue::UInt64(Some(u64::MAX))),
        ],
        DataType::UInt64,
        512,
    ));
    assert!(uint64_values.iter().all(Option::is_some));
}

fn stress_public_udfs(number_rows: usize) {
    let int64_uniform = primitive_values::<Int64Type>(&invoke_array(
        &Int64Uniform::new(),
        vec![
            scalar(ScalarValue::Int64(Some(-10_000))),
            scalar(ScalarValue::Int64(Some(10_000))),
        ],
        DataType::Int64,
        number_rows,
    ));
    assert!(
        int64_uniform
            .iter()
            .all(|value| value.is_some_and(|value| (-10_000..=10_000).contains(&value)))
    );

    let uint64_uniform = primitive_values::<UInt64Type>(&invoke_array(
        &UInt64Uniform::new(),
        vec![
            scalar(ScalarValue::UInt64(Some(0))),
            scalar(ScalarValue::UInt64(Some(u64::MAX))),
        ],
        DataType::UInt64,
        number_rows,
    ));
    assert!(uint64_uniform.iter().all(Option::is_some));

    let float64_uniform = primitive_values::<Float64Type>(&invoke_array(
        &Float64Uniform::new(),
        vec![
            scalar(ScalarValue::Float64(Some(-1_000_000.0))),
            scalar(ScalarValue::Float64(Some(1_000_000.0))),
        ],
        DataType::Float64,
        number_rows,
    ));
    assert!(float64_uniform.iter().all(|value| {
        value
            .is_some_and(|value| value.is_finite() && (-1_000_000.0..=1_000_000.0).contains(&value))
    }));

    let normal_values = primitive_values::<Float64Type>(&invoke_array(
        &Float64Normal::new(),
        vec![
            scalar(ScalarValue::Float64(Some(5.0))),
            scalar(ScalarValue::Float64(Some(2.0))),
        ],
        DataType::Float64,
        number_rows,
    ))
    .into_iter()
    .map(Option::unwrap)
    .collect::<Vec<_>>();
    let observed_mean = mean(&normal_values);
    let observed_stddev = stddev(&normal_values, observed_mean);
    assert!((4.75..=5.25).contains(&observed_mean));
    assert!((1.75..=2.25).contains(&observed_stddev));

    let int64_normal = primitive_values::<Int64Type>(&invoke_array(
        &Int64Normal::new(),
        vec![
            scalar(ScalarValue::Int64(Some(0))),
            scalar(ScalarValue::Int64(Some(200))),
            scalar(ScalarValue::Int64(Some(100))),
            scalar(ScalarValue::Float64(Some(25.0))),
        ],
        DataType::Int64,
        number_rows,
    ));
    assert!(
        int64_normal
            .iter()
            .all(|value| value.is_some_and(|value| (0..=200).contains(&value)))
    );

    let uint64_normal = primitive_values::<UInt64Type>(&invoke_array(
        &UInt64Normal::new(),
        vec![
            scalar(ScalarValue::UInt64(Some(0))),
            scalar(ScalarValue::UInt64(Some(200))),
            scalar(ScalarValue::UInt64(Some(100))),
            scalar(ScalarValue::Float64(Some(25.0))),
        ],
        DataType::UInt64,
        number_rows,
    ));
    assert!(
        uint64_normal
            .iter()
            .all(|value| value.is_some_and(|value| (0..=200).contains(&value)))
    );

    let bool_values = bool_values(&invoke_array(
        &Bool::new(),
        vec![scalar(ScalarValue::Float64(Some(0.25)))],
        DataType::Boolean,
        number_rows,
    ));
    let true_count = bool_values
        .iter()
        .filter(|value| **value == Some(true))
        .count();
    let true_rate = true_count as f64 / bool_values.len() as f64;
    assert!((0.20..=0.30).contains(&true_rate));

    let utf8_values = string_values(&invoke_array(
        &Utf8::new(),
        vec![
            scalar(ScalarValue::Utf8(Some("ABC123".to_owned()))),
            scalar(ScalarValue::Int64(Some(4))),
            scalar(ScalarValue::Int64(Some(24))),
        ],
        DataType::Utf8,
        number_rows,
    ));
    assert!(utf8_values.iter().all(|value| {
        value.as_ref().is_some_and(|value| {
            let length = value.chars().count();
            (4..=24).contains(&length)
                && value.chars().all(|character| "ABC123".contains(character))
        })
    }));

    let choice_values = string_values(&invoke_array(
        &Choice::new(),
        vec![scalar_choice_utf8(&[
            "UTC",
            "America/New_York",
            "Europe/London",
        ])],
        DataType::Utf8,
        number_rows,
    ));
    let observed_choices = choice_values
        .iter()
        .map(|value| value.as_deref().unwrap())
        .collect::<HashSet<_>>();
    assert_eq!(
        observed_choices,
        HashSet::from(["UTC", "America/New_York", "Europe/London"])
    );

    let date_values = primitive_values::<Date32Type>(&invoke_array(
        &Date32::new(),
        vec![
            scalar(ScalarValue::Date32(Some(-100_000))),
            scalar(ScalarValue::Date32(Some(100_000))),
        ],
        DataType::Date32,
        number_rows,
    ));
    assert!(
        date_values
            .iter()
            .all(|value| value.is_some_and(|value| (-100_000..=100_000).contains(&value)))
    );

    let timestamp_type = DataType::Timestamp(TimeUnit::Millisecond, None);
    let timestamp_values = primitive_values::<TimestampMillisecondType>(&invoke_array(
        &TimestampMillisecond::new(),
        vec![
            scalar(ScalarValue::TimestampMillisecond(Some(-1_000_000), None)),
            scalar(ScalarValue::TimestampMillisecond(Some(1_000_000), None)),
        ],
        timestamp_type,
        number_rows,
    ));
    assert!(
        timestamp_values
            .iter()
            .all(|value| value.is_some_and(|value| (-1_000_000..=1_000_000).contains(&value)))
    );
}

#[test]
fn stress_public_udfs_over_large_batches() {
    stress_public_udfs(STRESS_ROWS);
}

#[test]
#[ignore]
fn soak_public_udfs_over_repeated_large_batches() {
    let iterations = std::env::var("RANDGEN_SOAK_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(50);
    let rows = std::env::var("RANDGEN_SOAK_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8192);

    for _ in 0..iterations {
        stress_public_udfs(rows);
    }
}
