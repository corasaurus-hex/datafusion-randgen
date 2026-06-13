use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{
    ArrayRef, Date32Array, Float64Array, Int64Array, TimestampMillisecondArray, UInt64Array,
};
use arrow_schema::{DataType, Field, TimeUnit};
use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use datafusion_common::{ScalarValue, config::ConfigOptions};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_randgen::{
    Bool, Choice, Date32, Float64Normal, Float64Uniform, Int64Normal, Int64Uniform,
    TimestampMillisecond, UInt64Normal, UInt64Uniform, Utf8,
};

const ROWS: usize = 16_384;

fn array<T>(array: T) -> ArrayRef
where
    T: arrow_array::Array + 'static,
{
    Arc::new(array)
}

fn args(
    args: Vec<ColumnarValue>,
    return_type: DataType,
    return_name: &'static str,
) -> ScalarFunctionArgs {
    let arg_fields = args
        .iter()
        .enumerate()
        .map(|(index, arg)| Arc::new(Field::new(format!("arg{index}"), arg.data_type(), true)))
        .collect();

    ScalarFunctionArgs {
        args,
        arg_fields,
        number_rows: ROWS,
        return_field: Arc::new(Field::new(return_name, return_type, true)),
        config_options: Arc::new(ConfigOptions::default()),
    }
}

fn finish(value: datafusion_common::Result<ColumnarValue>) {
    let value = value.unwrap();
    let ColumnarValue::Array(array) = value else {
        panic!("expected array result");
    };
    black_box(array.len());
}

fn bench_udf_invocation(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: &str,
    udf: &impl ScalarUDFImpl,
    invocation_args: Vec<ColumnarValue>,
    return_type: DataType,
    return_name: &'static str,
) {
    group.bench_function(BenchmarkId::new(id, ROWS), |b| {
        b.iter(|| {
            finish(udf.invoke_with_args(args(
                invocation_args.clone(),
                return_type.clone(),
                return_name,
            )));
        });
    });
}

fn bench_array_scalar_pair(
    group: &mut BenchmarkGroup<'_, WallTime>,
    base_id: &str,
    udf: &impl ScalarUDFImpl,
    array_args: Vec<ColumnarValue>,
    scalar_args: Vec<ColumnarValue>,
    return_type: DataType,
    return_name: &'static str,
) {
    bench_udf_invocation(
        group,
        base_id,
        udf,
        array_args,
        return_type.clone(),
        return_name,
    );
    bench_udf_invocation(
        group,
        &format!("{base_id}_scalar_args"),
        udf,
        scalar_args,
        return_type,
        return_name,
    );
}

fn scalar_choice(values: &[&str]) -> ColumnarValue {
    let values = values
        .iter()
        .map(|value| ScalarValue::Utf8(Some((*value).to_owned())))
        .collect::<Vec<_>>();

    ColumnarValue::Scalar(ScalarValue::List(ScalarValue::new_list_nullable(
        &values,
        &DataType::Utf8,
    )))
}

fn scalar_choice_int64(values: &[i64]) -> ColumnarValue {
    let values = values
        .iter()
        .map(|value| ScalarValue::Int64(Some(*value)))
        .collect::<Vec<_>>();

    ColumnarValue::Scalar(ScalarValue::List(ScalarValue::new_list_nullable(
        &values,
        &DataType::Int64,
    )))
}

fn scalar_choice_float64(values: &[f64]) -> ColumnarValue {
    let values = values
        .iter()
        .map(|value| ScalarValue::Float64(Some(*value)))
        .collect::<Vec<_>>();

    ColumnarValue::Scalar(ScalarValue::List(ScalarValue::new_list_nullable(
        &values,
        &DataType::Float64,
    )))
}

fn bench_udfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("udfs");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let int_min = ColumnarValue::Array(array(Int64Array::from(vec![Some(1_i64); ROWS])));
    let int_max = ColumnarValue::Array(array(Int64Array::from(vec![Some(1_000_i64); ROWS])));
    let int_min_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(1)));
    let int_max_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(1_000)));
    let int64_uniform = Int64Uniform::new();
    group.throughput(Throughput::Elements(ROWS as u64));
    bench_array_scalar_pair(
        &mut group,
        "randgen_int64_uniform",
        &int64_uniform,
        vec![int_min.clone(), int_max.clone()],
        vec![int_min_scalar.clone(), int_max_scalar.clone()],
        DataType::Int64,
        "randgen_int64_uniform",
    );

    let uint_min = ColumnarValue::Array(array(UInt64Array::from(vec![Some(1_u64); ROWS])));
    let uint_max = ColumnarValue::Array(array(UInt64Array::from(vec![Some(1_000_u64); ROWS])));
    let uint_min_scalar = ColumnarValue::Scalar(ScalarValue::UInt64(Some(1)));
    let uint_max_scalar = ColumnarValue::Scalar(ScalarValue::UInt64(Some(1_000)));
    let uint64_uniform = UInt64Uniform::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_uint64_uniform",
        &uint64_uniform,
        vec![uint_min.clone(), uint_max.clone()],
        vec![uint_min_scalar.clone(), uint_max_scalar.clone()],
        DataType::UInt64,
        "randgen_uint64_uniform",
    );

    let float_min = ColumnarValue::Array(array(Float64Array::from(vec![Some(1.0_f64); ROWS])));
    let float_max = ColumnarValue::Array(array(Float64Array::from(vec![Some(1_000.0_f64); ROWS])));
    let float_min_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0)));
    let float_max_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(1_000.0)));
    let float64_uniform = Float64Uniform::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_float64_uniform",
        &float64_uniform,
        vec![float_min.clone(), float_max.clone()],
        vec![float_min_scalar.clone(), float_max_scalar.clone()],
        DataType::Float64,
        "randgen_float64_uniform",
    );

    let normal_mean = ColumnarValue::Array(array(Float64Array::from(vec![Some(10.0_f64); ROWS])));
    let normal_stddev = ColumnarValue::Array(array(Float64Array::from(vec![Some(2.0_f64); ROWS])));
    let normal_mean_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(10.0)));
    let normal_stddev_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(2.0)));
    let float64_normal = Float64Normal::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_float64_normal",
        &float64_normal,
        vec![normal_mean.clone(), normal_stddev.clone()],
        vec![normal_mean_scalar.clone(), normal_stddev_scalar.clone()],
        DataType::Float64,
        "randgen_float64_normal",
    );

    let int_normal_min = ColumnarValue::Array(array(Int64Array::from(vec![Some(0_i64); ROWS])));
    let int_normal_max = ColumnarValue::Array(array(Int64Array::from(vec![Some(20_i64); ROWS])));
    let int_normal_mean = ColumnarValue::Array(array(Int64Array::from(vec![Some(10_i64); ROWS])));
    let int_normal_min_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(0)));
    let int_normal_max_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(20)));
    let int_normal_mean_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(10)));
    let int64_normal = Int64Normal::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_int64_normal",
        &int64_normal,
        vec![
            int_normal_min.clone(),
            int_normal_max.clone(),
            int_normal_mean.clone(),
            normal_stddev.clone(),
        ],
        vec![
            int_normal_min_scalar.clone(),
            int_normal_max_scalar.clone(),
            int_normal_mean_scalar.clone(),
            normal_stddev_scalar.clone(),
        ],
        DataType::Int64,
        "randgen_int64_normal",
    );

    let uint_normal_min = ColumnarValue::Array(array(UInt64Array::from(vec![Some(0_u64); ROWS])));
    let uint_normal_max = ColumnarValue::Array(array(UInt64Array::from(vec![Some(20_u64); ROWS])));
    let uint_normal_mean = ColumnarValue::Array(array(UInt64Array::from(vec![Some(10_u64); ROWS])));
    let uint_normal_min_scalar = ColumnarValue::Scalar(ScalarValue::UInt64(Some(0)));
    let uint_normal_max_scalar = ColumnarValue::Scalar(ScalarValue::UInt64(Some(20)));
    let uint_normal_mean_scalar = ColumnarValue::Scalar(ScalarValue::UInt64(Some(10)));
    let uint64_normal = UInt64Normal::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_uint64_normal",
        &uint64_normal,
        vec![
            uint_normal_min.clone(),
            uint_normal_max.clone(),
            uint_normal_mean.clone(),
            normal_stddev.clone(),
        ],
        vec![
            uint_normal_min_scalar.clone(),
            uint_normal_max_scalar.clone(),
            uint_normal_mean_scalar.clone(),
            normal_stddev_scalar.clone(),
        ],
        DataType::UInt64,
        "randgen_uint64_normal",
    );

    let probability = ColumnarValue::Array(array(Float64Array::from(vec![Some(0.5_f64); ROWS])));
    let probability_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(0.5)));
    let bool_udf = Bool::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_bool",
        &bool_udf,
        vec![probability.clone()],
        vec![probability_scalar.clone()],
        DataType::Boolean,
        "randgen_bool",
    );

    let utf8_characters = ColumnarValue::Scalar(ScalarValue::Utf8(Some(
        "abcdefghijklmnopqrstuvwxyz0123456789".to_owned(),
    )));
    let utf8_min = ColumnarValue::Array(array(Int64Array::from(vec![Some(12_i64); ROWS])));
    let utf8_max = ColumnarValue::Array(array(Int64Array::from(vec![Some(24_i64); ROWS])));
    let utf8_min_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(12)));
    let utf8_max_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(24)));
    let utf8 = Utf8::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_utf8",
        &utf8,
        vec![utf8_characters.clone(), utf8_min.clone(), utf8_max.clone()],
        vec![
            utf8_characters.clone(),
            utf8_min_scalar.clone(),
            utf8_max_scalar.clone(),
        ],
        DataType::Utf8,
        "randgen_utf8",
    );

    let choice = Choice::new();
    let choice_values = scalar_choice(&[
        "UTC",
        "America/New_York",
        "Europe/London",
        "Asia/Tokyo",
        "Australia/Sydney",
    ]);
    bench_udf_invocation(
        &mut group,
        "randgen_choice_utf8",
        &choice,
        vec![choice_values.clone()],
        DataType::Utf8,
        "randgen_choice",
    );
    let choice_int64_values = scalar_choice_int64(&[10, 20, 30, 40, 50]);
    bench_udf_invocation(
        &mut group,
        "randgen_choice_int64",
        &choice,
        vec![choice_int64_values.clone()],
        DataType::Int64,
        "randgen_choice",
    );
    let choice_float64_values = scalar_choice_float64(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    bench_udf_invocation(
        &mut group,
        "randgen_choice_float64",
        &choice,
        vec![choice_float64_values.clone()],
        DataType::Float64,
        "randgen_choice",
    );

    let date_min = ColumnarValue::Array(array(Date32Array::from(vec![Some(19_723_i32); ROWS])));
    let date_max = ColumnarValue::Array(array(Date32Array::from(vec![Some(19_753_i32); ROWS])));
    let date_min_scalar = ColumnarValue::Scalar(ScalarValue::Date32(Some(19_723)));
    let date_max_scalar = ColumnarValue::Scalar(ScalarValue::Date32(Some(19_753)));
    let date32 = Date32::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_date32",
        &date32,
        vec![date_min.clone(), date_max.clone()],
        vec![date_min_scalar.clone(), date_max_scalar.clone()],
        DataType::Date32,
        "randgen_date32",
    );

    let timestamp_type = DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into()));
    let timestamp_timezone = Some("+00:00".into());
    let timestamp_min = ColumnarValue::Array(array(
        TimestampMillisecondArray::from(vec![Some(1_704_067_200_000_i64); ROWS])
            .with_timezone("+00:00"),
    ));
    let timestamp_max = ColumnarValue::Array(array(
        TimestampMillisecondArray::from(vec![Some(1_704_153_600_000_i64); ROWS])
            .with_timezone("+00:00"),
    ));
    let timestamp_min_scalar = ColumnarValue::Scalar(ScalarValue::TimestampMillisecond(
        Some(1_704_067_200_000),
        timestamp_timezone.clone(),
    ));
    let timestamp_max_scalar = ColumnarValue::Scalar(ScalarValue::TimestampMillisecond(
        Some(1_704_153_600_000),
        timestamp_timezone,
    ));
    let timestamp_millisecond = TimestampMillisecond::new();
    bench_array_scalar_pair(
        &mut group,
        "randgen_timestamp_millisecond",
        &timestamp_millisecond,
        vec![timestamp_min.clone(), timestamp_max.clone()],
        vec![timestamp_min_scalar.clone(), timestamp_max_scalar.clone()],
        timestamp_type.clone(),
        "randgen_timestamp_millisecond",
    );

    group.finish();
}

criterion_group!(benches, bench_udfs);
criterion_main!(benches);
