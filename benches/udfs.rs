use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{ArrayRef, Date32Array, Float64Array, Int64Array, TimestampMillisecondArray};
use arrow_schema::{DataType, Field, TimeUnit};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datafusion_common::{ScalarValue, config::ConfigOptions};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_randgen::{
    Bool, Choice, Date32, Float64Normal, Float64Uniform, Int64Uniform, TimestampMillisecond, Utf8,
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
    group.bench_function(BenchmarkId::new("randgen_int64_uniform", ROWS), |b| {
        b.iter(|| {
            finish(int64_uniform.invoke_with_args(args(
                vec![int_min.clone(), int_max.clone()],
                DataType::Int64,
                "randgen_int64_uniform",
            )));
        });
    });
    group.bench_function(
        BenchmarkId::new("randgen_int64_uniform_scalar_args", ROWS),
        |b| {
            b.iter(|| {
                finish(int64_uniform.invoke_with_args(args(
                    vec![int_min_scalar.clone(), int_max_scalar.clone()],
                    DataType::Int64,
                    "randgen_int64_uniform",
                )));
            });
        },
    );

    let float_min = ColumnarValue::Array(array(Float64Array::from(vec![Some(1.0_f64); ROWS])));
    let float_max = ColumnarValue::Array(array(Float64Array::from(vec![Some(1_000.0_f64); ROWS])));
    let float_min_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0)));
    let float_max_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(1_000.0)));
    let float64_uniform = Float64Uniform::new();
    group.bench_function(BenchmarkId::new("randgen_float64_uniform", ROWS), |b| {
        b.iter(|| {
            finish(float64_uniform.invoke_with_args(args(
                vec![float_min.clone(), float_max.clone()],
                DataType::Float64,
                "randgen_float64_uniform",
            )));
        });
    });
    group.bench_function(
        BenchmarkId::new("randgen_float64_uniform_scalar_args", ROWS),
        |b| {
            b.iter(|| {
                finish(float64_uniform.invoke_with_args(args(
                    vec![float_min_scalar.clone(), float_max_scalar.clone()],
                    DataType::Float64,
                    "randgen_float64_uniform",
                )));
            });
        },
    );

    let normal_mean = ColumnarValue::Array(array(Float64Array::from(vec![Some(10.0_f64); ROWS])));
    let normal_stddev = ColumnarValue::Array(array(Float64Array::from(vec![Some(2.0_f64); ROWS])));
    let normal_mean_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(10.0)));
    let normal_stddev_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(2.0)));
    let float64_normal = Float64Normal::new();
    group.bench_function(BenchmarkId::new("randgen_float64_normal", ROWS), |b| {
        b.iter(|| {
            finish(float64_normal.invoke_with_args(args(
                vec![normal_mean.clone(), normal_stddev.clone()],
                DataType::Float64,
                "randgen_float64_normal",
            )));
        });
    });
    group.bench_function(
        BenchmarkId::new("randgen_float64_normal_scalar_args", ROWS),
        |b| {
            b.iter(|| {
                finish(float64_normal.invoke_with_args(args(
                    vec![normal_mean_scalar.clone(), normal_stddev_scalar.clone()],
                    DataType::Float64,
                    "randgen_float64_normal",
                )));
            });
        },
    );

    let probability = ColumnarValue::Array(array(Float64Array::from(vec![Some(0.5_f64); ROWS])));
    let probability_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(0.5)));
    let bool_udf = Bool::new();
    group.bench_function(BenchmarkId::new("randgen_bool", ROWS), |b| {
        b.iter(|| {
            finish(bool_udf.invoke_with_args(args(
                vec![probability.clone()],
                DataType::Boolean,
                "randgen_bool",
            )));
        });
    });
    group.bench_function(BenchmarkId::new("randgen_bool_scalar_args", ROWS), |b| {
        b.iter(|| {
            finish(bool_udf.invoke_with_args(args(
                vec![probability_scalar.clone()],
                DataType::Boolean,
                "randgen_bool",
            )));
        });
    });

    let utf8_characters = ColumnarValue::Scalar(ScalarValue::Utf8(Some(
        "abcdefghijklmnopqrstuvwxyz0123456789".to_owned(),
    )));
    let utf8_min = ColumnarValue::Array(array(Int64Array::from(vec![Some(12_i64); ROWS])));
    let utf8_max = ColumnarValue::Array(array(Int64Array::from(vec![Some(24_i64); ROWS])));
    let utf8_min_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(12)));
    let utf8_max_scalar = ColumnarValue::Scalar(ScalarValue::Int64(Some(24)));
    let utf8 = Utf8::new();
    group.bench_function(BenchmarkId::new("randgen_utf8", ROWS), |b| {
        b.iter(|| {
            finish(utf8.invoke_with_args(args(
                vec![utf8_characters.clone(), utf8_min.clone(), utf8_max.clone()],
                DataType::Utf8,
                "randgen_utf8",
            )));
        });
    });
    group.bench_function(BenchmarkId::new("randgen_utf8_scalar_args", ROWS), |b| {
        b.iter(|| {
            finish(utf8.invoke_with_args(args(
                vec![
                    utf8_characters.clone(),
                    utf8_min_scalar.clone(),
                    utf8_max_scalar.clone(),
                ],
                DataType::Utf8,
                "randgen_utf8",
            )));
        });
    });

    let choice = Choice::new();
    let choice_values = scalar_choice(&[
        "UTC",
        "America/New_York",
        "Europe/London",
        "Asia/Tokyo",
        "Australia/Sydney",
    ]);
    group.bench_function(BenchmarkId::new("randgen_choice_utf8", ROWS), |b| {
        b.iter(|| {
            finish(choice.invoke_with_args(args(
                vec![choice_values.clone()],
                DataType::Utf8,
                "randgen_choice",
            )));
        });
    });

    let date_min = ColumnarValue::Array(array(Date32Array::from(vec![Some(19_723_i32); ROWS])));
    let date_max = ColumnarValue::Array(array(Date32Array::from(vec![Some(19_753_i32); ROWS])));
    let date_min_scalar = ColumnarValue::Scalar(ScalarValue::Date32(Some(19_723)));
    let date_max_scalar = ColumnarValue::Scalar(ScalarValue::Date32(Some(19_753)));
    let date32 = Date32::new();
    group.bench_function(BenchmarkId::new("randgen_date32", ROWS), |b| {
        b.iter(|| {
            finish(date32.invoke_with_args(args(
                vec![date_min.clone(), date_max.clone()],
                DataType::Date32,
                "randgen_date32",
            )));
        });
    });
    group.bench_function(BenchmarkId::new("randgen_date32_scalar_args", ROWS), |b| {
        b.iter(|| {
            finish(date32.invoke_with_args(args(
                vec![date_min_scalar.clone(), date_max_scalar.clone()],
                DataType::Date32,
                "randgen_date32",
            )));
        });
    });

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
    group.bench_function(
        BenchmarkId::new("randgen_timestamp_millisecond", ROWS),
        |b| {
            b.iter(|| {
                finish(timestamp_millisecond.invoke_with_args(args(
                    vec![timestamp_min.clone(), timestamp_max.clone()],
                    timestamp_type.clone(),
                    "randgen_timestamp_millisecond",
                )));
            });
        },
    );
    group.bench_function(
        BenchmarkId::new("randgen_timestamp_millisecond_scalar_args", ROWS),
        |b| {
            b.iter(|| {
                finish(timestamp_millisecond.invoke_with_args(args(
                    vec![timestamp_min_scalar.clone(), timestamp_max_scalar.clone()],
                    timestamp_type.clone(),
                    "randgen_timestamp_millisecond",
                )));
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_udfs);
criterion_main!(benches);
