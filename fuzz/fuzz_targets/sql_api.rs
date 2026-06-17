#![no_main]

use std::collections::HashSet;
use std::sync::Arc;

use arbitrary::{Arbitrary, Unstructured};
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Float64Type, Int64Type, TimestampMillisecondType, UInt32Type,
    UInt64Type,
};
use arrow_array::{
    Array, BooleanArray, PrimitiveArray, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use datafusion_common::Result;
use datafusion_randgen::{all_udafs, all_udfs};
use libfuzzer_sys::fuzz_target;
use tokio::runtime::{Builder, Runtime};

thread_local! {
    static RUNTIME: Runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
}

#[derive(Arbitrary, Debug)]
enum SqlCase {
    Int64Uniform {
        min: i16,
        span: u8,
        rows: u8,
    },
    UInt64Uniform {
        min: u32,
        span: u8,
        rows: u8,
    },
    Float64Uniform {
        min: i16,
        width: u8,
        rows: u8,
    },
    Float64Normal {
        mean: i16,
        stddev: u8,
        rows: u8,
    },
    Int64Normal {
        min: i16,
        span: u8,
        mean_offset: u8,
        stddev: u8,
        rows: u8,
    },
    UInt64Normal {
        min: u16,
        span: u8,
        mean_offset: u8,
        stddev: u8,
        rows: u8,
    },
    Bool {
        probability: u8,
        rows: u8,
    },
    Utf8 {
        alphabet: u8,
        min_length: u8,
        extra_length: u8,
        rows: u8,
    },
    ChoiceInt64 {
        choices: [i16; 4],
        len: u8,
        rows: u8,
    },
    NullableInt64 {
        value: i16,
        probability: u8,
        rows: u8,
    },
    Date32 {
        rows: u8,
    },
    TimestampMillisecond {
        rows: u8,
    },
    ColumnChoiceUInt32 {
        values: [u32; 4],
        len: u8,
        rows: u8,
    },
    ColumnChoiceUInt64 {
        values: [u64; 4],
        len: u8,
        rows: u8,
    },
    InvalidRange,
    InvalidProbability,
}

fuzz_target!(|bytes: &[u8]| {
    let mut unstructured = Unstructured::new(bytes);
    let Ok(case) = SqlCase::arbitrary(&mut unstructured) else {
        return;
    };
    case.run();
});

impl SqlCase {
    fn run(self) {
        match self {
            Self::Int64Uniform { min, span, rows } => {
                let rows = bounded_rows(rows);
                let min = i64::from(min);
                let max = min + i64::from(span);
                let sql = format!(
                    "SELECT randgen_int64_uniform({min}, {max}) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid int64 uniform SQL");
                assert_eq!(row_count(&batches), rows);
                assert_range::<Int64Type>(&batches, 0, min, max);
            }
            Self::UInt64Uniform { min, span, rows } => {
                let rows = bounded_rows(rows);
                let min = u64::from(min);
                let max = min + u64::from(span);
                let sql = format!(
                    "SELECT randgen_uint64_uniform({min}, {max}) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid uint64 uniform SQL");
                assert_eq!(row_count(&batches), rows);
                assert_range::<UInt64Type>(&batches, 0, min, max);
            }
            Self::Float64Uniform { min, width, rows } => {
                let rows = bounded_rows(rows);
                let min = f64::from(min) / 10.0;
                let max = min + f64::from(width) / 10.0;
                let sql = format!(
                    "SELECT randgen_float64_uniform({min}, {max}) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid float64 uniform SQL");
                assert_eq!(row_count(&batches), rows);
                assert!(
                    primitive_values::<Float64Type>(&batches, 0)
                        .iter()
                        .all(|value| {
                            value.is_some_and(|value| {
                                value.is_finite() && (min..=max).contains(&value)
                            })
                        })
                );
            }
            Self::Float64Normal { mean, stddev, rows } => {
                let rows = bounded_rows(rows);
                let mean = f64::from(mean) / 10.0;
                let stddev = f64::from(stddev.max(1)) / 10.0;
                let sql = format!(
                    "SELECT randgen_float64_normal({mean}, {stddev}) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid float64 normal SQL");
                assert_eq!(row_count(&batches), rows);
                assert!(
                    primitive_values::<Float64Type>(&batches, 0)
                        .iter()
                        .all(|value| value.is_some_and(f64::is_finite))
                );
            }
            Self::Int64Normal {
                min,
                span,
                mean_offset,
                stddev,
                rows,
            } => {
                let rows = bounded_rows(rows);
                let min = i64::from(min);
                let max = min + i64::from(span);
                let mean = min + i64::from(mean_offset % span.saturating_add(1));
                let stddev = f64::from(stddev.max(1)) / 10.0;
                let sql = format!(
                    "SELECT randgen_int64_normal({min}, {max}, {mean}, {stddev}) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid int64 normal SQL");
                assert_eq!(row_count(&batches), rows);
                assert_range::<Int64Type>(&batches, 0, min, max);
            }
            Self::UInt64Normal {
                min,
                span,
                mean_offset,
                stddev,
                rows,
            } => {
                let rows = bounded_rows(rows);
                let min = u64::from(min);
                let max = min + u64::from(span);
                let mean = min + u64::from(mean_offset % span.saturating_add(1));
                let stddev = f64::from(stddev.max(1)) / 10.0;
                let sql = format!(
                    "SELECT randgen_uint64_normal({min}, {max}, {mean}, {stddev}) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid uint64 normal SQL");
                assert_eq!(row_count(&batches), rows);
                assert_range::<UInt64Type>(&batches, 0, min, max);
            }
            Self::Bool { probability, rows } => {
                let rows = bounded_rows(rows);
                let probability = f64::from(probability) / 255.0;
                let sql =
                    format!("SELECT randgen_bool({probability}) FROM generate_series(1, {rows})");
                let batches = collect(&sql).expect("valid bool SQL");
                assert_eq!(row_count(&batches), rows);
                let values = bool_values(&batches, 0);
                assert!(values.iter().all(Option::is_some));
                if probability == 0.0 {
                    assert!(values.iter().all(|value| *value == Some(false)));
                }
                if probability == 1.0 {
                    assert!(values.iter().all(|value| *value == Some(true)));
                }
            }
            Self::Utf8 {
                alphabet,
                min_length,
                extra_length,
                rows,
            } => {
                let rows = bounded_rows(rows);
                let alphabet = alphabet_for(alphabet);
                let min_length = min_length % 8;
                let max_length = min_length + extra_length % 8;
                let sql = format!(
                    "SELECT randgen_utf8('{alphabet}', {min_length}, {max_length}) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid utf8 SQL");
                assert_eq!(row_count(&batches), rows);
                let allowed = alphabet.chars().collect::<HashSet<_>>();
                assert!(string_values(&batches, 0).iter().all(|value| {
                    value.as_ref().is_some_and(|value| {
                        (usize::from(min_length)..=usize::from(max_length))
                            .contains(&value.chars().count())
                            && value.chars().all(|character| allowed.contains(&character))
                    })
                }));
            }
            Self::ChoiceInt64 { choices, len, rows } => {
                let rows = bounded_rows(rows);
                let len = usize::from(len % 4) + 1;
                let choices = choices[..len]
                    .iter()
                    .map(|value| i64::from(*value))
                    .collect::<Vec<_>>();
                let choices_sql = choices
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let sql = format!(
                    "SELECT randgen_choice([{choices_sql}]) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid choice SQL");
                assert_eq!(row_count(&batches), rows);
                let allowed = choices.iter().copied().collect::<HashSet<_>>();
                assert!(
                    primitive_values::<Int64Type>(&batches, 0)
                        .iter()
                        .all(|value| value.is_some_and(|value| allowed.contains(&value)))
                );
            }
            Self::NullableInt64 {
                value,
                probability,
                rows,
            } => {
                let rows = bounded_rows(rows);
                let value = i64::from(value);
                let probability = f64::from(probability) / 255.0;
                let sql = format!(
                    "SELECT randgen_nullable({value}, {probability}) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid nullable SQL");
                assert_eq!(row_count(&batches), rows);
                assert!(
                    primitive_values::<Int64Type>(&batches, 0)
                        .iter()
                        .all(|output| output.is_none() || *output == Some(value))
                );
            }
            Self::Date32 { rows } => {
                let rows = bounded_rows(rows);
                let sql = format!(
                    "SELECT randgen_date32(to_date('2024-01-01'), to_date('2024-01-31')) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid date32 SQL");
                assert_eq!(row_count(&batches), rows);
                assert_range::<Date32Type>(&batches, 0, 19723, 19753);
            }
            Self::TimestampMillisecond { rows } => {
                let rows = bounded_rows(rows);
                let sql = format!(
                    "SELECT randgen_timestamp_millisecond(to_timestamp_millis('2024-01-01T00:00:00Z'), to_timestamp_millis('2024-01-02T00:00:00Z')) FROM generate_series(1, {rows})"
                );
                let batches = collect(&sql).expect("valid timestamp SQL");
                assert_eq!(row_count(&batches), rows);
                assert_range::<TimestampMillisecondType>(
                    &batches,
                    0,
                    1_704_067_200_000,
                    1_704_153_600_000,
                );
            }
            Self::ColumnChoiceUInt32 { values, len, rows } => {
                let rows = bounded_rows(rows);
                let len = usize::from(len % 4) + 1;
                let values = values[..len].to_vec();
                let batches = collect_with_uint32_source(&values, rows)
                    .expect("valid uint32 column choice SQL");
                assert_eq!(row_count(&batches), rows);
                let allowed = values.iter().copied().collect::<HashSet<_>>();
                assert!(
                    primitive_values::<UInt32Type>(&batches, 0)
                        .iter()
                        .all(|value| value.is_some_and(|value| allowed.contains(&value)))
                );
            }
            Self::ColumnChoiceUInt64 { values, len, rows } => {
                let rows = bounded_rows(rows);
                let len = usize::from(len % 4) + 1;
                let values = values[..len].to_vec();
                let batches = collect_with_uint64_source(&values, rows)
                    .expect("valid uint64 column choice SQL");
                assert_eq!(row_count(&batches), rows);
                let allowed = values.iter().copied().collect::<HashSet<_>>();
                assert!(
                    primitive_values::<UInt64Type>(&batches, 0)
                        .iter()
                        .all(|value| value.is_some_and(|value| allowed.contains(&value)))
                );
            }
            Self::InvalidRange => {
                expect_error("SELECT randgen_int64_uniform(10, 1) FROM generate_series(1, 1)");
            }
            Self::InvalidProbability => {
                expect_error("SELECT randgen_bool(0.5, -0.1) FROM generate_series(1, 1)");
            }
        }
    }
}

fn bounded_rows(rows: u8) -> usize {
    usize::from(rows % 16) + 1
}

fn alphabet_for(index: u8) -> &'static str {
    match index % 4 {
        0 => "A",
        1 => "ABC",
        2 => "AABC",
        _ => "01xyz",
    }
}

fn context_with_all_functions() -> SessionContext {
    let ctx = SessionContext::new();
    for udf in all_udfs() {
        ctx.register_udf(udf);
    }
    for udaf in all_udafs() {
        ctx.register_udaf(udaf);
    }
    ctx
}

fn collect(sql: &str) -> Result<Vec<RecordBatch>> {
    RUNTIME.with(|runtime| {
        runtime.block_on(async {
            let ctx = context_with_all_functions();
            let df = ctx.sql(sql).await?;
            df.collect().await
        })
    })
}

fn collect_with_uint32_source(values: &[u32], rows: usize) -> Result<Vec<RecordBatch>> {
    let values = values.to_vec();
    RUNTIME.with(|runtime| {
        runtime.block_on(async move {
            let ctx = context_with_all_functions();
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt32, false)]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(UInt32Array::from(values))],
            )?;
            let table = MemTable::try_new(schema, vec![vec![batch]])?;
            ctx.register_table("source", Arc::new(table))?;
            let sql = format!(
                "WITH choices AS (SELECT randgen_roaring_agg(id) AS ids FROM source) \
                 SELECT randgen_column_choice(ids) FROM choices, generate_series(1, {rows})"
            );
            let df = ctx.sql(&sql).await?;
            df.collect().await
        })
    })
}

fn collect_with_uint64_source(values: &[u64], rows: usize) -> Result<Vec<RecordBatch>> {
    let values = values.to_vec();
    RUNTIME.with(|runtime| {
        runtime.block_on(async move {
            let ctx = context_with_all_functions();
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(UInt64Array::from(values))],
            )?;
            let table = MemTable::try_new(schema, vec![vec![batch]])?;
            ctx.register_table("source", Arc::new(table))?;
            let sql = format!(
                "WITH choices AS (SELECT randgen_roaring_agg(id) AS ids FROM source) \
                 SELECT randgen_column_choice(ids) FROM choices, generate_series(1, {rows})"
            );
            let df = ctx.sql(&sql).await?;
            df.collect().await
        })
    })
}

fn expect_error(sql: &str) {
    assert!(collect(sql).is_err(), "{sql}");
}

fn row_count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn primitive_values<T>(batches: &[RecordBatch], column_index: usize) -> Vec<Option<T::Native>>
where
    T: ArrowPrimitiveType,
{
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column_index)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .expect("primitive column")
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn bool_values(batches: &[RecordBatch], column_index: usize) -> Vec<Option<bool>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column_index)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("boolean column")
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn string_values(batches: &[RecordBatch], column_index: usize) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column_index)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string column")
                .iter()
                .map(|value| value.map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn assert_range<T>(batches: &[RecordBatch], column_index: usize, min: T::Native, max: T::Native)
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    assert!(
        primitive_values::<T>(batches, column_index)
            .iter()
            .all(|value| value.is_some_and(|value| min <= value && value <= max))
    );
}
