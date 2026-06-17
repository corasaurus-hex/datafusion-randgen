use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Float64Type, Int64Type, TimestampMillisecondType, UInt64Type,
};
use arrow_array::{Array, BooleanArray, PrimitiveArray, RecordBatch, StringArray};
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
use arrow_array::{UInt32Array, UInt64Array};
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
use arrow_schema::{DataType, Field, Schema};
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use datafusion_common::Result;
use datafusion_randgen::{all_udafs, all_udfs};
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
use std::collections::HashSet;
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
use std::sync::Arc;

const SQL_STRESS_ROWS: usize = 4096;

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

async fn collect(sql: &str) -> Result<Vec<RecordBatch>> {
    let ctx = context_with_all_functions();
    let df = ctx.sql(sql).await?;
    df.collect().await
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

async fn stress_sql_public_udfs(number_rows: usize) {
    let sql = format!(
        "SELECT \
            randgen_int64_uniform(-10000, 10000) AS int64_uniform, \
            randgen_uint64_uniform(0, 18446744073709551615) AS uint64_uniform, \
            randgen_float64_uniform(-1000000.0, 1000000.0) AS float64_uniform, \
            randgen_float64_normal(5.0, 2.0) AS float64_normal, \
            randgen_int64_normal(0, 200, 100, 25.0) AS int64_normal, \
            randgen_uint64_normal(0, 200, 100, 25.0) AS uint64_normal, \
            randgen_bool(0.25) AS bool_value, \
            randgen_utf8('ABC123', 4, 24) AS text_value, \
            randgen_choice(['UTC', 'America/New_York', 'Europe/London']) AS choice_value, \
            randgen_nullable(7, 0.25) AS nullable_value, \
            randgen_date32(to_date('1700-01-01'), to_date('2300-01-01')) AS date_value, \
            randgen_timestamp_millisecond(to_timestamp_millis('1900-01-01T00:00:00Z'), to_timestamp_millis('2100-01-01T00:00:00Z')) AS timestamp_value \
         FROM generate_series(1, {number_rows})"
    );
    let batches = collect(&sql).await.unwrap();

    assert_eq!(row_count(&batches), number_rows);
    assert!(
        primitive_values::<Int64Type>(&batches, 0)
            .iter()
            .all(|value| value.is_some_and(|value| (-10000..=10000).contains(&value)))
    );
    assert!(
        primitive_values::<UInt64Type>(&batches, 1)
            .iter()
            .all(Option::is_some)
    );
    assert!(
        primitive_values::<Float64Type>(&batches, 2)
            .iter()
            .all(|value| value.is_some_and(
                |value| value.is_finite() && (-1000000.0..=1000000.0).contains(&value)
            ))
    );
    assert!(
        primitive_values::<Float64Type>(&batches, 3)
            .iter()
            .all(|value| value.is_some_and(f64::is_finite))
    );
    assert!(
        primitive_values::<Int64Type>(&batches, 4)
            .iter()
            .all(|value| value.is_some_and(|value| (0..=200).contains(&value)))
    );
    assert!(
        primitive_values::<UInt64Type>(&batches, 5)
            .iter()
            .all(|value| value.is_some_and(|value| (0..=200).contains(&value)))
    );
    assert!(bool_values(&batches, 6).iter().all(Option::is_some));
    assert!(string_values(&batches, 7).iter().all(|value| {
        value.as_ref().is_some_and(|value| {
            let length = value.chars().count();
            (4..=24).contains(&length)
                && value.chars().all(|character| "ABC123".contains(character))
        })
    }));
    assert!(string_values(&batches, 8).iter().all(|value| matches!(
        value.as_deref(),
        Some("UTC" | "America/New_York" | "Europe/London")
    )));
    assert!(
        primitive_values::<Int64Type>(&batches, 9)
            .iter()
            .all(|value| value.is_none() || *value == Some(7))
    );
    assert!(
        primitive_values::<Date32Type>(&batches, 10)
            .iter()
            .all(Option::is_some)
    );
    assert!(
        primitive_values::<TimestampMillisecondType>(&batches, 11)
            .iter()
            .all(Option::is_some)
    );
}

#[tokio::test]
async fn sql_public_scalar_generators_obey_documented_invariants() {
    let batches = collect(
        "SELECT \
            randgen_int64_uniform(-10, 10) AS int64_uniform, \
            randgen_uint64_uniform(0, 18446744073709551615) AS uint64_uniform, \
            randgen_float64_uniform(-1.5, 2.5) AS float64_uniform, \
            randgen_float64_normal(10.0, 2.0) AS float64_normal, \
            randgen_int64_normal(0, 20, 10, 2.0) AS int64_normal, \
            randgen_uint64_normal(0, 20, 10, 2.0) AS uint64_normal, \
            randgen_bool(1.0) AS bool_true, \
            randgen_utf8('ABC', 2, 6) AS text_value, \
            randgen_choice(['UTC', 'Europe/London']) AS choice_value, \
            randgen_nullable(7, 0.0) AS nullable_value, \
            randgen_date32(to_date('2024-01-01'), to_date('2024-01-31')) AS date_value, \
            randgen_timestamp_millisecond(to_timestamp_millis('2024-01-01T00:00:00Z'), to_timestamp_millis('2024-01-02T00:00:00Z')) AS timestamp_value \
         FROM generate_series(1, 128)",
    )
    .await
    .unwrap();

    assert_eq!(row_count(&batches), 128);

    assert!(
        primitive_values::<Int64Type>(&batches, 0)
            .iter()
            .all(|value| value.is_some_and(|value| (-10..=10).contains(&value)))
    );
    assert!(
        primitive_values::<UInt64Type>(&batches, 1)
            .iter()
            .all(Option::is_some)
    );
    assert!(
        primitive_values::<Float64Type>(&batches, 2)
            .iter()
            .all(|value| value
                .is_some_and(|value| value.is_finite() && (-1.5..=2.5).contains(&value)))
    );
    assert!(
        primitive_values::<Float64Type>(&batches, 3)
            .iter()
            .all(|value| value.is_some_and(f64::is_finite))
    );
    assert!(
        primitive_values::<Int64Type>(&batches, 4)
            .iter()
            .all(|value| value.is_some_and(|value| (0..=20).contains(&value)))
    );
    assert!(
        primitive_values::<UInt64Type>(&batches, 5)
            .iter()
            .all(|value| value.is_some_and(|value| (0..=20).contains(&value)))
    );
    assert!(
        bool_values(&batches, 6)
            .iter()
            .all(|value| *value == Some(true))
    );
    assert!(string_values(&batches, 7).iter().all(|value| {
        value.as_ref().is_some_and(|value| {
            (2..=6).contains(&value.chars().count())
                && value.chars().all(|character| "ABC".contains(character))
        })
    }));
    assert!(
        string_values(&batches, 8)
            .iter()
            .all(|value| { matches!(value.as_deref(), Some("UTC" | "Europe/London")) })
    );
    assert!(
        primitive_values::<Int64Type>(&batches, 9)
            .iter()
            .all(|value| *value == Some(7))
    );
    assert!(
        primitive_values::<Date32Type>(&batches, 10)
            .iter()
            .all(|value| value.is_some_and(|value| (19723..=19753).contains(&value)))
    );
    assert!(
        primitive_values::<TimestampMillisecondType>(&batches, 11)
            .iter()
            .all(|value| value
                .is_some_and(|value| (1_704_067_200_000..=1_704_153_600_000).contains(&value)))
    );
}

#[tokio::test]
async fn sql_row_wise_arguments_and_native_nullability_work_together() {
    let batches = collect(
        "SELECT \
            randgen_int64_uniform(min_value, max_value, null_probability) AS int_value, \
            randgen_bool(probability, null_probability) AS bool_value, \
            randgen_utf8(characters, min_length, max_length, null_probability) AS text_value, \
            randgen_date32(min_date, max_date, null_probability) AS date_value \
         FROM (VALUES \
            (-5, -5, 0.0, 1.0, 'A', 2, 2, to_date('2024-01-01'), to_date('2024-01-01')), \
            (10, 10, 1.0, 0.0, 'B', 3, 3, to_date('2024-01-02'), to_date('2024-01-02')), \
            (CAST(NULL AS BIGINT), 20, 0.0, 1.0, 'C', 1, 1, CAST(NULL AS DATE), to_date('2024-01-03')) \
         ) AS t(min_value, max_value, null_probability, probability, characters, min_length, max_length, min_date, max_date)",
    )
    .await
    .unwrap();

    assert_eq!(
        primitive_values::<Int64Type>(&batches, 0),
        vec![Some(-5), None, None]
    );
    assert_eq!(bool_values(&batches, 1), vec![Some(true), None, Some(true)]);
    assert_eq!(
        string_values(&batches, 2),
        vec![Some("AA".to_owned()), None, Some("C".to_owned())]
    );
    assert_eq!(
        primitive_values::<Date32Type>(&batches, 3),
        vec![Some(19723), None, None]
    );
}

#[tokio::test]
async fn sql_adversarial_invalid_parameters_return_errors() {
    let invalid_queries = [
        "SELECT randgen_int64_uniform(10, 1) FROM generate_series(1, 1)",
        "SELECT randgen_uint64_uniform(-1, 10) FROM generate_series(1, 1)",
        "SELECT randgen_float64_uniform(CAST('NaN' AS DOUBLE), 1.0) FROM generate_series(1, 1)",
        "SELECT randgen_float64_normal(0.0, 0.0) FROM generate_series(1, 1)",
        "SELECT randgen_int64_normal(10, 1, 5, 1.0) FROM generate_series(1, 1)",
        "SELECT randgen_uint64_normal(0, 10, -1, 1.0) FROM generate_series(1, 1)",
        "SELECT randgen_bool(1.1) FROM generate_series(1, 1)",
        "SELECT randgen_utf8('', 1, 1) FROM generate_series(1, 1)",
        "SELECT randgen_choice([]) FROM generate_series(1, 1)",
        "SELECT randgen_nullable(7, -0.1) FROM generate_series(1, 1)",
        "SELECT randgen_date32(to_date('2024-01-02'), to_date('2024-01-01')) FROM generate_series(1, 1)",
        "SELECT randgen_timestamp_millisecond(to_timestamp_millis('2024-01-02T00:00:00Z'), to_timestamp_millis('2024-01-01T00:00:00Z')) FROM generate_series(1, 1)",
    ];

    for sql in invalid_queries {
        assert!(collect(sql).await.is_err(), "{sql}");
    }
}

#[tokio::test]
async fn stress_sql_public_udfs_over_large_batches() {
    stress_sql_public_udfs(SQL_STRESS_ROWS).await;
}

#[tokio::test]
#[ignore]
async fn soak_sql_public_udfs_over_repeated_large_batches() {
    let iterations = std::env::var("RANDGEN_SQL_SOAK_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(25);
    let rows = std::env::var("RANDGEN_SQL_SOAK_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096);

    for _ in 0..iterations {
        stress_sql_public_udfs(rows).await;
    }
}

#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
#[tokio::test]
async fn sql_column_choice_round_trip_obeys_documented_invariants() {
    let ctx = context_with_all_functions();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id32", DataType::UInt32, true),
        Field::new("id64", DataType::UInt64, true),
    ]));
    let high = u64::from(u32::MAX) + 37;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(UInt32Array::from(vec![Some(3), Some(5), Some(5), None])),
            Arc::new(UInt64Array::from(vec![
                Some(2),
                Some(high),
                Some(high),
                None,
            ])),
        ],
    )
    .unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table("source", Arc::new(table)).unwrap();

    let df = ctx
        .sql(
            "WITH choices AS (SELECT randgen_roaring_agg(id32) AS ids32, randgen_roaring_agg(id64) AS ids64 FROM source) \
             SELECT randgen_column_choice(ids32) AS choice32, randgen_column_choice(ids64) AS choice64 \
             FROM choices, generate_series(1, 128)",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let allowed32 = HashSet::from([3, 5]);
    let allowed64 = HashSet::from([2, high]);
    assert!(
        primitive_values::<arrow_array::types::UInt32Type>(&batches, 0)
            .iter()
            .all(|value| value.is_some_and(|value| allowed32.contains(&value)))
    );
    assert!(
        primitive_values::<UInt64Type>(&batches, 1)
            .iter()
            .all(|value| value.is_some_and(|value| allowed64.contains(&value)))
    );
}

#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
#[tokio::test]
async fn stress_sql_column_choice_over_large_source_and_sample_batches() {
    let ctx = context_with_all_functions();
    let source_rows = 4096;
    let sample_rows = 2048;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id32", DataType::UInt32, true),
        Field::new("id64", DataType::UInt64, true),
    ]));

    let id32 = UInt32Array::from(
        (0..source_rows)
            .map(|row| {
                if row % 17 == 0 {
                    None
                } else {
                    Some((row % 1024) as u32)
                }
            })
            .collect::<Vec<_>>(),
    );
    let id64 = UInt64Array::from(
        (0..source_rows)
            .map(|row| {
                if row % 19 == 0 {
                    None
                } else {
                    Some(u64::from(u32::MAX) + (row % 1024) as u64)
                }
            })
            .collect::<Vec<_>>(),
    );
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(id32), Arc::new(id64)]).unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table("source", Arc::new(table)).unwrap();

    let df = ctx
        .sql(&format!(
            "WITH choices AS (SELECT randgen_roaring_agg(id32) AS ids32, randgen_roaring_agg(id64) AS ids64 FROM source) \
             SELECT randgen_column_choice(ids32) AS choice32, randgen_column_choice(ids64) AS choice64 \
             FROM choices, generate_series(1, {sample_rows})"
        ))
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(row_count(&batches), sample_rows);
    assert!(
        primitive_values::<arrow_array::types::UInt32Type>(&batches, 0)
            .iter()
            .all(|value| value.is_some_and(|value| value < 1024))
    );
    assert!(
        primitive_values::<UInt64Type>(&batches, 1)
            .iter()
            .all(|value| value.is_some_and(|value| (u64::from(u32::MAX)
                ..u64::from(u32::MAX) + 1024)
                .contains(&value)))
    );
}
