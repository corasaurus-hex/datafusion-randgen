#![cfg(feature = "column-choice-parquet")]

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_randgen::{all_udfs, column_choice_udf};
use parquet::arrow::ArrowWriter;

fn unique_path(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "datafusion_randgen_{test_name}_{}_{}",
        std::process::id(),
        nanos
    ))
}

fn write_parquet(test_name: &str, field: Field, column: ArrayRef) -> PathBuf {
    let path = unique_path(test_name);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn path_sql(path: &Path) -> String {
    sql_string(&path.to_string_lossy())
}

fn column_values_u32(batches: &[RecordBatch]) -> Vec<u32> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .iter()
                .map(Option::unwrap)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn column_values_u64(batches: &[RecordBatch]) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .iter()
                .map(Option::unwrap)
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn collect_with_column_choice(sql: &str) -> Vec<RecordBatch> {
    let ctx = SessionContext::new();
    ctx.register_udf(column_choice_udf());
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

async fn collect_error(sql: &str) -> String {
    let ctx = SessionContext::new();
    ctx.register_udf(column_choice_udf());
    match ctx.sql(sql).await {
        Ok(df) => match df.collect().await {
            Ok(_) => panic!("query unexpectedly succeeded"),
            Err(error) => error.to_string(),
        },
        Err(error) => error.to_string(),
    }
}

#[tokio::test]
async fn samples_distinct_non_null_uint32_values() {
    let path = write_parquet(
        "uint32_distinct",
        Field::new("id", DataType::UInt32, true),
        Arc::new(UInt32Array::from(vec![
            Some(7),
            Some(7),
            Some(7),
            Some(7),
            Some(7),
            Some(7),
            Some(7),
            Some(7),
            Some(7),
            Some(7),
            None,
            Some(99),
        ])),
    );
    let sql = format!(
        "SELECT randgen_column_choice({}, 'id') AS id FROM generate_series(1, 512)",
        path_sql(&path)
    );

    let batches = collect_with_column_choice(&sql).await;

    assert_eq!(batches[0].schema().field(0).data_type(), &DataType::UInt32);
    let values = column_values_u32(&batches);
    assert_eq!(values.len(), 512);
    assert!(values.iter().all(|value| [7, 99].contains(value)));
    let rare_value_count = values.iter().filter(|value| **value == 99).count();
    assert!(rare_value_count > 100);
}

#[tokio::test]
async fn samples_uint64_values_including_max() {
    let path = write_parquet(
        "uint64_max",
        Field::new("id", DataType::UInt64, true),
        Arc::new(UInt64Array::from(vec![Some(9), None, Some(u64::MAX)])),
    );
    let sql = format!(
        "SELECT randgen_column_choice({}, 'id') AS id FROM generate_series(1, 256)",
        path_sql(&path)
    );

    let batches = collect_with_column_choice(&sql).await;

    assert_eq!(batches[0].schema().field(0).data_type(), &DataType::UInt64);
    let values = column_values_u64(&batches);
    assert_eq!(values.len(), 256);
    assert!(values.iter().all(|value| [9, u64::MAX].contains(value)));
    assert_eq!(
        values.iter().copied().collect::<HashSet<_>>(),
        HashSet::from([9, u64::MAX])
    );
}

#[tokio::test]
async fn all_udfs_registers_column_choice_when_feature_is_enabled() {
    let path = write_parquet(
        "all_udfs",
        Field::new("id", DataType::UInt32, false),
        Arc::new(UInt32Array::from(vec![1, 2, 3])),
    );
    let sql = format!(
        "SELECT randgen_column_choice({}, 'id') AS id FROM generate_series(1, 8)",
        path_sql(&path)
    );
    let ctx = SessionContext::new();
    for udf in all_udfs() {
        ctx.register_udf(udf);
    }

    let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();

    assert_eq!(column_values_u32(&batches).len(), 8);
}

#[tokio::test]
async fn errors_when_source_column_is_all_null() {
    let path = write_parquet(
        "all_null",
        Field::new("id", DataType::UInt32, true),
        Arc::new(UInt32Array::from(vec![None::<u32>, None])),
    );
    let sql = format!(
        "SELECT randgen_column_choice({}, 'id') AS id FROM generate_series(1, 1)",
        path_sql(&path)
    );

    let error = collect_error(&sql).await;

    assert!(error.contains("requires at least one non-null source value"));
}

#[tokio::test]
async fn errors_when_source_column_is_missing() {
    let path = write_parquet(
        "missing_column",
        Field::new("id", DataType::UInt32, false),
        Arc::new(UInt32Array::from(vec![1, 2, 3])),
    );
    let sql = format!(
        "SELECT randgen_column_choice({}, 'missing') AS id FROM generate_series(1, 1)",
        path_sql(&path)
    );

    let error = collect_error(&sql).await;

    assert!(error.contains("could not find column missing"));
}

#[tokio::test]
async fn errors_when_source_column_type_is_unsupported() {
    let path = write_parquet(
        "unsupported_type",
        Field::new("id", DataType::Int64, false),
        Arc::new(Int64Array::from(vec![1, 2, 3])),
    );
    let sql = format!(
        "SELECT randgen_column_choice({}, 'id') AS id FROM generate_series(1, 1)",
        path_sql(&path)
    );

    let error = collect_error(&sql).await;

    assert!(error.contains("supports UInt32 and UInt64 columns"));
}

#[tokio::test]
async fn errors_when_source_path_is_not_scalar() {
    let path = write_parquet(
        "non_scalar_path",
        Field::new("id", DataType::UInt32, false),
        Arc::new(UInt32Array::from(vec![1, 2, 3])),
    );
    let sql = format!(
        "SELECT randgen_column_choice(path, 'id') AS id \
         FROM (SELECT {} AS path FROM generate_series(1, 1))",
        path_sql(&path)
    );

    let error = collect_error(&sql).await;

    assert!(error.contains("requires scalar source_path"));
}

#[tokio::test]
async fn errors_when_column_name_is_not_scalar() {
    let path = write_parquet(
        "non_scalar_column",
        Field::new("id", DataType::UInt32, false),
        Arc::new(UInt32Array::from(vec![1, 2, 3])),
    );
    let sql = format!(
        "SELECT randgen_column_choice({}, column_name) AS id \
         FROM (SELECT 'id' AS column_name FROM generate_series(1, 1))",
        path_sql(&path)
    );

    let error = collect_error(&sql).await;

    assert!(error.contains("requires scalar column_name"));
}

#[tokio::test]
async fn errors_when_scalar_arguments_are_null() {
    let path = write_parquet(
        "null_scalars",
        Field::new("id", DataType::UInt32, false),
        Arc::new(UInt32Array::from(vec![1, 2, 3])),
    );
    let null_path_sql = "SELECT randgen_column_choice(CAST(NULL AS VARCHAR), 'id') AS id \
         FROM generate_series(1, 1)"
        .to_owned();
    let null_column_sql = format!(
        "SELECT randgen_column_choice({}, CAST(NULL AS VARCHAR)) AS id \
         FROM generate_series(1, 1)",
        path_sql(&path)
    );

    let null_path_error = collect_error(&null_path_sql).await;
    let null_column_error = collect_error(&null_column_sql).await;

    assert!(
        null_path_error.contains("requires scalar source_path"),
        "{null_path_error}"
    );
    assert!(
        null_column_error.contains("requires scalar column_name"),
        "{null_column_error}"
    );
}
