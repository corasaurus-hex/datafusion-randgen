#![cfg(feature = "column-choice-arrow-ipc")]

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_ipc::writer::{FileWriter, StreamWriter};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use datafusion_randgen::{all_udfs, column_choice_udf};

fn unique_path(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "datafusion_randgen_arrow_ipc_{test_name}_{}_{}",
        std::process::id(),
        nanos
    ))
}

fn batch(field: Field, column: ArrayRef) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![field]));
    RecordBatch::try_new(schema, vec![column]).unwrap()
}

fn write_arrow_file(test_name: &str, field: Field, column: ArrayRef) -> PathBuf {
    let path = unique_path(test_name);
    let batch = batch(field, column);
    let file = File::create(&path).unwrap();
    let mut writer = FileWriter::try_new(file, batch.schema_ref()).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    path
}

fn write_arrow_stream(test_name: &str, field: Field, column: ArrayRef) -> PathBuf {
    let path = unique_path(test_name);
    let batch = batch(field, column);
    let file = File::create(&path).unwrap();
    let mut writer = StreamWriter::try_new(file, batch.schema_ref()).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
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
async fn samples_distinct_non_null_uint32_values_from_arrow_file() {
    let path = write_arrow_file(
        "file_uint32_distinct",
        Field::new("id", DataType::UInt32, true),
        Arc::new(UInt32Array::from(vec![Some(7), Some(7), None, Some(99)])),
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
async fn samples_uint64_values_from_arrow_stream() {
    let path = write_arrow_stream(
        "stream_uint64_max",
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
async fn all_udfs_registers_column_choice_when_arrow_feature_is_enabled() {
    let path = write_arrow_file(
        "all_udfs_arrow",
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
async fn optional_null_probability_can_null_arrow_ipc_values() {
    let path = write_arrow_stream(
        "nullable_arrow_stream",
        Field::new("id", DataType::UInt32, false),
        Arc::new(UInt32Array::from(vec![1, 2, 3])),
    );
    let sql = format!(
        "SELECT randgen_column_choice({}, 'id', 1.0) AS id FROM generate_series(1, 8)",
        path_sql(&path)
    );

    let batches = collect_with_column_choice(&sql).await;

    assert_eq!(batches[0].schema().field(0).data_type(), &DataType::UInt32);
    for batch in batches {
        let column = batch.column(0);
        assert_eq!(column.null_count(), column.len());
    }
}

#[tokio::test]
async fn errors_when_arrow_source_column_type_is_unsupported() {
    let path = write_arrow_file(
        "unsupported_arrow_type",
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
async fn errors_when_file_format_is_not_supported() {
    let path = unique_path("not_supported");
    std::fs::write(&path, b"not a supported columnar source").unwrap();
    let sql = format!(
        "SELECT randgen_column_choice({}, 'id') AS id FROM generate_series(1, 1)",
        path_sql(&path)
    );

    let error = collect_error(&sql).await;

    assert!(error.contains("could not identify source format"));
}
